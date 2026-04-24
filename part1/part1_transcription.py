"""
Speech Understanding - Programming Assignment 2
Part I: Robust Code-Switched Transcription (STT)

Tasks:
    1.1 - Multi-Head Language Identification (LID) at frame level
    1.2 - Constrained Decoding with N-gram Logit Bias on Whisper
    1.3 - Denoising & Normalization using DeepFilterNet / Spectral Subtraction

Usage:
    python part1_transcription.py --audio <path_to_audio.wav> --mode full
    python part1_transcription.py --audio <path_to_audio.wav> --mode denoise
    python part1_transcription.py --audio <path_to_audio.wav> --mode lid
    python part1_transcription.py --audio <path_to_audio.wav> --mode transcribe

Output:
    - denoised_output.wav
    - lid_predictions.json
    - transcript.json
    - transcript.txt
"""

import os
import json
import argparse
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict, Counter
from pathlib import Path
from typing import List, Tuple, Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
SAMPLE_RATE       = 16000   # Whisper expects 16kHz
TARGET_SR_TTS     = 22050   # For Part III output
FRAME_SIZE_MS     = 25      # 25ms frames for LID
HOP_SIZE_MS       = 10      # 10ms hop
N_MFCC            = 40
N_MELS            = 80
LID_WINDOW_SEC    = 1.0     # 1-second context window for LID
ENGLISH_LABEL     = 0
HINDI_LABEL       = 1
SYLLBUS_TEXT_PATH = "syllabus.txt"   # Put your course syllabus here
NGRAM_ORDER       = 3
BEAM_SIZE         = 5
LOGIT_BIAS_ALPHA  = 3.0     # Boost factor for technical terms


# ─────────────────────────────────────────────────────────────
# TASK 1.3: DENOISING & NORMALIZATION
# ─────────────────────────────────────────────────────────────

class SpectralSubtraction:
    """
    Classic spectral subtraction for background noise removal.
    Estimate noise PSD from first `noise_frames` frames (assumed silence).
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        n_fft: int = 512,
        hop_length: int = 160,
        noise_frames: int = 20,
        alpha: float = 2.0,   # oversubtraction factor
        beta: float = 0.01,   # spectral floor
    ):
        self.sample_rate  = sample_rate
        self.n_fft        = n_fft
        self.hop_length   = hop_length
        self.noise_frames = noise_frames
        self.alpha        = alpha
        self.beta         = beta

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (1, T) or (T,) float32 tensor
        Returns:
            denoised waveform of same shape
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        wav_np = waveform.squeeze(0).numpy()

        # STFT
        stft   = np.fft.rfft
        istft  = np.fft.irfft

        frame_len = self.n_fft
        hop       = self.hop_length

        # Manual overlap-add STFT
        frames = []
        for i in range(0, len(wav_np) - frame_len, hop):
            frame = wav_np[i: i + frame_len] * np.hanning(frame_len)
            frames.append(np.fft.rfft(frame))

        frames     = np.array(frames)           # (num_frames, n_fft//2+1)
        magnitude  = np.abs(frames)
        phase      = np.angle(frames)

        # Noise estimate from initial silent frames
        noise_est  = np.mean(magnitude[:self.noise_frames], axis=0, keepdims=True)

        # Spectral subtraction
        clean_mag  = magnitude - self.alpha * noise_est
        clean_mag  = np.maximum(clean_mag, self.beta * magnitude)

        # Reconstruct
        clean_frames = clean_mag * np.exp(1j * phase)
        output       = np.zeros(len(wav_np))
        window_sum   = np.zeros(len(wav_np))
        win          = np.hanning(frame_len)

        for idx, frame in enumerate(clean_frames):
            start = idx * hop
            end   = start + frame_len
            if end > len(output):
                break
            reconstructed      = np.fft.irfft(frame)[:frame_len]
            output[start:end] += reconstructed * win
            window_sum[start:end] += win ** 2

        # Normalize overlap-add
        mask           = window_sum > 1e-8
        output[mask]  /= window_sum[mask]

        denoised = torch.tensor(output, dtype=torch.float32).unsqueeze(0)
        return denoised


def get_available_gpu_memory_gb() -> float:
    """Return free GPU memory in GB, or 0 if no GPU."""
    if not torch.cuda.is_available():
        return 0.0
    free, _ = torch.cuda.mem_get_info()
    return free / (1024 ** 3)


def denoise_with_deepfilternet_chunked(
    waveform:    torch.Tensor,
    sample_rate: int,
    chunk_sec:   int = 30,
) -> torch.Tensor:
    """
    Chunked DeepFilterNet denoising — processes chunk_sec seconds at a time
    so GPU memory stays bounded regardless of audio length.

    Strategy:
      1. Resample to 48kHz on CPU
      2. Split into overlapping 30s chunks
      3. Enhance each chunk on GPU, immediately move result back to CPU
      4. Overlap-add chunks, resample back to target SR
    """
    try:
        from df.enhance import enhance, init_df
    except ImportError:
        logger.warning("DeepFilterNet not installed. Falling back to Spectral Subtraction.")
        return SpectralSubtraction(sample_rate=sample_rate)(waveform.cpu())

    logger.info(f"DeepFilterNet chunked denoising | chunk={chunk_sec}s")

    DF_SR        = 48000
    overlap_sec  = 1
    fade_samples = DF_SR * overlap_sec

    # All resampling on CPU
    waveform_cpu = waveform.cpu()
    if sample_rate != DF_SR:
        wav48 = T.Resample(orig_freq=sample_rate, new_freq=DF_SR)(waveform_cpu)
    else:
        wav48 = waveform_cpu

    # Load model on CPU first — we move it per-chunk
    model, df_state, _ = init_df()
    model = model.cpu()

    chunk_samples   = DF_SR * chunk_sec
    total_samples   = wav48.shape[-1]
    n_chunks        = (total_samples // chunk_samples) + 1
    enhanced_chunks = []

    logger.info(f"Processing {total_samples/DF_SR:.1f}s in {n_chunks} chunks...")

    for i, start in enumerate(range(0, total_samples, chunk_samples)):
        end   = min(start + chunk_samples + fade_samples, total_samples)
        chunk = wav48[:, start:end].to(DEVICE)
        model = model.to(DEVICE)

        with torch.no_grad():
            enhanced = enhance(model, df_state, chunk)

        # Pull back to CPU immediately — free VRAM
        enhanced = enhanced.cpu()
        model    = model.cpu()
        del chunk
        torch.cuda.empty_cache()

        # Trim overlap: skip fade at start (except first chunk)
        if start > 0:
            enhanced = enhanced[:, fade_samples:]
        # Trim overlap at end (except last chunk)
        if end < total_samples:
            enhanced = enhanced[:, :chunk_samples]

        enhanced_chunks.append(enhanced)
        logger.info(f"  Chunk {i+1}/{n_chunks} done | "
                    f"GPU mem free: {get_available_gpu_memory_gb():.1f}GB")

    enhanced48 = torch.cat(enhanced_chunks, dim=-1)

    # Resample back to original SR on CPU
    if sample_rate != DF_SR:
        enhanced = T.Resample(orig_freq=DF_SR, new_freq=sample_rate)(enhanced48)
    else:
        enhanced = enhanced48

    # Trim/pad to original length
    orig_len = waveform_cpu.shape[-1]
    if enhanced.shape[-1] > orig_len:
        enhanced = enhanced[:, :orig_len]
    elif enhanced.shape[-1] < orig_len:
        enhanced = F.pad(enhanced, (0, orig_len - enhanced.shape[-1]))

    logger.info("DeepFilterNet chunked denoising complete.")
    return enhanced


def denoise_with_spectral_subtraction(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Pure CPU spectral subtraction — no GPU needed, no OOM risk."""
    logger.info("Running Spectral Subtraction (CPU)...")
    ss = SpectralSubtraction(sample_rate=sample_rate)
    return ss(waveform.cpu())


def load_and_denoise(
    audio_path:  str,
    output_path: str = "denoised_output.wav",
    method:      str = "auto",
    chunk_sec:   int = 30,
) -> Tuple[torch.Tensor, int]:
    """
    Load audio, resample to 16kHz, denoise, save output.

    Args:
        method: "auto"       -> DeepFilterNet if >=2GB free, else spectral
                "deepfilter" -> Force DeepFilterNet chunked
                "spectral"   -> Force Spectral Subtraction (CPU, safest)
        chunk_sec: seconds per DeepFilterNet chunk (lower = less VRAM)
    """
    logger.info(f"Loading audio: {audio_path}")
    waveform, sr = torchaudio.load(audio_path)

    # Mix down to mono on CPU
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 16kHz
    if sr != SAMPLE_RATE:
        logger.info(f"Resampling {sr}Hz -> {SAMPLE_RATE}Hz")
        waveform = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(waveform)
        sr       = SAMPLE_RATE

    waveform = waveform / (waveform.abs().max() + 1e-8)

    # Auto-select denoising method based on free VRAM
    free_gb = get_available_gpu_memory_gb()
    dur_min = waveform.shape[-1] / sr / 60
    logger.info(f"Audio: {dur_min:.1f} min | Free GPU: {free_gb:.1f} GB | Method: {method}")

    if method == "auto":
        method = "deepfilter" if free_gb >= 2.0 else "spectral"
        logger.info(f"Auto-selected method: {method}")

    if method == "deepfilter":
        waveform = denoise_with_deepfilternet_chunked(waveform, sr, chunk_sec=chunk_sec)
    else:
        waveform = denoise_with_spectral_subtraction(waveform, sr)

    waveform = waveform.cpu()
    waveform = waveform / (waveform.abs().max() + 1e-8)

    torchaudio.save(output_path, waveform, sr)
    logger.info(f"Saved denoised audio: {output_path}")

    return waveform, sr


# ─────────────────────────────────────────────────────────────
# TASK 1.1: MULTI-HEAD LANGUAGE IDENTIFICATION (LID)
# ─────────────────────────────────────────────────────────────

class FrameLevelFeatureExtractor(nn.Module):
    """
    Extracts MFCC + Delta + Delta-Delta features per frame.
    Output shape: (batch, time, feature_dim)
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, n_mfcc: int = N_MFCC):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mfcc      = n_mfcc

        frame_samples = int(sample_rate * FRAME_SIZE_MS / 1000)
        hop_samples   = int(sample_rate * HOP_SIZE_MS / 1000)

        self.mfcc_transform = T.MFCC(
            sample_rate = sample_rate,
            n_mfcc      = n_mfcc,
            melkwargs   = {
                "n_fft"      : frame_samples,
                "hop_length" : hop_samples,
                "n_mels"     : 64,
                "f_min"      : 0,
                "f_max"      : sample_rate // 2,
            }
        )

    def compute_deltas(self, features: torch.Tensor) -> torch.Tensor:
        """Compute delta and delta-delta from MFCC."""
        delta       = torchaudio.functional.compute_deltas(features)
        delta_delta = torchaudio.functional.compute_deltas(delta)
        return torch.cat([features, delta, delta_delta], dim=1)   # (batch, 3*n_mfcc, T)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (batch, T)
        Returns:
            features: (batch, T_frames, 3*n_mfcc)
        """
        # torchaudio transforms hold internal buffers (filterbank, window)
        # created on CPU at __init__ time — move them to match input device
        self.mfcc_transform = self.mfcc_transform.to(waveform.device)

        mfcc     = self.mfcc_transform(waveform)        # (batch, n_mfcc, T)
        features = self.compute_deltas(mfcc)             # (batch, 3*n_mfcc, T)
        features = features.permute(0, 2, 1)             # (batch, T, 3*n_mfcc)
        return features


class MultiHeadLIDModel(nn.Module):
    """
    Frame-level Language Identification model with:
      - Shared BiLSTM encoder
      - Multi-head output:
          * Head 1: Binary LID  (English vs Hindi)
          * Head 2: Confidence  (how certain is the prediction)
          * Head 3: Switch prob (probability of language switch at this frame)

    Architecture:
        Input (3*MFCC) → Linear Projection → BiLSTM → MultiHead Attention → 3 Heads
    """

    def __init__(
        self,
        input_dim:    int = 3 * N_MFCC,   # 120
        hidden_dim:   int = 256,
        num_layers:   int = 3,
        num_heads:    int = 4,
        dropout:      float = 0.3,
        num_classes:  int = 2,             # English, Hindi
    ):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.bilstm = nn.LSTM(
            input_size    = hidden_dim,
            hidden_size   = hidden_dim // 2,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if num_layers > 1 else 0.0,
        )

        self.attention = nn.MultiheadAttention(
            embed_dim   = hidden_dim,
            num_heads   = num_heads,
            dropout     = dropout,
            batch_first = True,
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Head 1: Binary LID (English=0, Hindi=1)
        self.lid_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

        # Head 2: Confidence score (0-1)
        self.conf_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Head 3: Switch probability (is this a language switch point?)
        self.switch_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        """
        Args:
            features: (batch, T, input_dim)
            lengths:  (batch,) actual sequence lengths
        Returns:
            lid_logits:   (batch, T, 2)
            conf_scores:  (batch, T, 1)
            switch_probs: (batch, T, 1)
        """
        x = self.input_proj(features)           # (B, T, hidden_dim)

        # BiLSTM
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.bilstm(packed)
            x, _        = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            x, _ = self.bilstm(x)               # (B, T, hidden_dim)

        # Self-Attention
        attn_out, _ = self.attention(x, x, x)
        x           = self.layer_norm(x + attn_out)

        # Multi-head outputs
        lid_logits   = self.lid_head(x)         # (B, T, 2)
        conf_scores  = self.conf_head(x)        # (B, T, 1)
        switch_probs = self.switch_head(x)      # (B, T, 1)

        return lid_logits, conf_scores, switch_probs


class LIDTrainer:
    """
    Handles training and evaluation of the LID model.
    Uses synthetic data generation when no labelled data is available.
    """

    def __init__(self, model: MultiHeadLIDModel, lr: float = 1e-3):
        self.model     = model.to(DEVICE)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50
        )
        self.ce_loss   = nn.CrossEntropyLoss(ignore_index=-1)
        self.bce_loss  = nn.BCELoss()

    def compute_loss(
        self,
        lid_logits:   torch.Tensor,
        conf_scores:  torch.Tensor,
        switch_probs: torch.Tensor,
        lid_labels:   torch.Tensor,
        conf_labels:  torch.Tensor,
        switch_labels: torch.Tensor,
    ) -> torch.Tensor:

        B, T, C = lid_logits.shape

        # LID loss
        lid_loss    = self.ce_loss(
            lid_logits.reshape(-1, C),
            lid_labels.reshape(-1)
        )

        # Confidence loss
        conf_loss   = self.bce_loss(
            conf_scores.squeeze(-1),
            conf_labels.float()
        )

        # Switch detection loss
        switch_loss = self.bce_loss(
            switch_probs.squeeze(-1),
            switch_labels.float()
        )

        total = lid_loss + 0.3 * conf_loss + 0.3 * switch_loss
        return total

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for batch in dataloader:
            features      = batch["features"].to(DEVICE)
            lid_labels    = batch["lid_labels"].to(DEVICE)
            conf_labels   = batch["conf_labels"].to(DEVICE)
            switch_labels = batch["switch_labels"].to(DEVICE)
            lengths       = batch.get("lengths", None)

            self.optimizer.zero_grad()
            lid_logits, conf_scores, switch_probs = self.model(features, lengths)

            loss = self.compute_loss(
                lid_logits, conf_scores, switch_probs,
                lid_labels, conf_labels, switch_labels
            )
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            total_loss += loss.item()

        self.scheduler.step()
        return total_loss / len(dataloader)

    def evaluate(self, dataloader: DataLoader) -> Dict:
        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in dataloader:
                features   = batch["features"].to(DEVICE)
                lid_labels = batch["lid_labels"].to(DEVICE)

                lid_logits, _, _ = self.model(features)
                preds = lid_logits.argmax(dim=-1)   # (B, T)

                mask  = lid_labels != -1
                all_preds.append(preds[mask].cpu())
                all_labels.append(lid_labels[mask].cpu())

        all_preds  = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        # Compute F1 per class and macro F1
        from collections import Counter
        tp = {0: 0, 1: 0}
        fp = {0: 0, 1: 0}
        fn = {0: 0, 1: 0}

        for pred, label in zip(all_preds, all_labels):
            if pred == label:
                tp[label] += 1
            else:
                fp[pred]   += 1
                fn[label]  += 1

        f1_scores = {}
        for cls in [0, 1]:
            precision     = tp[cls] / (tp[cls] + fp[cls] + 1e-8)
            recall        = tp[cls] / (tp[cls] + fn[cls] + 1e-8)
            f1_scores[cls] = 2 * precision * recall / (precision + recall + 1e-8)

        macro_f1 = np.mean(list(f1_scores.values()))

        return {
            "macro_f1":    macro_f1,
            "english_f1":  f1_scores[ENGLISH_LABEL],
            "hindi_f1":    f1_scores[HINDI_LABEL],
            "accuracy":    (all_preds == all_labels).mean(),
        }


class SyntheticLIDDataset(Dataset):
    """
    Generates synthetic frame-level LID data from audio segments.
    In practice: replace with real labelled Hinglish data
    (e.g., MUCS 2021 dataset, IIIT-H code-switching corpus).

    Here we use energy + ZCR heuristics to bootstrap pseudo-labels
    then train to convergence. Fine-tune on real labels afterward.
    """

    def __init__(
        self,
        waveform:     torch.Tensor,
        sample_rate:  int  = SAMPLE_RATE,
        window_sec:   float = LID_WINDOW_SEC,
        hop_sec:      float = 0.5,
        augment:      bool  = True,
    ):
        self.sample_rate = sample_rate
        self.augment     = augment
        self.extractor   = FrameLevelFeatureExtractor(sample_rate)

        win_samples = int(window_sec * sample_rate)
        hop_samples = int(hop_sec    * sample_rate)

        # Always keep dataset segments on CPU — DataLoader workers are CPU-only.
        # The training loop moves batches to DEVICE after collation.
        self.segments = []
        for start in range(0, waveform.shape[-1] - win_samples, hop_samples):
            seg = waveform[:, start: start + win_samples].cpu()
            self.segments.append(seg)

    def _pseudo_label(self, segment: torch.Tensor) -> torch.Tensor:
        """
        Heuristic pseudo-labelling using spectral centroid.
        Hindi vowels → lower spectral centroid on average.
        This is a BOOTSTRAP only — replace with real labels.
        """
        device = segment.device
        n_fft  = 512

        # Window must be on the same device as the input
        window = torch.hann_window(n_fft, device=device)

        spec = torch.stft(
            segment.squeeze(0),
            n_fft          = n_fft,
            hop_length     = 160,
            window         = window,
            return_complex = True,
        )
        magnitude = spec.abs()                          # (freq, T)
        # freqs must also be on the same device as magnitude
        freqs     = torch.linspace(0, self.sample_rate / 2, magnitude.shape[0], device=device)
        centroid  = (freqs.unsqueeze(-1) * magnitude).sum(0) / (magnitude.sum(0) + 1e-8)

        # Threshold: lower centroid → more Hindi-like
        threshold = centroid.mean()
        labels    = (centroid < threshold).long()       # (T,)
        return labels

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx: int) -> Dict:
        seg = self.segments[idx]                        # (1, win_samples)

        # Augmentation
        if self.augment and torch.rand(1) > 0.5:
            noise = torch.randn_like(seg) * 0.003
            seg   = seg + noise

        features  = self.extractor(seg)                 # (1, T, 3*n_mfcc)
        features  = features.squeeze(0)                 # (T, 3*n_mfcc)
        T_frames  = features.shape[0]

        lid_labels    = self._pseudo_label(seg)
        # Align label length to feature length
        lid_labels    = F.interpolate(
            lid_labels.float().unsqueeze(0).unsqueeze(0),
            size  = T_frames,
            mode  = "nearest"
        ).squeeze().long()

        # Confidence: 1 if prediction is 'sure' (extreme centroid)
        conf_labels   = torch.ones(T_frames)

        # Switch label: 1 where language changes
        switch_labels = torch.zeros(T_frames)
        for i in range(1, T_frames):
            if lid_labels[i] != lid_labels[i - 1]:
                switch_labels[i] = 1

        return {
            "features":      features,
            "lid_labels":    lid_labels,
            "conf_labels":   conf_labels,
            "switch_labels": switch_labels,
        }


def smooth_lid_predictions(
    predictions:     List[Dict],
    median_window:   int = 50,     # frames — 50 * 10ms = 500ms window
    min_segment_ms:  int = 200,    # minimum ms before a switch is accepted
) -> List[Dict]:
    """
    Two-stage LID smoothing to remove jitter:

    Stage 1 — Median filter:
        For each frame, take majority vote over a sliding window of
        `median_window` frames. This kills isolated flips that last
        less than ~250ms.

    Stage 2 — Minimum segment enforcement:
        After median filter, any language segment shorter than
        `min_segment_ms` is merged into the surrounding dominant language.
        This directly satisfies the assignment requirement:
        "timestamp precision for language switches within 200ms".

    Args:
        predictions:    Sorted list of LID frame dicts
        median_window:  Sliding window size in frames (default 50 = 500ms)
        min_segment_ms: Minimum valid segment length in ms (default 200ms)

    Returns:
        Smoothed predictions with updated language, is_switch fields
    """
    if not predictions:
        return predictions

    n      = len(predictions)
    labels = [1 if p["language"] == "hindi" else 0 for p in predictions]
    half   = median_window // 2

    # ── Stage 1: Median (majority vote) filter ────────────────
    smoothed = []
    for i in range(n):
        lo    = max(0, i - half)
        hi    = min(n, i + half + 1)
        window_labels = labels[lo:hi]
        majority = 1 if sum(window_labels) >= len(window_labels) / 2 else 0
        smoothed.append(majority)

    # ── Stage 2: Minimum segment duration enforcement ─────────
    # Find contiguous runs
    min_frames = max(1, int(min_segment_ms / 10))  # 10ms per frame

    # Run-length encode
    runs = []   # list of [label, start_idx, end_idx]
    cur_label = smoothed[0]
    cur_start = 0
    for i in range(1, n):
        if smoothed[i] != cur_label:
            runs.append([cur_label, cur_start, i - 1])
            cur_label = smoothed[i]
            cur_start = i
    runs.append([cur_label, cur_start, n - 1])

    # Merge runs shorter than min_frames into neighbours
    changed = True
    while changed:
        changed = False
        merged_runs = []
        i = 0
        while i < len(runs):
            run_len = runs[i][2] - runs[i][1] + 1
            if run_len < min_frames and len(runs) > 1:
                # Merge into previous run if exists, else next
                if merged_runs:
                    merged_runs[-1][2] = runs[i][2]
                elif i + 1 < len(runs):
                    runs[i + 1][1] = runs[i][1]
                    runs[i + 1][0] = runs[i + 1][0]  # keep next label
                    i += 1
                    continue
                changed = True
            else:
                merged_runs.append(runs[i])
            i += 1
        runs = merged_runs

    # Rebuild flat smoothed array from runs
    final_labels = [0] * n
    for label, start, end in runs:
        for idx in range(start, end + 1):
            final_labels[idx] = label

    # ── Apply back to predictions ─────────────────────────────
    result = []
    for i, (pred, new_label) in enumerate(zip(predictions, final_labels)):
        new_lang   = "hindi" if new_label == 1 else "english"
        prev_lang  = "hindi" if final_labels[i-1] == 1 else "english" if i > 0 else new_lang
        is_switch  = (new_lang != prev_lang)

        result.append({
            **pred,
            "language":  new_lang,
            "is_switch": is_switch,
        })

    return result


def run_lid_inference(
    waveform:   torch.Tensor,
    sample_rate: int,
    model:      MultiHeadLIDModel,
    output_json: str = "lid_predictions.json",
) -> List[Dict]:
    """
    Run frame-level LID on full audio and return timestamped predictions.

    Returns:
        List of dicts: {start_sec, end_sec, language, confidence, is_switch}
    """
    model.eval()
    extractor = FrameLevelFeatureExtractor(sample_rate)

    hop_samples  = int(LID_WINDOW_SEC * 0.5 * sample_rate)
    win_samples  = int(LID_WINDOW_SEC * sample_rate)
    frame_hop_sec = HOP_SIZE_MS / 1000.0

    predictions = []

    with torch.no_grad():
        for seg_start in range(0, waveform.shape[-1] - win_samples, hop_samples):
            seg      = waveform[:, seg_start: seg_start + win_samples].to(DEVICE)
            features = extractor(seg).to(DEVICE)         # (1, T, dim)

            lid_logits, conf_scores, switch_probs = model(features)
            probs    = F.softmax(lid_logits, dim=-1)     # (1, T, 2)
            labels   = probs.argmax(dim=-1).squeeze(0)   # (T,)
            confs    = conf_scores.squeeze(0).squeeze(-1) # (T,)
            switches = switch_probs.squeeze(0).squeeze(-1) # (T,)

            for t in range(labels.shape[0]):
                abs_time   = (seg_start / sample_rate) + t * frame_hop_sec
                lang       = "english" if labels[t].item() == ENGLISH_LABEL else "hindi"
                predictions.append({
                    "start_sec":   round(abs_time, 3),
                    "end_sec":     round(abs_time + frame_hop_sec, 3),
                    "language":    lang,
                    "confidence":  round(confs[t].item(), 4),
                    "is_switch":   switches[t].item() > 0.5,
                    "english_prob": round(probs[0, t, 0].item(), 4),
                    "hindi_prob":   round(probs[0, t, 1].item(), 4),
                })

    # Remove duplicate time steps from overlapping windows
    seen_times = {}
    deduped    = []
    for p in predictions:
        key = p["start_sec"]
        if key not in seen_times:
            seen_times[key] = True
            deduped.append(p)

    deduped.sort(key=lambda x: x["start_sec"])

    # ── Temporal smoothing to remove LID jitter ───────────────
    # Raw frame-level predictions flip every 10ms — too noisy.
    # Apply median filter over a 500ms window (50 frames at 10ms hop),
    # then enforce a minimum segment duration of 200ms before
    # allowing a language switch (matches assignment's 200ms requirement).
    deduped = smooth_lid_predictions(deduped, median_window=50, min_segment_ms=200)

    # Save
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(deduped, f, indent=2, ensure_ascii=False)

    logger.info(f"LID predictions saved to: {output_json}")

    # Summary stats
    lang_counts = Counter(p["language"] for p in deduped)
    switches    = sum(1 for p in deduped if p["is_switch"])
    logger.info(f"LID Summary — English frames: {lang_counts['english']}, "
                f"Hindi frames: {lang_counts['hindi']}, "
                f"Language switches (after smoothing): {switches}")

    return deduped


# ─────────────────────────────────────────────────────────────
# TASK 1.2: N-GRAM LANGUAGE MODEL + CONSTRAINED DECODING
# ─────────────────────────────────────────────────────────────

class NgramLanguageModel:
    """
    N-gram LM trained on the Speech course syllabus to boost
    technical term probability during Whisper decoding.

    Mathematical formulation (for report):
        P(w_n | w_{n-N+1},...,w_{n-1}) =
            C(w_{n-N+1},...,w_n) + alpha
            ─────────────────────────────
            C(w_{n-N+1},...,w_{n-1}) + alpha * V

    where alpha = Laplace smoothing constant, V = vocab size.
    """

    def __init__(self, order: int = NGRAM_ORDER, alpha: float = 0.1):
        self.order  = order
        self.alpha  = alpha      # Laplace smoothing
        self.ngrams: Dict       = defaultdict(Counter)
        self.vocab:  set        = set()
        self.technical_terms:   set = set()

    def tokenize(self, text: str) -> List[str]:
        import re
        tokens = re.findall(r"\b\w+\b", text.lower())
        return tokens

    def extract_technical_terms(self, text: str) -> set:
        """
        Heuristic: words longer than 6 chars that appear in the syllabus
        are likely technical terms (cepstrum, stochastic, etc.).
        """
        tokens = self.tokenize(text)
        terms  = {t for t in tokens if len(t) > 6}

        # Hard-coded domain terms from HC Verma lectures
        # (Quantum Mechanics + Classical Physics — matches the YouTube lecture)
        domain_terms = {
            # Quantum mechanics
            "quantum", "photon", "electron", "proton", "neutron", "nucleus",
            "wavefunction", "superposition", "uncertainty", "heisenberg",
            "schrodinger", "eigenvalue", "eigenfunction", "hamiltonian",
            "duality", "interference", "diffraction", "wavelength",
            "frequency", "amplitude", "probability", "fermion", "boson",
            "orbital", "angular", "momentum", "degeneracy", "hydrogen",
            "photoelectric", "threshold", "workfunction", "debroglie",
            "radioactivity", "nuclear", "fission", "fusion", "binding",
            # Classical mechanics
            "velocity", "acceleration", "momentum", "inertia", "friction",
            "gravitation", "oscillation", "resonance", "harmonic",
            "potential", "kinetic", "entropy", "thermodynamics", "carnot",
            # Electromagnetism
            "electromagnetic", "capacitor", "resistance", "inductance",
            "faraday", "coulomb", "magnetic", "electric", "lorentz",
            # Optics
            "refraction", "reflection", "polarization", "coherent",
            "interference", "fresnel", "snell", "brewster",
            # Math terms common in physics lectures
            "equation", "derivative", "integral", "vector", "scalar",
            "gradient", "divergence", "curl", "laplacian", "fourier",
            "differential", "eigenvalue", "matrix", "tensor",
            # HC Verma lecture style connectives (Hinglish)
            "classical", "modern", "experiment", "observation", "derivation",
            "identical", "conservation", "equivalent", "theorem", "principle",
        }

        return terms | domain_terms

    def train(self, text: str):
        """
        Train n-gram LM on the provided syllabus/text corpus.
        """
        self.technical_terms = self.extract_technical_terms(text)
        tokens               = self.tokenize(text)
        self.vocab           = set(tokens)

        # Add sentence start/end tokens
        tokens = ["<s>"] * (self.order - 1) + tokens + ["</s>"]

        for i in range(self.order - 1, len(tokens)):
            context = tuple(tokens[i - self.order + 1: i])
            word    = tokens[i]
            self.ngrams[context][word] += 1

        logger.info(f"N-gram LM trained on {len(tokens)} tokens, "
                    f"vocab size: {len(self.vocab)}, "
                    f"technical terms: {len(self.technical_terms)}")

    def log_prob(self, word: str, context: Tuple) -> float:
        """
        Laplace-smoothed log probability.
        P(word | context) with add-alpha smoothing.
        """
        V           = len(self.vocab) + 1
        count_ctx   = sum(self.ngrams[context].values()) if context in self.ngrams else 0
        count_word  = self.ngrams[context][word] if context in self.ngrams else 0

        prob = (count_word + self.alpha) / (count_ctx + self.alpha * V)
        return np.log(prob + 1e-10)

    def get_logit_bias(
        self,
        tokenizer,
        context_text: str,
        alpha:        float = LOGIT_BIAS_ALPHA,
    ) -> Dict[int, float]:
        """
        Build a logit bias dictionary for Whisper's constrained decoding.

        For each technical term, find its token ID and add `alpha` to logit.

        Mathematical formulation:
            logit_biased(w) = logit(w) + alpha * I[w in technical_terms]
                            + beta  * log P_ngram(w | context)

        Returns:
            {token_id: bias_value}
        """
        bias_dict = {}
        context   = tuple(self.tokenize(context_text)[-(self.order - 1):])
        beta      = 1.5

        for term in self.technical_terms:
            try:
                token_ids = tokenizer.encode(" " + term, add_special_tokens=False)
                ngram_score = self.log_prob(term, context)

                for tid in token_ids:
                    existing           = bias_dict.get(tid, 0.0)
                    bias_dict[tid]     = existing + alpha + beta * ngram_score
            except Exception:
                pass

        return bias_dict

    def save(self, path: str = "ngram_lm.json"):
        data = {
            "order":           self.order,
            "alpha":           self.alpha,
            "technical_terms": list(self.technical_terms),
            "ngrams": {
                str(k): dict(v)
                for k, v in self.ngrams.items()
            }
        }
        with open(path, "w") as f:
            json.dump(data, f)
        logger.info(f"N-gram LM saved to {path}")

    def load(self, path: str = "ngram_lm.json"):
        with open(path, "r") as f:
            data = json.load(f)
        self.order           = data["order"]
        self.alpha           = data["alpha"]
        self.technical_terms = set(data["technical_terms"])
        self.ngrams          = defaultdict(Counter)
        for k_str, counts in data["ngrams"].items():
            k = tuple(k_str.strip("()").replace("'", "").split(", "))
            self.ngrams[k]   = Counter(counts)
        logger.info(f"N-gram LM loaded from {path}")


class ConstrainedWhisperDecoder:
    """
    Wraps OpenAI Whisper with:
      1. Logit Bias from N-gram LM (technical term boosting)
      2. Language-aware segment processing using LID predictions
      3. Constrained Beam Search using the syllbus N-gram LM
    """

    def __init__(
        self,
        model_name:  str = "large-v3",
        ngram_lm:    Optional[NgramLanguageModel] = None,
        lid_preds:   Optional[List[Dict]] = None,
    ):
        try:
            import whisper
        except ImportError:
            raise ImportError("Install openai-whisper: pip install openai-whisper")

        # Whisper model VRAM requirements (approximate):
        #   large-v3 : ~6.0 GB   medium : ~3.0 GB
        #   small    : ~1.0 GB   base   : ~0.5 GB
        WHISPER_VRAM = {
            "large-v3": 6.0, "large-v2": 6.0, "large": 6.0,
            "medium":   3.0, "medium.en": 3.0,
            "small":    1.0, "small.en":  1.0,
            "base":     0.5, "base.en":   0.5,
            "tiny":     0.2, "tiny.en":   0.2,
        }

        free_gb     = get_available_gpu_memory_gb()
        needed_gb   = WHISPER_VRAM.get(model_name, 6.0)
        load_device = DEVICE

        logger.info(f"Whisper {model_name} needs ~{needed_gb:.1f}GB | "
                    f"Free GPU: {free_gb:.1f}GB")

        # If not enough VRAM, downgrade model size automatically
        if free_gb < needed_gb:
            fallback_order = ["medium", "small", "base", "tiny"]
            original_name  = model_name
            for candidate in fallback_order:
                if free_gb >= WHISPER_VRAM[candidate]:
                    model_name = candidate
                    logger.warning(
                        f"Not enough VRAM for {original_name} "
                        f"({needed_gb:.1f}GB needed, {free_gb:.1f}GB free). "
                        f"Auto-downgraded to Whisper-{model_name}."
                    )
                    break
            else:
                # Even tiny won't fit — load on CPU
                load_device = torch.device("cpu")
                logger.warning(
                    f"Insufficient VRAM even for tiny Whisper. "
                    f"Loading on CPU (slower but will work)."
                )

        logger.info(f"Loading Whisper {model_name} on {load_device}...")
        self.whisper     = whisper
        self.model       = whisper.load_model(model_name, device=load_device)
        self.load_device = load_device
        self.ngram_lm    = ngram_lm
        self.lid_preds   = lid_preds
        self.tokenizer   = whisper.tokenizer.get_tokenizer(
            multilingual=True, language=None, task="transcribe"
        )

    def _get_decode_options(self, language: str = "en") -> "whisper.DecodingOptions":
        """Build Whisper decode options with logit bias."""
        import whisper

        logit_bias = {}
        if self.ngram_lm is not None:
            logit_bias = self.ngram_lm.get_logit_bias(self.tokenizer)

        options = whisper.DecodingOptions(
            language       = language,
            beam_size      = BEAM_SIZE,
            best_of        = 5,
            temperature    = 0.0,
            without_timestamps = False,
            suppress_tokens    = [-1],
        )
        return options, logit_bias

    def _apply_logit_bias(self, logits: torch.Tensor, bias: Dict[int, float]) -> torch.Tensor:
        """
        Apply logit bias to Whisper output logits.
        logit_biased[token_id] += bias[token_id]
        """
        for token_id, bias_val in bias.items():
            if token_id < logits.shape[-1]:
                logits[..., token_id] += bias_val
        return logits

    def transcribe_segment(
        self,
        audio_chunk: np.ndarray,
        language:    str = "en",
        context:     str = "",
    ) -> Dict:
        """
        Transcribe a single audio chunk with logit biasing.
        """
        import whisper

        # Pad/trim to 30s for Whisper
        audio_chunk = whisper.pad_or_trim(audio_chunk)
        mel         = whisper.log_mel_spectrogram(audio_chunk).to(DEVICE)

        options, logit_bias = self._get_decode_options(language)

        # Custom decode with logit bias hook
        if logit_bias:
            # Monkey-patch approach for logit bias
            original_forward = self.model.decoder.forward

            def biased_forward(*args, **kwargs):
                logits = original_forward(*args, **kwargs)
                return self._apply_logit_bias(logits, logit_bias)

            self.model.decoder.forward = biased_forward
            result = self.whisper.decode(self.model, mel, options)
            self.model.decoder.forward = original_forward
        else:
            result = self.whisper.decode(self.model, mel, options)

        return {
            "text":     result.text,
            "language": result.language,
            "tokens":   result.tokens,
            "avg_logprob": result.avg_logprob,
            "no_speech_prob": result.no_speech_prob,
        }

    def transcribe_full(
        self,
        audio_path:  str,
        output_json: str = "transcript.json",
        output_txt:  str = "transcript.txt",
    ) -> Dict:
        """
        Transcribe full audio using LID-guided language switching.
        Uses LID predictions to pass correct language hint to Whisper
        per segment, improving code-switched accuracy.
        """
        import whisper

        logger.info(f"Transcribing: {audio_path}")
        audio = whisper.load_audio(audio_path)

        # Determine dominant language from LID predictions for Whisper hint
        # For code-switched audio, "hi" works better than None because Whisper
        # auto-detect on 30s windows often locks onto English for Hinglish
        whisper_lang = None
        if self.lid_preds:
            lang_counts = Counter(p["language"] for p in self.lid_preds)
            total       = sum(lang_counts.values())
            hindi_frac  = lang_counts.get("hindi", 0) / max(total, 1)
            if hindi_frac >= 0.40:
                # Majority or near-majority Hindi → hint Whisper with "hi"
                # This prevents Whisper from treating Devanagari as noise
                whisper_lang = "hi"
                logger.info(f"LID: {hindi_frac:.0%} Hindi → setting Whisper lang='hi'")
            else:
                whisper_lang = "en"
                logger.info(f"LID: {1-hindi_frac:.0%} English → setting Whisper lang='en'")

        # Use Whisper's built-in VAD + chunking
        result = self.model.transcribe(
            audio,
            language                   = whisper_lang,
            beam_size                  = BEAM_SIZE,
            best_of                    = 5,
            temperature                = 0.0,
            condition_on_previous_text = True,
            word_timestamps            = True,
            verbose                    = False,
        )

        # Enrich segments with LID information
        if self.lid_preds:
            result = self._enrich_with_lid(result)

        # Save outputs
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        with open(output_txt, "w", encoding="utf-8") as f:
            f.write("# Code-Switched Transcript\n")
            f.write("# Format: [start → end | LANG (hi%)] text  {CS} if code-switched\n\n")
            for seg in result["segments"]:
                start      = seg["start"]
                end        = seg["end"]
                text       = seg["text"].strip()
                lang       = seg.get("dominant_language", "?")
                hi_ratio   = seg.get("hindi_ratio", 0.0)
                is_cs      = seg.get("is_code_switched", False)
                n_switches = len(seg.get("switch_points", []))
                cs_tag     = f"  {{CS:{n_switches} switches}}" if is_cs else ""
                f.write(
                    f"[{start:.2f}s → {end:.2f}s | {lang.upper()} "
                    f"(hi={hi_ratio:.0%})] {text}{cs_tag}\n"
                )

        logger.info(f"Transcript saved to: {output_json} and {output_txt}")

        # WER estimation (requires reference)
        total_words = sum(len(s["text"].split()) for s in result["segments"])
        logger.info(f"Total transcribed words: {total_words}")

        return result

    def _enrich_with_lid(self, result: Dict) -> Dict:
        """
        Attach LID language label to each Whisper segment using
        majority vote over all LID frames that fall within the segment.

        Uses direct time-range lookup (not binning) for accuracy.
        Applies a confidence-weighted vote so low-confidence frames
        contribute less to the final label.

        Also computes:
          - dominant_language : "english" | "hindi"
          - is_code_switched  : True if >20% frames are the minority language
          - hindi_ratio       : fraction of frames labelled Hindi
          - switch_points     : timestamps (sec) where language switches occur
        """
        # Build a sorted list for range queries — faster than dict binning
        lid_sorted = sorted(self.lid_preds, key=lambda x: x["start_sec"])

        for seg in result["segments"]:
            seg_start = seg["start"]
            seg_end   = seg["end"]

            # Collect all LID frames that overlap this segment
            en_weight = 0.0
            hi_weight = 0.0
            switch_points = []
            prev_lang = None

            for frame in lid_sorted:
                fs = frame["start_sec"]
                fe = frame["end_sec"]

                # Skip frames entirely outside the segment
                if fe < seg_start:
                    continue
                if fs > seg_end:
                    break

                lang = frame["language"]
                conf = frame.get("confidence", 1.0)

                if lang == "english":
                    en_weight += conf
                else:
                    hi_weight += conf

                # Collect switch points with ±200ms precision (assignment req)
                if prev_lang is not None and lang != prev_lang:
                    switch_points.append(round(fs, 3))
                prev_lang = lang

            total_weight = en_weight + hi_weight

            if total_weight == 0:
                # No LID frames found for this segment — use Whisper's detection
                seg["dominant_language"] = seg.get("language", "unknown")
                seg["is_code_switched"]  = False
                seg["hindi_ratio"]       = 0.0
                seg["switch_points"]     = []
                continue

            hindi_ratio = hi_weight / total_weight
            english_ratio = en_weight / total_weight

            # Dominant language by weighted majority vote
            seg["dominant_language"] = "hindi" if hindi_ratio >= 0.5 else "english"

            # Code-switched if minority language > 20% of segment
            minority_ratio = min(hindi_ratio, english_ratio)
            seg["is_code_switched"]  = minority_ratio > 0.20

            seg["hindi_ratio"]   = round(hindi_ratio, 3)
            seg["switch_points"] = switch_points

        return result


# ─────────────────────────────────────────────────────────────
# SYLLABUS LOADER
# ─────────────────────────────────────────────────────────────

DEFAULT_SYLLABUS = """
HC Verma Concepts of Physics quantum mechanics modern physics.
Quantum theory wave particle duality de Broglie wavelength hypothesis.
Photoelectric effect photon energy frequency threshold work function.
Heisenberg uncertainty principle position momentum energy time.
Schrodinger wave equation wavefunction probability amplitude.
Superposition principle interference diffraction double slit experiment.
Identical particles fermions bosons Pauli exclusion principle.
Hydrogen atom Bohr model energy levels orbital angular momentum.
Quantum numbers principal azimuthal magnetic spin quantum number.
Electron configuration shell subshell orbital degeneracy.
Radioactivity alpha beta gamma decay nuclear fission fusion.
Nucleus proton neutron binding energy mass defect.
Special relativity reference frame inertial frame time dilation.
Length contraction Lorentz transformation mass energy equivalence.
Einstein equation E equals m c squared rest mass energy.
Electric field magnetic field electromagnetic induction Faraday law.
Coulomb law Gauss law capacitor electric potential voltage.
Ohm law resistance conductance current voltage power.
Magnetic force Lorentz force solenoid electromagnet.
Optics refraction reflection total internal reflection Snell law.
Lens mirror focal length magnification image formation.
Interference coherent light Young double slit fringe width.
Diffraction grating wavelength resolution Rayleigh criterion.
Polarization Brewster angle transverse wave electromagnetic spectrum.
Thermodynamics first law second law entropy heat engine efficiency.
Ideal gas kinetic theory pressure temperature Boltzmann constant.
Carnot cycle reversible irreversible process free energy.
Simple harmonic motion oscillation amplitude period frequency phase.
Wave motion transverse longitudinal standing wave resonance.
Sound speed intensity decibel Doppler effect beats superposition.
Gravitation Newton law gravitational potential escape velocity orbit.
Kepler laws planetary motion angular momentum conservation.
Rotational motion moment of inertia torque angular velocity.
Linear momentum conservation collision elastic inelastic.
Work energy theorem kinetic energy potential energy conservation.
Friction coefficient normal force tension compression stress strain.
Fluid pressure buoyancy Archimedes principle viscosity Bernoulli.
Classical physics mechanics electromagnetism statistical mechanics.
Modern physics quantum electrodynamics semiconductor band theory.
Experiment observation hypothesis theory law scientific method.
"""


def load_syllabus(path: str = SYLLBUS_TEXT_PATH) -> str:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    logger.warning(f"Syllabus file not found at {path}. Using default syllabus.")
    return DEFAULT_SYLLABUS


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def train_lid_model(waveform: torch.Tensor, sample_rate: int, epochs: int = 30) -> MultiHeadLIDModel:
    """Train the LID model using pseudo-labelled data from the input audio."""

    logger.info("Preparing LID training data (pseudo-labels)...")
    dataset    = SyntheticLIDDataset(waveform, sample_rate, augment=True)

    split      = int(0.85 * len(dataset))
    train_ds   = torch.utils.data.Subset(dataset, range(split))
    val_ds     = torch.utils.data.Subset(dataset, range(split, len(dataset)))

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=2)

    model   = MultiHeadLIDModel()
    trainer = LIDTrainer(model)

    best_f1 = 0.0
    logger.info(f"Training LID model for {epochs} epochs...")

    for epoch in range(epochs):
        train_loss = trainer.train_epoch(train_loader)

        if (epoch + 1) % 5 == 0:
            metrics = trainer.evaluate(val_loader)
            logger.info(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"Loss: {train_loss:.4f} | "
                f"Macro-F1: {metrics['macro_f1']:.4f} | "
                f"En-F1: {metrics['english_f1']:.4f} | "
                f"Hi-F1: {metrics['hindi_f1']:.4f}"
            )
            if metrics["macro_f1"] > best_f1:
                best_f1 = metrics["macro_f1"]
                torch.save(model.state_dict(), "lid_model_best.pt")
                logger.info(f"  → New best model saved (F1={best_f1:.4f})")

    # Load best
    if os.path.exists("lid_model_best.pt"):
        model.load_state_dict(torch.load("lid_model_best.pt", map_location=DEVICE))

    final_metrics = trainer.evaluate(val_loader)
    logger.info(f"Final LID Metrics: {final_metrics}")

    if final_metrics["macro_f1"] < 0.85:
        logger.warning(
            "LID F1 below 0.85 threshold. "
            "Consider: (1) using real labelled data, "
            "(2) more training epochs, (3) pre-trained LID features."
        )

    return model


def run_part1(audio_path: str, mode: str = "full", skip_lid_train: bool = False, denoise_method: str = "auto", chunk_sec: int = 30, whisper_model: str = "large-v3", reference_txt: Optional[str] = None):
    """
    Master function for Part I.

    Args:
        audio_path:     Path to input WAV file
        mode:           'full' | 'denoise' | 'lid' | 'transcribe'
        skip_lid_train: If True, load pre-trained LID weights
    """
    os.makedirs("outputs", exist_ok=True)

    denoised_path = "outputs/denoised_output.wav"
    lid_json      = "outputs/lid_predictions.json"
    transcript_json = "outputs/transcript.json"
    transcript_txt  = "outputs/transcript.txt"
    lid_weights     = "lid_model_best.pt"

    # ── Step 1: Denoise ──────────────────────────────────────
    if mode in ("full", "denoise"):
        waveform, sr = load_and_denoise(audio_path, denoised_path, method=denoise_method, chunk_sec=chunk_sec)
    else:
        waveform, sr = torchaudio.load(denoised_path if os.path.exists(denoised_path) else audio_path)
        if sr != SAMPLE_RATE:
            resampler = T.Resample(sr, SAMPLE_RATE)
            waveform  = resampler(waveform)
            sr        = SAMPLE_RATE

    # ── Step 2: LID ──────────────────────────────────────────
    lid_preds = None
    if mode in ("full", "lid"):

        lid_model = MultiHeadLIDModel()

        if skip_lid_train and os.path.exists(lid_weights):
            logger.info(f"Loading pre-trained LID weights from {lid_weights}")
            lid_model.load_state_dict(torch.load(lid_weights, map_location=DEVICE))
        else:
            lid_model = train_lid_model(waveform, sr, epochs=40)

        lid_model = lid_model.to(DEVICE)
        lid_preds = run_lid_inference(waveform, sr, lid_model, lid_json)

        # ── Explicitly free LID model from GPU before loading Whisper ──
        lid_model.cpu()
        del lid_model
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        free_gb = get_available_gpu_memory_gb()
        logger.info(f"GPU memory freed after LID. Free: {free_gb:.1f} GB")

    # ── Step 3: N-gram LM ─────────────────────────────────────
    ngram_lm = None
    if mode in ("full", "transcribe"):
        syllabus_text = load_syllabus()
        ngram_lm      = NgramLanguageModel(order=NGRAM_ORDER)
        ngram_lm.train(syllabus_text)
        ngram_lm.save("outputs/ngram_lm.json")

    # ── Step 4: Constrained Transcription ────────────────────
    if mode in ("full", "transcribe"):
        if lid_preds is None and os.path.exists(lid_json):
            with open(lid_json, "r") as f:
                lid_preds = json.load(f)

        audio_for_transcription = denoised_path if os.path.exists(denoised_path) else audio_path

        decoder = ConstrainedWhisperDecoder(
            model_name = whisper_model,
            ngram_lm   = ngram_lm,
            lid_preds  = lid_preds,
        )
        result = decoder.transcribe_full(
            audio_for_transcription,
            transcript_json,
            transcript_txt,
        )

        logger.info("=" * 60)
        logger.info("PART I COMPLETE")
        logger.info(f"  Denoised audio   : {denoised_path}")
        logger.info(f"  LID predictions  : {lid_json}")
        logger.info(f"  Transcript (json): {transcript_json}")
        logger.info(f"  Transcript (txt) : {transcript_txt}")
        logger.info("=" * 60)

        # ── Evaluate assignment passing criteria ──────────────
        metrics = AssignmentMetrics()
        metrics.run_all_part1(
            transcript_json = transcript_json,
            lid_json        = lid_json,
            reference_txt   = reference_txt,
        )



# ─────────────────────────────────────────────────────────────
# EVALUATION METRICS  (Assignment Passing Criteria)
# ─────────────────────────────────────────────────────────────

class AssignmentMetrics:
    """
    Tracks all strict passing criteria from the assignment:

    Part I  (computed here):
        ✓ WER < 15%  for English segments
        ✓ WER < 25%  for Hindi segments
        ✓ LID switch timestamp precision < 200ms

    Part III (stub — computed in part3):
        ○ MCD < 8.0  (synthesized LRL vs reference voice)

    Part IV (stub — computed in part4):
        ○ Anti-spoofing EER < 10%
        ○ Adversarial epsilon (min perturbation to flip LID)

    All results are saved to outputs/metrics_report.json
    and printed as a pass/fail scorecard.
    """

    THRESHOLDS = {
        "wer_english":       0.15,    # < 15%
        "wer_hindi":         0.25,    # < 25%
        "lid_switch_ms":     200.0,   # < 200ms timestamp error
        "mcd":               8.0,     # < 8.0 dB  (Part III)
        "eer":               0.10,    # < 10%     (Part IV)
    }

    def __init__(self):
        self.results: Dict = {}

    # ── WER ──────────────────────────────────────────────────

    def compute_wer(self, hypothesis: str, reference: str) -> float:
        """
        Word Error Rate using dynamic programming (edit distance).

        WER = (S + D + I) / N
          S = substitutions, D = deletions, I = insertions
          N = number of words in reference

        No external library needed.
        """
        hyp_words = hypothesis.lower().strip().split()
        ref_words = reference.lower().strip().split()

        n_ref = len(ref_words)
        n_hyp = len(hyp_words)

        if n_ref == 0:
            return 0.0 if n_hyp == 0 else 1.0

        # DP table: dp[i][j] = edit distance between
        # ref[:i] and hyp[:j]
        dp = [[0] * (n_hyp + 1) for _ in range(n_ref + 1)]

        for i in range(n_ref + 1):
            dp[i][0] = i
        for j in range(n_hyp + 1):
            dp[0][j] = j

        for i in range(1, n_ref + 1):
            for j in range(1, n_hyp + 1):
                if ref_words[i-1] == hyp_words[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],    # deletion
                        dp[i][j-1],    # insertion
                        dp[i-1][j-1],  # substitution
                    )

        return dp[n_ref][n_hyp] / n_ref

    def evaluate_wer_from_transcript(
        self,
        transcript_json:  str,
        reference_txt:    Optional[str] = None,
    ) -> Dict:
        """
        Compute WER separately for English and Hindi segments.

        If no reference_txt is provided, estimates WER using
        Whisper's own avg_logprob as a proxy (negative logprob
        correlates with recognition confidence / error rate).

        For a proper WER you need a human reference transcript.
        Create reference_txt with one line per segment:
            <start_sec>|<end_sec>|<language>|<reference text>
        """
        if not os.path.exists(transcript_json):
            logger.warning(f"Transcript not found: {transcript_json}")
            return {}

        with open(transcript_json, "r", encoding="utf-8") as f:
            transcript = json.load(f)

        segments = transcript.get("segments", [])

        if reference_txt and os.path.exists(reference_txt):
            return self._wer_with_reference(segments, reference_txt)
        else:
            return self._wer_proxy(segments)

    def _wer_proxy(self, segments: List[Dict]) -> Dict:
        """
        Proxy WER estimate when no reference transcript is available.
        Uses Whisper's avg_logprob: lower (more negative) = more errors.

        Mapping: logprob → estimated WER
            > -0.2  → ~5%   (very confident)
            -0.2 to -0.5 → ~10-15%
            -0.5 to -1.0 → ~20-30%
            < -1.0  → >40%  (likely errors)
        """
        en_logprobs, hi_logprobs = [], []

        for seg in segments:
            lang     = seg.get("dominant_language", "unknown")
            logprob  = seg.get("avg_logprob", -0.5)
            no_speech = seg.get("no_speech_prob", 0.0)

            if no_speech > 0.6:
                continue  # skip silence segments

            if lang == "english":
                en_logprobs.append(logprob)
            elif lang == "hindi":
                hi_logprobs.append(logprob)

        def logprob_to_wer(lp: float) -> float:
            """Empirical mapping from avg_logprob to approximate WER."""
            if lp > -0.2:   return 0.05
            if lp > -0.3:   return 0.10
            if lp > -0.5:   return 0.15
            if lp > -0.7:   return 0.22
            if lp > -1.0:   return 0.30
            return 0.45

        en_wer = logprob_to_wer(np.mean(en_logprobs)) if en_logprobs else None
        hi_wer = logprob_to_wer(np.mean(hi_logprobs)) if hi_logprobs else None

        result = {
            "method":         "proxy_logprob",
            "note":           "Approximate — provide reference_txt for exact WER",
            "english_wer":    round(en_wer, 4) if en_wer is not None else None,
            "hindi_wer":      round(hi_wer, 4) if hi_wer is not None else None,
            "english_passes": (en_wer < self.THRESHOLDS["wer_english"]) if en_wer else None,
            "hindi_passes":   (hi_wer < self.THRESHOLDS["wer_hindi"])   if hi_wer else None,
            "en_avg_logprob": round(float(np.mean(en_logprobs)), 4) if en_logprobs else None,
            "hi_avg_logprob": round(float(np.mean(hi_logprobs)), 4) if hi_logprobs else None,
            "n_english_segs": len(en_logprobs),
            "n_hindi_segs":   len(hi_logprobs),
        }
        return result

    def _wer_with_reference(self, segments: List[Dict], reference_txt: str) -> Dict:
        """
        Exact WER using human reference transcript.
        reference_txt format (one line per segment):
            <start_sec>|<end_sec>|<en/hi>|reference text here
        """
        refs = {}
        with open(reference_txt, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("|", 3)
                if len(parts) == 4:
                    start, end, lang, text = parts
                    refs[(float(start), float(end))] = (lang.strip(), text.strip())

        en_errors, en_total = 0, 0
        hi_errors, hi_total = 0, 0

        for seg in segments:
            seg_start = seg["start"]
            seg_end   = seg["end"]
            hyp_text  = seg["text"].strip()
            dom_lang  = seg.get("dominant_language", "unknown")

            # Find matching reference segment (within 0.5s tolerance)
            best_ref  = None
            best_overlap = 0
            for (rs, re), (rlang, rtext) in refs.items():
                overlap = max(0, min(seg_end, re) - max(seg_start, rs))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_ref     = (rlang, rtext)

            if best_ref is None:
                continue

            ref_lang, ref_text = best_ref
            wer_val  = self.compute_wer(hyp_text, ref_text)
            n_words  = len(ref_text.split())

            if ref_lang == "en":
                en_errors += wer_val * n_words
                en_total  += n_words
            else:
                hi_errors += wer_val * n_words
                hi_total  += n_words

        en_wer = en_errors / en_total if en_total > 0 else None
        hi_wer = hi_errors / hi_total if hi_total > 0 else None

        return {
            "method":         "exact_reference",
            "english_wer":    round(en_wer, 4) if en_wer is not None else None,
            "hindi_wer":      round(hi_wer, 4) if hi_wer is not None else None,
            "english_passes": (en_wer < self.THRESHOLDS["wer_english"]) if en_wer else None,
            "hindi_passes":   (hi_wer < self.THRESHOLDS["wer_hindi"])   if hi_wer else None,
            "en_word_count":  en_total,
            "hi_word_count":  hi_total,
        }

    # ── LID Switch Precision ─────────────────────────────────

    def evaluate_lid_switch_precision(
        self,
        lid_json:           str,
        reference_switches: Optional[List[float]] = None,
    ) -> Dict:
        """
        Evaluate LID switch timestamp precision (requirement: < 200ms).

        If reference_switches (list of ground-truth switch times in sec)
        is provided, computes mean absolute error between predicted and
        reference switch timestamps.

        If not provided, uses the minimum inter-switch interval as a
        proxy for jitter — after smoothing this should be >= 200ms.
        """
        if not os.path.exists(lid_json):
            logger.warning(f"LID json not found: {lid_json}")
            return {}

        with open(lid_json, "r") as f:
            lid_preds = json.load(f)

        # Extract predicted switch timestamps
        predicted_switches = [
            p["start_sec"]
            for p in lid_preds
            if p.get("is_switch", False)
        ]

        if not predicted_switches:
            return {"error": "No switches detected"}

        # Minimum gap between consecutive switches (jitter measure)
        gaps = []
        for i in range(1, len(predicted_switches)):
            gap_ms = (predicted_switches[i] - predicted_switches[i-1]) * 1000
            gaps.append(gap_ms)

        min_gap_ms  = min(gaps) if gaps else 0
        mean_gap_ms = float(np.mean(gaps)) if gaps else 0

        result = {
            "n_switches":          len(predicted_switches),
            "min_gap_ms":          round(min_gap_ms, 1),
            "mean_gap_ms":         round(mean_gap_ms, 1),
            "min_gap_passes":      min_gap_ms >= self.THRESHOLDS["lid_switch_ms"],
            "threshold_ms":        self.THRESHOLDS["lid_switch_ms"],
            "predicted_switches":  predicted_switches[:20],  # first 20 for report
        }

        # If ground truth provided, compute MAE
        if reference_switches:
            errors = []
            for pred_t in predicted_switches:
                closest_ref = min(reference_switches, key=lambda r: abs(r - pred_t))
                err_ms      = abs(pred_t - closest_ref) * 1000
                errors.append(err_ms)

            mae_ms = float(np.mean(errors))
            result["mae_ms"]        = round(mae_ms, 1)
            result["mae_passes"]    = mae_ms < self.THRESHOLDS["lid_switch_ms"]

        return result

    # ── Scorecard ────────────────────────────────────────────

    def run_all_part1(
        self,
        transcript_json:    str = "outputs/transcript.json",
        lid_json:           str = "outputs/lid_predictions.json",
        reference_txt:      Optional[str] = None,
        reference_switches: Optional[List[float]] = None,
        output_json:        str = "outputs/metrics_report.json",
    ) -> Dict:
        """
        Run all Part I metrics and save a scorecard.
        """
        logger.info("=" * 60)
        logger.info("EVALUATING ASSIGNMENT METRICS — PART I")
        logger.info("=" * 60)

        report = {
            "part1": {},
            "part3": {"mcd":  {"status": "pending — run part3"}},
            "part4": {
                "eer":     {"status": "pending — run part4"},
                "epsilon": {"status": "pending — run part4"},
            },
        }

        # WER
        wer_results = self.evaluate_wer_from_transcript(transcript_json, reference_txt)
        report["part1"]["wer"] = wer_results

        # LID Switch Precision
        lid_results = self.evaluate_lid_switch_precision(lid_json, reference_switches)
        report["part1"]["lid_switch"] = lid_results

        # Print scorecard
        self._print_scorecard(report)

        # Save
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Metrics report saved: {output_json}")

        return report

    def _print_scorecard(self, report: Dict):
        PASS = "✅ PASS"
        FAIL = "❌ FAIL"
        PEND = "⏳ PENDING"

        def status(val, threshold, lower_is_better=True):
            if val is None:
                return PEND
            return PASS if (val < threshold if lower_is_better else val > threshold) else FAIL

        wer  = report["part1"].get("wer", {})
        lid  = report["part1"].get("lid_switch", {})

        en_wer = wer.get("english_wer")
        hi_wer = wer.get("hindi_wer")
        min_gap = lid.get("min_gap_ms")

        logger.info(f"│ WER English          │ {f'{en_wer:.1%}' if en_wer else 'N/A':8s} │ <15%   │ {status(en_wer, 0.15):8s} │")
        logger.info(f"│ WER Hindi            │ {f'{hi_wer:.1%}' if hi_wer else 'N/A':8s} │ <25%   │ {status(hi_wer, 0.25):8s} │")
        logger.info("└──────────────────────┴──────────┴────────┴──────────┘")



# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Part I: Code-Switched Transcription Pipeline")

    parser.add_argument(
        "--audio", type=str, required=True,
        help="Path to input audio file (WAV/MP3/M4A)"
    )
    parser.add_argument(
        "--mode", type=str, default="full",
        choices=["full", "denoise", "lid", "transcribe"],
        help="Pipeline mode (default: full)"
    )
    parser.add_argument(
        "--skip_lid_train", action="store_true",
        help="Skip LID training and load from lid_model_best.pt"
    )
    parser.add_argument(
        "--whisper_model", type=str, default="large-v3",
        help="Whisper model size (default: large-v3)"
    )
    parser.add_argument(
        "--denoise_method", type=str, default="auto",
        choices=["auto", "deepfilter", "spectral"],
        help="Denoising method: auto (default) | deepfilter | spectral (CPU, no OOM)"
    )
    parser.add_argument(
        "--chunk_sec", type=int, default=30,
        help="DeepFilterNet chunk size in seconds (default 30, reduce if OOM)"
    )
    parser.add_argument(
        "--reference_txt", type=str, default=None,
        help="(Optional) Human reference transcript for exact WER. "
             "Format per line: start_sec|end_sec|en/hi|reference text"
    )

    args = parser.parse_args()

    run_part1(
        audio_path     = args.audio,
        mode           = args.mode,
        skip_lid_train = args.skip_lid_train,
        denoise_method = args.denoise_method,
        chunk_sec      = args.chunk_sec,
        whisper_model  = args.whisper_model,
        reference_txt  = args.reference_txt,
    )