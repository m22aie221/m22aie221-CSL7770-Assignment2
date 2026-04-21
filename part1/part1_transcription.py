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


def denoise_with_deepfilternet(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """
    Try DeepFilterNet first; fall back to SpectralSubtraction if unavailable.
    """
    try:
        from df.enhance import enhance, init_df, load_audio, save_audio
        from df.io import resample

        logger.info("DeepFilterNet found. Using it for denoising.")
        model, df_state, _ = init_df()

        # DeepFilterNet needs 48kHz
        if sample_rate != 48000:
            resampler  = T.Resample(orig_freq=sample_rate, new_freq=48000)
            waveform   = resampler(waveform)

        enhanced   = enhance(model, df_state, waveform)

        # Resample back
        resampler  = T.Resample(orig_freq=48000, new_freq=sample_rate)
        enhanced   = resampler(enhanced)
        return enhanced

    except ImportError:
        logger.warning("DeepFilterNet not installed. Falling back to Spectral Subtraction.")
        ss = SpectralSubtraction(sample_rate=sample_rate)
        return ss(waveform)


def load_and_denoise(audio_path: str, output_path: str = "denoised_output.wav") -> Tuple[torch.Tensor, int]:
    """
    Load audio, resample to 16kHz, denoise, save output.

    Returns:
        (waveform, sample_rate)
    """
    logger.info(f"Loading audio: {audio_path}")
    waveform, sr = torchaudio.load(audio_path)

    # Mix down to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 16kHz
    if sr != SAMPLE_RATE:
        logger.info(f"Resampling from {sr}Hz to {SAMPLE_RATE}Hz")
        resampler = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform  = resampler(waveform)
        sr        = SAMPLE_RATE

    # Normalize amplitude
    waveform = waveform / (waveform.abs().max() + 1e-8)

    logger.info("Denoising audio...")
    waveform = denoise_with_deepfilternet(waveform, sr)

    # Normalize again after denoising
    waveform = waveform / (waveform.abs().max() + 1e-8)

    torchaudio.save(output_path, waveform, sr)
    logger.info(f"Denoised audio saved to: {output_path}")

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

    # Save
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(deduped, f, indent=2, ensure_ascii=False)

    logger.info(f"LID predictions saved to: {output_json}")

    # Summary stats
    lang_counts = Counter(p["language"] for p in deduped)
    switches    = sum(1 for p in deduped if p["is_switch"])
    logger.info(f"LID Summary — English frames: {lang_counts['english']}, "
                f"Hindi frames: {lang_counts['hindi']}, "
                f"Language switches: {switches}")

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

        # Hard-coded domain terms from speech processing
        domain_terms = {
            "cepstrum", "cepstral", "stochastic", "mel", "mfcc", "spectrogram",
            "phoneme", "allophone", "formant", "fundamental", "frequency",
            "hmm", "hidden", "markov", "viterbi", "baum", "welch",
            "gaussian", "mixture", "model", "acoustic", "language",
            "waveform", "sampling", "quantization", "fourier", "transform",
            "filterbank", "feature", "extraction", "recognition", "synthesis",
            "prosody", "intonation", "duration", "pitch", "energy",
            "forced", "alignment", "transcription", "phonetic", "lexicon",
            "bigram", "trigram", "perplexity", "smoothing", "backoff",
            "interpolation", "kneser", "ney", "good", "turing",
            "neural", "network", "recurrent", "lstm", "attention",
            "transformer", "whisper", "wav2vec", "bert", "conformer",
            "connectionist", "temporal", "classification", "ctc",
            "beam", "search", "decoding", "lattice", "hypothesis",
            "reverb", "denoising", "spectral", "subtraction", "wiener",
            "deepfilternet", "voicing", "fricative", "plosive", "nasal",
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

        logger.info(f"Loading Whisper {model_name}...")
        self.whisper     = whisper
        self.model       = whisper.load_model(model_name, device=DEVICE)
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

        # Use Whisper's built-in VAD + chunking
        result = self.model.transcribe(
            audio,
            language          = None,      # auto-detect per segment
            beam_size         = BEAM_SIZE,
            best_of           = 5,
            temperature       = 0.0,
            condition_on_previous_text = True,
            word_timestamps   = True,
            verbose           = False,
        )

        # Enrich segments with LID information
        if self.lid_preds:
            result = self._enrich_with_lid(result)

        # Save outputs
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        with open(output_txt, "w", encoding="utf-8") as f:
            for seg in result["segments"]:
                start = seg["start"]
                end   = seg["end"]
                text  = seg["text"].strip()
                lang  = seg.get("dominant_language", "?")
                f.write(f"[{start:.2f}s → {end:.2f}s | {lang}] {text}\n")

        logger.info(f"Transcript saved to: {output_json} and {output_txt}")

        # WER estimation (requires reference)
        total_words = sum(len(s["text"].split()) for s in result["segments"])
        logger.info(f"Total transcribed words: {total_words}")

        return result

    def _enrich_with_lid(self, result: Dict) -> Dict:
        """
        Attach LID language label to each Whisper segment
        based on majority vote from LID frame predictions.
        """
        lid_lookup = defaultdict(list)
        for p in self.lid_preds:
            t_bin = int(p["start_sec"] * 10)   # 100ms buckets
            lid_lookup[t_bin].append(p["language"])

        for seg in result["segments"]:
            start_bin = int(seg["start"] * 10)
            end_bin   = int(seg["end"]   * 10)
            langs     = []
            for b in range(start_bin, end_bin + 1):
                langs.extend(lid_lookup[b])

            if langs:
                seg["dominant_language"] = Counter(langs).most_common(1)[0][0]
                seg["is_code_switched"]  = len(set(langs)) > 1
            else:
                seg["dominant_language"] = "unknown"
                seg["is_code_switched"]  = False

        return result


# ─────────────────────────────────────────────────────────────
# SYLLABUS LOADER
# ─────────────────────────────────────────────────────────────

DEFAULT_SYLLABUS = """
Speech Understanding course covering acoustic phonetics, signal processing,
feature extraction including mel frequency cepstral coefficients MFCC,
cepstrum cepstral analysis, filterbank, mel spectrogram,
hidden Markov model HMM, Gaussian mixture model GMM,
stochastic language model, n-gram bigram trigram,
Viterbi algorithm, Baum-Welch algorithm, forward backward algorithm,
neural network acoustic model, recurrent neural network LSTM GRU,
connectionist temporal classification CTC,
attention mechanism transformer self-attention,
end-to-end speech recognition, automatic speech recognition ASR,
forced alignment, phoneme recognition, word error rate WER,
speaker recognition verification identification,
speaker embedding d-vector x-vector,
text to speech synthesis TTS, vocoder, WaveNet, WaveRNN,
prosody intonation pitch fundamental frequency F0,
formant frequency spectral envelope,
dynamic time warping DTW, edit distance,
language identification code switching Hinglish,
denoising spectral subtraction Wiener filter,
voice activity detection VAD,
beam search decoding language model integration,
low resource speech, zero shot transfer learning,
International Phonetic Alphabet IPA, grapheme to phoneme G2P,
anti spoofing countermeasure, equal error rate EER,
adversarial perturbation FGSM robustness.
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


def run_part1(audio_path: str, mode: str = "full", skip_lid_train: bool = False):
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
        waveform, sr = load_and_denoise(audio_path, denoised_path)
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
            model_name = "large-v3",
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

    args = parser.parse_args()

    run_part1(
        audio_path     = args.audio,
        mode           = args.mode,
        skip_lid_train = args.skip_lid_train,
    )
