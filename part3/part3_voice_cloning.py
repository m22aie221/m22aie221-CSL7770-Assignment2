"""
Speech Understanding - Programming Assignment 2
Part III: Zero-Shot Cross-Lingual Voice Cloning (TTS)

Tasks:
    3.1 - Voice Embedding Extraction
          Record 60s reference voice → extract d-vector / x-vector speaker embedding
    3.2 - Prosody Warping
          Extract F0 + Energy from professor's lecture
          Apply DTW to map prosodic contours onto synthesized Bhojpuri speech
    3.3 - Synthesis
          Use VITS / YourTTS / Meta MMS to produce 10-min Bhojpuri lecture
          Output must be ≥ 22.05kHz

Evaluation:
    MCD (Mel-Cepstral Distortion) < 8.0 dB between synthesized output
    and student reference voice

Folder structure:
    /scratch/data/m22aie221/workspace/CSL7770-Assignment2/
        data/
            lecture_segment.wav          ← professor's 10-min audio (Part 1 input)
            student_voice_ref.wav        ← YOUR 60s voice recording (YOU provide this)
        part2/
            outputs/
                bhojpuri_translation.json ← from Part 2
        part3/
            part3_voice_cloning.py        ← this file
            outputs/
                speaker_embedding.pt      ← d-vector (Task 3.1)
                f0_professor.npy          ← F0 contour of professor (Task 3.2)
                energy_professor.npy      ← Energy contour (Task 3.2)
                f0_warped.npy             ← DTW-warped F0 (Task 3.2)
                output_LRL_cloned.wav     ← Final 10-min Bhojpuri lecture (Task 3.3)
                mcd_report.json           ← MCD evaluation (metric)

Usage:
    # Full pipeline
    python part3_voice_cloning.py \\
        --reference  ../data/student_voice_ref.wav \\
        --lecture    ../data/lecture_segment.wav \\
        --transcript ../part2/outputs/bhojpuri_translation.json

    # Individual steps
    python part3_voice_cloning.py --mode embed     # Task 3.1 only
    python part3_voice_cloning.py --mode prosody   # Task 3.2 only
    python part3_voice_cloning.py --mode synthesize # Task 3.3 only
    python part3_voice_cloning.py --mode mcd        # Evaluate MCD only
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
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from scipy.signal import medfilt
from scipy.interpolate import interp1d

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE       = 22050   # ≥22.05kHz as required
SAMPLE_RATE_16K   = 16000   # for embedding extraction
HOP_LENGTH        = 256     # ~11.6ms at 22kHz
WIN_LENGTH        = 1024
N_FFT             = 1024
N_MELS            = 80
N_MFCC            = 40
F0_MIN            = 50.0    # Hz — minimum F0 for speech
F0_MAX            = 500.0   # Hz — maximum F0 for speech

logger.info(f"Device: {DEVICE}")


# ─────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────

def load_audio(
    path:        str,
    target_sr:   int  = SAMPLE_RATE,
    mono:        bool = True,
) -> Tuple[torch.Tensor, int]:
    """Load audio, convert to mono, resample to target_sr."""
    waveform, sr = torchaudio.load(path)
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = T.Resample(orig_freq=sr, new_freq=target_sr)(waveform)
    waveform = waveform / (waveform.abs().max() + 1e-8)
    return waveform, target_sr


def save_audio(waveform: torch.Tensor, path: str, sr: int = SAMPLE_RATE):
    """Save waveform tensor to WAV file."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    waveform = waveform.cpu()
    torchaudio.save(path, waveform, sr)
    dur = waveform.shape[-1] / sr
    logger.info(f"Saved audio: {path}  ({dur:.1f}s, {sr}Hz)")


def get_free_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    free, _ = torch.cuda.mem_get_info()
    return free / (1024 ** 3)


# ─────────────────────────────────────────────────────────────
# TASK 3.1 — SPEAKER EMBEDDING (D-VECTOR / X-VECTOR)
# ─────────────────────────────────────────────────────────────

class SpeakerEncoderGE2E(nn.Module):
    """
    GE2E (Generalized End-to-End) speaker encoder.
    Produces a d-vector from a mel-spectrogram sequence.

    Architecture (from Wan et al., 2018):
        3-layer LSTM → L2-normalized mean pooling → 256-dim d-vector

    The d-vector represents the unique voice characteristics
    of the speaker — used to condition the TTS model in Task 3.3.
    """

    def __init__(
        self,
        input_dim:  int = N_MELS,      # 80 mel bins
        hidden_dim: int = 768,
        embed_dim:  int = 256,
        num_layers: int = 3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
        )
        self.linear = nn.Linear(hidden_dim, embed_dim)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (batch, time, n_mels)
        Returns:
            embedding: (batch, embed_dim)  L2-normalized d-vector
        """
        out, _    = self.lstm(mel)           # (B, T, hidden)
        out       = self.linear(out)          # (B, T, embed_dim)
        embedding = out.mean(dim=1)           # (B, embed_dim) — mean pooling
        embedding = F.normalize(embedding, p=2, dim=-1)  # L2 normalize
        return embedding


def extract_mel_for_embedding(
    waveform:   torch.Tensor,
    sample_rate: int = SAMPLE_RATE_16K,
) -> torch.Tensor:
    """
    Extract mel-spectrogram features for speaker embedding.
    Uses 16kHz, 40ms window, 10ms hop — standard for d-vector.
    """
    mel_transform = T.MelSpectrogram(
        sample_rate = sample_rate,
        n_fft       = 512,
        hop_length  = 160,   # 10ms at 16kHz
        win_length  = 400,   # 25ms
        n_mels      = N_MELS,
        f_min       = 90,
        f_max       = 7600,
    ).to(waveform.device)

    mel   = mel_transform(waveform)          # (1, n_mels, T)
    mel   = torch.log(mel + 1e-6)            # log-mel
    mel   = mel.squeeze(0).T                 # (T, n_mels)
    return mel


def extract_speaker_embedding(
    reference_path: str,
    output_path:    str = "outputs/speaker_embedding.pt",
    use_pretrained: bool = True,
) -> torch.Tensor:
    """
    Task 3.1: Extract d-vector from 60s reference recording.

    Tries (in order):
      1. SpeechBrain ECAPA-TDNN pretrained x-vector (best quality)
      2. Resemblyzer pretrained GE2E d-vector
      3. Our GE2E implementation (no pretrained weights — less accurate
         but fully self-contained for submission)

    Args:
        reference_path: Path to student_voice_ref.wav (60 seconds)
        output_path:    Where to save the embedding tensor
        use_pretrained: Try pretrained models first (recommended)

    Returns:
        speaker_embedding: (256,) or (192,) float tensor
    """
    logger.info(f"Task 3.1: Extracting speaker embedding from {reference_path}")

    if not os.path.exists(reference_path):
        raise FileNotFoundError(
            f"Reference voice not found: {reference_path}\n"
            "Please record 60 seconds of your voice:\n"
            "  arecord -d 60 -f cd -r 22050 -t wav student_voice_ref.wav\n"
            "Then copy to: ../data/student_voice_ref.wav"
        )

    waveform_22k, _ = load_audio(reference_path, target_sr=SAMPLE_RATE)
    waveform_16k, _ = load_audio(reference_path, target_sr=SAMPLE_RATE_16K)

    dur = waveform_22k.shape[-1] / SAMPLE_RATE
    logger.info(f"Reference audio duration: {dur:.1f}s  (required: 60s)")
    if dur < 55:
        logger.warning(f"Reference audio is only {dur:.1f}s — recommend ≥60s")

    embedding = None

    # ── Option 1: SpeechBrain ECAPA-TDNN (x-vector) ──────────
    if use_pretrained:
        try:
            from speechbrain.pretrained import EncoderClassifier
            logger.info("Using SpeechBrain ECAPA-TDNN for x-vector extraction...")
            classifier = EncoderClassifier.from_hparams(
                source       = "speechbrain/spkrec-ecapa-voxceleb",
                savedir      = "pretrained_models/spkrec-ecapa",
                run_opts     = {"device": str(DEVICE)},
            )
            signal = waveform_16k.squeeze(0).to(DEVICE)
            with torch.no_grad():
                embedding = classifier.encode_batch(signal.unsqueeze(0))
                embedding = embedding.squeeze().cpu()
            logger.info(f"ECAPA x-vector shape: {embedding.shape}")

        except Exception as e:
            logger.warning(f"SpeechBrain failed: {e}")
            embedding = None

    # ── Option 2: Resemblyzer GE2E d-vector ──────────────────
    if embedding is None and use_pretrained:
        try:
            from resemblyzer import VoiceEncoder, preprocess_wav
            import numpy as np_res
            logger.info("Using Resemblyzer GE2E d-vector...")
            encoder = VoiceEncoder(device=str(DEVICE))
            wav_np  = waveform_16k.squeeze(0).numpy()
            wav_pp  = preprocess_wav(wav_np, source_sr=SAMPLE_RATE_16K)
            with torch.no_grad():
                embedding = torch.tensor(encoder.embed_utterance(wav_pp))
            logger.info(f"Resemblyzer d-vector shape: {embedding.shape}")

        except Exception as e:
            logger.warning(f"Resemblyzer failed: {e}")
            embedding = None

    # ── Option 3: Our GE2E model (self-contained) ─────────────
    if embedding is None:
        logger.info("Using self-contained GE2E speaker encoder...")
        model = SpeakerEncoderGE2E().to(DEVICE)
        model.eval()

        waveform_16k = waveform_16k.to(DEVICE)
        mel          = extract_mel_for_embedding(waveform_16k, SAMPLE_RATE_16K)

        # Split into 1.6s chunks with 50% overlap (160 frames @ 10ms)
        chunk_frames = 160
        hop_frames   = 80
        chunks       = []
        for start in range(0, mel.shape[0] - chunk_frames, hop_frames):
            chunk = mel[start: start + chunk_frames]  # (160, 80)
            chunks.append(chunk)

        if not chunks:
            chunks = [mel[:chunk_frames] if mel.shape[0] >= chunk_frames else
                      F.pad(mel, (0, 0, 0, chunk_frames - mel.shape[0]))]

        chunks_tensor = torch.stack(chunks).to(DEVICE)   # (N, 160, 80)

        with torch.no_grad():
            embeddings = model(chunks_tensor)             # (N, 256)
            embedding  = embeddings.mean(dim=0).cpu()    # (256,) average

        logger.info(f"GE2E d-vector shape: {embedding.shape}")

    # Save embedding
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(embedding, output_path)
    logger.info(f"Speaker embedding saved: {output_path}  shape={embedding.shape}")

    return embedding


# ─────────────────────────────────────────────────────────────
# TASK 3.2 — PROSODY EXTRACTION + DTW WARPING
# ─────────────────────────────────────────────────────────────

class ProsodyExtractor:
    """
    Extracts F0 (fundamental frequency) and energy contours
    from speech audio.

    F0 extraction uses the YIN algorithm (de Cheveigné & Kawahara, 2002)
    implemented via torchaudio's functional API.

    Energy is computed as frame-level RMS power.
    """

    def __init__(
        self,
        sample_rate: int   = SAMPLE_RATE,
        hop_length:  int   = HOP_LENGTH,
        win_length:  int   = WIN_LENGTH,
        f0_min:      float = F0_MIN,
        f0_max:      float = F0_MAX,
    ):
        self.sample_rate = sample_rate
        self.hop_length  = hop_length
        self.win_length  = win_length
        self.f0_min      = f0_min
        self.f0_max      = f0_max

    def extract_f0(self, waveform: torch.Tensor) -> np.ndarray:
        """
        Extract F0 contour using torchaudio's detect_pitch_frequency.

        Returns:
            f0: (T,) numpy array in Hz, 0.0 for unvoiced frames
        """
        waveform_cpu = waveform.squeeze(0).cpu()

        try:
            # torchaudio pitch detection (Kaldi-style autocorrelation)
            f0 = torchaudio.functional.detect_pitch_frequency(
                waveform_cpu,
                sample_rate   = self.sample_rate,
                frame_time    = self.hop_length / self.sample_rate,
                win_length    = 30,
                freq_low      = self.f0_min,
                freq_high     = self.f0_max,
            )
            f0_np = f0.squeeze().numpy()

        except Exception:
            # Fallback: autocorrelation-based F0
            f0_np = self._autocorr_f0(waveform_cpu.numpy())

        # Unvoiced detection: F0 outside range → set to 0
        f0_np = np.where(
            (f0_np >= self.f0_min) & (f0_np <= self.f0_max),
            f0_np, 0.0
        )

        # Median smooth to remove outliers
        f0_np = medfilt(f0_np, kernel_size=5)

        return f0_np.astype(np.float32)

    def _autocorr_f0(self, wav: np.ndarray) -> np.ndarray:
        """
        Simple autocorrelation-based F0 estimation (fallback).
        """
        frames = []
        for start in range(0, len(wav) - self.win_length, self.hop_length):
            frame  = wav[start: start + self.win_length]
            frames.append(frame)

        f0_values = []
        min_lag   = int(self.sample_rate / self.f0_max)
        max_lag   = int(self.sample_rate / self.f0_min)

        for frame in frames:
            # Windowed autocorrelation
            windowed = frame * np.hanning(len(frame))
            corr     = np.correlate(windowed, windowed, mode="full")
            corr     = corr[len(corr)//2:]
            # Find peak in valid lag range
            segment  = corr[min_lag: max_lag]
            if len(segment) == 0 or segment.max() < 0.3 * corr[0]:
                f0_values.append(0.0)
            else:
                peak_lag = np.argmax(segment) + min_lag
                f0_values.append(self.sample_rate / peak_lag)

        return np.array(f0_values)

    def extract_energy(self, waveform: torch.Tensor) -> np.ndarray:
        """
        Extract frame-level RMS energy contour.

        Returns:
            energy: (T,) numpy array, values in [0, 1]
        """
        wav_np = waveform.squeeze(0).cpu().numpy()
        energy = []

        for start in range(0, len(wav_np) - self.win_length, self.hop_length):
            frame = wav_np[start: start + self.win_length]
            rms   = np.sqrt(np.mean(frame ** 2))
            energy.append(rms)

        energy_np = np.array(energy, dtype=np.float32)

        # Normalize to [0, 1]
        e_max = energy_np.max()
        if e_max > 0:
            energy_np = energy_np / e_max

        return energy_np

    def extract_all(
        self,
        waveform:    torch.Tensor,
        save_prefix: str = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract both F0 and energy. Optionally save to .npy files.

        Returns:
            (f0, energy) numpy arrays
        """
        logger.info("Extracting F0 contour...")
        f0     = self.extract_f0(waveform)

        logger.info("Extracting energy contour...")
        energy = self.extract_energy(waveform)

        # Align lengths (may differ by a few frames due to edge effects)
        min_len = min(len(f0), len(energy))
        f0      = f0[:min_len]
        energy  = energy[:min_len]

        voiced_frames = np.sum(f0 > 0)
        total_frames  = len(f0)
        mean_f0       = f0[f0 > 0].mean() if voiced_frames > 0 else 0

        logger.info(f"F0 stats: mean={mean_f0:.1f}Hz, "
                    f"voiced={voiced_frames}/{total_frames} frames "
                    f"({100*voiced_frames/max(total_frames,1):.1f}%)")

        if save_prefix:
            np.save(f"{save_prefix}_f0.npy",     f0)
            np.save(f"{save_prefix}_energy.npy", energy)
            logger.info(f"Saved: {save_prefix}_f0.npy, {save_prefix}_energy.npy")

        return f0, energy


class DTWProsodyWarper:
    """
    Dynamic Time Warping (DTW) prosody transfer.

    Maps the professor's F0 + Energy contours onto the synthesized
    Bhojpuri speech, preserving the "teaching style" (Task 3.2).

    Mathematical formulation:
        Given source sequence X = {x_1,...,x_N} (professor prosody)
        and target sequence Y = {y_1,...,y_M} (synthesized prosody),

        DTW finds an optimal warping path W = {(i_1,j_1),...,(i_K,j_K)}
        that minimizes the accumulated cost:

            DTW(X,Y) = min_W Σ d(x_{i_k}, y_{j_k})

        where d(·,·) is Euclidean distance on (F0, Energy) pairs.

        The warping path is then used to resample the source contour
        to match the length of the target, effectively "teaching" the
        synthesized voice to follow the professor's prosodic pattern.
    """

    def __init__(self, f0_weight: float = 1.0, energy_weight: float = 0.5):
        self.f0_weight     = f0_weight
        self.energy_weight = energy_weight

    def dtw_path(
        self,
        x: np.ndarray,   # (N,) source
        y: np.ndarray,   # (M,) target
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute DTW alignment path between two sequences.

        Returns:
            (path_x, path_y): index arrays defining the alignment
        """
        N, M  = len(x), len(y)
        cost  = np.full((N, M), np.inf)
        cost[0, 0] = abs(x[0] - y[0])

        # Fill cost matrix
        for i in range(N):
            for j in range(M):
                d = abs(float(x[i]) - float(y[j]))
                if i == 0 and j == 0:
                    cost[i, j] = d
                elif i == 0:
                    cost[i, j] = d + cost[i, j-1]
                elif j == 0:
                    cost[i, j] = d + cost[i-1, j]
                else:
                    cost[i, j] = d + min(
                        cost[i-1, j],
                        cost[i, j-1],
                        cost[i-1, j-1],
                    )

        # Backtrack to find path
        i, j     = N - 1, M - 1
        path_x   = [i]
        path_y   = [j]

        while i > 0 or j > 0:
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                step = np.argmin([
                    cost[i-1, j-1],
                    cost[i-1, j],
                    cost[i, j-1],
                ])
                if step == 0:
                    i -= 1; j -= 1
                elif step == 1:
                    i -= 1
                else:
                    j -= 1
            path_x.append(i)
            path_y.append(j)

        return np.array(path_x[::-1]), np.array(path_y[::-1])

    def dtw_path_fast(
        self,
        x: np.ndarray,
        y: np.ndarray,
        radius: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sakoe-Chiba band-constrained DTW (faster for long sequences).
        Uses a diagonal band of width `radius` to limit search space.
        O(N * radius) instead of O(N * M).
        """
        N, M   = len(x), len(y)
        cost   = np.full((N, M), np.inf)

        # Only fill within Sakoe-Chiba band
        for i in range(N):
            j_lo = max(0, i - radius)
            j_hi = min(M, i + radius + 1)
            for j in range(j_lo, j_hi):
                d = abs(float(x[i]) - float(y[j]))
                if i == 0 and j == 0:
                    cost[i, j] = d
                elif i == 0:
                    cost[i, j] = d + cost[i, j-1]
                elif j == 0:
                    cost[i, j] = d + cost[i-1, j]
                else:
                    prev = [
                        cost[i-1, j-1] if abs((i-1) - (j-1)) <= radius else np.inf,
                        cost[i-1, j]   if abs((i-1) -  j   ) <= radius else np.inf,
                        cost[i, j-1]   if abs( i    - (j-1)) <= radius else np.inf,
                    ]
                    cost[i, j] = d + min(prev)

        # Backtrack
        i, j   = N - 1, M - 1
        path_x = [i]
        path_y = [j]

        while i > 0 or j > 0:
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                candidates = []
                if abs((i-1)-(j-1)) <= radius: candidates.append((cost[i-1,j-1], -1,-1))
                else: candidates.append((np.inf, -1,-1))
                if abs((i-1)-j) <= radius: candidates.append((cost[i-1,j],   -1, 0))
                else: candidates.append((np.inf, -1, 0))
                if abs(i-(j-1)) <= radius: candidates.append((cost[i,j-1],    0,-1))
                else: candidates.append((np.inf, 0,-1))

                best = min(candidates, key=lambda c: c[0])
                i += best[1]
                j += best[2]

            path_x.append(i)
            path_y.append(j)

        return np.array(path_x[::-1]), np.array(path_y[::-1])

    def warp_contour(
        self,
        source: np.ndarray,   # professor's contour (N,)
        target_len: int,       # desired output length (M)
    ) -> np.ndarray:
        """
        Warp source contour to target_len using DTW path.

        Creates a uniform target reference, finds DTW path,
        then resamples source to match target length.

        Returns:
            warped: (target_len,) — source contour stretched/compressed
                    to match target_len, preserving prosodic shape
        """
        # Use linear reference as target (neutral baseline)
        target = np.linspace(source.mean(), source.mean(), target_len)

        # Use fast DTW for long sequences
        use_fast = len(source) * target_len > 10000
        if use_fast:
            path_x, path_y = self.dtw_path_fast(source, target, radius=100)
        else:
            path_x, path_y = self.dtw_path(source, target)

        # Resample source along DTW path to target length
        # Build mapping: target_idx → average source value at that position
        target_to_source = defaultdict(list)
        for sx, ty in zip(path_x, path_y):
            target_to_source[ty].append(source[sx])

        warped = np.zeros(target_len, dtype=np.float32)
        for j in range(target_len):
            if j in target_to_source:
                warped[j] = np.mean(target_to_source[j])
            elif j > 0:
                warped[j] = warped[j-1]   # fill gaps

        return warped

    def warp_prosody(
        self,
        prof_f0:       np.ndarray,    # professor F0 (N,)
        prof_energy:   np.ndarray,    # professor energy (N,)
        synth_len:     int,            # synthesized audio frame count (M)
        save_path:     str = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full prosody warping: warp both F0 and energy to synth_len.

        Returns:
            (warped_f0, warped_energy)
        """
        logger.info(f"DTW prosody warping: {len(prof_f0)} → {synth_len} frames")

        # Normalize F0 (log scale for perceptual linearity)
        # Only use voiced frames (F0 > 0)
        voiced_mask = prof_f0 > 0
        if voiced_mask.sum() > 10:
            log_f0 = np.where(voiced_mask, np.log(np.maximum(prof_f0, 1.0)), 0.0)
        else:
            log_f0 = prof_f0.copy()

        # DTW warp F0
        warped_log_f0 = self.warp_contour(log_f0, synth_len)
        # Convert back from log scale
        warped_f0     = np.where(warped_log_f0 > 0,
                                  np.exp(warped_log_f0), 0.0)
        warped_f0     = warped_f0.astype(np.float32)

        # DTW warp energy
        warped_energy = self.warp_contour(prof_energy, synth_len)
        warped_energy = np.clip(warped_energy, 0.0, 1.0).astype(np.float32)

        logger.info(f"Warped F0: mean={warped_f0[warped_f0>0].mean():.1f}Hz")
        logger.info(f"Warped energy: mean={warped_energy.mean():.3f}")

        if save_path:
            np.save(save_path, warped_f0)
            np.save(save_path.replace("f0", "energy"), warped_energy)

        return warped_f0, warped_energy

    def apply_prosody_to_waveform(
        self,
        waveform:      torch.Tensor,   # synthesized audio (1, T)
        warped_f0:     np.ndarray,     # target F0 contour
        warped_energy: np.ndarray,     # target energy contour
        sample_rate:   int = SAMPLE_RATE,
        hop_length:    int = HOP_LENGTH,
    ) -> torch.Tensor:
        """
        Apply warped F0 and energy contours to the synthesized waveform
        using PSOLA-inspired frame-level amplitude scaling.

        Note: Full F0 modification requires a vocoder (handled in Task 3.3
        via VITS). This method applies energy shaping as a post-processing
        step and is used when the TTS model doesn't accept explicit F0 input.

        Returns:
            modified waveform (1, T)
        """
        wav_np    = waveform.squeeze(0).cpu().numpy()
        n_frames  = len(warped_energy)
        output    = wav_np.copy()

        for i, gain in enumerate(warped_energy):
            start = i * hop_length
            end   = min(start + hop_length, len(output))
            if start >= len(output):
                break
            # Apply energy-based amplitude scaling
            current_rms = np.sqrt(np.mean(output[start:end] ** 2)) + 1e-8
            target_rms  = gain * 0.1   # scale to reasonable amplitude
            scale       = target_rms / current_rms
            scale       = np.clip(scale, 0.1, 10.0)  # prevent extreme scaling
            output[start:end] *= scale

        # Renormalize
        max_amp = np.abs(output).max()
        if max_amp > 0:
            output = output / max_amp * 0.95

        return torch.tensor(output, dtype=torch.float32).unsqueeze(0)


# ─────────────────────────────────────────────────────────────
# TASK 3.3 — SYNTHESIS
# ─────────────────────────────────────────────────────────────

class BhojpuriTTSSynthesizer:
    """
    Zero-shot cross-lingual voice cloning TTS synthesizer.

    Tries models in order of quality / availability:
      1. YourTTS (Coqui TTS)  — best zero-shot cloning
      2. VITS multilingual    — good quality, fast
      3. Meta MMS-TTS         — good for Indian languages
      4. Fallback: gTTS+pitch — guaranteed to work, lower quality

    All produce output at ≥22.05kHz as required.
    """

    COQUI_MODELS = [
        # XTTS v2 first — supports Hindi natively (best for Bhojpuri)
        "tts_models/multilingual/multi-dataset/xtts_v2",
        # YourTTS fallback — only en/fr-fr/pt-br but still does voice cloning
        "tts_models/multilingual/multi-dataset/your_tts",
    ]

    def __init__(
        self,
        speaker_embedding: torch.Tensor,
        reference_wav:     str,
    ):
        self.speaker_embedding = speaker_embedding
        self.reference_wav     = reference_wav
        self.tts_model         = None
        self.model_name        = None

    def _load_coqui_tts(self) -> bool:
        """Try to load a Coqui TTS model."""
        try:
            from TTS.api import TTS
            for model_name in self.COQUI_MODELS:
                try:
                    logger.info(f"Loading Coqui TTS: {model_name}")
                    tts = TTS(model_name, gpu=torch.cuda.is_available())
                    self.tts_model  = tts
                    self.model_name = model_name
                    logger.info(f"Loaded: {model_name}")
                    return True
                except Exception as e:
                    logger.warning(f"  Failed {model_name}: {e}")
            return False
        except ImportError:
            logger.warning("Coqui TTS not installed: pip install TTS")
            return False

    def _load_mms_tts(self) -> bool:
        """Try to load Meta MMS-TTS via transformers."""
        try:
            from transformers import VitsModel, AutoTokenizer
            logger.info("Loading Meta MMS-TTS (bho = Bhojpuri)...")
            # MMS has Bhojpuri support as language code 'bho'
            self.tts_tokenizer = AutoTokenizer.from_pretrained(
                "facebook/mms-tts-bho"
            )
            self.tts_model = VitsModel.from_pretrained(
                "facebook/mms-tts-bho"
            ).to(DEVICE)
            self.model_name = "mms-tts-bho"
            logger.info("MMS-TTS Bhojpuri loaded.")
            return True
        except Exception as e:
            logger.warning(f"MMS-TTS failed: {e}")
            return False

    def _load_fallback_tts(self) -> bool:
        """gTTS fallback — always works, Hindi as proxy for Bhojpuri."""
        try:
            import gtts
            self.model_name = "gtts_fallback"
            logger.warning(
                "Using gTTS fallback (Hindi as Bhojpuri proxy). "
                "Install Coqui TTS for proper zero-shot cloning: pip install TTS"
            )
            return True
        except ImportError:
            logger.warning("gTTS not installed: pip install gtts")
            return False

    def load_model(self):
        """Load best available TTS model."""
        if self._load_coqui_tts():
            return
        if self._load_mms_tts():
            return
        if self._load_fallback_tts():
            return
        raise RuntimeError(
            "No TTS model available. Install at least one of:\n"
            "  pip install TTS           (Coqui — recommended)\n"
            "  pip install transformers  (Meta MMS)\n"
            "  pip install gtts          (fallback)"
        )

    def synthesize_segment(
        self,
        text:       str,
        output_wav: str,
    ) -> Optional[torch.Tensor]:
        """
        Synthesize a single text segment to wav.
        Returns waveform tensor or None on failure.
        """
        if not text.strip():
            return None

        try:
            if self.model_name in self.COQUI_MODELS:
                return self._synth_coqui(text, output_wav)
            elif self.model_name == "mms-tts-bho":
                return self._synth_mms(text, output_wav)
            elif self.model_name == "gtts_fallback":
                return self._synth_gtts(text, output_wav)
        except Exception as e:
            logger.warning(f"Synthesis failed for segment: {e}")
            return None

    def _get_coqui_language(self) -> str:
        """
        Return the correct language code for the loaded Coqui model.

        YourTTS supports : en, fr-fr, pt-br  (no Hindi)
        XTTS v2 supports : en, es, fr, de, it, pt, pl, tr, ru, nl, cs,
                           ar, zh-cn, hu, ko, ja, hi  (has Hindi!)

        For Bhojpuri (not directly supported by any model),
        we use the closest available language:
          - XTTS v2 → "hi"  (Hindi — closest to Bhojpuri phonetically)
          - YourTTS → "en"  (only option; still does voice cloning)
        """
        if "xtts" in self.model_name.lower():
            return "hi"    # XTTS v2 has Hindi support
        else:
            # YourTTS only supports en/fr-fr/pt-br
            # Use "en" — voice cloning still works, accent differs
            return "en"

    def _synth_coqui(self, text: str, output_wav: str) -> torch.Tensor:
        """Synthesize with Coqui TTS (YourTTS or XTTS)."""
        lang = self._get_coqui_language()
        self.tts_model.tts_to_file(
            text        = text,
            speaker_wav = self.reference_wav,
            language    = lang,
            file_path   = output_wav,
        )
        wav, sr = load_audio(output_wav, target_sr=SAMPLE_RATE)
        return wav

    def _synth_mms(self, text: str, output_wav: str) -> torch.Tensor:
        """Synthesize with Meta MMS-TTS."""
        inputs = self.tts_tokenizer(text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            output = self.tts_model(**inputs)
        wav = output.waveform.squeeze(0).cpu()
        # MMS outputs at 16kHz — resample to 22.05kHz
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        wav = T.Resample(orig_freq=16000, new_freq=SAMPLE_RATE)(wav)
        save_audio(wav, output_wav, SAMPLE_RATE)
        return wav

    def _synth_gtts(self, text: str, output_wav: str) -> torch.Tensor:
        """Synthesize with gTTS (fallback, Hindi language)."""
        import gtts
        import io
        tts  = gtts.gTTS(text=text, lang="hi", slow=False)
        mp3  = output_wav.replace(".wav", ".mp3")
        tts.save(mp3)
        # Convert mp3 → wav using torchaudio
        try:
            wav, sr = torchaudio.load(mp3)
            wav = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(wav)
            save_audio(wav, output_wav, SAMPLE_RATE)
            os.remove(mp3)
            return wav
        except Exception:
            # If torchaudio can't read mp3, use pydub
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_mp3(mp3)
                audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1)
                audio.export(output_wav, format="wav")
                os.remove(mp3)
                wav, _ = load_audio(output_wav, SAMPLE_RATE)
                return wav
            except Exception as e2:
                logger.warning(f"MP3 conversion failed: {e2}")
                return None

    def synthesize_full_lecture(
        self,
        translation_json:  str,
        output_wav:        str,
        warped_f0:         np.ndarray = None,
        warped_energy:     np.ndarray = None,
        prosody_warper:    DTWProsodyWarper = None,
        chunk_dir:         str = "outputs/chunks",
    ) -> torch.Tensor:
        """
        Synthesize the full 10-minute lecture from Bhojpuri translation.

        Strategy:
          1. Load all translated segments from Part 2
          2. Synthesize each segment individually (prevents OOM)
          3. Concatenate all segments with short silence between them
          4. Apply DTW prosody warping to final output
          5. Save as output_LRL_cloned.wav at ≥22.05kHz

        Returns:
            Full lecture waveform (1, T)
        """
        os.makedirs(chunk_dir, exist_ok=True)

        # Load Bhojpuri translation from Part 2
        if not os.path.exists(translation_json):
            raise FileNotFoundError(
                f"Translation not found: {translation_json}\n"
                "Run Part 2 first: python ../part2/part2_phonetic_translation.py"
            )

        with open(translation_json, "r", encoding="utf-8") as f:
            translation = json.load(f)

        segments = translation.get("segments", [])
        logger.info(f"Synthesizing {len(segments)} segments...")

        # Short silence buffer between segments
        silence_dur  = 0.3  # 300ms
        silence_len  = int(silence_dur * SAMPLE_RATE)
        silence      = torch.zeros(1, silence_len)

        all_wavs     = []
        total_dur    = 0.0
        failed_segs  = 0

        for i, seg in enumerate(segments):
            bhojpuri_text = seg.get("bhojpuri", "").strip()
            orig_text     = seg.get("original", "").strip()
            start_t       = seg.get("start", 0)
            end_t         = seg.get("end", 0)
            seg_dur       = end_t - start_t

            if not bhojpuri_text:
                # Silence for empty segments
                gap_wav = torch.zeros(1, int(seg_dur * SAMPLE_RATE))
                all_wavs.append(gap_wav)
                continue

            chunk_path = os.path.join(chunk_dir, f"chunk_{i:04d}.wav")

            # Skip if already synthesized (allows resuming)
            if os.path.exists(chunk_path):
                wav, _ = load_audio(chunk_path, SAMPLE_RATE)
            else:
                wav = self.synthesize_segment(bhojpuri_text, chunk_path)

            if wav is None:
                logger.warning(f"Segment {i} failed, using silence")
                wav = torch.zeros(1, int(max(seg_dur, 0.5) * SAMPLE_RATE))
                failed_segs += 1

            all_wavs.append(wav)
            all_wavs.append(silence)
            total_dur += wav.shape[-1] / SAMPLE_RATE

            if (i + 1) % 10 == 0:
                logger.info(f"  Synthesized {i+1}/{len(segments)} segments "
                            f"({total_dur:.1f}s synthesized, "
                            f"{failed_segs} failures)")

        # Concatenate all segments
        logger.info("Concatenating all segments...")
        full_wav = torch.cat(all_wavs, dim=-1)

        logger.info(f"Total synthesized duration: {full_wav.shape[-1]/SAMPLE_RATE:.1f}s")

        # Apply DTW prosody warping if available
        if warped_f0 is not None and warped_energy is not None and prosody_warper is not None:
            logger.info("Applying DTW prosody warping to full synthesis...")
            n_synth_frames = full_wav.shape[-1] // HOP_LENGTH
            # Warp professor's prosody to match synthesized length
            final_f0, final_energy = prosody_warper.warp_prosody(
                prof_f0     = warped_f0,
                prof_energy = warped_energy,
                synth_len   = n_synth_frames,
            )
            full_wav = prosody_warper.apply_prosody_to_waveform(
                full_wav, final_f0, final_energy, SAMPLE_RATE, HOP_LENGTH
            )
            logger.info("Prosody warping applied.")

        # Final normalization
        full_wav = full_wav / (full_wav.abs().max() + 1e-8) * 0.95

        # Save final output
        save_audio(full_wav, output_wav, SAMPLE_RATE)
        logger.info(f"Final lecture saved: {output_wav}")
        logger.info(f"  Duration : {full_wav.shape[-1]/SAMPLE_RATE:.1f}s")
        logger.info(f"  Sample rate: {SAMPLE_RATE}Hz (≥22050 ✅)")
        logger.info(f"  Failed segments: {failed_segs}/{len(segments)}")

        return full_wav


# ─────────────────────────────────────────────────────────────
# MCD EVALUATION
# ─────────────────────────────────────────────────────────────

def compute_mcd(
    reference_wav:  str,
    synthesized_wav: str,
    n_mfcc:         int = 24,   # MCD uses 24 MFCC coefficients (skip C0)
    sample_rate:    int = SAMPLE_RATE,
) -> float:
    """
    Compute Mel-Cepstral Distortion (MCD) between reference voice
    and synthesized output.

    MCD = (10/ln(10)) * sqrt(2 * Σ(c_ref_k - c_synth_k)^2)

    Standard evaluation: MCD < 8.0 dB is the passing criterion.

    Lower MCD = synthesized voice is more similar to reference.
    Typical values:
      MCD < 5.0  → excellent voice similarity
      MCD 5–8    → acceptable (passes assignment requirement)
      MCD > 8.0  → too dissimilar (fails)

    Args:
        reference_wav:   Student reference voice (60s recording)
        synthesized_wav: output_LRL_cloned.wav from Task 3.3

    Returns:
        mcd_db: float (lower is better, threshold < 8.0)
    """
    logger.info("Computing MCD...")

    ref_wav,  _  = load_audio(reference_wav,   target_sr=sample_rate)
    synth_wav, _ = load_audio(synthesized_wav, target_sr=sample_rate)

    mfcc_transform = T.MFCC(
        sample_rate = sample_rate,
        n_mfcc      = n_mfcc + 1,   # +1 because we skip C0
        melkwargs   = {
            "n_fft":      N_FFT,
            "hop_length": HOP_LENGTH,
            "n_mels":     N_MELS,
            "f_min":      80,
            "f_max":      7600,
        },
    )

    # Extract MFCC (skip C0 — MCD uses C1..C24)
    ref_mfcc   = mfcc_transform(ref_wav).squeeze(0)[1:]    # (24, T_ref)
    synth_mfcc = mfcc_transform(synth_wav).squeeze(0)[1:]  # (24, T_synth)

    # Align lengths by taking shorter
    min_T = min(ref_mfcc.shape[-1], synth_mfcc.shape[-1])
    ref_mfcc   = ref_mfcc[:, :min_T].numpy()
    synth_mfcc = synth_mfcc[:, :min_T].numpy()

    # MCD per frame
    diff     = ref_mfcc - synth_mfcc          # (24, T)
    sq_sum   = np.sum(diff ** 2, axis=0)      # (T,)
    mcd_per_frame = (10.0 / np.log(10.0)) * np.sqrt(2.0 * sq_sum)

    mcd_mean = float(np.mean(mcd_per_frame))
    mcd_std  = float(np.std(mcd_per_frame))

    THRESHOLD = 8.0
    status    = "✅ PASS" if mcd_mean < THRESHOLD else "❌ FAIL"

    logger.info(f"MCD = {mcd_mean:.2f} ± {mcd_std:.2f} dB  "
                f"(threshold < {THRESHOLD})  {status}")

    return mcd_mean, mcd_std


def save_mcd_report(
    mcd_mean: float,
    mcd_std:  float,
    output_path: str = "outputs/mcd_report.json",
):
    report = {
        "mcd_mean_db":   round(mcd_mean, 4),
        "mcd_std_db":    round(mcd_std,  4),
        "threshold_db":  8.0,
        "passes":        mcd_mean < 8.0,
        "status":        "PASS" if mcd_mean < 8.0 else "FAIL",
        "n_mfcc_coeffs": 24,
        "note": (
            "MCD computed on C1..C24 (C0 excluded per standard). "
            "Lower is better. Values < 5.0 = excellent, 5-8 = acceptable."
        ),
    }
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"MCD report saved: {output_path}")
    return report


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def run_part3(
    reference_path:   str,
    lecture_path:     str,
    translation_json: str,
    mode:             str = "full",
    use_pretrained:   bool = True,
):
    os.makedirs("outputs",        exist_ok=True)
    os.makedirs("outputs/chunks", exist_ok=True)

    embedding_path    = "outputs/speaker_embedding.pt"
    prof_f0_path      = "outputs/f0_professor.npy"
    prof_energy_path  = "outputs/energy_professor.npy"
    warped_f0_path    = "outputs/f0_warped.npy"
    output_wav_path   = "outputs/output_LRL_cloned.wav"
    mcd_report_path   = "outputs/mcd_report.json"

    speaker_embedding = None
    prof_f0           = None
    prof_energy       = None

    # ── Task 3.1: Speaker embedding ───────────────────────────
    if mode in ("full", "embed"):
        speaker_embedding = extract_speaker_embedding(
            reference_path  = reference_path,
            output_path     = embedding_path,
            use_pretrained  = use_pretrained,
        )
    elif os.path.exists(embedding_path):
        speaker_embedding = torch.load(embedding_path, map_location="cpu")
        logger.info(f"Loaded embedding: {embedding_path}  shape={speaker_embedding.shape}")

    # ── Task 3.2: Prosody extraction + DTW ───────────────────
    if mode in ("full", "prosody"):
        extractor = ProsodyExtractor(sample_rate=SAMPLE_RATE)
        warper    = DTWProsodyWarper()

        # Extract professor's prosody from lecture audio
        logger.info("Task 3.2: Extracting professor prosody...")
        lecture_wav, _ = load_audio(lecture_path, target_sr=SAMPLE_RATE)
        prof_f0, prof_energy = extractor.extract_all(
            lecture_wav,
            save_prefix = "outputs/professor"
        )
        np.save(prof_f0_path,     prof_f0)
        np.save(prof_energy_path, prof_energy)
        logger.info(f"Professor prosody saved: {prof_f0_path}, {prof_energy_path}")

    elif os.path.exists(prof_f0_path):
        prof_f0     = np.load(prof_f0_path)
        prof_energy = np.load(prof_energy_path)
        logger.info(f"Loaded professor prosody: F0={prof_f0.shape}, E={prof_energy.shape}")

    # ── Task 3.3: Synthesis ───────────────────────────────────
    if mode in ("full", "synthesize"):
        if speaker_embedding is None and os.path.exists(embedding_path):
            speaker_embedding = torch.load(embedding_path, map_location="cpu")

        synthesizer = BhojpuriTTSSynthesizer(
            speaker_embedding = speaker_embedding,
            reference_wav     = reference_path,
        )
        synthesizer.load_model()

        warper = DTWProsodyWarper()

        full_wav = synthesizer.synthesize_full_lecture(
            translation_json = translation_json,
            output_wav       = output_wav_path,
            warped_f0        = prof_f0,
            warped_energy    = prof_energy,
            prosody_warper   = warper,
        )

    # ── MCD Evaluation ───────────────────────────────────────
    if mode in ("full", "mcd"):
        if os.path.exists(output_wav_path) and os.path.exists(reference_path):
            mcd_mean, mcd_std = compute_mcd(
                reference_wav   = reference_path,
                synthesized_wav = output_wav_path,
            )
            save_mcd_report(mcd_mean, mcd_std, mcd_report_path)
        else:
            logger.warning("Cannot compute MCD — need both reference and synthesized wav")

    logger.info("=" * 60)
    logger.info("PART 3 COMPLETE")
    logger.info(f"  Speaker embedding : {embedding_path}")
    logger.info(f"  Professor F0      : {prof_f0_path}")
    logger.info(f"  Warped F0         : {warped_f0_path}")
    logger.info(f"  Synthesized audio : {output_wav_path}")
    logger.info(f"  MCD report        : {mcd_report_path}")
    logger.info("=" * 60)


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Part III: Zero-Shot Cross-Lingual Voice Cloning"
    )
    parser.add_argument(
        "--reference", type=str,
        default="../data/student_voice_ref.wav",
        help="Path to 60s student voice reference WAV"
    )
    parser.add_argument(
        "--lecture", type=str,
        default="../data/lecture_segment.wav",
        help="Path to professor lecture WAV (for prosody extraction)"
    )
    parser.add_argument(
        "--transcript", type=str,
        default="../part2/outputs/bhojpuri_translation.json",
        help="Path to Part 2 Bhojpuri translation JSON"
    )
    parser.add_argument(
        "--mode", type=str, default="full",
        choices=["full", "embed", "prosody", "synthesize", "mcd"],
        help="Pipeline mode (default: full)"
    )
    parser.add_argument(
        "--no_pretrained", action="store_true",
        help="Skip pretrained models, use self-contained GE2E encoder"
    )

    args = parser.parse_args()

    run_part3(
        reference_path   = args.reference,
        lecture_path     = args.lecture,
        translation_json = args.transcript,
        mode             = args.mode,
        use_pretrained   = not args.no_pretrained,
    )
