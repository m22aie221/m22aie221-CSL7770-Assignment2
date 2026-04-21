"""
Speech Understanding - Programming Assignment 2
Part IV: Adversarial Robustness & Spoofing Detection

Tasks:
    4.1 - Anti-Spoofing Classifier (Countermeasure / CM)
          Features : LFCC (Linear Frequency Cepstral Coefficients)
                     CQCC (Constant-Q Cepstral Coefficients)
          Model    : Light CNN + GRU classifier
          Metric   : EER (Equal Error Rate) < 10%
          Test set : student real voice vs Task 3.3 cloned output

    4.2 - Adversarial Noise Injection
          Method   : FGSM (Fast Gradient Sign Method)
          Target   : LID model from Part 1 (flip Hindi → English)
          Constraint: SNR > 40dB (perturbation inaudible to humans)
          Report   : minimum epsilon (ε) required to flip LID prediction

Folder structure:
    /scratch/data/m22aie221/workspace/CSL7770-Assignment2/
        data/
            student_voice_ref.wav         ← real voice (bona fide)
        part1/
            lid_model_best.pt             ← LID model weights
            outputs/
                lid_predictions.json
        part3/
            outputs/
                output_LRL_cloned.wav     ← synthesized voice (spoof)
        part4/
            part4_adversarial.py          ← this file
            outputs/
                lfcc_features_bonafide.npy
                lfcc_features_spoof.npy
                cm_model.pt               ← trained CM classifier
                eer_report.json           ← EER evaluation
                adversarial_segment.wav   ← perturbed 5s segment
                adversarial_report.json   ← FGSM epsilon results
                part4_metrics.json        ← combined scorecard

Usage:
    # Full pipeline
    python part4_adversarial.py \\
        --reference   ../data/student_voice_ref.wav \\
        --synthesized ../part3/outputs/output_LRL_cloned.wav \\
        --lid_model   ../part1/lid_model_best.pt \\
        --lecture     ../data/lecture_segment.wav

    # Individual modes
    python part4_adversarial.py --mode spoof    # Task 4.1 only
    python part4_adversarial.py --mode fgsm     # Task 4.2 only
    python part4_adversarial.py --mode eval     # EER evaluation only
"""

import os
import json
import math
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
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 16000     # 16kHz for feature extraction
N_LFCC      = 60        # number of LFCC coefficients
N_CQCC      = 90        # CQCC bins (9 octaves × 10 bins)
HOP_LENGTH  = 160       # 10ms at 16kHz
WIN_LENGTH  = 400       # 25ms
N_FFT       = 512
SEGMENT_DUR = 5.0       # seconds — fixed segment length for CM

logger.info(f"Device: {DEVICE}")


# ─────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────

def load_audio(
    path:      str,
    target_sr: int  = SAMPLE_RATE,
    mono:      bool = True,
) -> Tuple[torch.Tensor, int]:
    waveform, sr = torchaudio.load(path)
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = T.Resample(orig_freq=sr, new_freq=target_sr)(waveform)
    waveform = waveform / (waveform.abs().max() + 1e-8)
    return waveform, target_sr


def compute_snr(original: torch.Tensor, perturbed: torch.Tensor) -> float:
    """
    Signal-to-Noise Ratio in dB.
    SNR = 10 * log10(||signal||² / ||noise||²)
    Requirement: SNR > 40dB (perturbation inaudible).
    """
    signal_power = (original ** 2).mean().item()
    noise        = perturbed - original
    noise_power  = (noise ** 2).mean().item()
    if noise_power < 1e-12:
        return float("inf")
    snr_db = 10.0 * math.log10(signal_power / (noise_power + 1e-12))
    return snr_db


def segment_audio(
    waveform:   torch.Tensor,
    sr:         int   = SAMPLE_RATE,
    seg_dur:    float = SEGMENT_DUR,
    overlap:    float = 0.0,
) -> List[torch.Tensor]:
    """Split waveform into fixed-length segments."""
    seg_len  = int(seg_dur * sr)
    hop      = int(seg_len * (1.0 - overlap))
    segments = []
    for start in range(0, waveform.shape[-1] - seg_len + 1, hop):
        segments.append(waveform[:, start: start + seg_len])
    # Last partial segment — pad to seg_len
    remainder = waveform.shape[-1] % seg_len
    if remainder > 0 and waveform.shape[-1] > seg_len:
        last = waveform[:, -remainder:]
        pad  = torch.zeros(1, seg_len - remainder)
        segments.append(torch.cat([last, pad], dim=-1))
    return segments


# ─────────────────────────────────────────────────────────────
# TASK 4.1 — FEATURE EXTRACTION: LFCC + CQCC
# ─────────────────────────────────────────────────────────────

class LFCCExtractor:
    """
    Linear Frequency Cepstral Coefficients (LFCC).

    Unlike MFCC which uses a mel (log) frequency scale,
    LFCC uses a linear filter bank — better for capturing
    fine-grained spectral differences between real and synthetic
    speech (used in ASVspoof challenge baselines).

    Pipeline:
        Waveform → STFT → Linear filterbank → Log → DCT → LFCC
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        n_lfcc:      int = N_LFCC,
        n_filter:    int = 70,          # linear filter bank size
        n_fft:       int = N_FFT,
        hop_length:  int = HOP_LENGTH,
        win_length:  int = WIN_LENGTH,
        f_min:       float = 0.0,
        f_max:       float = None,
    ):
        self.sample_rate = sample_rate
        self.n_lfcc      = n_lfcc
        self.n_filter    = n_filter
        self.n_fft       = n_fft
        self.hop_length  = hop_length
        self.win_length  = win_length
        self.f_min       = f_min
        self.f_max       = f_max or sample_rate / 2.0

        # Build linear filterbank (triangular, evenly spaced in Hz)
        self.filterbank  = self._build_linear_filterbank()

    def _build_linear_filterbank(self) -> torch.Tensor:
        """
        Build triangular linear filterbank matrix.
        Shape: (n_filter, n_fft//2 + 1)
        """
        n_bins    = self.n_fft // 2 + 1
        freq_bins = torch.linspace(0, self.sample_rate / 2, n_bins)
        centers   = torch.linspace(self.f_min, self.f_max, self.n_filter + 2)

        filterbank = torch.zeros(self.n_filter, n_bins)
        for m in range(1, self.n_filter + 1):
            f_left   = centers[m - 1]
            f_center = centers[m]
            f_right  = centers[m + 1]

            for k, f in enumerate(freq_bins):
                if f_left <= f <= f_center:
                    filterbank[m-1, k] = (f - f_left) / (f_center - f_left + 1e-8)
                elif f_center < f <= f_right:
                    filterbank[m-1, k] = (f_right - f) / (f_right - f_center + 1e-8)

        return filterbank  # (n_filter, n_bins)

    def extract(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract LFCC features from waveform.

        Args:
            waveform: (1, T) or (T,) float32

        Returns:
            lfcc: (n_lfcc * 3, T_frames)  — LFCC + Δ + ΔΔ
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        device = waveform.device
        fb     = self.filterbank.to(device)

        window = torch.hann_window(self.win_length, device=device)

        # STFT
        stft = torch.stft(
            waveform.squeeze(0),
            n_fft       = self.n_fft,
            hop_length  = self.hop_length,
            win_length  = self.win_length,
            window      = window,
            return_complex = True,
        )                                    # (n_fft//2+1, T)

        power_spec = stft.abs() ** 2         # (n_fft//2+1, T)

        # Apply linear filterbank
        filtered = torch.matmul(fb, power_spec)   # (n_filter, T)
        log_spec = torch.log(filtered + 1e-6)     # log energy

        # DCT to get cepstral coefficients
        # DCT-II: c[k] = Σ log_spec[m] * cos(π*k*(m+0.5)/n_filter)
        T_frames = log_spec.shape[-1]
        m_idx    = torch.arange(self.n_filter, device=device).float()
        k_idx    = torch.arange(self.n_lfcc,   device=device).float()

        dct_mat  = torch.cos(
            math.pi * k_idx.unsqueeze(1) * (m_idx.unsqueeze(0) + 0.5)
            / self.n_filter
        )   # (n_lfcc, n_filter)

        lfcc = torch.matmul(dct_mat, log_spec)   # (n_lfcc, T)

        # Append delta and delta-delta
        delta       = torchaudio.functional.compute_deltas(lfcc.unsqueeze(0)).squeeze(0)
        delta_delta = torchaudio.functional.compute_deltas(delta.unsqueeze(0)).squeeze(0)

        features = torch.cat([lfcc, delta, delta_delta], dim=0)  # (3*n_lfcc, T)
        return features

    def extract_fixed_length(
        self,
        waveform: torch.Tensor,
        target_frames: int = 300,   # ~3s at 10ms hop
    ) -> torch.Tensor:
        """
        Extract LFCC and return fixed-length feature matrix.
        Pads or truncates to target_frames.
        """
        features = self.extract(waveform)   # (3*n_lfcc, T)
        T        = features.shape[-1]

        if T >= target_frames:
            features = features[:, :target_frames]
        else:
            pad      = torch.zeros(features.shape[0], target_frames - T,
                                   device=features.device)
            features = torch.cat([features, pad], dim=-1)

        return features    # (3*n_lfcc, target_frames)


class CQCCExtractor:
    """
    Constant-Q Cepstral Coefficients (CQCC).

    The CQT (Constant-Q Transform) provides geometrically-spaced
    frequency bins — better frequency resolution at low frequencies
    and better time resolution at high frequencies.

    This makes CQCC more sensitive to the artifacts introduced by
    neural vocoders (which tend to leave traces in high-frequency bands).

    Pipeline:
        Waveform → CQT (log-frequency bins) → Log → DCT → CQCC

    Reference: Todisco et al., "Constant Q cepstral coefficients:
    A spoofing countermeasure for automatic speaker verification",
    Computer Speech & Language, 2017.
    """

    def __init__(
        self,
        sample_rate: int   = SAMPLE_RATE,
        n_bins:      int   = N_CQCC,      # total CQT bins
        bins_per_oct: int  = 10,           # bins per octave
        f_min:       float = 32.7,         # C1
        n_cqcc:      int   = 30,           # output cepstral coefficients
        hop_length:  int   = HOP_LENGTH,
    ):
        self.sample_rate  = sample_rate
        self.n_bins       = n_bins
        self.bins_per_oct = bins_per_oct
        self.f_min        = f_min
        self.n_cqcc       = n_cqcc
        self.hop_length   = hop_length
        self.Q            = 1.0 / (2 ** (1.0 / bins_per_oct) - 1)

    def _cqt_approximate(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Approximate CQT via STFT with variable window lengths.
        For each frequency bin, uses a window of length proportional
        to 1/frequency (constant Q property).

        Returns:
            cqt_mag: (n_bins, T) magnitude spectrogram
        """
        device = waveform.device
        wav_np  = waveform.squeeze(0).cpu().numpy()
        n_frames = (len(wav_np) - self.hop_length) // self.hop_length + 1

        cqt_mag = np.zeros((self.n_bins, n_frames), dtype=np.float32)

        for b in range(self.n_bins):
            freq_b   = self.f_min * (2 ** (b / self.bins_per_oct))
            win_len  = int(self.Q * self.sample_rate / freq_b)
            win_len  = max(win_len, 8)
            win_len  = min(win_len, len(wav_np))

            window = np.hanning(win_len)

            for t in range(n_frames):
                start = t * self.hop_length
                end   = start + win_len
                if end > len(wav_np):
                    frame = np.pad(wav_np[start:], (0, end - len(wav_np)))
                else:
                    frame = wav_np[start:end]

                frame  = frame[:win_len] * window
                # Single-frequency DFT at freq_b
                t_idx  = np.arange(len(frame))
                phase  = 2 * np.pi * freq_b * t_idx / self.sample_rate
                mag    = abs(np.sum(frame * np.exp(-1j * phase)))
                cqt_mag[b, t] = mag

        return torch.tensor(cqt_mag, device=device)

    def extract(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract CQCC features.

        Returns:
            cqcc: (n_cqcc * 3, T_frames)  — CQCC + Δ + ΔΔ
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        cqt_mag   = self._cqt_approximate(waveform)          # (n_bins, T)
        log_cqt   = torch.log(cqt_mag + 1e-6)

        # DCT to get cepstral coefficients
        device   = log_cqt.device
        m_idx    = torch.arange(self.n_bins,  device=device).float()
        k_idx    = torch.arange(self.n_cqcc,  device=device).float()

        dct_mat  = torch.cos(
            math.pi * k_idx.unsqueeze(1) * (m_idx.unsqueeze(0) + 0.5)
            / self.n_bins
        )   # (n_cqcc, n_bins)

        cqcc = torch.matmul(dct_mat, log_cqt)   # (n_cqcc, T)

        # Append deltas
        delta       = torchaudio.functional.compute_deltas(cqcc.unsqueeze(0)).squeeze(0)
        delta_delta = torchaudio.functional.compute_deltas(delta.unsqueeze(0)).squeeze(0)

        return torch.cat([cqcc, delta, delta_delta], dim=0)   # (3*n_cqcc, T)

    def extract_fixed_length(
        self,
        waveform: torch.Tensor,
        target_frames: int = 300,
    ) -> torch.Tensor:
        features = self.extract(waveform)
        T        = features.shape[-1]
        if T >= target_frames:
            return features[:, :target_frames]
        pad = torch.zeros(features.shape[0], target_frames - T,
                          device=features.device)
        return torch.cat([features, pad], dim=-1)


# ─────────────────────────────────────────────────────────────
# TASK 4.1 — CM CLASSIFIER MODEL
# ─────────────────────────────────────────────────────────────

class AntiSpoofingCM(nn.Module):
    """
    Countermeasure (CM) classifier for spoof detection.

    Architecture:
        LFCC+CQCC features → LightCNN blocks → GRU → FC → Binary output
        (Bona Fide = 0, Spoof = 1)

    Inspired by the ASVspoof 2021 baseline (LFCC-LCNN).

    Input:  concatenated LFCC + CQCC features
            shape: (batch, feature_dim, time_frames)
    Output: (batch, 2)  logits for [bonafide, spoof]
    """

    def __init__(
        self,
        lfcc_dim:    int = N_LFCC * 3,     # 180
        cqcc_dim:    int = 30 * 3,          # 90
        hidden_dim:  int = 128,
        gru_dim:     int = 64,
        dropout:     float = 0.3,
    ):
        super().__init__()

        in_dim = lfcc_dim + cqcc_dim       # 270

        # LightCNN frontend — captures local spectro-temporal patterns
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv1d(in_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            # Block 2
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            # Block 3
            nn.Conv1d(128, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(2),
        )

        # GRU to capture temporal dependencies
        self.gru = nn.GRU(
            input_size  = hidden_dim,
            hidden_size = gru_dim,
            num_layers  = 2,
            batch_first = True,
            bidirectional = True,
            dropout     = dropout,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(gru_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, feature_dim, time_frames)
        Returns:
            logits: (batch, 2)
        """
        x = self.cnn(x)              # (B, hidden_dim, T//8)
        x = x.permute(0, 2, 1)      # (B, T//8, hidden_dim)
        x, _ = self.gru(x)          # (B, T//8, gru_dim*2)
        x = x.mean(dim=1)           # (B, gru_dim*2) — mean pooling over time
        return self.classifier(x)   # (B, 2)

    def get_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get spoof score (probability of being spoof).
        Higher score = more likely synthetic/spoofed.
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)[:, 1]   # spoof probability


# ─────────────────────────────────────────────────────────────
# TASK 4.1 — FEATURE EXTRACTION PIPELINE
# ─────────────────────────────────────────────────────────────

def extract_cm_features(
    wav_path:      str,
    lfcc:          LFCCExtractor,
    cqcc:          CQCCExtractor,
    label:         int,           # 0=bonafide, 1=spoof
    target_frames: int = 300,
    max_segments:  int = 50,
) -> List[Tuple[torch.Tensor, int]]:
    """
    Extract LFCC+CQCC feature pairs from an audio file.
    Splits audio into 5s segments, extracts features from each.

    Returns:
        List of (feature_tensor, label) pairs
        feature_tensor shape: (lfcc_dim + cqcc_dim, target_frames)
    """
    waveform, sr = load_audio(wav_path, SAMPLE_RATE)
    segments     = segment_audio(waveform, sr, seg_dur=SEGMENT_DUR)
    segments     = segments[:max_segments]

    features_list = []
    logger.info(f"Extracting features from {wav_path} "
                f"({len(segments)} segments, label={'bonafide' if label==0 else 'spoof'})")

    for i, seg in enumerate(segments):
        try:
            seg_device = seg.to(DEVICE)

            lfcc_feat = lfcc.extract_fixed_length(seg_device, target_frames)
            # CQCC is slow (per-bin DFT) — run on CPU
            cqcc_feat = cqcc.extract_fixed_length(seg.cpu(), target_frames).to(DEVICE)

            combined  = torch.cat([lfcc_feat, cqcc_feat], dim=0)  # (270, T)
            features_list.append((combined.cpu(), label))

        except Exception as e:
            logger.warning(f"  Segment {i} failed: {e}")

    logger.info(f"  Extracted {len(features_list)} feature vectors")
    return features_list


# ─────────────────────────────────────────────────────────────
# TASK 4.1 — TRAINING + EER EVALUATION
# ─────────────────────────────────────────────────────────────

class CMTrainer:
    """
    Trains and evaluates the Anti-Spoofing CM classifier.
    """

    def __init__(self, model: AntiSpoofingCM, lr: float = 1e-3):
        self.model     = model.to(DEVICE)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=30
        )
        # Weighted cross-entropy to handle class imbalance
        self.criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, 1.0]).to(DEVICE)
        )

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for features, labels in loader:
            features = features.to(DEVICE)
            labels   = labels.to(DEVICE)
            self.optimizer.zero_grad()
            logits   = self.model(features)
            loss     = self.criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            total_loss += loss.item()
        self.scheduler.step()
        return total_loss / max(len(loader), 1)

    def get_scores_and_labels(
        self,
        loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get spoof scores and ground-truth labels for EER computation."""
        self.model.eval()
        all_scores = []
        all_labels = []
        with torch.no_grad():
            for features, labels in loader:
                features = features.to(DEVICE)
                scores   = self.model.get_score(features)
                all_scores.extend(scores.cpu().numpy())
                all_labels.extend(labels.numpy())
        return np.array(all_scores), np.array(all_labels)

    def compute_eer(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Compute Equal Error Rate (EER).

        EER is the point where:
            FAR (False Accept Rate) = FRR (False Reject Rate)

        FAR = FP / (FP + TN)  — spoof accepted as bonafide
        FRR = FN / (FN + TP)  — bonafide rejected as spoof

        We sweep threshold from 0 to 1 and find crossover point.

        Returns:
            (eer, threshold_at_eer)
        """
        thresholds = np.linspace(0.0, 1.0, 1000)
        far_list   = []
        frr_list   = []

        bonafide_scores = scores[labels == 0]
        spoof_scores    = scores[labels == 1]

        for thresh in thresholds:
            # FAR: spoof accepted (score < thresh means classified as bonafide)
            fa  = np.sum(spoof_scores < thresh)
            far = fa / max(len(spoof_scores), 1)

            # FRR: bonafide rejected (score >= thresh means classified as spoof)
            fr  = np.sum(bonafide_scores >= thresh)
            frr = fr / max(len(bonafide_scores), 1)

            far_list.append(far)
            frr_list.append(frr)

        far_arr = np.array(far_list)
        frr_arr = np.array(frr_list)

        # Find EER: where |FAR - FRR| is minimized
        diff_idx   = np.argmin(np.abs(far_arr - frr_arr))
        eer        = (far_arr[diff_idx] + frr_arr[diff_idx]) / 2.0
        threshold  = thresholds[diff_idx]

        return float(eer), float(threshold)

    def train_and_evaluate(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        epochs:       int = 40,
        model_path:   str = "outputs/cm_model.pt",
    ) -> Dict:
        """Full training loop with best-model checkpointing."""

        best_eer  = 1.0
        best_epoch = 0

        logger.info(f"Training CM classifier for {epochs} epochs...")

        for epoch in range(epochs):
            loss = self.train_epoch(train_loader)

            if (epoch + 1) % 5 == 0:
                scores, labels = self.get_scores_and_labels(val_loader)
                eer, thresh    = self.compute_eer(scores, labels)
                status         = "✅" if eer < 0.10 else "❌"

                logger.info(
                    f"Epoch {epoch+1:3d}/{epochs} | "
                    f"Loss: {loss:.4f} | "
                    f"EER: {eer:.4f} ({eer*100:.1f}%) {status}"
                )

                if eer < best_eer:
                    best_eer   = eer
                    best_epoch = epoch + 1
                    torch.save(self.model.state_dict(), model_path)
                    logger.info(f"  → Best CM model saved (EER={best_eer*100:.1f}%)")

        # Load best
        if os.path.exists(model_path):
            self.model.load_state_dict(
                torch.load(model_path, map_location=DEVICE)
            )

        # Final evaluation
        scores, labels = self.get_scores_and_labels(val_loader)
        eer, thresh    = self.compute_eer(scores, labels)

        # Compute additional metrics
        preds    = (scores >= thresh).astype(int)
        tp       = np.sum((preds == 1) & (labels == 1))
        tn       = np.sum((preds == 0) & (labels == 0))
        fp       = np.sum((preds == 1) & (labels == 0))
        fn       = np.sum((preds == 0) & (labels == 1))
        accuracy = (tp + tn) / max(len(labels), 1)

        result = {
            "eer":            round(eer, 4),
            "eer_percent":    round(eer * 100, 2),
            "threshold":      round(thresh, 4),
            "accuracy":       round(accuracy, 4),
            "best_epoch":     best_epoch,
            "tp": int(tp), "tn": int(tn),
            "fp": int(fp), "fn": int(fn),
            "passes":         eer < 0.10,
            "threshold_req":  0.10,
        }

        self._log_eer_report(result)
        return result

    def _log_eer_report(self, result: Dict):
        status = "✅ PASS" if result["passes"] else "❌ FAIL"
        logger.info("─" * 50)
        logger.info(f"ANTI-SPOOFING CM RESULTS")
        logger.info(f"  EER        : {result['eer_percent']:.1f}%  "
                    f"(threshold < 10%)  {status}")
        logger.info(f"  Accuracy   : {result['accuracy']*100:.1f}%")
        logger.info(f"  Threshold  : {result['threshold']:.4f}")
        logger.info(f"  TP={result['tp']}  TN={result['tn']}  "
                    f"FP={result['fp']}  FN={result['fn']}")
        logger.info("─" * 50)


def run_antispoof_task(
    bonafide_wav:  str,
    spoof_wav:     str,
    model_path:    str = "outputs/cm_model.pt",
    report_path:   str = "outputs/eer_report.json",
    target_frames: int = 300,
    epochs:        int = 40,
) -> Dict:
    """
    Full Task 4.1 pipeline:
      1. Extract LFCC + CQCC from bonafide (real) and spoof (synthesized)
      2. Train CM classifier
      3. Evaluate EER

    Returns:
        EER report dict
    """
    lfcc = LFCCExtractor(sample_rate=SAMPLE_RATE, n_lfcc=N_LFCC)
    cqcc = CQCCExtractor(sample_rate=SAMPLE_RATE, n_bins=N_CQCC, n_cqcc=30)

    logger.info("Task 4.1: Extracting features for anti-spoofing CM...")

    # Extract features
    bonafide_feats = extract_cm_features(
        bonafide_wav, lfcc, cqcc, label=0,
        target_frames=target_frames, max_segments=60
    )
    spoof_feats    = extract_cm_features(
        spoof_wav, lfcc, cqcc, label=1,
        target_frames=target_frames, max_segments=60
    )

    all_feats = bonafide_feats + spoof_feats
    if len(all_feats) < 10:
        raise ValueError(
            f"Not enough features extracted: {len(all_feats)}. "
            "Check that both audio files exist and have sufficient duration."
        )

    # Shuffle and split 80/20
    import random
    random.shuffle(all_feats)
    split      = int(0.8 * len(all_feats))
    train_data = all_feats[:split]
    val_data   = all_feats[split:]

    # Build DataLoaders
    def to_loader(data: List, batch_size=16, shuffle=True) -> DataLoader:
        feats  = torch.stack([f for f, _ in data])
        labels = torch.tensor([l for _, l in data], dtype=torch.long)
        ds     = TensorDataset(feats, labels)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = to_loader(train_data, shuffle=True)
    val_loader   = to_loader(val_data,   shuffle=False)

    logger.info(f"Train: {len(train_data)} | Val: {len(val_data)} segments")

    # Save features for inspection
    os.makedirs("outputs", exist_ok=True)
    np.save("outputs/lfcc_features_bonafide.npy",
            np.array([f[:N_LFCC*3].numpy() for f, l in bonafide_feats]))
    np.save("outputs/lfcc_features_spoof.npy",
            np.array([f[:N_LFCC*3].numpy() for f, l in spoof_feats]))
    logger.info("LFCC features saved for report visualization.")

    # Train model
    model   = AntiSpoofingCM(lfcc_dim=N_LFCC*3, cqcc_dim=30*3)
    trainer = CMTrainer(model)
    result  = trainer.train_and_evaluate(
        train_loader, val_loader,
        epochs=epochs, model_path=model_path
    )

    # Save report
    with open(report_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"EER report saved: {report_path}")

    return result


# ─────────────────────────────────────────────────────────────
# TASK 4.2 — ADVERSARIAL NOISE INJECTION (FGSM)
# ─────────────────────────────────────────────────────────────

class LIDModelWrapper(nn.Module):
    """
    Wraps the Part 1 LID model for gradient computation.
    Needed because FGSM requires gradients w.r.t. the input waveform.

    We rebuild the feature extractor + LID forward pass in a
    single differentiable graph so torch.autograd can compute
    ∂loss/∂input_waveform.
    """

    def __init__(self, lid_model_path: str, sample_rate: int = 16000):
        super().__init__()
        self.sample_rate = sample_rate

        # Import Part 1 classes
        import sys
        sys.path.insert(0, "../part1")

        try:
            from part1_transcription import MultiHeadLIDModel, FrameLevelFeatureExtractor
            self.feature_extractor = FrameLevelFeatureExtractor(sample_rate)
            self.lid_model         = MultiHeadLIDModel()
            state = torch.load(lid_model_path, map_location="cpu")
            self.lid_model.load_state_dict(state)
            logger.info(f"LID model loaded from {lid_model_path}")
        except ImportError as e:
            raise ImportError(
                f"Cannot import Part 1 LID model: {e}\n"
                "Ensure part1_transcription.py is in ../part1/"
            )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (1, T) float32
        Returns:
            lid_logits: (1, T_frames, 2)  — averaged over time → (1, 2)

        Note on cuDNN RNN + backward():
            cuDNN's optimised RNN kernel only supports backward() in
            TRAIN mode. When this wrapper is called from fgsm_step(),
            the model is already switched to train() by fgsm_step before
            calling forward(). When called from _get_dominant_class()
            inside torch.no_grad(), no backward is needed so eval() is fine.
        """
        # Move transforms to same device as input
        self.feature_extractor = self.feature_extractor.to(waveform.device)
        self.lid_model         = self.lid_model.to(waveform.device)

        features = self.feature_extractor(waveform)       # (1, T, 120)
        logits, _, _ = self.lid_model(features)           # (1, T, 2)
        # Average over time frames for a single segment prediction
        avg_logits = logits.mean(dim=1)                   # (1, 2)
        return avg_logits


class FGSMAttacker:
    """
    Fast Gradient Sign Method (FGSM) adversarial attack on LID model.

    FGSM (Goodfellow et al., 2014):
        x_adv = x + ε * sign(∇_x J(θ, x, y_target))

    where:
        x       = original audio segment
        ε       = perturbation magnitude
        J       = cross-entropy loss
        y_target = target class (ENGLISH = 0, to flip Hindi → English)
        ∇_x J   = gradient of loss w.r.t. input

    Constraint: SNR > 40dB
        10 * log10(||x||² / ||ε * sign(∇)||²) > 40
        This limits ε to inaudible levels.

    Task: Find minimum ε such that a Hindi segment is
          misclassified as English by the LID model.
    """

    ENGLISH_LABEL = 0
    HINDI_LABEL   = 1

    def __init__(self, lid_wrapper: LIDModelWrapper):
        self.model = lid_wrapper

    def _get_dominant_class(self, waveform: torch.Tensor) -> int:
        """Get current LID prediction for a waveform segment."""
        self.model.eval()
        with torch.no_grad():
            # Temporarily set all submodules to eval
            # (in case fgsm_step left some in train mode)
            for m in self.model.modules():
                m.training = False
            logits = self.model(waveform)
            return logits.argmax(dim=-1).item()

    def fgsm_step(
        self,
        waveform:    torch.Tensor,   # (1, T) — must require_grad
        target:      int,            # target class (0=English)
        epsilon:     float,
    ) -> torch.Tensor:
        """
        Single FGSM step.

        cuDNN RNN (LSTM/GRU) only allows backward() in TRAIN mode.
        So we temporarily switch to train() just for the forward+backward
        pass, then restore eval() immediately after.

        Returns:
            perturbation: (1, T) — the noise to add
        """
        # cuDNN RNN backward requires train mode — switch temporarily
        self.model.train()
        # Disable dropout stochasticity during FGSM (deterministic grads)
        for m in self.model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.eval()

        waveform = waveform.clone().detach().requires_grad_(True)

        logits = self.model(waveform)                # (1, 2)
        target_tensor = torch.tensor([target],
                                      dtype=torch.long,
                                      device=waveform.device)

        # Loss: minimize cross-entropy to target class
        loss = F.cross_entropy(logits, target_tensor)
        loss.backward()

        # FGSM: take gradient sign, scale by epsilon
        perturbation = epsilon * waveform.grad.sign()

        # Restore eval mode
        self.model.eval()

        return perturbation.detach()

    def find_minimum_epsilon(
        self,
        waveform:     torch.Tensor,    # 5-second Hindi segment (1, T)
        epsilon_min:  float = 1e-6,
        epsilon_max:  float = 1e-2,
        n_steps:      int   = 30,
        snr_threshold: float = 40.0,   # must stay above 40dB
    ) -> Dict:
        """
        Binary search for minimum epsilon that:
          1. Flips LID prediction from Hindi → English
          2. Maintains SNR > 40dB (inaudible perturbation)

        Returns:
            {
              epsilon: float,
              snr_db: float,
              original_class: str,
              adversarial_class: str,
              success: bool,
              epsilon_scan: list of {eps, class, snr_db}
            }
        """
        waveform = waveform.to(DEVICE)

        # Verify original prediction is Hindi
        orig_class = self._get_dominant_class(waveform)
        orig_name  = "hindi"   if orig_class == self.HINDI_LABEL   else "english"
        target     = self.ENGLISH_LABEL   # we want to flip to English

        logger.info(f"Original LID prediction: {orig_name}")
        if orig_class == self.ENGLISH_LABEL:
            logger.warning(
                "Segment is already predicted as English. "
                "Try a different segment with clear Hindi content."
            )

        # Scan epsilon values (log-spaced)
        epsilons   = np.logspace(
            np.log10(epsilon_min),
            np.log10(epsilon_max),
            n_steps
        )

        scan_results = []
        min_eps_flip = None
        adv_waveform_best = None

        for eps in epsilons:
            # Apply FGSM
            perturb   = self.fgsm_step(waveform, target, eps)
            adv_wav   = (waveform + perturb).clamp(-1.0, 1.0)

            snr_db    = compute_snr(waveform, adv_wav)
            adv_class = self._get_dominant_class(adv_wav)
            adv_name  = "hindi" if adv_class == self.HINDI_LABEL else "english"

            flipped   = (adv_class == self.ENGLISH_LABEL and
                         orig_class == self.HINDI_LABEL)
            snr_ok    = snr_db >= snr_threshold

            scan_results.append({
                "epsilon":          round(float(eps), 8),
                "predicted_class":  adv_name,
                "snr_db":           round(snr_db, 2),
                "flipped":          flipped,
                "snr_passes":       snr_ok,
            })

            logger.info(
                f"  ε={eps:.2e} | class={adv_name:7s} | "
                f"SNR={snr_db:.1f}dB | "
                f"{'FLIP✓' if flipped else 'no flip':6s} | "
                f"{'SNR✓' if snr_ok else 'SNR✗':5s}"
            )

            # Track minimum epsilon that flips AND satisfies SNR
            if flipped and snr_ok and min_eps_flip is None:
                min_eps_flip      = eps
                adv_waveform_best = adv_wav.detach().cpu()

        # If we never got a clean flip with SNR constraint,
        # find best flip ignoring SNR (for reporting)
        flip_any = next(
            (r for r in scan_results if r["flipped"]), None
        )

        success = min_eps_flip is not None

        result = {
            "original_class":      orig_name,
            "target_class":        "english",
            "min_epsilon":         round(float(min_eps_flip), 8) if min_eps_flip else None,
            "snr_at_min_epsilon":  None,
            "success":             success,
            "snr_threshold_db":    snr_threshold,
            "epsilon_scan":        scan_results,
            "note": (
                "FGSM applied to 5s Hindi segment from lecture. "
                "min_epsilon is smallest ε that flips LID to English "
                "while keeping SNR > 40dB."
            ),
        }

        if min_eps_flip is not None:
            # Get SNR at min_eps_flip
            for r in scan_results:
                if abs(r["epsilon"] - min_eps_flip) < 1e-10:
                    result["snr_at_min_epsilon"] = r["snr_db"]
                    break

        if not success and flip_any:
            result["flip_without_snr_constraint"] = flip_any
            logger.warning(
                "Could not flip LID with SNR > 40dB. "
                f"Minimum ε to flip (ignoring SNR): {flip_any['epsilon']:.2e}, "
                f"SNR={flip_any['snr_db']:.1f}dB"
            )

        return result, adv_waveform_best

    def iterative_fgsm(
        self,
        waveform:    torch.Tensor,
        target:      int   = 0,     # ENGLISH
        epsilon:     float = 1e-4,
        alpha:       float = 1e-5,  # step size per iteration
        n_iter:      int   = 20,
        snr_threshold: float = 40.0,
    ) -> Tuple[torch.Tensor, float]:
        """
        Iterative FGSM (I-FGSM / PGD) for stronger attack.
        Useful if single-step FGSM cannot flip within SNR budget.

        x_adv^0 = x
        x_adv^{t+1} = clip(x_adv^t + α * sign(∇J), x-ε, x+ε)

        Returns:
            (adversarial_waveform, final_snr_db)
        """
        waveform    = waveform.to(DEVICE)
        adv_wav     = waveform.clone().detach()
        orig_wav    = waveform.clone().detach()

        for i in range(n_iter):
            perturb = self.fgsm_step(adv_wav, target, alpha)
            adv_wav = adv_wav + perturb
            # Project back into epsilon ball
            adv_wav = torch.clamp(adv_wav, orig_wav - epsilon, orig_wav + epsilon)
            adv_wav = torch.clamp(adv_wav, -1.0, 1.0).detach()

            snr_db  = compute_snr(orig_wav, adv_wav)
            cls     = self._get_dominant_class(adv_wav)

            if cls == target and snr_db >= snr_threshold:
                logger.info(
                    f"I-FGSM converged at iter {i+1}: "
                    f"class={cls}, SNR={snr_db:.1f}dB"
                )
                break

        final_snr = compute_snr(orig_wav, adv_wav)
        return adv_wav.cpu(), final_snr


def run_fgsm_task(
    lecture_wav:    str,
    lid_model_path: str,
    output_wav:     str = "outputs/adversarial_segment.wav",
    report_path:    str = "outputs/adversarial_report.json",
    segment_sec:    float = 5.0,
    segment_start:  float = 30.0,   # pick a clearly Hindi segment
) -> Dict:
    """
    Full Task 4.2 pipeline:
      1. Extract a 5s Hindi segment from lecture
      2. Run FGSM epsilon scan
      3. Save adversarial audio and report

    Returns:
        Adversarial attack report dict
    """
    logger.info("Task 4.2: Adversarial noise injection (FGSM)")

    # Load LID model wrapper
    lid_wrapper = LIDModelWrapper(lid_model_path)
    lid_wrapper.eval()

    # Extract 5-second Hindi segment
    waveform, sr = load_audio(lecture_wav, target_sr=16000)
    start_sample = int(segment_start * sr)
    end_sample   = start_sample + int(segment_sec * sr)

    if end_sample > waveform.shape[-1]:
        start_sample = 0
        end_sample   = int(segment_sec * sr)

    segment = waveform[:, start_sample:end_sample]

    logger.info(f"Using lecture segment: {segment_start:.1f}s – "
                f"{segment_start + segment_sec:.1f}s")

    # FGSM epsilon scan
    attacker   = FGSMAttacker(lid_wrapper)

    logger.info("Running FGSM epsilon scan (single-step)...")
    result, adv_wav = attacker.find_minimum_epsilon(
        segment,
        epsilon_min   = 1e-6,
        epsilon_max   = 5e-3,
        n_steps       = 25,
        snr_threshold = 40.0,
    )

    # If single-step failed with SNR constraint, try iterative
    if not result["success"]:
        logger.info("Single-step FGSM failed SNR constraint. "
                    "Trying iterative FGSM (I-FGSM)...")
        adv_wav_iter, iter_snr = attacker.iterative_fgsm(
            segment,
            target        = FGSMAttacker.ENGLISH_LABEL,
            epsilon       = 1e-4,
            alpha         = 5e-6,
            n_iter        = 50,
            snr_threshold = 40.0,
        )
        iter_class = attacker._get_dominant_class(adv_wav_iter.to(DEVICE))
        if iter_class == FGSMAttacker.ENGLISH_LABEL and iter_snr >= 40.0:
            adv_wav = adv_wav_iter
            result["ifgsm_success"]  = True
            result["ifgsm_snr_db"]   = round(iter_snr, 2)
            result["success"]        = True
            result["min_epsilon"]    = 1e-4
            logger.info(f"I-FGSM succeeded: SNR={iter_snr:.1f}dB")

    # Save adversarial audio
    if adv_wav is not None:
        if adv_wav.dim() == 1:
            adv_wav = adv_wav.unsqueeze(0)
        torchaudio.save(output_wav, adv_wav.cpu().float(), sr)
        result["adversarial_wav"] = output_wav
        logger.info(f"Adversarial audio saved: {output_wav}")

        # Also save original segment for comparison
        orig_path = output_wav.replace(".wav", "_original.wav")
        torchaudio.save(orig_path, segment.cpu().float(), sr)
        result["original_wav"] = orig_path

    # Log summary
    status = "✅" if result["success"] else "⚠️"
    logger.info("─" * 50)
    logger.info(f"FGSM RESULTS {status}")
    logger.info(f"  Original class  : {result['original_class']}")
    logger.info(f"  Min epsilon     : {result['min_epsilon']}")
    logger.info(f"  SNR at min eps  : {result.get('snr_at_min_epsilon')}dB")
    logger.info(f"  SNR threshold   : {result['snr_threshold_db']}dB")
    logger.info(f"  Success         : {result['success']}")
    logger.info("─" * 50)

    # Save report
    with open(report_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"FGSM report saved: {report_path}")

    return result


# ─────────────────────────────────────────────────────────────
# COMBINED SCORECARD
# ─────────────────────────────────────────────────────────────

def save_part4_metrics(
    eer_report:  Dict,
    fgsm_report: Dict,
    output_path: str = "outputs/part4_metrics.json",
):
    """Save combined Part 4 metrics scorecard."""

    eer     = eer_report.get("eer", None)
    eps     = fgsm_report.get("min_epsilon", None)
    snr     = fgsm_report.get("snr_at_min_epsilon", None)

    PASS = "PASS"
    FAIL = "FAIL"
    PEND = "PENDING"

    metrics = {
        "anti_spoofing_cm": {
            "eer":          eer,
            "eer_percent":  round(eer * 100, 2) if eer else None,
            "threshold":    0.10,
            "status":       PASS if (eer and eer < 0.10) else FAIL,
            "details":      eer_report,
        },
        "adversarial_robustness": {
            "min_epsilon":      eps,
            "snr_db":           snr,
            "snr_threshold_db": 40.0,
            "flip_success":     fgsm_report.get("success", False),
            "details":          fgsm_report,
        },
        "summary": {
            "eer_passes":   (eer is not None and eer < 0.10),
            "fgsm_success": fgsm_report.get("success", False),
        }
    }

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("=" * 60)
    logger.info("PART 4 SCORECARD")
    logger.info("=" * 60)
    eer_s = f"{eer*100:.1f}%" if eer else "N/A"
    eps_s = f"{eps:.2e}"      if eps else "N/A"
    snr_s = f"{snr:.1f}dB"   if snr else "N/A"
    logger.info(f"  EER         : {eer_s:8s}  (< 10% required)  "
                f"{'✅ PASS' if eer and eer < 0.10 else '❌ FAIL'}")
    logger.info(f"  Min epsilon : {eps_s:8s}")
    logger.info(f"  SNR         : {snr_s:8s}  (> 40dB required)")
    logger.info("=" * 60)

    logger.info(f"Part 4 metrics saved: {output_path}")
    return metrics


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def run_part4(
    reference_wav:  str,
    synthesized_wav: str,
    lid_model_path: str,
    lecture_wav:    str,
    mode:           str = "full",
    epochs:         int = 40,
    segment_start:  float = 30.0,
):
    os.makedirs("outputs", exist_ok=True)

    eer_report_path  = "outputs/eer_report.json"
    fgsm_report_path = "outputs/adversarial_report.json"
    metrics_path     = "outputs/part4_metrics.json"

    eer_report  = {}
    fgsm_report = {}

    # ── Task 4.1: Anti-Spoofing CM ────────────────────────────
    if mode in ("full", "spoof", "eval"):
        if not os.path.exists(reference_wav):
            raise FileNotFoundError(
                f"Reference wav not found: {reference_wav}\n"
                "Record 60s of your voice and save to ../data/student_voice_ref.wav"
            )
        if not os.path.exists(synthesized_wav):
            raise FileNotFoundError(
                f"Synthesized wav not found: {synthesized_wav}\n"
                "Run Part 3 first to generate output_LRL_cloned.wav"
            )

        eer_report = run_antispoof_task(
            bonafide_wav = reference_wav,
            spoof_wav    = synthesized_wav,
            model_path   = "outputs/cm_model.pt",
            report_path  = eer_report_path,
            epochs       = epochs,
        )

    elif os.path.exists(eer_report_path):
        with open(eer_report_path) as f:
            eer_report = json.load(f)
        logger.info(f"Loaded EER report: EER={eer_report.get('eer_percent')}%")

    # ── Task 4.2: FGSM Adversarial Attack ────────────────────
    if mode in ("full", "fgsm"):
        if not os.path.exists(lid_model_path):
            raise FileNotFoundError(
                f"LID model not found: {lid_model_path}\n"
                "Run Part 1 first to train and save lid_model_best.pt"
            )
        if not os.path.exists(lecture_wav):
            raise FileNotFoundError(
                f"Lecture wav not found: {lecture_wav}"
            )

        fgsm_report = run_fgsm_task(
            lecture_wav    = lecture_wav,
            lid_model_path = lid_model_path,
            output_wav     = "outputs/adversarial_segment.wav",
            report_path    = fgsm_report_path,
            segment_start  = segment_start,
        )

    elif os.path.exists(fgsm_report_path):
        with open(fgsm_report_path) as f:
            fgsm_report = json.load(f)

    # ── Combined scorecard ────────────────────────────────────
    if eer_report and fgsm_report:
        save_part4_metrics(eer_report, fgsm_report, metrics_path)


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Part IV: Adversarial Robustness & Spoofing Detection"
    )
    parser.add_argument(
        "--reference", type=str,
        default="../data/student_voice_ref.wav",
        help="Student real voice WAV (bona fide)"
    )
    parser.add_argument(
        "--synthesized", type=str,
        default="../part3/outputs/output_LRL_cloned.wav",
        help="Synthesized Bhojpuri lecture WAV (spoof)"
    )
    parser.add_argument(
        "--lid_model", type=str,
        default="../part1/lid_model_best.pt",
        help="Path to trained LID model weights from Part 1"
    )
    parser.add_argument(
        "--lecture", type=str,
        default="../data/lecture_segment.wav",
        help="Original lecture WAV (source of Hindi segments for FGSM)"
    )
    parser.add_argument(
        "--mode", type=str, default="full",
        choices=["full", "spoof", "fgsm", "eval"],
        help="Pipeline mode (default: full)"
    )
    parser.add_argument(
        "--epochs", type=int, default=40,
        help="CM classifier training epochs (default: 40)"
    )
    parser.add_argument(
        "--segment_start", type=float, default=30.0,
        help="Start time (sec) of Hindi segment for FGSM (default: 30.0)"
    )

    args = parser.parse_args()

    run_part4(
        reference_wav   = args.reference,
        synthesized_wav = args.synthesized,
        lid_model_path  = args.lid_model,
        lecture_wav     = args.lecture,
        mode            = args.mode,
        epochs          = args.epochs,
        segment_start   = args.segment_start,
    )
