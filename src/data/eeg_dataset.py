"""EEG Dataset with lazy loading and bandpass filtering."""
import json
import logging
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import BandpassFilter, ZScoreNormalize, ArtifactRejection

logger = logging.getLogger(__name__)


class EEGDataset(Dataset):
    """Lazy-loading EEG dataset for open-access EEG benchmarks.

    Expects preprocessed data stored as HDF5 with structure:
        /{subject_id}/signals  -> (n_windows, n_channels, n_samples)
        /{subject_id}/labels   -> (n_windows,)
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        split_file: Optional[str] = None,
        sample_rate: int = 125,
        window_samples: int = 500,
        channels: int = 2,
        bandpass_low: float = 0.3,
        bandpass_high: float = 35.0,
        transform: Optional[object] = None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.sample_rate = sample_rate
        self.window_samples = window_samples
        self.channels = channels
        self.transform = transform

        self.bandpass = BandpassFilter(
            low=bandpass_low, high=bandpass_high, fs=sample_rate
        )
        self.normalize = ZScoreNormalize()

        # Load split information
        self.subjects = self._load_split(split_file, split)

        # Build index: list of (subject_id, window_idx)
        self.index = self._build_index()

        # Preload all data into RAM to eliminate HDF5 I/O during training
        self.signals, self.labels = self._preload()

        logger.info(
            f"EEGDataset [{split}]: {len(self.subjects)} subjects, "
            f"{len(self.index)} windows (preloaded to RAM)"
        )

    def _load_split(self, split_file: Optional[str], split: str) -> list[str]:
        if split_file and Path(split_file).exists():
            with open(split_file) as f:
                splits = json.load(f)
            return splits[split]
        # Fallback: list all subjects in the HDF5 file
        h5_path = self.data_dir / "data.h5"
        if h5_path.exists():
            with h5py.File(h5_path, "r") as f:
                return list(f.keys())
        return []

    def _build_index(self) -> list[tuple[str, int]]:
        index = []
        h5_path = self.data_dir / "data.h5"
        if not h5_path.exists():
            logger.warning(f"Data file not found: {h5_path}")
            return index
        with h5py.File(h5_path, "r") as f:
            for subj in self.subjects:
                if subj in f:
                    n_windows = f[subj]["signals"].shape[0]
                    index.extend([(subj, i) for i in range(n_windows)])
        return index

    def _preload(self):
        """Load all signals and labels into RAM (entire subjects at once)."""
        h5_path = self.data_dir / "data.h5"
        if not h5_path.exists() or len(self.index) == 0:
            return [], []
        all_signals = [None] * len(self.index)
        all_labels = [None] * len(self.index)
        # Group indices by subject for bulk reads
        subj_ranges = {}
        for i, (subj, win_idx) in enumerate(self.index):
            if subj not in subj_ranges:
                subj_ranges[subj] = []
            subj_ranges[subj].append((i, win_idx))
        with h5py.File(h5_path, "r") as f:
            for subj, entries in subj_ranges.items():
                sigs = f[subj]["signals"][:]  # read entire subject at once
                labs = f[subj]["labels"][:]
                for i, win_idx in entries:
                    all_signals[i] = sigs[win_idx].astype(np.float32)
                    all_labels[i] = int(labs[win_idx])
        logger.info(f"Preloaded {len(all_signals)} windows into RAM")
        return all_signals, all_labels

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        signal = self.signals[idx]  # (C, T) already float32
        label = self.labels[idx]

        # Per-channel z-score (bandpass already applied during preprocessing)
        signal = self.normalize(signal)

        signal = torch.from_numpy(signal)
        label = torch.tensor(label, dtype=torch.long)

        sample = {
            "signal": signal,       # (C, T)
            "label": label,         # scalar
            "subject_id": self.index[idx][0],
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
