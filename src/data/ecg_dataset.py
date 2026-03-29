"""ECG Dataset with lazy loading for MIMIC-IV-ECG."""
import json
import logging
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import BandpassFilter, ZScoreNormalize

logger = logging.getLogger(__name__)


class ECGDataset(Dataset):
    """Lazy-loading ECG dataset for MIMIC-IV-ECG.

    Expects preprocessed data stored as HDF5 with structure:
        /{record_id}/signals  -> (n_windows, n_leads, n_samples)
        /{record_id}/labels   -> (n_windows,)
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        split_file: Optional[str] = None,
        sample_rate: int = 500,
        window_samples: int = 1000,
        channels: int = 12,
        bandpass_low: float = 0.5,
        bandpass_high: float = 40.0,
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

        self.subjects = self._load_split(split_file, split)
        self.index = self._build_index()
        logger.info(
            f"ECGDataset [{split}]: {len(self.subjects)} records, "
            f"{len(self.index)} windows"
        )

    def _load_split(self, split_file: Optional[str], split: str) -> list[str]:
        if split_file and Path(split_file).exists():
            with open(split_file) as f:
                splits = json.load(f)
            return splits[split]
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
            for rec in self.subjects:
                if rec in f:
                    n_windows = f[rec]["signals"].shape[0]
                    index.extend([(rec, i) for i in range(n_windows)])
        return index

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        rec_id, win_idx = self.index[idx]
        h5_path = self.data_dir / "data.h5"

        with h5py.File(h5_path, "r") as f:
            signal = f[rec_id]["signals"][win_idx]  # (C, T)
            label = int(f[rec_id]["labels"][win_idx])

        signal = signal.astype(np.float32)
        signal = self.bandpass(signal)
        signal = self.normalize(signal)

        signal = torch.from_numpy(signal)
        label = torch.tensor(label, dtype=torch.long)

        sample = {
            "signal": signal,
            "label": label,
            "record_id": rec_id,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
