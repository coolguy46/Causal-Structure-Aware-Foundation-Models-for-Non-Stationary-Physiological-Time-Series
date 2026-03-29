"""Generate synthetic data for testing the full pipeline without real datasets.

Creates fake HDF5 files with the same structure as preprocessed real data.
Useful for verifying the training loop, model, and evaluation on Vast.ai
before real data is downloaded.
"""
import argparse
import json
import logging
from pathlib import Path

import h5py
import numpy as np

from src.data.splits import subject_stratified_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_synthetic_eeg(
    output_dir: str,
    n_subjects: int = 50,
    n_windows_per_subject: int = 100,
    n_channels: int = 2,
    window_samples: int = 3000,
    n_classes: int = 5,
    sample_rate: float = 100.0,
):
    """Generate synthetic EEG data mimicking Sleep-EDF structure."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    h5_path = out_path / "data.h5"
    subject_ids = []

    logger.info(f"Generating synthetic EEG: {n_subjects} subjects, "
                f"{n_windows_per_subject} windows each")

    with h5py.File(h5_path, "w") as h5f:
        for i in range(n_subjects):
            subj_id = f"synth_eeg_{i:04d}"
            subject_ids.append(subj_id)

            # Generate signals with frequency content
            t = np.linspace(0, window_samples / sample_rate,
                          window_samples, dtype=np.float32)

            signals = []
            labels = []
            for w in range(n_windows_per_subject):
                label = np.random.randint(0, n_classes)

                # Create signal with class-dependent frequency content
                signal = np.zeros((n_channels, window_samples), dtype=np.float32)
                for ch in range(n_channels):
                    # Base rhythm depends on class
                    freq = 2.0 + label * 3.0 + ch * 0.5  # Hz
                    amplitude = 1.0 + 0.2 * label
                    signal[ch] = amplitude * np.sin(2 * np.pi * freq * t)

                    # Add harmonics
                    signal[ch] += 0.3 * np.sin(2 * np.pi * (freq * 2) * t)

                    # Add noise
                    signal[ch] += np.random.randn(window_samples).astype(np.float32) * 0.5

                signals.append(signal)
                labels.append(label)

            signals = np.stack(signals)  # (n_windows, C, T)
            labels = np.array(labels, dtype=np.int64)

            grp = h5f.create_group(subj_id)
            grp.create_dataset("signals", data=signals, dtype="float32")
            grp.create_dataset("labels", data=labels, dtype="int64")

    logger.info(f"Saved to {h5_path}")

    # Create splits
    subject_stratified_split(
        subject_ids, output_path=str(out_path / "splits.json")
    )


def generate_synthetic_ecg(
    output_dir: str,
    n_subjects: int = 50,
    n_windows_per_subject: int = 100,
    n_channels: int = 12,
    window_samples: int = 1000,
    n_classes: int = 5,
    sample_rate: float = 500.0,
):
    """Generate synthetic ECG data mimicking PTB-XL structure."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    h5_path = out_path / "data.h5"
    record_ids = []

    logger.info(f"Generating synthetic ECG: {n_subjects} records, "
                f"{n_windows_per_subject} windows each")

    with h5py.File(h5_path, "w") as h5f:
        for i in range(n_subjects):
            rec_id = f"synth_ecg_{i:04d}"
            record_ids.append(rec_id)

            t = np.linspace(0, window_samples / sample_rate,
                          window_samples, dtype=np.float32)

            signals = []
            labels = []
            for w in range(n_windows_per_subject):
                label = np.random.randint(0, n_classes)

                signal = np.zeros((n_channels, window_samples), dtype=np.float32)
                for ch in range(n_channels):
                    # QRS-like morphology with class variation
                    heart_rate = 60 + label * 10 + np.random.randn() * 5
                    freq = heart_rate / 60.0
                    signal[ch] = np.sin(2 * np.pi * freq * t)
                    signal[ch] += 0.5 * np.sin(2 * np.pi * 2 * freq * t)
                    signal[ch] += np.random.randn(window_samples).astype(np.float32) * 0.3

                signals.append(signal)
                labels.append(label)

            signals = np.stack(signals)
            labels = np.array(labels, dtype=np.int64)

            grp = h5f.create_group(rec_id)
            grp.create_dataset("signals", data=signals, dtype="float32")
            grp.create_dataset("labels", data=labels, dtype="int64")

    logger.info(f"Saved to {h5_path}")
    subject_stratified_split(record_ids, output_path=str(out_path / "splits.json"))


def generate_synthetic_seizure(
    output_dir: str,
    n_subjects: int = 50,
    n_windows_per_subject: int = 100,
    n_channels: int = 23,
    window_samples: int = 512,
    sample_rate: float = 256.0,
):
    """Generate synthetic seizure EEG data mimicking CHB-MIT structure."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    h5_path = out_path / "data.h5"
    subject_ids = []

    logger.info(f"Generating synthetic CHB-MIT: {n_subjects} subjects")

    with h5py.File(h5_path, "w") as h5f:
        for i in range(n_subjects):
            subj_id = f"synth_chbmit_{i:04d}"
            subject_ids.append(subj_id)

            t = np.linspace(0, window_samples / sample_rate,
                          window_samples, dtype=np.float32)

            signals = []
            labels = []
            for w in range(n_windows_per_subject):
                # Binary: 0 = normal, 1 = seizure (10% seizure rate)
                label = 1 if np.random.random() < 0.1 else 0

                signal = np.zeros((n_channels, window_samples), dtype=np.float32)
                for ch in range(n_channels):
                    if label == 1:
                        # Seizure: high-frequency, high-amplitude
                        signal[ch] = 3.0 * np.sin(2 * np.pi * 15 * t)
                        signal[ch] += 2.0 * np.sin(2 * np.pi * 25 * t)
                    else:
                        # Normal: low-frequency, moderate amplitude
                        signal[ch] = np.sin(2 * np.pi * 8 * t)
                        signal[ch] += 0.5 * np.sin(2 * np.pi * 3 * t)

                    signal[ch] += np.random.randn(window_samples).astype(np.float32) * 0.5

                signals.append(signal)
                labels.append(label)

            signals = np.stack(signals)
            labels = np.array(labels, dtype=np.int64)

            grp = h5f.create_group(subj_id)
            grp.create_dataset("signals", data=signals, dtype="float32")
            grp.create_dataset("labels", data=labels, dtype="int64")

    logger.info(f"Saved to {h5_path}")
    subject_stratified_split(subject_ids, output_path=str(out_path / "splits.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic data for testing")
    parser.add_argument("--data_root", default="/workspace/data")
    parser.add_argument(
        "--dataset", choices=["sleep_edf", "chbmit", "ptbxl", "all"], default="all"
    )
    parser.add_argument("--n_subjects", type=int, default=50)
    parser.add_argument("--n_windows", type=int, default=100)
    args = parser.parse_args()

    generators = {
        "sleep_edf": lambda: generate_synthetic_eeg(
            f"{args.data_root}/sleep_edf",
            n_subjects=args.n_subjects,
            n_windows_per_subject=args.n_windows,
        ),
        "chbmit": lambda: generate_synthetic_seizure(
            f"{args.data_root}/chbmit",
            n_subjects=args.n_subjects,
            n_windows_per_subject=args.n_windows,
        ),
        "ptbxl": lambda: generate_synthetic_ecg(
            f"{args.data_root}/ptbxl",
            n_subjects=args.n_subjects,
            n_windows_per_subject=args.n_windows,
        ),
    }

    if args.dataset == "all":
        for name, fn in generators.items():
            fn()
    else:
        generators[args.dataset]()

    logger.info("Done! Synthetic data ready for training.")
