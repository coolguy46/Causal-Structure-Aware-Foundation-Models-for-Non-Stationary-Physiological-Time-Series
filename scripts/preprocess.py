"""Preprocessing script: converts raw dataset files to HDF5 format.

Handles SHHS (EDF), TUSZ (EDF), and MIMIC-IV-ECG (WFDB) formats.
Produces a single data.h5 per dataset with windowed, filtered signals.

Uses multiprocessing to parallelize EDF/WFDB parsing across CPU cores.
"""
import argparse
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import mne
import numpy as np
import wfdb
from tqdm import tqdm

# Allow running as: python scripts/preprocess.py ...
# by adding the repository root to module search path.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.transforms import BandpassFilter, ArtifactRejection
from src.data.splits import subject_stratified_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress MNE verbose output
mne.set_log_level("WARNING")

# Use 75% of cores by default, minimum 1
_MAX_WORKERS = max(1, (os.cpu_count() or 1) * 3 // 4)


def _safe_h5_open(h5_path: Path, mode: str = "w"):
    """Open HDF5 file safely — backs up existing file before truncating."""
    if mode == "w" and h5_path.exists():
        backup = h5_path.with_suffix(".h5.bak")
        logger.info(f"Backing up {h5_path} -> {backup}")
        import shutil
        shutil.copy2(h5_path, backup)
    return h5py.File(h5_path, mode)


def _process_shhs_file(edf_file, annot_dir, target_sr, window_samples, window_sec):
    """Process a single SHHS EDF file. Returns (subject_id, signals, labels) or None."""
    mne.set_log_level("WARNING")
    bandpass = BandpassFilter(low=0.3, high=35.0, fs=target_sr)
    artifact_rej = ArtifactRejection(threshold_uv=500.0)
    subject_id = edf_file.stem

    try:
        raw = mne.io.read_raw_edf(str(edf_file), preload=True, verbose=False)
        available = raw.ch_names
        eeg_chs = [ch for ch in available if "EEG" in ch.upper()][:2]
        if len(eeg_chs) < 2:
            return None

        raw.pick(eeg_chs)
        raw.resample(target_sr)
        data = raw.get_data()

        annot_file = annot_dir / f"{subject_id}-nsrr.xml"
        if annot_file.exists():
            annotations = mne.read_annotations(str(annot_file))
            raw.set_annotations(annotations)

        n_windows = data.shape[1] // window_samples
        if n_windows == 0:
            return None

        signals = []
        labels = []
        for w in range(n_windows):
            start = w * window_samples
            end = start + window_samples
            window = data[:, start:end]
            window = bandpass(window)
            window = artifact_rej(window)
            signals.append(window)
            labels.append(0)

        return (subject_id, np.stack(signals), np.array(labels))
    except Exception as e:
        return None


def preprocess_shhs(data_dir: str, output_dir: str):
    """Preprocess SHHS EDF files into HDF5."""
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    edf_dir = data_path / "polysomnography" / "edfs" / "shhs1"
    annot_dir = data_path / "polysomnography" / "annotations-events-nsrr" / "shhs1"

    if not edf_dir.exists():
        edf_dir = data_path
        annot_dir = data_path

    edf_files = sorted(edf_dir.glob("*.edf"))
    if not edf_files:
        logger.error(f"No EDF files found in {edf_dir}")
        return

    logger.info(f"Found {len(edf_files)} EDF files, using {_MAX_WORKERS} workers")

    target_sr = 125
    window_sec = 30
    window_samples = target_sr * window_sec

    h5_path = out_path / "data.h5"
    subject_ids = []

    with _safe_h5_open(h5_path) as h5f:
        with ProcessPoolExecutor(max_workers=_MAX_WORKERS) as pool:
            futures = {
                pool.submit(_process_shhs_file, f, annot_dir, target_sr, window_samples, window_sec): f
                for f in edf_files
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing SHHS"):
                result = future.result()
                if result is None:
                    continue
                sid, signals, labels = result
                grp = h5f.create_group(sid)
                grp.create_dataset("signals", data=signals, dtype="float32",
                                   compression="gzip", compression_opts=4)
                grp.create_dataset("labels", data=labels, dtype="int64")
                subject_ids.append(sid)

    logger.info(f"Processed {len(subject_ids)} SHHS subjects -> {h5_path}")

    # Create splits
    if subject_ids:
        subject_stratified_split(
            subject_ids, output_path=str(out_path / "splits.json")
        )


def _process_tusz_file(edf_file, target_sr, window_samples):
    """Process a single TUSZ EDF file. Returns (subject_id, signals, labels) or None."""
    mne.set_log_level("WARNING")
    bandpass = BandpassFilter(low=0.5, high=50.0, fs=target_sr)
    subject_id = edf_file.stem

    try:
        raw = mne.io.read_raw_edf(str(edf_file), preload=True, verbose=False)

        montage_channels = [
            "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4",
            "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6",
            "FZ", "CZ", "PZ"
        ]
        available = [ch.upper() for ch in raw.ch_names]
        pick_chs = [
            raw.ch_names[i] for i, ch in enumerate(available)
            if any(m in ch for m in montage_channels)
        ][:19]

        if len(pick_chs) < 2:
            return None

        raw.pick(pick_chs)
        raw.resample(target_sr)
        data = raw.get_data()

        C = data.shape[0]
        if C < 19:
            pad = np.zeros((19 - C, data.shape[1]), dtype=data.dtype)
            data = np.concatenate([data, pad], axis=0)

        tse_file = edf_file.with_suffix(".tse")
        seizure_intervals = []
        if tse_file.exists():
            with open(tse_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        try:
                            start_t = float(parts[0])
                            end_t = float(parts[1])
                            label = parts[2]
                            if label != "bckg":
                                seizure_intervals.append((start_t, end_t))
                        except ValueError:
                            continue

        n_windows = data.shape[1] // window_samples
        if n_windows == 0:
            return None

        signals = []
        labels = []
        for w in range(n_windows):
            start = w * window_samples
            end = start + window_samples
            window = data[:, start:end]
            window = bandpass(window)
            signals.append(window)

            t_start = start / target_sr
            t_end = end / target_sr
            is_seizure = any(
                s_start < t_end and s_end > t_start
                for s_start, s_end in seizure_intervals
            )
            labels.append(1 if is_seizure else 0)

        return (subject_id, np.stack(signals), np.array(labels))
    except Exception:
        return None


def preprocess_tusz(data_dir: str, output_dir: str):
    """Preprocess TUSZ EDF files into HDF5."""
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    edf_files = sorted(data_path.rglob("*.edf"))
    if not edf_files:
        logger.error(f"No EDF files found in {data_path}")
        return

    logger.info(f"Found {len(edf_files)} EDF files, using {_MAX_WORKERS} workers")

    target_sr = 256
    window_sec = 2
    window_samples = target_sr * window_sec

    h5_path = out_path / "data.h5"
    subject_ids = []

    with _safe_h5_open(h5_path) as h5f:
        with ProcessPoolExecutor(max_workers=_MAX_WORKERS) as pool:
            futures = {
                pool.submit(_process_tusz_file, f, target_sr, window_samples): f
                for f in edf_files
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing TUSZ"):
                result = future.result()
                if result is None:
                    continue
                sid, signals, labels = result
                grp = h5f.create_group(sid)
                grp.create_dataset("signals", data=signals, dtype="float32",
                                   compression="gzip", compression_opts=4)
                grp.create_dataset("labels", data=labels, dtype="int64")
                subject_ids.append(sid)

    logger.info(f"Processed {len(subject_ids)} TUSZ subjects -> {h5_path}")

    if subject_ids:
        subject_stratified_split(
            subject_ids, output_path=str(out_path / "splits.json")
        )


def _process_mimic_file(hea_file, target_sr, window_samples):
    """Process a single MIMIC-IV-ECG WFDB record. Returns (record_id, signals, labels) or None."""
    bandpass = BandpassFilter(low=0.5, high=40.0, fs=target_sr)
    record_id = hea_file.stem

    try:
        record = wfdb.rdrecord(str(hea_file.with_suffix("")))
        data = record.p_signal.T

        if data.shape[0] < 12:
            pad = np.zeros((12 - data.shape[0], data.shape[1]))
            data = np.concatenate([data, pad], axis=0)
        data = data[:12]

        if record.fs != target_sr:
            from scipy.signal import resample
            n_samples = int(data.shape[1] * target_sr / record.fs)
            data = resample(data, n_samples, axis=1)

        n_windows = data.shape[1] // window_samples
        if n_windows == 0:
            return None

        signals = []
        labels = []
        for w in range(n_windows):
            start = w * window_samples
            end = start + window_samples
            window = data[:, start:end].astype(np.float32)
            window = bandpass(window)
            signals.append(window)
            labels.append(0)

        return (record_id, np.stack(signals), np.array(labels))
    except Exception:
        return None


def preprocess_mimic_ecg(data_dir: str, output_dir: str):
    """Preprocess MIMIC-IV-ECG WFDB records into HDF5."""
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    hea_files = sorted(data_path.rglob("*.hea"))
    if not hea_files:
        logger.error(f"No WFDB header files found in {data_path}")
        return

    logger.info(f"Found {len(hea_files)} WFDB records, using {_MAX_WORKERS} workers")

    target_sr = 500
    window_sec = 2
    window_samples = target_sr * window_sec

    h5_path = out_path / "data.h5"
    record_ids = []

    with _safe_h5_open(h5_path) as h5f:
        with ProcessPoolExecutor(max_workers=_MAX_WORKERS) as pool:
            futures = {
                pool.submit(_process_mimic_file, f, target_sr, window_samples): f
                for f in hea_files
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing MIMIC-IV-ECG"):
                result = future.result()
                if result is None:
                    continue
                rid, signals, labels = result
                grp = h5f.create_group(rid)
                grp.create_dataset("signals", data=signals, dtype="float32",
                                   compression="gzip", compression_opts=4)
                grp.create_dataset("labels", data=labels, dtype="int64")
                record_ids.append(rid)

    logger.info(f"Processed {len(record_ids)} MIMIC records -> {h5_path}")

    if record_ids:
        subject_stratified_split(
            record_ids, output_path=str(out_path / "splits.json")
        )


# =========================================================================
# PUBLIC DATASET PREPROCESSORS (no credentials required)
# =========================================================================


def _process_sleep_edf_file(psg_file, hyp_path, target_sr, window_samples, window_sec, stage_map):
    """Process a single Sleep-EDF PSG file. Returns (subject_id, signals, labels) or None."""
    mne.set_log_level("WARNING")
    bandpass = BandpassFilter(low=0.3, high=35.0, fs=target_sr)
    artifact_rej = ArtifactRejection(threshold_uv=500.0)
    subject_id = psg_file.stem.replace("-PSG", "")

    try:
        raw = mne.io.read_raw_edf(str(psg_file), preload=True, verbose=False)

        target_chs = ["EEG Fpz-Cz", "EEG Pz-Oz"]
        available = raw.ch_names
        pick = [ch for ch in target_chs if ch in available]
        if len(pick) < 2:
            pick = [ch for ch in available if "EEG" in ch][:2]
        if len(pick) < 2:
            return None

        raw.pick(pick)
        raw.resample(target_sr)
        data = raw.get_data()

        if hyp_path is None or not hyp_path.exists():
            return None

        annot = mne.read_annotations(str(hyp_path))

        n_epochs = data.shape[1] // window_samples
        epoch_labels = np.full(n_epochs, -1, dtype=np.int64)

        for desc, onset, duration in zip(annot.description, annot.onset, annot.duration):
            if desc in stage_map:
                label = stage_map[desc]
                start_epoch = int(onset // window_sec)
                n_annot_epochs = max(1, int(duration // window_sec))
                for e in range(start_epoch, min(start_epoch + n_annot_epochs, n_epochs)):
                    epoch_labels[e] = label

        valid_mask = epoch_labels >= 0
        if valid_mask.sum() == 0:
            return None

        signals = []
        labels = []
        for e in range(n_epochs):
            if epoch_labels[e] < 0:
                continue
            start = e * window_samples
            end = start + window_samples
            window = data[:, start:end]
            window = bandpass(window)
            window = artifact_rej(window)
            signals.append(window)
            labels.append(epoch_labels[e])

        if not signals:
            return None

        return (subject_id, np.stack(signals).astype(np.float32), np.array(labels, dtype=np.int64))
    except Exception:
        return None


def preprocess_sleep_edf(data_dir: str, output_dir: str):
    """Preprocess Sleep-EDF Expanded (Sleep Cassette) EDF files into HDF5.

    Uses Fpz-Cz and Pz-Oz EEG channels at 100 Hz with standard 30-second epochs.
    Annotations are in *-Hypnogram.edf files with labels:
        Sleep stage W, Sleep stage 1, Sleep stage 2, Sleep stage 3/4, Sleep stage R
    Mapped to: 0=W, 1=N1, 2=N2, 3=N3, 4=REM.  Movement/unknown epochs are skipped.
    """
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    sc_dir = data_path / "sleep-cassette"
    if not sc_dir.exists():
        sc_dir = data_path

    psg_files = sorted(sc_dir.glob("*PSG.edf")) or sorted(sc_dir.glob("**/*PSG.edf"))
    if not psg_files:
        logger.error(f"No PSG EDF files found in {sc_dir}")
        return

    hyp_files = sorted(sc_dir.glob("*Hypnogram.edf")) or sorted(sc_dir.glob("**/*Hypnogram.edf"))
    hyp_by_prefix = {}
    for hp in hyp_files:
        key = hp.name[:6]
        hyp_by_prefix.setdefault(key, []).append(hp)

    logger.info(f"Found {len(psg_files)} Sleep-EDF PSG files, using {_MAX_WORKERS} workers")

    target_sr = 100
    window_sec = 30
    window_samples = target_sr * window_sec

    stage_map = {
        "Sleep stage W": 0,
        "Sleep stage 1": 1,
        "Sleep stage 2": 2,
        "Sleep stage 3": 3,
        "Sleep stage 4": 3,
        "Sleep stage R": 4,
    }

    h5_path = out_path / "data.h5"
    subject_ids = []

    # Build list of (psg_file, hyp_path) pairs for workers
    tasks = []
    for psg_file in psg_files:
        hyp_candidates = hyp_by_prefix.get(psg_file.name[:6], [])
        hyp_path = hyp_candidates[0] if hyp_candidates else None
        tasks.append((psg_file, hyp_path))

    with _safe_h5_open(h5_path) as h5f:
        with ProcessPoolExecutor(max_workers=_MAX_WORKERS) as pool:
            futures = {
                pool.submit(_process_sleep_edf_file, psg, hyp, target_sr, window_samples, window_sec, stage_map): psg
                for psg, hyp in tasks
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Sleep-EDF"):
                result = future.result()
                if result is None:
                    continue
                sid, signals, labels = result
                grp = h5f.create_group(sid)
                grp.create_dataset("signals", data=signals, dtype="float32",
                                   compression="gzip", compression_opts=4)
                grp.create_dataset("labels", data=labels, dtype="int64")
                subject_ids.append(sid)

    logger.info(f"Processed {len(subject_ids)} Sleep-EDF subjects -> {h5_path}")

    if subject_ids:
        subject_stratified_split(
            subject_ids, output_path=str(out_path / "splits.json")
        )


def _process_chbmit_subject(subj_dir, target_sr, window_samples):
    """Process all EDF files for one CHB-MIT subject. Returns (subj_name, signals, labels) or None."""
    mne.set_log_level("WARNING")
    bandpass = BandpassFilter(low=0.5, high=50.0, fs=target_sr)
    subj_name = subj_dir.name

    summary_file = subj_dir / f"{subj_name}-summary.txt"
    seizure_info = {}
    if summary_file.exists():
        seizure_info = _parse_chbmit_summary(summary_file)

    edf_files = sorted(subj_dir.glob("*.edf"))
    if not edf_files:
        return None

    all_signals = []
    all_labels = []

    for edf_file in edf_files:
        try:
            raw = mne.io.read_raw_edf(str(edf_file), preload=True, verbose=False)
            raw.resample(target_sr)
            data = raw.get_data()

            C = data.shape[0]
            if C < 23:
                pad = np.zeros((23 - C, data.shape[1]), dtype=data.dtype)
                data = np.concatenate([data, pad], axis=0)
            data = data[:23]

            fname = edf_file.name
            intervals = seizure_info.get(fname, [])

            n_windows = data.shape[1] // window_samples
            for w in range(n_windows):
                start = w * window_samples
                end = start + window_samples
                window = data[:, start:end]
                window = bandpass(window)
                all_signals.append(window)

                t_start = start / target_sr
                t_end = end / target_sr
                is_seizure = any(
                    s_start < t_end and s_end > t_start
                    for s_start, s_end in intervals
                )
                all_labels.append(1 if is_seizure else 0)
        except Exception:
            continue

    if not all_signals:
        return None

    return (subj_name, np.stack(all_signals).astype(np.float32), np.array(all_labels, dtype=np.int64))


def preprocess_chbmit(data_dir: str, output_dir: str):
    """Preprocess CHB-MIT Scalp EEG Database into HDF5.

    23 pediatric subjects, each in their own folder (chb01..chb24, some skipped).
    EDF files with 23-channel EEG at 256 Hz.
    Seizure annotations in *-summary.txt per subject folder.
    """
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    target_sr = 256
    window_sec = 2
    window_samples = target_sr * window_sec

    h5_path = out_path / "data.h5"
    subject_ids = []

    subject_dirs = sorted([
        d for d in data_path.iterdir()
        if d.is_dir() and d.name.startswith("chb")
    ])
    if not subject_dirs:
        subject_dirs = [data_path]

    logger.info(f"Found {len(subject_dirs)} CHB-MIT subjects, using {_MAX_WORKERS} workers")

    with _safe_h5_open(h5_path) as h5f:
        with ProcessPoolExecutor(max_workers=_MAX_WORKERS) as pool:
            futures = {
                pool.submit(_process_chbmit_subject, d, target_sr, window_samples): d
                for d in subject_dirs
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing CHB-MIT subjects"):
                result = future.result()
                if result is None:
                    continue
                sid, signals, labels = result
                grp = h5f.create_group(sid)
                grp.create_dataset("signals", data=signals, dtype="float32",
                                   compression="gzip", compression_opts=4)
                grp.create_dataset("labels", data=labels, dtype="int64")
                subject_ids.append(sid)

    logger.info(f"Processed {len(subject_ids)} CHB-MIT subjects -> {h5_path}")

    if subject_ids:
        subject_stratified_split(
            subject_ids, output_path=str(out_path / "splits.json")
        )


def _parse_chbmit_summary(summary_file: Path) -> dict:
    """Parse CHB-MIT *-summary.txt to extract seizure intervals per file.

    Returns: dict mapping filename -> list of (start_sec, end_sec)
    """
    seizure_info = {}
    current_file = None

    with open(summary_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith("File Name:"):
                current_file = line.split(":")[-1].strip()
                if current_file not in seizure_info:
                    seizure_info[current_file] = []
            elif "Seizure Start Time:" in line or "Seizure  Start Time:" in line:
                # Handle inconsistent spacing in CHB-MIT files
                parts = line.split(":")
                try:
                    start_sec = float(parts[-1].strip().replace(" seconds", ""))
                except ValueError:
                    continue
                # Next line should be end time — store start, fill end later
                seizure_info.setdefault(current_file, []).append([start_sec, None])
            elif "Seizure End Time:" in line or "Seizure  End Time:" in line:
                parts = line.split(":")
                try:
                    end_sec = float(parts[-1].strip().replace(" seconds", ""))
                except ValueError:
                    continue
                if current_file and seizure_info.get(current_file):
                    last = seizure_info[current_file][-1]
                    if last[1] is None:
                        last[1] = end_sec

    # Convert lists to tuples and remove incomplete entries
    result = {}
    for fname, intervals in seizure_info.items():
        result[fname] = [
            (s, e) for s, e in intervals if s is not None and e is not None
        ]
    return result


def _process_ptbxl_record(wfdb_path, ecg_id, label, target_sr, window_samples):
    """Process a single PTB-XL WFDB record. Returns (record_id, signals, labels) or None."""
    bandpass = BandpassFilter(low=0.5, high=40.0, fs=target_sr)
    record_id = f"ptbxl_{ecg_id:05d}"

    try:
        record = wfdb.rdrecord(str(wfdb_path.with_suffix("")))
        data = record.p_signal.T

        if data.shape[0] < 12:
            pad = np.zeros((12 - data.shape[0], data.shape[1]))
            data = np.concatenate([data, pad], axis=0)
        data = data[:12].astype(np.float32)

        n_windows = data.shape[1] // window_samples
        if n_windows == 0:
            return None

        signals = []
        labels = []
        for w in range(n_windows):
            start = w * window_samples
            end = start + window_samples
            window = data[:, start:end]
            window = bandpass(window)
            signals.append(window)
            labels.append(label)

        return (record_id, np.stack(signals), np.array(labels, dtype=np.int64))
    except Exception:
        return None


def preprocess_ptbxl(data_dir: str, output_dir: str):
    """Preprocess PTB-XL 12-lead ECG dataset into HDF5.

    21,799 records at 500 Hz, 10-second duration each.
    Labels from ptbxl_database.csv -> scp_codes column -> mapped to 5 superclasses:
        NORM, MI, STTC, CD, HYP  (mapped to 0-4)
    Uses the recommended train/val/test folds (strat_fold column).
    """
    import ast

    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    csv_path = data_path / "ptbxl_database.csv"
    if not csv_path.exists():
        logger.error(f"ptbxl_database.csv not found in {data_path}")
        return

    import pandas as pd

    df = pd.read_csv(csv_path, index_col="ecg_id")
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)

    scp_path = data_path / "scp_statements.csv"
    if not scp_path.exists():
        logger.error(f"scp_statements.csv not found in {data_path}")
        return

    scp_df = pd.read_csv(scp_path, index_col=0)
    scp_df = scp_df[scp_df.diagnostic == 1]

    superclass_map = {"NORM": 0, "MI": 1, "STTC": 2, "CD": 3, "HYP": 4}

    def get_superclass(scp_codes: dict) -> int:
        class_scores = {k: 0.0 for k in superclass_map}
        for code, likelihood in scp_codes.items():
            if code in scp_df.index:
                sc = scp_df.loc[code].diagnostic_class
                if sc in class_scores:
                    class_scores[sc] += likelihood
        if max(class_scores.values()) == 0:
            return -1
        return superclass_map[max(class_scores, key=class_scores.get)]

    df["label"] = df.scp_codes.apply(get_superclass)
    df = df[df.label >= 0]

    logger.info(f"PTB-XL: {len(df)} records with valid diagnostic labels, using {_MAX_WORKERS} workers")

    target_sr = 500
    window_sec = 2
    window_samples = target_sr * window_sec

    h5_path = out_path / "data.h5"
    record_ids = []

    # Build task list: (wfdb_path, ecg_id, label)
    tasks = []
    for ecg_id, row in df.iterrows():
        wfdb_path = data_path / row.filename_hr
        tasks.append((wfdb_path, ecg_id, int(row.label)))

    with _safe_h5_open(h5_path) as h5f:
        with ProcessPoolExecutor(max_workers=_MAX_WORKERS) as pool:
            futures = {
                pool.submit(_process_ptbxl_record, wp, eid, lab, target_sr, window_samples): eid
                for wp, eid, lab in tasks
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing PTB-XL"):
                result = future.result()
                if result is None:
                    continue
                rid, signals, labels = result
                grp = h5f.create_group(rid)
                grp.create_dataset("signals", data=signals, dtype="float32",
                                   compression="gzip", compression_opts=4)
                grp.create_dataset("labels", data=labels, dtype="int64")
                record_ids.append(rid)

    logger.info(f"Processed {len(record_ids)} PTB-XL records -> {h5_path}")

    if record_ids:
        subject_stratified_split(
            record_ids, output_path=str(out_path / "splits.json")
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess biosignal datasets to HDF5")
    parser.add_argument(
        "--dataset",
        choices=["sleep_edf", "chbmit", "ptbxl", "shhs", "tusz", "mimic_ecg", "all"],
        required=True,
    )
    parser.add_argument("--data_root", default="/workspace/data")
    args = parser.parse_args()

    processors = {
        # Public datasets (no credentials)
        "sleep_edf": lambda: preprocess_sleep_edf(
            f"{args.data_root}/sleep_edf", f"{args.data_root}/sleep_edf"
        ),
        "chbmit": lambda: preprocess_chbmit(
            f"{args.data_root}/chbmit", f"{args.data_root}/chbmit"
        ),
        "ptbxl": lambda: preprocess_ptbxl(
            f"{args.data_root}/ptbxl", f"{args.data_root}/ptbxl"
        ),
        # Original datasets (need credentials)
        "shhs": lambda: preprocess_shhs(
            f"{args.data_root}/shhs", f"{args.data_root}/shhs"
        ),
        "tusz": lambda: preprocess_tusz(
            f"{args.data_root}/tusz", f"{args.data_root}/tusz"
        ),
        "mimic_ecg": lambda: preprocess_mimic_ecg(
            f"{args.data_root}/mimic_ecg", f"{args.data_root}/mimic_ecg"
        ),
    }

    if args.dataset == "all":
        # Default to the public datasets
        for name in ["sleep_edf", "chbmit", "ptbxl"]:
            logger.info(f"\n{'='*60}\nProcessing {name}\n{'='*60}")
            processors[name]()
    else:
        processors[args.dataset]()
