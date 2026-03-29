"""Tests for the data pipeline."""
import json
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from src.data.transforms import ZScoreNormalize, BandpassFilter, ArtifactRejection
from src.data.splits import subject_stratified_split


class TestZScoreNormalize:
    def test_basic(self):
        norm = ZScoreNormalize()
        x = np.random.randn(2, 100).astype(np.float32) * 10 + 5
        out = norm(x)
        assert out.shape == x.shape
        np.testing.assert_allclose(out.mean(axis=-1), 0.0, atol=1e-5)
        np.testing.assert_allclose(out.std(axis=-1), 1.0, atol=1e-5)

    def test_zero_std(self):
        norm = ZScoreNormalize()
        x = np.ones((2, 100), dtype=np.float32)
        out = norm(x)
        # Should not produce NaN
        assert not np.any(np.isnan(out))


class TestBandpassFilter:
    def test_shape_preserved(self):
        bp = BandpassFilter(low=0.5, high=40.0, fs=125.0)
        x = np.random.randn(2, 500).astype(np.float32)
        out = bp(x)
        assert out.shape == x.shape
        assert out.dtype == np.float32

    def test_removes_dc(self):
        bp = BandpassFilter(low=0.5, high=40.0, fs=125.0)
        # Signal with large DC offset
        x = np.ones((1, 500), dtype=np.float32) * 100
        x += np.random.randn(1, 500).astype(np.float32) * 0.1
        out = bp(x)
        # DC should be removed
        assert abs(out.mean()) < 10


class TestArtifactRejection:
    def test_rejects_large_amplitude(self):
        ar = ArtifactRejection(threshold_uv=100.0)
        x = np.ones((2, 100), dtype=np.float32) * 50  # below threshold
        x[0, :] = 200  # above threshold on channel 0
        out = ar(x)
        np.testing.assert_array_equal(out[0], 0.0)
        np.testing.assert_array_equal(out[1], x[1])


class TestSubjectSplit:
    def test_no_leakage(self):
        ids = [f"subj_{i:03d}" for i in range(100)]
        splits = subject_stratified_split(ids, seed=42)
        train_set = set(splits["train"])
        val_set = set(splits["val"])
        test_set = set(splits["test"])
        # No overlap
        assert len(train_set & val_set) == 0
        assert len(train_set & test_set) == 0
        assert len(val_set & test_set) == 0
        # All subjects accounted for
        assert len(train_set | val_set | test_set) == 100

    def test_reproducible(self):
        ids = [f"subj_{i}" for i in range(50)]
        s1 = subject_stratified_split(ids, seed=42)
        s2 = subject_stratified_split(ids, seed=42)
        assert s1 == s2

    def test_saves_json(self):
        ids = [f"s{i}" for i in range(20)]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "splits.json")
            splits = subject_stratified_split(ids, output_path=path)
            assert Path(path).exists()
            with open(path) as f:
                loaded = json.load(f)
            assert loaded == splits
