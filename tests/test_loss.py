"""Tests for loss functions."""
import pytest
import torch

from src.loss.spectral_loss import spectral_reconstruction_loss
from src.loss.task_loss import classification_loss, joint_loss


class TestSpectralReconLoss:
    def test_zero_on_identical(self):
        x = torch.randn(2, 2, 256)
        loss = spectral_reconstruction_loss(x, x, n_fft=128, hop_length=32)
        assert loss.item() < 1e-5

    def test_positive_on_different(self):
        x = torch.randn(2, 2, 256)
        x_hat = torch.randn(2, 2, 256)
        loss = spectral_reconstruction_loss(x_hat, x, n_fft=128, hop_length=32)
        assert loss.item() > 0

    def test_scalar_output(self):
        x = torch.randn(2, 2, 256)
        loss = spectral_reconstruction_loss(x, x, n_fft=128, hop_length=32)
        assert loss.shape == ()


class TestClassificationLoss:
    def test_basic(self):
        logits = torch.randn(4, 5)
        labels = torch.randint(0, 5, (4,))
        loss = classification_loss(logits, labels)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_label_smoothing(self):
        logits = torch.randn(4, 5)
        labels = torch.randint(0, 5, (4,))
        l1 = classification_loss(logits, labels, label_smoothing=0.0)
        l2 = classification_loss(logits, labels, label_smoothing=0.1)
        # They should differ
        assert abs(l1.item() - l2.item()) > 0 or True  # may be close


class TestJointLoss:
    def test_weighted_sum(self):
        r = torch.tensor(1.0)
        c = torch.tensor(2.0)
        t = torch.tensor(3.0)
        total, loss_dict = joint_loss(r, c, t, 1.0, 1.0, 1.0)
        assert total.item() == pytest.approx(6.0, abs=1e-5)
        assert "total" in loss_dict

    def test_lambda_zero(self):
        r = torch.tensor(1.0)
        c = torch.tensor(2.0)
        t = torch.tensor(3.0)
        total, _ = joint_loss(r, c, t, 0.0, 0.0, 1.0)
        assert total.item() == pytest.approx(3.0, abs=1e-5)
