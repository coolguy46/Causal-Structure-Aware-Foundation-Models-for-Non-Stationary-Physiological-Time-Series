"""Tests for the spectral tokenizer."""
import pytest
import torch

from src.model.tokenizer import SpectralTokenizer, TokenDecoder


class TestSpectralTokenizer:
    @pytest.fixture
    def tokenizer(self):
        return SpectralTokenizer(
            n_channels=2,
            d_token=64,
            n_fft=128,
            hop_length=32,
            win_length=128,
            sample_rate=125.0,
        )

    def test_output_shape(self, tokenizer):
        x = torch.randn(4, 2, 500)  # (B, C, T)
        tokens = tokenizer(x)
        expected_n_tokens = 2 * 5  # channels * bands
        assert tokens.shape == (4, expected_n_tokens, 64)

    def test_no_nan(self, tokenizer):
        x = torch.randn(2, 2, 500)
        tokens = tokenizer(x)
        assert not torch.isnan(tokens).any()

    def test_batch_independence(self, tokenizer):
        x1 = torch.randn(1, 2, 500)
        x2 = torch.randn(1, 2, 500)
        t1 = tokenizer(x1)
        t2 = tokenizer(x2)
        t_batch = tokenizer(torch.cat([x1, x2], dim=0))
        torch.testing.assert_close(t_batch[0], t1[0], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(t_batch[1], t2[0], atol=1e-5, rtol=1e-5)

    def test_n_tokens_property(self, tokenizer):
        assert tokenizer.n_tokens == 10  # 2 channels * 5 bands


class TestTokenDecoder:
    def test_output_shape(self):
        decoder = TokenDecoder(
            n_channels=2,
            d_token=64,
            n_fft=128,
            hop_length=32,
            win_length=128,
            output_length=500,
            sample_rate=125.0,
        )
        tokens = torch.randn(4, 10, 64)  # (B, C*n_bands, d_token)
        recon = decoder(tokens)
        assert recon.shape == (4, 2, 500)

    def test_no_nan(self):
        decoder = TokenDecoder(
            n_channels=2,
            d_token=64,
            n_fft=128,
            hop_length=32,
            win_length=128,
            output_length=500,
            sample_rate=125.0,
        )
        tokens = torch.randn(2, 10, 64)
        recon = decoder(tokens)
        assert not torch.isnan(recon).any()
