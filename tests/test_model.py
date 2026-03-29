"""Integration test: full model forward pass."""
import pytest
import torch

from src.model.full_model import CausalBiosignalModel


class TestFullModel:
    @pytest.fixture
    def model(self):
        return CausalBiosignalModel(
            n_channels=2,
            n_classes=5,
            d_token=64,
            window_samples=500,
            sample_rate=125.0,
            n_fft=128,
            hop_length=32,
            win_length=128,
            graph_n_layers=1,
            graph_n_heads=2,
            graph_dropout=0.0,
            tf_n_layers=2,
            tf_n_heads=4,
            tf_d_ff=128,
            tf_dropout=0.0,
            cls_hidden=64,
        )

    def test_forward_shape(self, model):
        x = torch.randn(2, 2, 500)
        output = model(x)
        assert output["logits"].shape == (2, 5)
        assert output["recon"].shape == (2, 2, 500)
        assert output["edge_probs"].shape[0] == 2
        # adj, tokens, embeddings always returned for loss computation
        assert output["adj"].shape[0] == 2
        assert output["tokens"].shape == (2, 10, 64)  # 2 ch * 5 bands
        assert output["embeddings"].shape == (2, 10, 64)

    def test_forward_eval_skips_decoder(self, model):
        model.eval()
        x = torch.randn(2, 2, 500)
        output = model(x)
        assert output["logits"].shape == (2, 5)
        assert output["recon"] is None
        model.train()

    def test_forward_with_graph(self, model):
        x = torch.randn(2, 2, 500)
        output = model(x, return_graph=True)
        assert "adj" in output
        assert output["adj"].shape[0] == 2

    def test_backward(self, model):
        x = torch.randn(2, 2, 500)
        output = model(x)  # train mode: recon is computed
        # Include recon in loss so decoder gets gradients (mirrors real training)
        loss = output["logits"].sum() + output["recon"].sum()
        loss.backward()
        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_freeze_backbone(self, model):
        model_with_adapter = CausalBiosignalModel(
            n_channels=2, n_classes=5, d_token=64,
            window_samples=500, sample_rate=125.0,
            n_fft=128, hop_length=32, win_length=128,
            graph_n_layers=1, graph_n_heads=2,
            tf_n_layers=2, tf_n_heads=4,
            tf_d_ff=128, use_adapter=True,
        )
        model_with_adapter.freeze_backbone()

        n_trainable = sum(
            p.numel() for p in model_with_adapter.parameters() if p.requires_grad
        )
        n_total = sum(p.numel() for p in model_with_adapter.parameters())
        assert n_trainable < n_total
        assert n_trainable > 0  # adapter params should be trainable
