"""Tests for the causal graph inference module."""
import pytest
import torch

from src.model.causal_graph import CausalGraphInferencer, StraightThroughThreshold


class TestCausalGraphInferencer:
    @pytest.fixture
    def inferencer(self):
        return CausalGraphInferencer(
            d_model=64,
            n_layers=1,
            n_heads=4,
            dropout=0.0,
            sparsity_threshold=0.5,
        )

    def test_output_shapes(self, inferencer):
        tokens = torch.randn(2, 10, 64)
        adj, edge_probs = inferencer(tokens)
        assert adj.shape == (2, 10, 10)
        assert edge_probs.shape == (2, 10, 10)

    def test_no_self_loops(self, inferencer):
        tokens = torch.randn(2, 10, 64)
        adj, _ = inferencer(tokens)
        for b in range(2):
            for i in range(10):
                assert adj[b, i, i] == 0.0

    def test_binary_adjacency(self, inferencer):
        tokens = torch.randn(2, 10, 64)
        adj, _ = inferencer(tokens)
        unique_vals = torch.unique(adj)
        assert all(v in [0.0, 1.0] for v in unique_vals)

    def test_edge_probs_bounded(self, inferencer):
        tokens = torch.randn(2, 10, 64)
        _, edge_probs = inferencer(tokens)
        assert (edge_probs >= 0).all()
        assert (edge_probs <= 1).all()

    def test_sparsity_loss(self, inferencer):
        edge_probs = torch.rand(2, 10, 10)
        loss = inferencer.sparsity_loss(edge_probs)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_dag_loss(self, inferencer):
        edge_probs = torch.rand(2, 10, 10)
        loss = inferencer.dag_loss(edge_probs)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_gradient_flows(self, inferencer):
        """Verify straight-through estimator allows gradient flow."""
        tokens = torch.randn(2, 10, 64, requires_grad=True)
        adj, edge_probs = inferencer(tokens)
        loss = adj.sum()
        loss.backward()
        assert tokens.grad is not None


class TestStraightThroughThreshold:
    def test_forward(self):
        x = torch.tensor([0.3, 0.5, 0.7, 0.9])
        out = StraightThroughThreshold.apply(x, 0.5)
        expected = torch.tensor([0.0, 0.0, 1.0, 1.0])
        torch.testing.assert_close(out, expected)

    def test_backward_passthrough(self):
        x = torch.tensor([0.3, 0.7], requires_grad=True)
        out = StraightThroughThreshold.apply(x, 0.5)
        out.sum().backward()
        # Gradient passes through unchanged
        torch.testing.assert_close(x.grad, torch.ones_like(x))
