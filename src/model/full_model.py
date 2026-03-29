"""Full Causal Biosignal Model: end-to-end integration of all components."""
import torch
import torch.nn as nn

from .tokenizer import SpectralTokenizer, TokenDecoder
from .causal_graph import CausalGraphInferencer
from .transformer import GraphConditionedTransformer, ClassificationHead
from .adapter import SubjectAdapter


class CausalBiosignalModel(nn.Module):
    """Full model: Tokenizer -> Graph Inference -> Graph Transformer -> Classifier.

    Combines:
        1. Spectral tokenizer (raw signal -> band tokens)
        2. Causal graph inferencer (tokens -> adjacency matrix)
        3. Graph-conditioned transformer (tokens + adj -> embeddings)
        4. Classification head (embeddings -> task logits)
        5. Token decoder (tokens -> reconstructed signal, for recon loss)
        6. Optional subject adapter
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        d_token: int = 128,
        window_samples: int = 500,
        sample_rate: float = 125.0,
        # Tokenizer
        n_fft: int = 256,
        hop_length: int = 64,
        win_length: int = 256,
        bands: dict | None = None,
        # Graph
        graph_n_layers: int = 2,
        graph_n_heads: int = 4,
        graph_dropout: float = 0.1,
        sparsity_threshold: float = 0.5,
        l1_lambda: float = 0.01,
        # Transformer
        tf_n_layers: int = 6,
        tf_n_heads: int = 8,
        tf_d_ff: int = 512,
        tf_dropout: float = 0.1,
        tf_gradient_checkpointing: bool = False,
        # Classifier
        cls_hidden: int = 256,
        cls_dropout: float = 0.3,
        # Adapter
        use_adapter: bool = False,
        adapter_hidden: int = 64,
        adapter_layers: int = 2,
    ):
        super().__init__()

        self.tokenizer = SpectralTokenizer(
            n_channels=n_channels,
            d_token=d_token,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            sample_rate=sample_rate,
            bands=bands,
        )

        self.decoder = TokenDecoder(
            n_channels=n_channels,
            d_token=d_token,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            output_length=window_samples,
            sample_rate=sample_rate,
            bands=bands,
        )

        self.graph_inferencer = CausalGraphInferencer(
            d_model=d_token,
            n_layers=graph_n_layers,
            n_heads=graph_n_heads,
            dropout=graph_dropout,
            sparsity_threshold=sparsity_threshold,
            l1_lambda=l1_lambda,
        )

        self.transformer = GraphConditionedTransformer(
            d_model=d_token,
            n_layers=tf_n_layers,
            n_heads=tf_n_heads,
            d_ff=tf_d_ff,
            dropout=tf_dropout,
            gradient_checkpointing=tf_gradient_checkpointing,
        )

        self.classifier = ClassificationHead(
            d_model=d_token,
            n_classes=n_classes,
            hidden_dim=cls_hidden,
            dropout=cls_dropout,
        )

        self.adapter = None
        if use_adapter:
            self.adapter = SubjectAdapter(
                d_model=d_token,
                hidden_dim=adapter_hidden,
                n_layers=adapter_layers,
            )

    def forward(
        self, x: torch.Tensor, return_graph: bool = False
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, C, T) raw signal
            return_graph: if True, include adjacency in output
        Returns:
            dict with keys: logits, recon, adj, edge_probs, tokens, embeddings
        """
        # 1. Tokenize
        tokens = self.tokenizer(x)  # (B, N, d_token)

        # 2. Infer causal graph
        adj, edge_probs = self.graph_inferencer(tokens)  # (B, N, N)

        # 3. Graph-conditioned transformer
        embeddings = self.transformer(tokens, adj)  # (B, N, d_token)

        # 4. Optional adapter
        if self.adapter is not None:
            embeddings = self.adapter(embeddings)

        # 5. Classification
        logits = self.classifier(embeddings)  # (B, n_classes)

        # 6. Reconstruction (from tokens, not embeddings)
        # Skip decoder during eval — saves iSTFT + linear per validation step
        if self.training:
            recon = self.decoder(tokens)  # (B, C, T)
        else:
            recon = None

        output = {
            "logits": logits,
            "recon": recon,
            "edge_probs": edge_probs,
            "adj": adj,
            "tokens": tokens,
            "embeddings": embeddings,
        }

        if not return_graph:
            # Still keep adj in output for loss computation,
            # but callers can ignore it
            pass

        return output

    def freeze_backbone(self):
        """Freeze everything except the adapter for subject adaptation."""
        for param in self.parameters():
            param.requires_grad = False
        if self.adapter is not None:
            for param in self.adapter.parameters():
                param.requires_grad = True

    def unfreeze_backbone(self):
        for param in self.parameters():
            param.requires_grad = True
