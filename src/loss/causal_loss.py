"""Causal consistency loss with proper do-calculus grounding.

Three components enforcing genuine causal structure:
1. Interventional consistency: do(X_i := c) should only affect descendants
   (Markov property under interventions — Pearl, 2009 Thm 3.4.1)
2. Counterfactual edge necessity: removing a high-probability edge should
   change target node representations (enforces edges carry causal info)
3. Distributional invariance: non-descendant representations should be
   invariant across *multiple* intervention values (connects to IRM)

Efficiency: all interventions are batched into a single transformer forward
pass by stacking along the batch dimension, eliminating Python loops.
"""
import torch
import torch.nn.functional as F


def compute_descendants(adj: torch.Tensor) -> torch.Tensor:
    """Compute descendant matrix via transitive closure.

    Args:
        adj: (B, N, N) adjacency matrix
    Returns:
        desc: (B, N, N) where desc[b,i,j]=1 means j is descendant of i
    """
    B, N, _ = adj.shape
    device = adj.device

    eye = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1)
    reach = torch.clamp(eye + adj, 0, 1)
    n_iters = max(1, (N - 1).bit_length())
    for _ in range(n_iters):
        reach = torch.clamp(torch.bmm(reach, reach), 0, 1)

    desc = reach * (1.0 - eye)
    return desc


def interventional_consistency_loss(
    adj: torch.Tensor,
    embeddings: torch.Tensor,
    transformer_fn,
    tokens: torch.Tensor,
    n_interventions: int = 3,
) -> torch.Tensor:
    """Batched do-calculus consistency: do(X_i := c) should only change descendants.

    All interventions are stacked along the batch dim for a single transformer call.
    """
    B, N, d = embeddings.shape
    device = embeddings.device

    desc = compute_descendants(adj)
    eye = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1)
    non_desc = (1.0 - desc) * (1.0 - eye)

    n_intervene = min(n_interventions, N)
    intervene_idx = torch.randint(0, N, (B, n_intervene), device=device)

    # Build all intervened token sets at once: (n_intervene * B, N, d)
    tokens_expanded = tokens.unsqueeze(0).expand(n_intervene, -1, -1, -1)  # (K, B, N, d)
    tokens_do = tokens_expanded.clone()
    batch_range = torch.arange(B, device=device)
    for k in range(n_intervene):
        tokens_do[k, batch_range, intervene_idx[:, k]] = 0.0

    # Single batched forward pass
    tokens_do_flat = tokens_do.reshape(n_intervene * B, N, d)
    adj_flat = adj.unsqueeze(0).expand(n_intervene, -1, -1, -1).reshape(n_intervene * B, N, N)
    emb_do_flat = transformer_fn(tokens_do_flat, adj_flat)  # (K*B, N, d)
    emb_do = emb_do_flat.reshape(n_intervene, B, N, d)

    # Compute penalties for all interventions at once
    change = (emb_do - embeddings.unsqueeze(0)).pow(2).sum(dim=-1)  # (K, B, N)
    masks = non_desc[batch_range.unsqueeze(0).expand(n_intervene, -1),
                     intervene_idx.T]  # (K, B, N)
    penalty = (change * masks).sum(dim=-1) / (masks.sum(dim=-1) + 1e-8)  # (K, B)

    return penalty.mean()


def counterfactual_edge_loss(
    adj: torch.Tensor,
    edge_probs: torch.Tensor,
    embeddings: torch.Tensor,
    transformer_fn,
    tokens: torch.Tensor,
    n_edges: int = 3,
) -> torch.Tensor:
    """Batched edge necessity: removing high-probability edges should change targets.

    All edge ablations are stacked along batch dim for a single transformer call.
    """
    B, N, d = embeddings.shape
    device = embeddings.device

    probs_flat = edge_probs.reshape(B, -1)

    # Sample edges to ablate
    sample_indices = torch.multinomial(probs_flat + 1e-8, n_edges)  # (B, K)
    src = sample_indices // N  # (B, K)
    tgt = sample_indices % N   # (B, K)

    # Build all ablated adj matrices at once: (K, B, N, N)
    adj_expanded = adj.unsqueeze(0).expand(n_edges, -1, -1, -1).clone()
    batch_range = torch.arange(B, device=device)
    for k in range(n_edges):
        adj_expanded[k, batch_range, src[:, k], tgt[:, k]] = 0.0

    # Single batched forward pass
    adj_flat = adj_expanded.reshape(n_edges * B, N, N)
    tokens_flat = tokens.unsqueeze(0).expand(n_edges, -1, -1, -1).reshape(n_edges * B, N, d)
    emb_ablated_flat = transformer_fn(tokens_flat, adj_flat)
    emb_ablated = emb_ablated_flat.reshape(n_edges, B, N, d)

    # Compute losses for all edges at once
    total_loss = torch.tensor(0.0, device=device)
    for k in range(n_edges):
        change_at_tgt = (emb_ablated[k, batch_range, tgt[:, k]] -
                         embeddings[batch_range, tgt[:, k]]).pow(2).sum(dim=-1)
        edge_prob_val = edge_probs[batch_range, src[:, k], tgt[:, k]]
        change_norm = torch.tanh(change_at_tgt)
        total_loss = total_loss + (edge_prob_val * (1.0 - change_norm)).mean()

    return total_loss / n_edges


def distributional_invariance_loss(
    adj: torch.Tensor,
    embeddings: torch.Tensor,
    transformer_fn,
    tokens: torch.Tensor,
    n_interventions: int = 3,
) -> torch.Tensor:
    """Batched IRM-inspired invariance: non-descendant representations should be
    invariant across multiple intervention values. Single transformer call.
    """
    B, N, d = embeddings.shape
    device = embeddings.device

    desc = compute_descendants(adj)
    eye = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1)
    non_desc = (1.0 - desc) * (1.0 - eye)

    node_idx = torch.randint(0, N, (1,), device=device).item()
    mask = non_desc[:, node_idx, :]  # (B, N)

    # Build all intervened token sets at once: (K, B, N, d)
    tokens_do = tokens.unsqueeze(0).expand(n_interventions, -1, -1, -1).clone()
    for k in range(n_interventions):
        tokens_do[k, :, node_idx] = torch.randn(B, d, device=device) * 2.0

    # Single batched forward pass
    tokens_flat = tokens_do.reshape(n_interventions * B, N, d)
    adj_flat = adj.unsqueeze(0).expand(n_interventions, -1, -1, -1).reshape(n_interventions * B, N, N)
    emb_flat = transformer_fn(tokens_flat, adj_flat)
    emb_do = emb_flat.reshape(n_interventions, B, N, d)

    variance = emb_do.var(dim=0).sum(dim=-1)  # (B, N)
    loss = (variance * mask).sum(dim=-1) / (mask.sum(dim=-1) + 1e-8)
    return loss.mean()


def causal_consistency_loss(
    adj: torch.Tensor,
    edge_probs: torch.Tensor,
    embeddings: torch.Tensor,
    transformer_fn,
    tokens: torch.Tensor,
    n_interventions: int = 5,
    intervention_type: str = "zero",
    use_edge_loss: bool = True,
    use_invariance_loss: bool = True,
) -> torch.Tensor:
    """Full causal consistency loss: L_interv + 0.5 * L_edge + 0.3 * L_invariance."""
    loss = interventional_consistency_loss(
        adj, embeddings, transformer_fn, tokens, n_interventions
    )

    if use_edge_loss:
        loss = loss + 0.5 * counterfactual_edge_loss(
            adj, edge_probs, embeddings, transformer_fn, tokens, n_edges=2
        )

    if use_invariance_loss:
        loss = loss + 0.3 * distributional_invariance_loss(
            adj, embeddings, transformer_fn, tokens, n_interventions=2
        )

    return loss
