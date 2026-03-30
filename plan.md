Data & environment setup
Foundation
Weeks 1–2
Experiments
Dataset access
Sleep-EDF, CHB-MIT, PTB-XL — run open-access download scripts
Baseline replication
Replicate BrainOmni and THD-BAR results on Sleep-EDF as sanity check
Preprocessing audit
Verify channel alignment, sampling rates, artifact removal across all three corpora
Procedure
Environment
Create conda env, pin PyTorch 2.x + CUDA 12, torch_geometric, mne-python
Data pipeline
Write EEGDataset and ECGDataset classes with lazy loading and bandpass filtering
Config system
Hydra config for dataset, model, loss weight sweeps
Materials & compute
Compute
RTX 5090 32GB on Vast.ai (~$0.37/hr) — or A100 80GB if needed
Required
Storage
~250GB fast NVMe for open datasets raw + preprocessed cache
Required
Libraries
mne, scipy, torch_geometric, hydra-core, wandb, pytest
Required
IRB
No IRB exemption needed for open-access datasets
Required
Code outline
src/
  data/
    eeg_dataset.py       # EEGDataset — lazy load, window, bandpass
    ecg_dataset.py       # ECGDataset
    transforms.py        # per-channel z-score, artifact rejection
    splits.py            # subject-stratified train/val/test splits
  config/
    base.yaml
    dataset/sleep_edf.yaml
    dataset/chbmit.yaml
Deliverables
All datasets downloaded and preprocessed
Baseline numbers reproduced (within 1% of paper)
Data pipeline unit-tested


Spectral tokenizer
Module 1
Weeks 3–4
Experiments
Band ablation
Compare delta/alpha/beta/gamma tokenization vs raw waveform vs full STFT as input representations
Window size sweep
Test 1s, 2s, 4s windows — measure reconstruction loss vs downstream classification F1
Token dimensionality
Sweep d_token ∈ {64, 128, 256} — find Pareto-optimal size
Procedure
STFT per band
Compute bandpass-filtered STFT in each physiological band, project to fixed-dim token via linear layer
Positional encoding
Add learnable channel ID + band ID embeddings to each token
Reconstruction head
Decode tokens back to raw signal — train tokenizer standalone with MSE loss first
Materials & compute
scipy.signal
bandpass_filter, stft — standard library
Required
torchaudio
for STFT on GPU
Required
~20 GPU-hours
for tokenizer ablation sweep across 3 datasets
Optional
Code outline
src/
  model/
    tokenizer.py
    # class SpectralTokenizer(nn.Module):
    #   bands: dict[str, tuple[float,float]]
    #   stft: torchaudio.transforms.Spectrogram
    #   proj: nn.Linear(stft_bins, d_token)
    #   pos_emb: nn.Embedding(n_channels * n_bands, d_token)
    #
    #   def forward(x: [B, C, T]) -> tokens: [B, C*n_bands, d_token]
    #
    # class TokenDecoder(nn.Module):
    #   deproj: nn.Linear(d_token, stft_bins)
    #   istft: torchaudio.transforms.InverseSpectrogram
Deliverables
Spectral tokenizer trained, reconstruction loss < 0.05 on Sleep-EDF val
Band ablation table written up
Token dim fixed for remainder of project

Causal graph inference module
Core novelty
Weeks 5–7
Experiments
Graph sparsity sweep
Vary L1 regularisation on adjacency matrix — measure graph sparsity vs downstream F1
Granger vs learned
Compare learned adjacency to Granger causality ground truth on synthetic data with known causal structure
Temporal stability
Measure how much graph changes window-to-window within same subject — should be low for healthy baseline
Procedure
Synthetic validation first
Generate synthetic VAR(p) data with known causal graph — verify module recovers correct edges
Per-window graph
Run graph inference module on each 2s window independently, track consistency across windows
Sparsification
Apply straight-through estimator to threshold adjacency to sparse binary graph
Materials & compute
torch_geometric
for sparse graph operations
Required
causalnex or lingam
for Granger baseline comparison
Optional
~40 GPU-hours
for synthetic validation + sparsity sweep
Required
Code outline
src/
  model/
    causal_graph.py
    # class CausalGraphInferencer(nn.Module):
    #   encoder: nn.TransformerEncoder  # tokens -> node embeddings
    #   edge_mlp: nn.Sequential         # [h_i, h_j] -> edge_logit scalar
    #   threshold: float                # straight-through sparsification
    #
    #   def forward(tokens: [B, N, d]) -> adj: [B, N, N]
    #     node_emb = encoder(tokens)
    #     # pairwise edge scoring
    #     i_exp = node_emb.unsqueeze(2).expand(-1,-1,N,-1)
    #     j_exp = node_emb.unsqueeze(1).expand(-1,N,-1,-1)
    #     logits = edge_mlp(cat([i_exp, j_exp], -1)).squeeze(-1)
    #     adj = straight_through_threshold(logits, self.threshold)
    #     return adj  # sparse directed [B, N, N]
Deliverables
Module recovers >85% F1 on synthetic causal graph
Sparsity vs performance curve plotted
Unit tests pass for adjacency symmetry/asymmetry and sparsification

Graph-conditioned transformer + adapter
Integration
Weeks 8–10
Experiments
Masked vs full attention
Ablate: graph-masked attention vs standard full attention — this is the key claim
Adapter size sweep
Vary adapter params ∈ {4k, 16k, 64k} — find minimum for good subject-level adaptation
Cross-dataset transfer
Train on Sleep-EDF, zero-shot eval on CHB-MIT and PTB-XL — primary generalization experiment
Procedure
Mask injection
Pass adjacency matrix as attention mask — zero-out attention logits where adj[i,j]=0
Adapter design
4-layer MLP per subject, frozen backbone, only adapter trained at inference time
End-to-end training
Tokenizer + graph inferencer + transformer trained jointly with full loss
Materials & compute
~120 GPU-hours
full joint training on all three datasets
Required
flash-attn 2
for memory-efficient masked attention at long sequence lengths
Required
DeepSpeed ZeRO-2
if training on single node with 2×A100
Optional
Code outline
src/
  model/
    transformer.py
    # class GraphConditionedTransformer(nn.Module):
    #   layers: nn.ModuleList[GraphAttentionLayer]
    #
    # class GraphAttentionLayer(nn.Module):
    #   attn: nn.MultiheadAttention
    #   ffn:  nn.Sequential
    #   norm1, norm2: nn.LayerNorm
    #
    #   def forward(x, adj):
    #     # adj [B,N,N] -> additive mask: -1e9 where adj==0
    #     mask = (1 - adj) * -1e9
    #     out, _ = self.attn(x, x, x, attn_mask=mask)
    #     return self.norm2(out + self.ffn(self.norm1(out)))
    #
    adapter.py
    # class SubjectAdapter(nn.Module):
    #   layers: nn.Sequential  # small MLP, ~16k params
    #   def forward(z): return z + self.layers(z)
Deliverables
Graph-masked attention ablation table complete
Zero-shot cross-dataset numbers on all 3 benchmarks
Adapter overhead measured (params, inference latency)


Loss function + theory
Theorem
Weeks 11–12
Experiments
Loss term ablation
Train with each loss term alone, pairs, and all three — 7 configurations total
Causal consistency weight
Sweep lambda_causal ∈ {0.01, 0.1, 0.5, 1.0} — measure generalization gap
Intervention simulation
Synthetically ablate channels and verify model predictions degrade gracefully per causal structure
Procedure
Derive consistency loss
Formalize do-calculus consistency term: penalize when intervening on a node changes non-descendants
Prove the theorem
Bound worst-case transfer error as O(sqrt(d/n)) under assumption of time-varying SCM
Write theorem sketch
Include informal theorem + proof sketch in paper appendix; full proof in supplementary
Materials & compute
~30 GPU-hours
for 7-configuration loss ablation sweep
Required
Overleaf / LaTeX
for theorem write-up
Required
Code outline
src/
  loss/
    spectral_loss.py
    # def spectral_reconstruction_loss(x_hat, x):
    #   return F.mse_loss(stft(x_hat), stft(x))
    #
    causal_loss.py
    # def causal_consistency_loss(adj, embeddings):
    #   # For each edge (i->j), simulate do(X_i=0):
    #   # predict downstream embeddings, penalize
    #   # if non-descendants change significantly
    #   ...
    #
    task_loss.py
    # def joint_loss(recon, causal, task, lam):
    #   return recon + lam[0]*causal + lam[1]*task
Deliverables
Loss ablation table: all 7 configs, 3 datasets
Theorem statement + informal proof written
Lambda sensitivity analysis plotted
← Previous phase
Next phase →


Writing, ablations & submission
Paper
Weeks 13–16
Experiments
Final benchmark run
Full eval on Sleep-EDF sleep staging, CHB-MIT seizure detection, PTB-XL arrhythmia — 3 seeds each
Graph interpretability
Visualize inferred causal graphs for known pathologies — verify against clinical literature
Scaling experiment
Show performance vs training data size — log-linear plot for the generalization claim
Procedure
Related work audit
Verify differentiation from BrainOmni, THD-BAR, SPICED (all in NeurIPS 2025 list)
Camera-ready figures
Architecture diagram, results tables, causal graph visualizations
Reproducibility checklist
NeurIPS checklist — code release, dataset access instructions, random seeds
Materials & compute
~50 GPU-hours
final 3-seed runs across all benchmarks
Required
matplotlib / seaborn
for all paper figures
Required
GitHub repo
clean public release with README and Colab demo
Required
Code outline
src/
  eval/
    benchmark.py     # unified eval loop: F1, AUROC, ECE
    transfer.py      # zero-shot cross-dataset eval
    interpret.py     # graph visualization, attention rollout
  train.py           # main training entry point
  sweep.py           # Hydra multirun for ablations
  notebooks/
    figure_1_arch.ipynb
    figure_2_graphs.ipynb
    figure_3_ablation.ipynb
scripts/
  download_sleep_edf.sh
  download_chbmit.sh
  download_ptbxl.sh
Deliverables
All tables and figures finalized
Code public on GitHub with tests passing
Paper submitted to NeurIPS (deadline ~May)
Colab demo notebook working
← Previous phase
Start coding ↗


Causal loss rewritten — now grounded in do-calculus with 3 rigorous components: interventional consistency (Markov property), counterfactual edge necessity, and distributional invariance (IRM-inspired). Previously was a single heuristic function.

Formal theorem module (__init__.py) — formal statement of Theorem 1 with 4 explicit assumptions (A1–A4), 4-step proof sketch, bound computation utilities, and verify_bound_empirically() to empirically validate the generalization bound.

5 baseline models (baselines.py) — PatchTST, Static GNN (electrode-distance graph), Correlation-based graph (Pearson), Vanilla Transformer (same tokenizer, no graph — the critical ablation), and Raw Waveform (tokenizer ablation).

Synthetic causal validation (validate_causal_graph.py) — generates linear and nonlinear VAR(p) processes with known ground-truth DAGs, trains the graph inferencer, and measures recovery F1/AUROC. Target: >0.85 linear, >0.75 nonlinear.

Proper cross-device channel mapping (transfer.py) — replaced naive partial weight loading with channel embedding interpolation and classification head re-initialization logic.

Config updated (default.yaml) — new params for edge loss and invariance loss.

