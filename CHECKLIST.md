# Pre-Paper Experiment Checklist

Everything you must run and verify **after training is complete** but **before writing the paper**.

---

## Phase 1: Core Training Runs (must complete first)

- [ ] **Train main model on Sleep-EDF** (sleep staging, 5-class EEG)
  ```bash
  python -m src.train dataset=sleep_edf
  ```
- [ ] **Train main model on CHB-MIT** (seizure detection, binary EEG)
  ```bash
  python -m src.train dataset=chbmit
  ```
- [ ] **Train main model on PTB-XL** (arrhythmia, 5-class ECG)
  ```bash
  python -m src.train dataset=ptbxl
  ```
- [ ] Each training run: 3 seeds × 3 datasets = 9 runs (use `seed=42,123,7` with `--multirun`)
  ```bash
  python -m src.train --multirun seed=42,123,7 dataset=sleep_edf,chbmit,ptbxl
  ```

## Phase 2: Baselines (same 3 seeds per dataset)

- [ ] **PatchTST baseline** on all 3 datasets
- [ ] **Vanilla Transformer** (same spectral tokenizer, no graph masking) — critical ablation
- [ ] **Static GNN** (electrode-distance graph)
- [ ] **Correlation-based graph** (Pearson correlation as adjacency)
- [ ] **Raw waveform + graph** (no spectral tokenizer) — tokenizer ablation

## Phase 3: Ablation Sweep

- [ ] Run full ablation sweep:
  ```bash
  python -m src.sweep
  ```
  This covers:
  - Loss ablations (no_causal, no_recon, no_task, task_only, causal_only, recon_only, full)
  - Lambda sweep (lambda_causal from 0.01 to 1.0)
  - Token dimension sweep (d_token 64, 128, 256)
  - Window size sweep (250, 500, 1000, 2000 samples)

## Phase 4: Transfer Evaluation

- [ ] Zero-shot transfer: Sleep-EDF → CHB-MIT
  ```bash
  python -m src.eval.transfer --checkpoint checkpoints/sleep_edf_best.pt --datasets chbmit
  ```
- [ ] Zero-shot transfer: Sleep-EDF → PTB-XL
- [ ] Zero-shot transfer: CHB-MIT → Sleep-EDF
- [ ] Zero-shot transfer: CHB-MIT → PTB-XL
- [ ] Few-shot adaptation (10/50/100 samples from target)

## Phase 5: Causal Structure Validation

- [ ] **Synthetic graph recovery** (linear VAR):
  ```bash
  python -m scripts.validate_causal_graph --n_nodes 10 --n_samples 10000
  ```
- [ ] **Synthetic graph recovery** (nonlinear VAR) — same script tests both
- [ ] **Target**: AUROC > 0.85 (linear), > 0.75 (nonlinear)
- [ ] **Graph stability across seeds**: extract graphs from 3 training seeds, compute edge agreement rate (should be > 80% for top edges)

## Phase 6: Theory Verification

- [ ] **Bound verification**: Run `src.theory.verify_bound_empirically()` on each trained model
  ```python
  from src.theory import verify_bound_empirically
  result = verify_bound_empirically(model, train_loader, test_loader)
  # bound_holds should be True (empirical gap < theoretical bound)
  ```
- [ ] Verify effective degree `d << D` (graph is actually sparse, not dense)
- [ ] **Key claim**: graph-masked model should have tighter generalization bound than vanilla transformer

## Phase 7: Statistical Significance

- [ ] All results reported with mean ± std across 3 seeds
- [ ] **Paired t-test** or **Wilcoxon signed-rank** between our model and each baseline
- [ ] p < 0.05 for main claims (our model vs vanilla transformer on each dataset)
- [ ] **Effect size** (Cohen's d) for key comparisons

## Phase 8: Figure Generation

### Required Figures
- [ ] **Fig 1**: Architecture overview (draw in PowerPoint/Figma, export PDF)
- [ ] **Fig 2**: Learned causal graphs for representative subjects (use `src.eval.interpret`)
  ```python
  from src.eval.interpret import visualize_causal_graph, plot_graph_stability
  visualize_causal_graph(adj_matrix, channel_names, save_path="figures/graph.pdf")
  ```
- [ ] **Fig 3**: Spectral tokenizer attention map / band decomposition visualization
- [ ] **Fig 4**: Ablation results as grouped bar chart
- [ ] **Fig 5**: Transfer performance heatmap (6 source→target combinations)
- [ ] **Fig 6**: Theoretical bound vs empirical generalization gap plot

### Required Tables
- [ ] **Table 1**: Main results — our model vs all baselines on 3 datasets (F1, AUROC, ECE)
- [ ] **Table 2**: Ablation study — each component removed
- [ ] **Table 3**: Cross-dataset transfer matrix
- [ ] **Table 4**: Synthetic causal graph recovery (AUROC, F1, precision, recall)
- [ ] **Table 5**: Computational cost comparison (params, FLOPs, training time)

## Phase 9: Reproducibility Checks

- [ ] **All random seeds fixed** (PyTorch, NumPy, Python random, CUDA)
- [ ] **Exact environment captured**: `pip freeze > requirements_frozen.txt`
- [ ] **All hyperparameters logged to W&B** (no manual tuning that isn't documented)
- [ ] **Data preprocessing is deterministic** (same splits, same normalization)
- [ ] **Model checkpoints saved** for all reported results

## Phase 10: Sanity Checks Before Writing

- [ ] Training curves look reasonable (loss decreases, no NaN/Inf)
- [ ] No data leakage: verify train/val/test subjects are disjoint
  ```python
  from src.data.splits import verify_no_leakage
  ```
- [ ] Model is not memorizing: training accuracy shouldn't be >> test accuracy
- [ ] Calibration (ECE) is reasonable (< 0.1 ideally)
- [ ] Learned graphs are interpretable: neuroscientifically plausible connections
  - EEG: frontal-parietal connections for attention tasks, hemispheric effects
  - ECG: known lead relationships (e.g., adjacent chest leads correlated)
- [ ] Distribution shift experiments: test under simulated non-stationarity

---

## Quick Reference: Key Numbers to Report

| Metric | What it proves |
|--------|---------------|
| F1 macro, AUROC | Task performance |
| Δ(ours - vanilla TF) | Graph masking helps |
| Δ(ours - static GNN) | Learned > fixed graph |
| Δ(ours - correlation) | Causal > correlational |
| Transfer F1 drop | Generalization claim |
| Bound holds (T/F) | Theory is valid |
| Graph AUROC (synthetic) | Graph inference works |
| ECE | Calibration quality |

---

## Estimated Compute Budget

| Run | Time (A100 80GB) | Count | Total |
|-----|-------------------|-------|-------|
| Main model (1 dataset, 1 seed) | ~4-8 hours | 9 | 36-72h |
| Baseline (1 dataset, 1 seed) | ~3-6 hours | 45 | 135-270h |
| Ablation sweep (per dataset) | ~20-30 hours | 3 | 60-90h |
| Transfer eval | ~30 min each | 6 | 3h |
| Causal validation | ~1-2 hours | 1 | 2h |
| **Total** | | | **~236-437 GPU-hours** |

Tip: Run baselines in parallel on multiple pods if budget allows.
