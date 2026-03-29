"""Main training entry point with Hydra config."""
import logging
import os
import random
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

from src.data.eeg_dataset import EEGDataset
from src.data.ecg_dataset import ECGDataset
from src.model.full_model import CausalBiosignalModel
from src.loss.spectral_loss import spectral_reconstruction_loss
from src.loss.causal_loss import causal_consistency_loss
from src.loss.task_loss import classification_loss, joint_loss
from src.eval.benchmark import evaluate_model

logger = logging.getLogger(__name__)


def build_dataset(cfg: DictConfig, split: str):
    if cfg.dataset.type == "eeg":
        return EEGDataset(
            data_dir=cfg.dataset.path,
            split=split,
            split_file=cfg.dataset.split_file,
            sample_rate=cfg.dataset.sample_rate,
            window_samples=cfg.dataset.window_samples,
            channels=cfg.dataset.channels,
            bandpass_low=cfg.dataset.bandpass_low,
            bandpass_high=cfg.dataset.bandpass_high,
        )
    elif cfg.dataset.type == "ecg":
        return ECGDataset(
            data_dir=cfg.dataset.path,
            split=split,
            split_file=cfg.dataset.split_file,
            sample_rate=cfg.dataset.sample_rate,
            window_samples=cfg.dataset.window_samples,
            channels=cfg.dataset.channels,
            bandpass_low=cfg.dataset.bandpass_low,
            bandpass_high=cfg.dataset.bandpass_high,
        )
    else:
        raise ValueError(f"Unknown dataset type: {cfg.dataset.type}")


def build_model(cfg: DictConfig) -> nn.Module:
    model_class = getattr(cfg.model, "model_class", "causal")

    if model_class == "causal":
        return _build_causal_model(cfg)

    from src.model.baselines import (
        PatchTSTBaseline,
        StaticGNNBaseline,
        CorrelationGraphBaseline,
        VanillaTransformerBaseline,
        RawWaveformBaseline,
    )

    common = dict(
        n_channels=cfg.dataset.channels,
        n_classes=cfg.dataset.num_classes,
        d_model=cfg.model.d_token,
        n_layers=cfg.model.transformer.n_layers,
        n_heads=cfg.model.transformer.n_heads,
        d_ff=cfg.model.transformer.d_ff,
        dropout=cfg.model.transformer.dropout,
    )

    if model_class == "patchtst":
        return PatchTSTBaseline(
            window_samples=cfg.dataset.window_samples,
            **common,
        )
    elif model_class == "vanilla_tf":
        return VanillaTransformerBaseline(
            window_samples=cfg.dataset.window_samples,
            sample_rate=cfg.dataset.sample_rate,
            n_fft=cfg.model.tokenizer.stft_n_fft,
            hop_length=cfg.model.tokenizer.stft_hop_length,
            win_length=cfg.model.tokenizer.stft_win_length,
            **common,
        )
    elif model_class == "static_gnn":
        return StaticGNNBaseline(
            n_bands=cfg.model.n_bands,
            **common,
        )
    elif model_class == "corr_graph":
        return CorrelationGraphBaseline(
            n_bands=cfg.model.n_bands,
            **common,
        )
    elif model_class == "raw_waveform":
        return RawWaveformBaseline(
            window_samples=cfg.dataset.window_samples,
            **common,
        )
    else:
        raise ValueError(f"Unknown model_class: {model_class}")


def _build_causal_model(cfg: DictConfig) -> CausalBiosignalModel:
    bands = None
    if hasattr(cfg.model.tokenizer, "bands"):
        bands = {
            k: tuple(v) for k, v in cfg.model.tokenizer.bands.items()
        }

    model = CausalBiosignalModel(
        n_channels=cfg.dataset.channels,
        n_classes=cfg.dataset.num_classes,
        d_token=cfg.model.d_token,
        window_samples=cfg.dataset.window_samples,
        sample_rate=cfg.dataset.sample_rate,
        n_fft=cfg.model.tokenizer.stft_n_fft,
        hop_length=cfg.model.tokenizer.stft_hop_length,
        win_length=cfg.model.tokenizer.stft_win_length,
        bands=bands,
        graph_n_layers=cfg.model.graph.n_layers,
        graph_n_heads=cfg.model.graph.n_heads,
        graph_dropout=cfg.model.graph.dropout,
        sparsity_threshold=cfg.model.graph.sparsity_threshold,
        l1_lambda=cfg.model.graph.l1_lambda,
        tf_n_layers=cfg.model.transformer.n_layers,
        tf_n_heads=cfg.model.transformer.n_heads,
        tf_d_ff=cfg.model.transformer.d_ff,
        tf_dropout=cfg.model.transformer.dropout,
        tf_gradient_checkpointing=getattr(cfg.model.transformer, "gradient_checkpointing", False),
        cls_hidden=cfg.model.classifier.hidden_dim,
        cls_dropout=cfg.model.classifier.dropout,
        use_adapter=False,
        adapter_hidden=cfg.model.adapter.hidden_dim,
        adapter_layers=cfg.model.adapter.n_layers,
    )
    return model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    cfg: DictConfig,
    device: torch.device,
    epoch: int,
    scaler: torch.amp.GradScaler | None = None,
    class_weights: torch.Tensor | None = None,
) -> dict:
    model.train()
    total_loss_accum = 0.0
    n_batches = 0

    is_causal = getattr(cfg.model, "model_class", "causal") == "causal"

    # Causal loss is expensive (extra transformer passes). Compute every K steps
    # to save ~60-80% wall time with negligible quality impact.
    causal_every = getattr(cfg.loss, "causal_every_n_steps", 4)
    global_step_offset = (epoch - 1) * len(loader)

    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    for batch_idx, batch in enumerate(pbar):
        signal = batch["signal"].to(device, non_blocking=True)  # (B, C, T)
        label = batch["label"].to(device, non_blocking=True)    # (B,)

        optimizer.zero_grad()

        if "bf16" in cfg.hardware.precision:
            dtype = torch.bfloat16
        elif "fp16" in cfg.hardware.precision:
            dtype = torch.float16
        else:
            dtype = torch.float32
        with torch.autocast(device_type="cuda", dtype=dtype):
            output = model(signal)

            # Task loss (always computed)
            task_l = classification_loss(
                output["logits"], label,
                label_smoothing=cfg.loss.task.label_smoothing,
                class_weights=class_weights,
            )

            if is_causal:
                # Reconstruction loss
                recon_loss = spectral_reconstruction_loss(
                    output["recon"], signal,
                    n_fft=cfg.model.tokenizer.stft_n_fft,
                    hop_length=cfg.model.tokenizer.stft_hop_length,
                )

                # Causal consistency loss — only every K steps
                tokens = output["tokens"]
                adj = output["adj"]
                edge_probs = output["edge_probs"]
                embeddings = output["embeddings"]

                step = global_step_offset + batch_idx
                if step % causal_every == 0:
                    causal_l = causal_consistency_loss(
                        adj=adj,
                        edge_probs=edge_probs,
                        embeddings=embeddings,
                        transformer_fn=model.transformer,
                        tokens=tokens,
                        n_interventions=cfg.loss.causal_consistency.n_interventions,
                        intervention_type=cfg.loss.causal_consistency.intervention_type,
                    )
                else:
                    causal_l = torch.tensor(0.0, device=device)

                # Sparsity and DAG losses (cheap — always compute)
                sparsity_l = model.graph_inferencer.sparsity_loss(edge_probs)
                dag_l = model.graph_inferencer.dag_loss(edge_probs)

                total, loss_dict = joint_loss(
                    recon_loss=recon_loss,
                    causal_loss=causal_l,
                    task_loss=task_l,
                    lambda_recon=cfg.loss.lambda_recon,
                    lambda_causal=cfg.loss.lambda_causal,
                    lambda_task=cfg.loss.lambda_task,
                    sparsity_loss=sparsity_l,
                    dag_loss=dag_l,
                )
            else:
                # Baselines: task loss only
                total = task_l
                loss_dict = {"task": task_l.item(), "total": task_l.item()}

        if scaler is not None:
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss_accum += loss_dict["total"]
        n_batches += 1

        pbar.set_postfix({
            "loss": f"{loss_dict['total']:.4f}",
            "task": f"{loss_dict['task']:.4f}",
        })

        if wandb.run:
            wandb.log({"train/" + k: v for k, v in loss_dict.items()})

    return {"avg_loss": total_loss_accum / max(n_batches, 1)}


@hydra.main(version_base="1.3", config_path="config", config_name="base")
def main(cfg: DictConfig):
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Seed — full reproducibility
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # W&B
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=cfg.wandb.mode,
    )

    # Data
    train_ds = build_dataset(cfg, "train")
    val_ds = build_dataset(cfg, "val")

    # Compute inverse-frequency class weights for imbalanced datasets (e.g., CHB-MIT)
    class_weights = None
    if getattr(cfg.dataset, "num_classes", 0) == 2 and len(train_ds) > 0:
        # Binary classification — likely imbalanced
        try:
            import h5py
            from collections import Counter
            label_counts = Counter()
            h5_path = Path(train_ds.data_dir) / "data.h5"
            if h5_path.exists():
                with h5py.File(h5_path, "r") as f:
                    for subj_id, _ in train_ds.index:
                        labels_arr = f[subj_id]["labels"][:]
                        for lbl in labels_arr:
                            label_counts[int(lbl)] += 1
            if label_counts:
                n_classes = cfg.dataset.num_classes
                total = sum(label_counts.values())
                weights = [total / (n_classes * label_counts.get(c, 1)) for c in range(n_classes)]
                class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
                logger.info(f"Class weights: {weights}")
        except Exception as e:
            logger.warning(f"Could not compute class weights: {e}")

    nw = cfg.num_workers
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=nw,
        pin_memory=True,
        drop_last=True,
        persistent_workers=nw > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.eval.batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=nw > 0,
    )

    # Model
    model = build_model(cfg)
    if cfg.hardware.compile:
        model = torch.compile(model, mode=cfg.hardware.compile_mode)
    model = model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {param_count:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    # Scheduler: cosine with warmup
    total_steps = len(train_loader) * cfg.train.epochs
    warmup_steps = len(train_loader) * cfg.train.warmup_epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # GradScaler for mixed precision — only needed for fp16, not bf16
    scaler = None
    if "fp16" in cfg.hardware.precision and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")

    # Paths — unique per dataset/seed/model to prevent overwrite in multirun
    model_class = getattr(cfg.model, "model_class", "causal")
    run_name = f"{model_class}_{cfg.dataset.name}_seed{cfg.seed}"
    ckpt_dir = Path(cfg.paths.checkpoint_dir) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, cfg.train.epochs + 1):
        logger.info(f"=== Epoch {epoch}/{cfg.train.epochs} ===")

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, cfg, device, epoch, scaler,
            class_weights=class_weights,
        )
        logger.info(f"Train: {train_metrics}")

        # Validation
        val_metrics = evaluate_model(model, val_loader, device, cfg)
        logger.info(f"Val: {val_metrics}")

        if wandb.run:
            wandb.log({"val/" + k: v for k, v in val_metrics.items()}, step=epoch)

        # Checkpointing
        val_f1 = val_metrics.get("f1_macro", 0.0)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            ckpt_path = ckpt_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
                "config": OmegaConf.to_container(cfg, resolve=True),
            }, ckpt_path)
            logger.info(f"Saved best model (F1={val_f1:.4f}) to {ckpt_path}")
        else:
            patience_counter += 1

        if patience_counter >= cfg.train.early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    wandb.finish()
    logger.info(f"Training complete. Best val F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    main()
