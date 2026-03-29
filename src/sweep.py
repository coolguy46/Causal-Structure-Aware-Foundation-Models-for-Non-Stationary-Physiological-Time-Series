"""Hydra multirun sweep runner for ablation studies."""
import subprocess
import sys


ABLATION_CONFIGS = {
    # Loss term ablation (7 configs)
    "recon_only": "loss.lambda_recon=1.0 loss.lambda_causal=0.0 loss.lambda_task=0.0",
    "causal_only": "loss.lambda_recon=0.0 loss.lambda_causal=1.0 loss.lambda_task=0.0",
    "task_only": "loss.lambda_recon=0.0 loss.lambda_causal=0.0 loss.lambda_task=1.0",
    "recon_causal": "loss.lambda_recon=1.0 loss.lambda_causal=0.1 loss.lambda_task=0.0",
    "recon_task": "loss.lambda_recon=1.0 loss.lambda_causal=0.0 loss.lambda_task=1.0",
    "causal_task": "loss.lambda_recon=0.0 loss.lambda_causal=0.1 loss.lambda_task=1.0",
    "all": "loss.lambda_recon=1.0 loss.lambda_causal=0.1 loss.lambda_task=1.0",

    # Causal lambda sweep
    "lambda_0.01": "loss.lambda_causal=0.01",
    "lambda_0.1": "loss.lambda_causal=0.1",
    "lambda_0.5": "loss.lambda_causal=0.5",
    "lambda_1.0": "loss.lambda_causal=1.0",

    # Token dim sweep
    "d64": "model.d_token=64",
    "d128": "model.d_token=128",
    "d256": "model.d_token=256",

    # Window size sweep (Sleep-EDF @ 100 Hz)
    "win_1s": "dataset.window_sec=1.0 dataset.window_samples=100",
    "win_2s": "dataset.window_sec=2.0 dataset.window_samples=200",
    "win_4s": "dataset.window_sec=4.0 dataset.window_samples=400",
}


def run_sweep(sweep_names: list[str], seeds: list[int] = [42, 123, 7]):
    """Run ablation sweep with multiple seeds."""
    for name in sweep_names:
        if name not in ABLATION_CONFIGS:
            print(f"Unknown sweep config: {name}")
            continue

        overrides = ABLATION_CONFIGS[name]
        for seed in seeds:
            cmd = (
                f"python -m src.train {overrides} "
                f"seed={seed} "
                f"wandb.project=causal-biosignal-ablation "
                f"paths.output_dir=/workspace/outputs/{name}/seed_{seed}"
            )
            print(f"\n{'='*60}")
            print(f"Running: {name} (seed={seed})")
            print(f"Command: {cmd}")
            print(f"{'='*60}\n")
            subprocess.run(cmd.split(), check=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweeps",
        nargs="+",
        default=["all"],
        choices=list(ABLATION_CONFIGS.keys()) + ["loss_ablation", "lambda_sweep", "token_dim", "window_size"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 7])
    args = parser.parse_args()

    # Expand sweep groups
    sweep_groups = {
        "loss_ablation": ["recon_only", "causal_only", "task_only", "recon_causal", "recon_task", "causal_task", "all"],
        "lambda_sweep": ["lambda_0.01", "lambda_0.1", "lambda_0.5", "lambda_1.0"],
        "token_dim": ["d64", "d128", "d256"],
        "window_size": ["win_1s", "win_2s", "win_4s"],
    }

    sweeps = []
    for s in args.sweeps:
        if s in sweep_groups:
            sweeps.extend(sweep_groups[s])
        else:
            sweeps.append(s)

    run_sweep(sweeps, args.seeds)
