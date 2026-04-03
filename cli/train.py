"""MIA Training CLI.

Trains an MIA classifier on pre-extracted features from multiple model/dataset
combinations. Supports holdout evaluation for generalization testing.

Usage:
    ltmia-train --config configs/train_attacker.yaml
"""

import argparse
from pathlib import Path

import torch

from ..utils import load_config, set_seed, get_device
from ..data import CombinedMIADataset, CombinedMIADatasetSimple
from ..models import create_mia_model
from ..training import train_mia_model
from ..evaluation import MIAEvaluator
from ..features import load_manifest


def run_training(cfg: dict) -> None:
    """Train MIA model on combined dataset from multiple model/dataset combinations.

    Args:
        cfg: Configuration dictionary containing training parameters.
    """
    seed = cfg.get("seed", 42)
    set_seed(seed)
    device = get_device()
    print(f"Device: {device}")

    manifest_path = Path(cfg["manifest_path"])
    save_path = Path(cfg.get("save_path", "outputs/mia_combined.pt"))
    save_path.parent.mkdir(parents=True, exist_ok=True)

    train_combinations = cfg.get("train_combinations", None)
    holdout_combinations = cfg.get("holdout_combinations", [])

    if holdout_combinations and train_combinations is None:
        all_entries = load_manifest(manifest_path)
        train_combinations = [k for k in all_entries.keys() if k not in holdout_combinations]

    model_cfg = cfg.get("model") or cfg.get("mia_model", {})
    architecture = model_cfg.get("architecture", "transformer")
    d_model = int(model_cfg.get("d_model", 128))
    nhead = int(model_cfg.get("nhead", 4))
    num_layers = int(model_cfg.get("num_layers", 2))
    dim_ff = int(model_cfg.get("dim_ff", 256))
    dropout = float(model_cfg.get("dropout", 0.1))
    hidden_dims = model_cfg.get("hidden_dims", [256, 128])

    train_cfg = cfg.get("train", {})
    epochs = int(train_cfg.get("epochs", 40))
    batch_size = int(train_cfg.get("batch_size", 256))
    lr = float(train_cfg.get("lr", 1e-3))
    balance_strategy = train_cfg.get("balance_strategy")
    label_smooth = float(train_cfg.get("label_smooth", 0.0))
    use_focal = bool(train_cfg.get("use_focal", False))
    focal_gamma = float(train_cfg.get("focal_gamma", 2.0))
    focal_alpha = train_cfg.get("focal_alpha", 0.25)
    if focal_alpha is not None:
        focal_alpha = float(focal_alpha)

    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    lr_scheduler = train_cfg.get("lr_scheduler", "cosine")
    warmup_epochs = int(train_cfg.get("warmup_epochs", 0))
    min_lr = float(train_cfg.get("min_lr", 1e-6))
    grad_clip = train_cfg.get("grad_clip", 1.0)
    if grad_clip is not None:
        grad_clip = float(grad_clip)

    print(f"\n{'='*60}")
    print("Combined MIA Training")
    print(f"{'='*60}")
    print(f"Manifest: {manifest_path}")
    print(f"Train combinations: {train_combinations or 'all'}")
    print(f"Holdout combinations: {holdout_combinations or 'none'}")
    print(f"Balance strategy: {balance_strategy}")
    print(f"{'='*60}\n")

    print("Loading training dataset...")
    train_ds_full = CombinedMIADataset(
        manifest_path=manifest_path,
        split="train",
        combinations=train_combinations,
    )
    train_ds = CombinedMIADatasetSimple(train_ds_full)

    print("Loading validation dataset...")
    val_ds_full = CombinedMIADataset(
        manifest_path=manifest_path,
        split="val",
        combinations=train_combinations,
    )
    val_ds = CombinedMIADatasetSimple(val_ds_full)

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")

    d_in = train_ds_full.get_feature_dim()
    seq_len = train_ds_full.get_sequence_length()
    print(f"Feature dim: {d_in}, Sequence length: {seq_len}")
    print(f"Architecture: {architecture}")

    mia = create_mia_model(
        architecture=architecture,
        d_in=d_in,
        seq_len=seq_len,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_ff=dim_ff,
        dropout=dropout,
        hidden_dims=hidden_dims,
    )

    sampler = None
    if balance_strategy == "uniform":
        print("Using uniform sampling across combinations")
        sampler = train_ds_full.get_balanced_sampler()

    print("\nTraining MIA model...")
    mia = train_mia_model(
        mia,
        train_ds,
        val_ds,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        label_smooth=label_smooth,
        use_focal=use_focal,
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
        sampler=sampler,
        weight_decay=weight_decay,
        lr_scheduler=lr_scheduler,
        warmup_epochs=warmup_epochs,
        min_lr=min_lr,
        grad_clip=grad_clip,
    )

    print("\nEvaluating on test set...")
    test_ds_full = CombinedMIADataset(
        manifest_path=manifest_path,
        split="test",
        combinations=train_combinations,
    )
    test_ds = CombinedMIADatasetSimple(test_ds_full)

    evaluator = MIAEvaluator(mia, device)
    test_results = evaluator.evaluate(test_ds, batch_size)

    def fmt(x):
        return f"{x:.4f}" if x is not None else "NA"

    print(f"\n{'='*60}")
    print("Test Results (In-Distribution)")
    print(f"{'='*60}")
    print(f"Accuracy: {test_results.accuracy:.4f}")
    print(f"AUC: {fmt(test_results.auc)}")
    print(f"TPR@1%FPR: {fmt(test_results.tpr_at_1pct)}")
    print(f"TPR@0.1%FPR: {fmt(test_results.tpr_at_01pct)}")

    if holdout_combinations:
        print(f"\n{'='*60}")
        print("Holdout Generalization Results")
        print(f"{'='*60}")

        holdout_results = evaluator.evaluate_per_combo(
            manifest_path, holdout_combinations, split="test", batch_size=batch_size
        )

        for combo_id, results in holdout_results.items():
            print(f"\n{combo_id}:")
            print(f"  Accuracy: {results.accuracy:.4f}")
            print(f"  AUC: {fmt(results.auc)}")
            print(f"  TPR@1%FPR: {fmt(results.tpr_at_1pct)}")
            print(f"  TPR@0.1%FPR: {fmt(results.tpr_at_01pct)}")

    torch.save(
        {
            "state_dict": mia.state_dict(),
            "d_in": d_in,
            "seq_len": seq_len,
            "architecture": architecture,
            "mia_hparams": {
                "d_model": d_model,
                "nhead": nhead,
                "num_layers": num_layers,
                "dim_ff": dim_ff,
                "dropout": dropout,
                "hidden_dims": hidden_dims,
            },
            "config": cfg,
            "train_combinations": train_combinations,
            "holdout_combinations": holdout_combinations,
            "test_metrics": {
                "accuracy": test_results.accuracy,
                "auc": test_results.auc,
                "tpr_at_1pct": test_results.tpr_at_1pct,
                "tpr_at_01pct": test_results.tpr_at_01pct,
            },
        },
        save_path,
    )
    print(f"\n[Saved] MIA classifier -> {save_path}")


def main():
    """Main entry point for the training script."""
    parser = argparse.ArgumentParser(
        description="Train MIA classifier on combined feature datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    ltmia-train --config configs/train_attacker.yaml

    ltmia-train \\
        --config configs/train_attacker.yaml \\
        --train-combos gpt2_ag_news gpt2_wikipedia \\
        --holdout gpt2_cnndm
        """,
    )

    parser.add_argument("--config", type=str, required=True, help="Path to training config YAML")
    parser.add_argument("--manifest", type=str, help="Override manifest path")
    parser.add_argument("--save-path", type=str, help="Override save path")
    parser.add_argument(
        "--train-combos",
        type=str,
        nargs="+",
        help="Specific combinations to train on",
    )
    parser.add_argument(
        "--holdout",
        type=str,
        nargs="+",
        help="Combinations to hold out for generalization testing",
    )
    parser.add_argument("--epochs", type=int, help="Override training epochs")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--seed", type=int, help="Override random seed")

    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.manifest:
        cfg["manifest_path"] = args.manifest
    if args.save_path:
        cfg["save_path"] = args.save_path
    if args.train_combos:
        cfg["train_combinations"] = args.train_combos
    if args.holdout:
        cfg["holdout_combinations"] = args.holdout
    if args.seed:
        cfg["seed"] = args.seed

    if "train" not in cfg:
        cfg["train"] = {}
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["train"]["batch_size"] = args.batch_size
    if args.lr:
        cfg["train"]["lr"] = args.lr

    run_training(cfg)


if __name__ == "__main__":
    main()
