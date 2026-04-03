#!/usr/bin/env python3
"""
Combined MIA Training Script

Trains an MIA classifier on pre-extracted features from multiple model/dataset
combinations. Supports holdout evaluation for generalization testing.

Usage:
    python train_combined.py --config configs/train_attacker.yaml
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

import utils as u


def load_config(path: str) -> dict:
    """Load configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Dictionary containing the configuration.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_training(cfg: dict):
    """Train MIA model on combined dataset from multiple model/dataset combinations.

    Args:
        cfg: Configuration dictionary containing training parameters including:
            - manifest_path: Path to the feature manifest
            - save_path: Where to save the trained model
            - train_combinations: List of combo_ids to train on
            - holdout_combinations: List of combo_ids for generalization testing
            - model/mia_model: Model architecture configuration
            - train: Training hyperparameters
    """
    seed = cfg.get("seed", 42)
    u.set_seed(seed)
    device = u.get_device()
    print(f"Device: {device}")

    manifest_path = Path(cfg["manifest_path"])
    save_path = Path(cfg.get("save_path", "outputs/mia_combined.pt"))
    save_path.parent.mkdir(parents=True, exist_ok=True)

    train_combinations = cfg.get("train_combinations", None)
    holdout_combinations = cfg.get("holdout_combinations", [])

    if holdout_combinations and train_combinations is None:
        all_entries = u.load_manifest(manifest_path)
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
    train_ds_full = u.CombinedMIADataset(
        manifest_path=manifest_path,
        split="train",
        combinations=train_combinations,
    )
    train_ds = u.CombinedMIADatasetSimple(train_ds_full)

    print("Loading validation dataset...")
    val_ds_full = u.CombinedMIADataset(
        manifest_path=manifest_path,
        split="val",
        combinations=train_combinations,
    )
    val_ds = u.CombinedMIADatasetSimple(val_ds_full)

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")

    d_in = train_ds_full.get_feature_dim()
    seq_len = train_ds_full.get_sequence_length()
    print(f"Feature dim: {d_in}, Sequence length: {seq_len}")
    print(f"Architecture: {architecture}")

    mia = u.create_mia_model(
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
    mia = u.train_mia_model(
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
    test_ds_full = u.CombinedMIADataset(
        manifest_path=manifest_path,
        split="test",
        combinations=train_combinations,
    )
    test_ds = u.CombinedMIADatasetSimple(test_ds_full)

    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False
    )

    mia.to(device)
    mia.eval()

    all_logits, all_labels = [], []
    with torch.no_grad():
        for feats, masks, labels in test_loader:
            feats = feats.to(device)
            masks = masks.to(device)
            logits = mia(feats, masks)
            all_logits.append(logits.cpu())
            all_labels.append(labels)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    probs = torch.sigmoid(all_logits)
    preds = (probs >= 0.5).float()
    test_acc = (preds == all_labels).float().mean().item()
    test_auc, test_tpr1, test_tpr01 = u.compute_curve_metrics(
        all_labels.numpy(), probs.numpy()
    )

    def fmt(x):
        return f"{x:.4f}" if x is not None else "NA"

    print(f"\n{'='*60}")
    print("Test Results (In-Distribution)")
    print(f"{'='*60}")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"AUC: {fmt(test_auc)}")
    print(f"TPR@1%FPR: {fmt(test_tpr1)}")
    print(f"TPR@0.1%FPR: {fmt(test_tpr01)}")

    if holdout_combinations:
        print(f"\n{'='*60}")
        print("Holdout Generalization Results")
        print(f"{'='*60}")

        for combo_id in holdout_combinations:
            try:
                holdout_ds_full = u.CombinedMIADataset(
                    manifest_path=manifest_path,
                    split="test",
                    combinations=[combo_id],
                )
                holdout_ds = u.CombinedMIADatasetSimple(holdout_ds_full)

                holdout_loader = torch.utils.data.DataLoader(
                    holdout_ds, batch_size=batch_size, shuffle=False
                )

                all_logits, all_labels = [], []
                with torch.no_grad():
                    for feats, masks, labels in holdout_loader:
                        feats = feats.to(device)
                        masks = masks.to(device)
                        logits = mia(feats, masks)
                        all_logits.append(logits.cpu())
                        all_labels.append(labels)

                all_logits = torch.cat(all_logits)
                all_labels = torch.cat(all_labels)
                probs = torch.sigmoid(all_logits)
                preds = (probs >= 0.5).float()
                acc = (preds == all_labels).float().mean().item()
                auc, tpr1, tpr01 = u.compute_curve_metrics(
                    all_labels.numpy(), probs.numpy()
                )

                print(f"\n{combo_id}:")
                print(f"  Accuracy: {acc:.4f}")
                print(f"  AUC: {fmt(auc)}")
                print(f"  TPR@1%FPR: {fmt(tpr1)}")
                print(f"  TPR@0.1%FPR: {fmt(tpr01)}")

            except Exception as e:
                print(f"\n{combo_id}: Error - {e}")

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
                "accuracy": test_acc,
                "auc": test_auc,
                "tpr_at_1pct": test_tpr1,
                "tpr_at_01pct": test_tpr01,
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
    python train_combined.py --config configs/train_attacker.yaml

    python train_combined.py \\
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
