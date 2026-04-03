"""
MIA Model Evaluation Script

Evaluates saved MIA classifiers on pre-extracted features from the manifest.
Supports per-combination breakdown and CSV result export.

Usage:
    python evaluate.py --config configs/evaluate_attacker.yaml
"""

import argparse
import csv
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


def load_mia_model(ckpt_path: Path, device: torch.device):
    """Load a saved MIA model from checkpoint.

    Args:
        ckpt_path: Path to the saved checkpoint file.
        device: Device to load the model onto.

    Returns:
        The loaded MIA model with weights restored.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    d_in = ckpt["d_in"]
    seq_len = ckpt.get("seq_len", 128)
    architecture = ckpt.get("architecture", "transformer")
    mia_hparams = ckpt.get("mia_hparams", {})
    mia = u.create_mia_model(
        architecture=architecture,
        d_in=d_in,
        seq_len=seq_len,
        d_model=mia_hparams.get("d_model", 128),
        nhead=mia_hparams.get("nhead", 4),
        num_layers=mia_hparams.get("num_layers", 2),
        dim_ff=mia_hparams.get("dim_ff", 256),
        dropout=mia_hparams.get("dropout", 0.1),
        hidden_dims=mia_hparams.get("hidden_dims", [256, 128]),
    )
    mia.load_state_dict(ckpt["state_dict"])
    return mia


def run_eval(cfg: dict):
    """Run evaluation on pre-extracted features from manifest.

    Args:
        cfg: Configuration dictionary containing eval settings.
    """
    device = u.get_device()
    print(f"Device: {device}")

    eval_cfg = cfg["eval"]
    ckpt_path = Path(eval_cfg["checkpoint"])
    manifest_path = Path(eval_cfg["manifest_path"])
    combinations = eval_cfg.get("combinations", None)
    split = eval_cfg.get("split", "test")
    batch_size = int(eval_cfg.get("batch_size", 256))

    mia = load_mia_model(ckpt_path, device)
    mia.to(device)
    mia.eval()

    ds_full = u.CombinedMIADataset(
        manifest_path=manifest_path,
        split=split,
        combinations=combinations,
    )
    ds = u.CombinedMIADatasetSimple(ds_full)

    print(f"Evaluating on {len(ds)} samples from {len(ds_full.combo_ids)} combinations")

    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_logits, all_labels = [], []
    with torch.no_grad():
        for feats, masks, labels in loader:
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
    auc, tpr1, tpr01 = u.compute_curve_metrics(all_labels.numpy(), probs.numpy())

    def fmt(x):
        return f"{x:.4f}" if x is not None else "NA"

    print(f"\n{'='*60}")
    print(f"Results ({split} split)")
    print(f"{'='*60}")
    print(f"Combinations: {combinations or 'all'}")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC: {fmt(auc)}")
    print(f"TPR@1%FPR: {fmt(tpr1)}")
    print(f"TPR@0.1%FPR: {fmt(tpr01)}")

    results_rows = []
    results_rows.append({
        "target_model": "all",
        "attack_model": ckpt_path.name,
        "dataset": split,
        "auc": auc,
        "tpr@1%": tpr1,
        "tpr@0.1%": tpr01,
        "config_yaml": "",
    })

    if eval_cfg.get("per_combo_breakdown", True):
        print(f"\nPer-Combination Breakdown:")
        print("-" * 60)

        for combo_id in ds_full.combo_ids:
            combo_ds_full = u.CombinedMIADataset(
                manifest_path=manifest_path,
                split=split,
                combinations=[combo_id],
            )
            combo_ds = u.CombinedMIADatasetSimple(combo_ds_full)
            combo_loader = torch.utils.data.DataLoader(
                combo_ds, batch_size=batch_size, shuffle=False
            )

            combo_logits, combo_labels = [], []
            with torch.no_grad():
                for feats, masks, labels in combo_loader:
                    feats = feats.to(device)
                    masks = masks.to(device)
                    logits = mia(feats, masks)
                    combo_logits.append(logits.cpu())
                    combo_labels.append(labels)

            combo_logits = torch.cat(combo_logits)
            combo_labels = torch.cat(combo_labels)
            combo_probs = torch.sigmoid(combo_logits)
            combo_preds = (combo_probs >= 0.5).float()
            combo_acc = (combo_preds == combo_labels).float().mean().item()
            combo_auc, combo_tpr1, combo_tpr01 = u.compute_curve_metrics(
                combo_labels.numpy(), combo_probs.numpy()
            )

            print(
                f"{combo_id:40s} | Acc: {combo_acc:.4f} | "
                f"AUC: {fmt(combo_auc)} | TPR@1%: {fmt(combo_tpr1)} | "
                f"TPR@0.1%: {fmt(combo_tpr01)}"
            )

            entry = ds_full.entries[combo_id]
            results_rows.append({
                "target_model": entry.model_name,
                "attack_model": ckpt_path.name,
                "dataset": entry.dataset_name,
                "auc": combo_auc,
                "tpr@1%": combo_tpr1,
                "tpr@0.1%": combo_tpr01,
                "config_yaml": "",
            })

    results_path = eval_cfg.get("results_path")
    if results_path:
        results_path = Path(results_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not results_path.exists()

        with results_path.open("a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "target_model",
                    "attack_model",
                    "dataset",
                    "auc",
                    "tpr@1%",
                    "tpr@0.1%",
                    "config_yaml",
                ])
            for row in results_rows:
                writer.writerow([
                    row["target_model"],
                    row["attack_model"],
                    row["dataset"],
                    fmt(row["auc"]),
                    fmt(row["tpr@1%"]),
                    fmt(row["tpr@0.1%"]),
                    row["config_yaml"],
                ])
        print(f"\n[Saved results] {results_path}")


def main():
    """Main entry point for the evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate saved MIA classifier on pre-extracted features",
    )

    parser.add_argument("--config", type=str, required=True, help="Path to eval config YAML")
    parser.add_argument("--checkpoint", type=str, help="Override checkpoint path")
    parser.add_argument("--combos", type=str, nargs="+", help="Specific combinations to evaluate")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], help="Data split")

    args = parser.parse_args()
    cfg = load_config(args.config)

    if "eval" not in cfg:
        cfg["eval"] = {}
    if args.checkpoint:
        cfg["eval"]["checkpoint"] = args.checkpoint
    if args.combos:
        cfg["eval"]["combinations"] = args.combos
    if args.split:
        cfg["eval"]["split"] = args.split

    run_eval(cfg)


if __name__ == "__main__":
    main()
