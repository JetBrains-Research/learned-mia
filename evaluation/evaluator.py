"""MIA model evaluation utilities."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .metrics import compute_curve_metrics
from ..data.datasets import MIADataset, CombinedMIADataset, CombinedMIADatasetSimple


@dataclass
class EvalResults:
    """Results from MIA evaluation."""

    accuracy: float
    auc: Optional[float]
    tpr_at_1pct: Optional[float]
    tpr_at_01pct: Optional[float]


class MIAEvaluator:
    """Evaluator for MIA models."""

    def __init__(self, model: nn.Module, device: torch.device):
        """Initialize evaluator.

        Args:
            model: The MIA model to evaluate.
            device: Device to run evaluation on.
        """
        self.model = model
        self.device = device

    def evaluate(self, dataset: Dataset, batch_size: int = 256) -> EvalResults:
        """Evaluate model on a dataset.

        Args:
            dataset: Dataset yielding (features, masks, labels).
            batch_size: Batch size for evaluation.

        Returns:
            EvalResults with accuracy and metrics.
        """
        self.model.to(self.device)
        self.model.eval()

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_logits, all_labels = [], []
        with torch.no_grad():
            for feats, masks, labels in loader:
                feats = feats.to(self.device)
                masks = masks.to(self.device)
                logits = self.model(feats, masks)
                all_logits.append(logits.cpu())
                all_labels.append(labels)

        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        probs = torch.sigmoid(all_logits)
        preds = (probs >= 0.5).float()
        acc = (preds == all_labels).float().mean().item()
        auc, tpr1, tpr01 = compute_curve_metrics(all_labels.numpy(), probs.numpy())

        return EvalResults(
            accuracy=acc,
            auc=auc,
            tpr_at_1pct=tpr1,
            tpr_at_01pct=tpr01,
        )

    def evaluate_per_combo(
        self,
        manifest_path: Path,
        combo_ids: List[str],
        split: str = "test",
        batch_size: int = 256,
    ) -> Dict[str, EvalResults]:
        """Evaluate on multiple combinations with per-combo breakdown.

        Args:
            manifest_path: Path to the manifest file.
            combo_ids: List of combination IDs to evaluate.
            split: Data split to use.
            batch_size: Batch size for evaluation.

        Returns:
            Dict mapping combo_id to EvalResults.
        """
        results = {}

        for combo_id in combo_ids:
            try:
                combo_ds_full = CombinedMIADataset(
                    manifest_path=manifest_path,
                    split=split,
                    combinations=[combo_id],
                )
                combo_ds = CombinedMIADatasetSimple(combo_ds_full)
                results[combo_id] = self.evaluate(combo_ds, batch_size)
            except Exception as e:
                print(f"Error evaluating {combo_id}: {e}")
                results[combo_id] = EvalResults(
                    accuracy=0.0, auc=None, tpr_at_1pct=None, tpr_at_01pct=None
                )

        return results


def evaluate_mia_model(
    mia_model: nn.Module,
    mem_feats: np.ndarray,
    mem_masks: np.ndarray,
    non_feats: np.ndarray,
    non_masks: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> tuple:
    """Evaluate MIA model on member and non-member features.

    Args:
        mia_model: The trained MIA model.
        mem_feats: Member features array.
        mem_masks: Member masks array.
        non_feats: Non-member features array.
        non_masks: Non-member masks array.
        device: Device to run evaluation on.
        batch_size: Batch size for evaluation.

    Returns:
        Tuple of (accuracy, auc, tpr_at_1pct, tpr_at_01pct).
    """
    mia_model = mia_model.to(device)
    mia_model.eval()

    labels = np.concatenate([np.ones(len(mem_feats)), np.zeros(len(non_feats))])
    feats = np.concatenate([mem_feats, non_feats], axis=0)
    masks = np.concatenate([mem_masks, non_masks], axis=0)

    ds = MIADataset(feats, masks, labels)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_logits, all_labels = [], []
    with torch.no_grad():
        for f, m, l in dl:
            f = f.to(device)
            m = m.to(device)
            logits = mia_model(f, m)
            all_logits.append(logits.detach().cpu())
            all_labels.append(l)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    probs = torch.sigmoid(all_logits)
    preds = (probs >= 0.5).float()
    acc = (preds == all_labels).float().mean().item()
    auc, tpr1, tpr01 = compute_curve_metrics(all_labels.numpy(), probs.numpy())
    return acc, auc, tpr1, tpr01
