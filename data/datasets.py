"""PyTorch Dataset classes for LT-MIA."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


class LMDataset(Dataset):
    """PyTorch Dataset for language model fine-tuning."""

    def __init__(self, encodings: Dict[str, torch.Tensor]):
        """Initialize with pre-tokenized encodings.

        Args:
            encodings: Dict with 'input_ids' and 'attention_mask' tensors.
        """
        self.encodings = encodings
        self.labels = encodings["input_ids"].clone()
        self.labels[self.encodings["attention_mask"] == 0] = -100

    def __len__(self):
        return self.encodings["input_ids"].size(0)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


class MIADataset(Dataset):
    """Simple in-memory dataset for MIA training."""

    def __init__(self, feats: np.ndarray, masks: np.ndarray, labels: np.ndarray):
        """Initialize dataset from numpy arrays.

        Args:
            feats: Feature array of shape (N, T, D).
            masks: Mask array of shape (N, T).
            labels: Label array of shape (N,).
        """
        self.feats = torch.from_numpy(feats)
        self.masks = torch.from_numpy(masks)
        self.labels = torch.from_numpy(labels.astype(np.float32))

    def __len__(self):
        return self.feats.size(0)

    def __getitem__(self, idx):
        return (
            self.feats[idx],
            self.masks[idx],
            self.labels[idx],
        )


class CombinedMIADataset(Dataset):
    """Dataset combining features from multiple model/dataset combinations."""

    def __init__(
        self,
        manifest_path: Path,
        split: str = "train",
        combinations: Optional[List[str]] = None,
        root_dir: Optional[Path] = None,
    ):
        """Initialize combined dataset.

        Args:
            manifest_path: Path to the manifest file.
            split: Data split to use ("train", "val", or "test").
            combinations: Optional list of combo_ids to include.
            root_dir: Root directory for feature paths (defaults to manifest parent).
        """
        from ..features.manifest import load_manifest

        self.manifest_path = Path(manifest_path)
        self.split = split
        self.root_dir = root_dir or self.manifest_path.parent
        all_entries = load_manifest(manifest_path)
        if combinations is not None:
            self.entries = {k: v for k, v in all_entries.items() if k in combinations}
        else:
            self.entries = all_entries

        if not self.entries:
            raise ValueError(f"No combinations found in manifest matching: {combinations}")

        self.combo_ids = list(self.entries.keys())
        self.n_combos = len(self.combo_ids)

        self._load_features()
        self._build_index()

        print(f"[CombinedMIADataset] Loaded {self.n_combos} combinations, {len(self)} total samples")

    def _load_features(self):
        """Load feature files for all combinations."""
        self.combo_data = {}

        for combo_id, entry in self.entries.items():
            feature_dir = self.root_dir / entry.path

            mem_feats = np.load(feature_dir / "members_feats.npy", mmap_mode='r')
            mem_masks = np.load(feature_dir / "members_masks.npy", mmap_mode='r')
            non_feats = np.load(feature_dir / "nonmembers_feats.npy", mmap_mode='r')
            non_masks = np.load(feature_dir / "nonmembers_masks.npy", mmap_mode='r')
            indices = np.load(feature_dir / f"{self.split}_indices.npy")

            self.combo_data[combo_id] = {
                "mem_feats": mem_feats,
                "mem_masks": mem_masks,
                "non_feats": non_feats,
                "non_masks": non_masks,
                "indices": indices,
                "combo_idx": self.combo_ids.index(combo_id),
            }

    def _build_index(self):
        """Build global index mapping: global_idx -> (combo_id, local_idx, label, combo_idx)."""
        self.index_map = []

        for combo_id, data in self.combo_data.items():
            indices = data["indices"]
            combo_idx = data["combo_idx"]
            for local_idx in indices:
                self.index_map.append((combo_id, int(local_idx), 1, combo_idx))
            for local_idx in indices:
                self.index_map.append((combo_id, int(local_idx), 0, combo_idx))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        combo_id, local_idx, label, combo_idx = self.index_map[idx]
        data = self.combo_data[combo_id]

        if label == 1:
            feats = data["mem_feats"][local_idx]
            mask = data["mem_masks"][local_idx]
        else:
            feats = data["non_feats"][local_idx]
            mask = data["non_masks"][local_idx]

        return (
            torch.from_numpy(feats.copy()),
            torch.from_numpy(mask.copy()),
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(combo_idx, dtype=torch.long),
        )

    def get_combo_weights(self) -> np.ndarray:
        """Get per-sample weights for balanced sampling across combinations.

        Returns:
            Weight array of shape (N,) where weights sum to N.
        """
        weights = np.zeros(len(self))
        combo_counts = {}
        for combo_id, local_idx, label, combo_idx in self.index_map:
            combo_counts[combo_id] = combo_counts.get(combo_id, 0) + 1
        for i, (combo_id, local_idx, label, combo_idx) in enumerate(self.index_map):
            weights[i] = 1.0 / combo_counts[combo_id]
        weights = weights / weights.sum() * len(self)

        return weights

    def get_balanced_sampler(self) -> WeightedRandomSampler:
        """Get a sampler for balanced training across combinations.

        Returns:
            WeightedRandomSampler with uniform sampling across combinations.
        """
        weights = self.get_combo_weights()
        return WeightedRandomSampler(
            weights=weights.tolist(),
            num_samples=len(self),
            replacement=True,
        )

    def get_feature_dim(self) -> int:
        """Get feature dimension (should be consistent across combinations)."""
        first_combo = self.combo_data[self.combo_ids[0]]
        return first_combo["mem_feats"].shape[-1]

    def get_sequence_length(self) -> int:
        """Get sequence length (should be consistent across combinations)."""
        first_combo = self.combo_data[self.combo_ids[0]]
        return first_combo["mem_feats"].shape[1]


class CombinedMIADatasetSimple(Dataset):
    """Simplified combined dataset that omits combo_idx for compatibility."""

    def __init__(self, combined_dataset: CombinedMIADataset):
        """Initialize wrapper around CombinedMIADataset.

        Args:
            combined_dataset: The underlying combined dataset.
        """
        self.dataset = combined_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        feats, mask, label, combo_idx = self.dataset[idx]
        return feats, mask, label

    def get_balanced_sampler(self):
        """Get balanced sampler from underlying dataset."""
        return self.dataset.get_balanced_sampler()
