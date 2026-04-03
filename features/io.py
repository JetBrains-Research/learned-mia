"""Feature I/O utilities."""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def save_features(
    output_dir: Path,
    members_feats: np.ndarray,
    members_masks: np.ndarray,
    nonmembers_feats: np.ndarray,
    nonmembers_masks: np.ndarray,
    members_token_ids: Optional[np.ndarray] = None,
    nonmembers_token_ids: Optional[np.ndarray] = None,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[int, int, int]:
    """Save extracted features to disk with train/val/test splits.

    Args:
        output_dir: Directory to save features.
        members_feats: Member feature array.
        members_masks: Member mask array.
        nonmembers_feats: Non-member feature array.
        nonmembers_masks: Non-member mask array.
        members_token_ids: Optional member token IDs.
        nonmembers_token_ids: Optional non-member token IDs.
        val_ratio: Fraction of data for validation.
        test_ratio: Fraction of data for testing.
        seed: Random seed for reproducible splits.

    Returns:
        Tuple of (train_size, val_size, test_size).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "members_feats.npy", members_feats.astype(np.float32))
    np.save(output_dir / "members_masks.npy", members_masks.astype(np.uint8))
    np.save(output_dir / "nonmembers_feats.npy", nonmembers_feats.astype(np.float32))
    np.save(output_dir / "nonmembers_masks.npy", nonmembers_masks.astype(np.uint8))
    if members_token_ids is not None:
        np.save(output_dir / "members_token_ids.npy", members_token_ids.astype(np.int64))
    if nonmembers_token_ids is not None:
        np.save(output_dir / "nonmembers_token_ids.npy", nonmembers_token_ids.astype(np.int64))
    n_samples = min(len(members_feats), len(nonmembers_feats))
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_samples)
    n_test = int(n_samples * test_ratio)
    n_val = int(n_samples * val_ratio)
    n_train = n_samples - n_test - n_val
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    np.save(output_dir / "train_indices.npy", train_idx)
    np.save(output_dir / "val_indices.npy", val_idx)
    np.save(output_dir / "test_indices.npy", test_idx)

    print(f"[Save] Features saved to {output_dir}")
    print(f"[Save] Split sizes: train={n_train}, val={n_val}, test={n_test}")

    return n_train, n_val, n_test


def load_features_mmap(
    feature_dir: Path,
    split: str = "train",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load features with memory mapping for efficient access.

    Args:
        feature_dir: Directory containing feature files.
        split: Which split to load indices for ("train", "val", or "test").

    Returns:
        Tuple of (mem_feats, mem_masks, non_feats, non_masks, indices).
    """
    feature_dir = Path(feature_dir)
    mem_feats = np.load(feature_dir / "members_feats.npy", mmap_mode='r')
    mem_masks = np.load(feature_dir / "members_masks.npy", mmap_mode='r')
    non_feats = np.load(feature_dir / "nonmembers_feats.npy", mmap_mode='r')
    non_masks = np.load(feature_dir / "nonmembers_masks.npy", mmap_mode='r')

    indices = np.load(feature_dir / f"{split}_indices.npy")

    return mem_feats, mem_masks, non_feats, non_masks, indices
