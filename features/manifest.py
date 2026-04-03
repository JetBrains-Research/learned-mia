"""Manifest and metadata management for extracted features."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict

import yaml


def make_combo_id(model_name: str, dataset_name: str) -> str:
    """Generate a unique combo ID from model and dataset names.

    Args:
        model_name: Name or path of the model.
        dataset_name: Name of the dataset.

    Returns:
        A sanitized string combining model and dataset names.
    """
    model_safe = model_name.replace("/", "_").replace("\\", "_").lower()
    dataset_safe = dataset_name.replace("/", "_").replace("\\", "_").replace("-", "_").lower()
    return f"{model_safe}_{dataset_safe}"


@dataclass
class ExtractionMetadata:
    """Metadata for an extracted feature set."""

    model_name: str
    dataset_name: str
    combo_id: str
    n_members: int
    n_nonmembers: int
    sequence_length: int
    top_k: int
    feature_dim: int
    ft_epochs: int
    ft_batch_size: int
    ft_lr: float
    seed: int
    extraction_timestamp: str = ""
    lora_config: Dict[str, Any] = field(default_factory=dict)
    train_size: int = 0
    val_size: int = 0
    test_size: int = 0


def save_extraction_metadata(metadata: ExtractionMetadata, output_dir: Path) -> None:
    """Save extraction metadata to YAML.

    Args:
        metadata: The metadata to save.
        output_dir: Directory to save the metadata file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "metadata.yaml"
    with open(meta_path, "w") as f:
        yaml.safe_dump(asdict(metadata), f, sort_keys=False)


def load_extraction_metadata(output_dir: Path) -> ExtractionMetadata:
    """Load extraction metadata from YAML.

    Args:
        output_dir: Directory containing the metadata file.

    Returns:
        Loaded ExtractionMetadata object.
    """
    meta_path = Path(output_dir) / "metadata.yaml"
    with open(meta_path, "r") as f:
        data = yaml.safe_load(f)
    return ExtractionMetadata(**data)


@dataclass
class ManifestEntry:
    """Entry in the feature manifest."""

    combo_id: str
    model_name: str
    dataset_name: str
    path: str
    n_members: int
    n_nonmembers: int
    feature_dim: int
    sequence_length: int
    train_size: int
    val_size: int
    test_size: int


def load_manifest(manifest_path: Path) -> Dict[str, ManifestEntry]:
    """Load the feature manifest.

    Args:
        manifest_path: Path to the manifest YAML file.

    Returns:
        Dict mapping combo_id to ManifestEntry.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return {}

    with open(manifest_path, "r") as f:
        data = yaml.safe_load(f) or {}

    entries = {}
    for combo_id, entry_data in data.get("combinations", {}).items():
        entries[combo_id] = ManifestEntry(combo_id=combo_id, **entry_data)

    return entries


def save_manifest(manifest_path: Path, entries: Dict[str, ManifestEntry]) -> None:
    """Save the feature manifest.

    Args:
        manifest_path: Path to save the manifest.
        entries: Dict mapping combo_id to ManifestEntry.
    """
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "combinations": {
            combo_id: {
                "model_name": e.model_name,
                "dataset_name": e.dataset_name,
                "path": e.path,
                "n_members": e.n_members,
                "n_nonmembers": e.n_nonmembers,
                "feature_dim": e.feature_dim,
                "sequence_length": e.sequence_length,
                "train_size": e.train_size,
                "val_size": e.val_size,
                "test_size": e.test_size,
            }
            for combo_id, e in entries.items()
        }
    }

    with open(manifest_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def update_manifest(manifest_path: Path, entry: ManifestEntry) -> None:
    """Add or update a manifest entry.

    Args:
        manifest_path: Path to the manifest file.
        entry: The entry to add or update.
    """
    entries = load_manifest(manifest_path)
    entries[entry.combo_id] = entry
    save_manifest(manifest_path, entries)
