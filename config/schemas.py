"""Configuration dataclasses for type-safe config handling."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation)."""

    use_lora: bool = False
    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    target_modules: Optional[List[str]] = None


@dataclass
class DataConfig:
    """Configuration for data loading."""

    n_members: int = 10000
    n_nonmembers: int = 10000
    n_ft_val: int = 500
    sequence_length: int = 128
    top_k: int = 20
    streaming: bool = False


@dataclass
class FinetuneConfig:
    """Configuration for model fine-tuning."""

    epochs: int = 3
    batch_size: int = 16
    lr: float = 5e-5
    grad_accum_steps: int = 1


@dataclass
class SplitConfig:
    """Configuration for train/val/test splits."""

    val_ratio: float = 0.1
    test_ratio: float = 0.1


@dataclass
class MIAModelConfig:
    """Configuration for MIA classifier architecture."""

    architecture: str = "transformer"
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 2
    dim_ff: int = 256
    dropout: float = 0.1
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])


@dataclass
class TrainingConfig:
    """Configuration for MIA model training."""

    epochs: int = 40
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.01
    lr_scheduler: str = "cosine"
    warmup_epochs: int = 0
    min_lr: float = 1e-6
    grad_clip: Optional[float] = 1.0
    balance_strategy: Optional[str] = None
    label_smooth: float = 0.0
    use_focal: bool = False
    focal_gamma: float = 2.0
    focal_alpha: Optional[float] = 0.25


@dataclass
class ExtractionDefaults:
    """Default values for feature extraction."""

    seed: int = 42
    n_members: int = 10000
    n_nonmembers: int = 10000
    n_ft_val: int = 500
    sequence_length: int = 128
    top_k: int = 20
    ft_epochs: int = 3
    ft_batch_size: int = 16
    ft_lr: float = 5e-5
    grad_accum_steps: int = 1
    inference_batch_size: int = 16
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    streaming: bool = False
    save_token_ids: bool = False


def build_extraction_config(
    combo: Dict[str, Any],
    defaults: Dict[str, Any],
    output_root: Path,
    manifest_path: Path,
) -> Dict[str, Any]:
    """Build a complete extraction config from combination config and defaults.

    Args:
        combo: Combination-specific configuration.
        defaults: Default values to use when not specified in combo.
        output_root: Root directory for output features.
        manifest_path: Path to the manifest file.

    Returns:
        Complete extraction configuration dictionary.
    """
    from ..features.manifest import make_combo_id

    model = combo["model"]
    dataset = combo["dataset"]
    combo_id = make_combo_id(model, dataset)

    combo_data_cfg = combo.get("data", {})
    defaults_data_cfg = defaults.get("data", {})

    return {
        "model": model,
        "dataset": dataset,
        "output_dir": str(output_root / combo_id),
        "manifest_path": str(manifest_path),
        "seed": combo.get("seed", defaults.get("seed", 42)),
        "data": {
            "n_members": combo.get("n_members", defaults.get("n_members", 10000)),
            "n_nonmembers": combo.get("n_nonmembers", defaults.get("n_nonmembers", 10000)),
            "n_ft_val": combo.get("n_ft_val", defaults.get("n_ft_val", 100)),
            "sequence_length": combo.get("sequence_length", defaults.get("sequence_length", 128)),
            "top_k": combo.get("top_k", defaults.get("top_k", 20)),
            "streaming": combo_data_cfg.get("streaming", defaults_data_cfg.get("streaming", False)),
        },
        "finetune": {
            "epochs": combo.get("ft_epochs", defaults.get("ft_epochs", 3)),
            "batch_size": combo.get("ft_batch_size", defaults.get("ft_batch_size", 8)),
            "lr": combo.get("ft_lr", defaults.get("ft_lr", 5e-5)),
            "grad_accum_steps": combo.get("grad_accum_steps", defaults.get("grad_accum_steps", 1)),
        },
        "splits": {
            "val_ratio": combo.get("val_ratio", defaults.get("val_ratio", 0.1)),
            "test_ratio": combo.get("test_ratio", defaults.get("test_ratio", 0.1)),
        },
        "inference_batch_size": combo.get(
            "inference_batch_size", defaults.get("inference_batch_size", 16)
        ),
        "lora": combo.get("lora", defaults.get("lora", {})),
        "save_token_ids": combo.get("save_token_ids", defaults.get("save_token_ids", False)),
    }
