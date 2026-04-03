"""
LT-MIA: Large-scale Transferable Membership Inference Attack

A framework for membership inference attacks on language models using
per-token feature extraction and transformer-based classification.
"""

__version__ = "0.1.0"

from .utils import set_seed, get_device, load_config
from .data import (
    load_data_splits,
    CombinedMIADataset,
    CombinedMIADatasetSimple,
)
from .models import (
    prepare_tokenizer_and_models,
    apply_lora_if_enabled,
    create_mia_model,
)
from .training import finetune_model_on_texts, train_mia_model
from .evaluation import MIAEvaluator, EvalResults, compute_curve_metrics
from .features import (
    extract_per_token_features_both,
    save_features,
    load_features_mmap,
    ExtractionMetadata,
    ManifestEntry,
    load_manifest,
    save_manifest,
    update_manifest,
    make_combo_id,
)
from .config import build_extraction_config

__all__ = [
    "set_seed",
    "get_device",
    "load_config",
    "load_data_splits",
    "CombinedMIADataset",
    "CombinedMIADatasetSimple",
    "prepare_tokenizer_and_models",
    "apply_lora_if_enabled",
    "create_mia_model",
    "finetune_model_on_texts",
    "train_mia_model",
    "MIAEvaluator",
    "EvalResults",
    "compute_curve_metrics",
    "extract_per_token_features_both",
    "save_features",
    "load_features_mmap",
    "ExtractionMetadata",
    "ManifestEntry",
    "load_manifest",
    "save_manifest",
    "update_manifest",
    "make_combo_id",
    "build_extraction_config",
]
