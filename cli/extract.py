"""Feature Extraction CLI.

Extracts per-token features for membership inference attacks by comparing
a fine-tuned target model against an untrained reference model.

Usage:
    ltmia-extract --config configs/extract_training_data.yaml
"""

import argparse
from datetime import datetime
from pathlib import Path

import torch

try:
    import transformers.models.rwkv.modeling_rwkv as rwkv_module
    rwkv_module.rwkv_cuda_kernel = None
    rwkv_module.load_wkv_cuda_kernel = lambda x: None
except ImportError:
    pass

from ..utils import load_config, set_seed, get_device
from ..data import load_data_splits
from ..models import prepare_tokenizer_and_models, apply_lora_if_enabled
from ..training import finetune_model_on_texts
from ..features import (
    extract_per_token_features_both,
    save_features,
    ExtractionMetadata,
    ManifestEntry,
    save_extraction_metadata,
    update_manifest,
    make_combo_id,
)


def run_extraction(cfg: dict) -> str:
    """Run feature extraction for a single model/dataset combination.

    Args:
        cfg: Configuration dictionary containing all extraction parameters.

    Returns:
        The combo_id string for the extracted features.
    """

    def require(d: dict, key: str, ctx: str = "config"):
        if key not in d:
            raise ValueError(f"Missing required key '{key}' in {ctx}")
        return d[key]

    seed = cfg.get("seed", 42)
    set_seed(seed)
    device = get_device()
    print(f"Device: {device}")

    model_name = cfg["model"]
    dataset_name = cfg["dataset"]
    output_dir = Path(cfg["output_dir"])
    manifest_path = Path(cfg.get("manifest_path", output_dir.parent / "manifest.yaml"))

    data_cfg = cfg.get("data")
    if data_cfg is None:
        raise ValueError("Missing required 'data' section in config")
    n_members = int(require(data_cfg, "n_members", "data"))
    n_nonmembers = int(require(data_cfg, "n_nonmembers", "data"))
    n_ft_val = int(require(data_cfg, "n_ft_val", "data"))
    sequence_length = int(require(data_cfg, "sequence_length", "data"))
    top_k = int(require(data_cfg, "top_k", "data"))
    streaming = bool(data_cfg.get("streaming", False))

    ft_cfg = cfg.get("finetune")
    if ft_cfg is None:
        raise ValueError("Missing required 'finetune' section in config")
    ft_epochs = int(require(ft_cfg, "epochs", "finetune"))
    ft_batch_size = int(require(ft_cfg, "batch_size", "finetune"))
    ft_lr = float(require(ft_cfg, "lr", "finetune"))
    grad_accum_steps = int(ft_cfg.get("grad_accum_steps", 1))

    splits_cfg = cfg.get("splits")
    if splits_cfg is None:
        raise ValueError("Missing required 'splits' section in config")
    val_ratio = float(require(splits_cfg, "val_ratio", "splits"))
    test_ratio = float(require(splits_cfg, "test_ratio", "splits"))

    lora_cfg = cfg.get("lora", {})
    inf_batch_size = int(require(cfg, "inference_batch_size", "config"))
    save_token_ids = cfg.get("save_token_ids", False)

    combo_id = make_combo_id(model_name, dataset_name)
    print(f"\n{'='*60}")
    print(f"Extraction: {combo_id}")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    print(f"Loading dataset: {dataset_name} (streaming={streaming})")
    members, nonmembers, ft_val = load_data_splits(
        dataset_name,
        n_members=n_members,
        n_nonmembers=n_nonmembers,
        n_val=n_ft_val,
        seed=seed,
        streaming=streaming,
    )
    print(f"  Members: {len(members)}, Non-members: {len(nonmembers)}, FT-val: {len(ft_val)}")

    print(f"Loading model: {model_name}")
    tokenizer, model_ref, model_tgt = prepare_tokenizer_and_models(model_name, dataset_name=dataset_name)
    model_tgt = apply_lora_if_enabled(model_tgt, lora_cfg)

    print(f"Fine-tuning target model...")
    finetune_model_on_texts(
        model_tgt,
        tokenizer,
        members,
        ft_val,
        device=device,
        epochs=ft_epochs,
        batch_size=ft_batch_size,
        lr=ft_lr,
        sequence_length=sequence_length,
        grad_accum_steps=grad_accum_steps,
    )

    model_ref.to(device)
    model_tgt.to(device)

    print(f"Extracting features for {len(members)} members...")
    mem_feats, mem_masks, mem_token_ids = extract_per_token_features_both(
        model_tgt,
        model_ref,
        tokenizer,
        members,
        device=device,
        batch_size=inf_batch_size,
        sequence_length=sequence_length,
        k=top_k,
    )
    print(f"  Member features shape: {mem_feats.shape}")

    print(f"Extracting features for {len(nonmembers)} non-members...")
    non_feats, non_masks, non_token_ids = extract_per_token_features_both(
        model_tgt,
        model_ref,
        tokenizer,
        nonmembers,
        device=device,
        batch_size=inf_batch_size,
        sequence_length=sequence_length,
        k=top_k,
    )
    print(f"  Non-member features shape: {non_feats.shape}")

    print(f"Saving features to {output_dir}...")
    train_size, val_size, test_size = save_features(
        output_dir,
        mem_feats,
        mem_masks,
        non_feats,
        non_masks,
        members_token_ids=mem_token_ids if save_token_ids else None,
        nonmembers_token_ids=non_token_ids if save_token_ids else None,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    metadata = ExtractionMetadata(
        model_name=model_name,
        dataset_name=dataset_name,
        combo_id=combo_id,
        n_members=len(mem_feats),
        n_nonmembers=len(non_feats),
        sequence_length=sequence_length,
        top_k=top_k,
        feature_dim=mem_feats.shape[-1],
        ft_epochs=ft_epochs,
        ft_batch_size=ft_batch_size,
        ft_lr=ft_lr,
        seed=seed,
        extraction_timestamp=datetime.now().isoformat(),
        lora_config=lora_cfg,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
    )
    save_extraction_metadata(metadata, output_dir)

    manifest_entry = ManifestEntry(
        combo_id=combo_id,
        model_name=model_name,
        dataset_name=dataset_name,
        path=str(output_dir.relative_to(manifest_path.parent)),
        n_members=len(mem_feats),
        n_nonmembers=len(non_feats),
        feature_dim=mem_feats.shape[-1],
        sequence_length=sequence_length,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
    )
    update_manifest(manifest_path, manifest_entry)
    print(f"Updated manifest: {manifest_path}")

    del model_ref, model_tgt, tokenizer
    torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"Extraction complete: {combo_id}")
    print(f"  Features: {output_dir}")
    print(f"  Manifest: {manifest_path}")
    print(f"{'='*60}\n")

    return combo_id


def main():
    """Main entry point for the feature extraction script."""
    parser = argparse.ArgumentParser(
        description="Extract MIA features for a model/dataset combination",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    ltmia-extract --config configs/extract/gpt2_agnews.yaml

    ltmia-extract \\
        --model gpt2 \\
        --dataset ag_news \\
        --output-dir data/features/gpt2_ag_news
        """,
    )

    parser.add_argument("--config", type=str, help="Path to extraction config YAML")
    parser.add_argument("--model", type=str, help="Model name/path (e.g., gpt2, EleutherAI/pythia-1.4b)")
    parser.add_argument("--dataset", type=str, help="Dataset name (e.g., ag_news, wikipedia, cnndm)")
    parser.add_argument("--output-dir", type=str, help="Output directory for features")
    parser.add_argument("--manifest-path", type=str, help="Path to manifest.yaml")
    parser.add_argument("--n-members", type=int, help="Number of member samples")
    parser.add_argument("--n-nonmembers", type=int, help="Number of non-member samples")
    parser.add_argument("--sequence-length", type=int, help="Sequence length")
    parser.add_argument("--top-k", type=int, help="Top-k for feature extraction")
    parser.add_argument("--ft-epochs", type=int, help="Fine-tuning epochs")
    parser.add_argument("--ft-batch-size", type=int, help="Fine-tuning batch size")
    parser.add_argument("--ft-lr", type=float, help="Fine-tuning learning rate")
    parser.add_argument("--seed", type=int, help="Random seed")

    args = parser.parse_args()

    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = {}

    if args.model:
        cfg["model"] = args.model
    if args.dataset:
        cfg["dataset"] = args.dataset
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.manifest_path:
        cfg["manifest_path"] = args.manifest_path
    if args.seed:
        cfg["seed"] = args.seed

    if "data" not in cfg:
        cfg["data"] = {}
    if args.n_members:
        cfg["data"]["n_members"] = args.n_members
    if args.n_nonmembers:
        cfg["data"]["n_nonmembers"] = args.n_nonmembers
    if args.sequence_length:
        cfg["data"]["sequence_length"] = args.sequence_length
    if args.top_k:
        cfg["data"]["top_k"] = args.top_k

    if "finetune" not in cfg:
        cfg["finetune"] = {}
    if args.ft_epochs:
        cfg["finetune"]["epochs"] = args.ft_epochs
    if args.ft_batch_size:
        cfg["finetune"]["batch_size"] = args.ft_batch_size
    if args.ft_lr:
        cfg["finetune"]["lr"] = args.ft_lr

    required = ["model", "dataset", "output_dir"]
    missing = [f for f in required if f not in cfg]
    if missing:
        parser.error(f"Missing required fields: {missing}. Provide via config or command-line.")

    run_extraction(cfg)


if __name__ == "__main__":
    main()
