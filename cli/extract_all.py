"""Batch Feature Extraction CLI.

Runs feature extraction for multiple model/dataset combinations defined in a config file.
Supports resumption via --skip-existing and selective runs via --only.

Usage:
    ltmia-extract-all --config configs/extract_training_data.yaml
"""

import argparse
import gc
import shutil
import traceback
from pathlib import Path

import torch

from ..utils import load_config
from ..config import build_extraction_config
from ..features import make_combo_id, load_manifest
from .extract import run_extraction


def clear_hf_cache() -> None:
    """Clear HuggingFace cache to free disk space.

    Clears CUDA cache, runs garbage collection, and removes the HuggingFace
    hub cache directory. This is useful when running many model extractions
    sequentially to prevent disk space exhaustion.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gc.collect()

    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    if hf_cache.exists():
        try:
            size_before = sum(f.stat().st_size for f in hf_cache.rglob("*") if f.is_file())
            size_gb = size_before / (1024**3)

            shutil.rmtree(hf_cache)
            hf_cache.mkdir(parents=True, exist_ok=True)
            print(f"  [Cache cleared: {size_gb:.2f} GB freed]")
        except Exception as e:
            print(f"  [Warning: Could not clear HF cache: {e}]")


def main():
    """Main entry point for batch feature extraction."""
    parser = argparse.ArgumentParser(
        description="Run feature extraction for all model/dataset combinations"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to batch extraction config"
    )
    parser.add_argument(
        "--only",
        type=str,
        nargs="+",
        help="Only run these combo_ids (e.g., gpt2_ag_news)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip combinations that already exist in manifest",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without actually running",
    )
    parser.add_argument(
        "--no-clear-cache",
        action="store_true",
        help="Disable automatic HuggingFace cache clearing between models",
    )

    args = parser.parse_args()
    cfg = load_config(args.config)

    output_root = Path(cfg.get("output_root", "data/features"))
    manifest_path = Path(cfg.get("manifest_path", output_root / "manifest.yaml"))

    defaults = cfg.get("defaults", {})

    combinations = cfg.get("combinations", [])

    if not combinations:
        print("No combinations defined in config!")
        return

    existing_combos = set()
    if args.skip_existing and manifest_path.exists():
        existing_combos = set(load_manifest(manifest_path).keys())

    if args.only:
        requested = set(args.only)
        combinations = [
            c for c in combinations
            if make_combo_id(c["model"], c["dataset"]) in requested
        ]

    print(f"\n{'='*60}")
    print("Batch Feature Extraction")
    print(f"{'='*60}")
    print(f"Output root: {output_root}")
    print(f"Manifest: {manifest_path}")
    print(f"Total combinations: {len(combinations)}")
    if args.skip_existing:
        print(f"Existing (will skip): {len(existing_combos)}")
    if args.no_clear_cache:
        print("Cache clearing: disabled")
    print(f"{'='*60}\n")

    results = {"success": [], "skipped": [], "failed": []}

    for i, combo_cfg in enumerate(combinations, 1):
        model = combo_cfg["model"]
        dataset = combo_cfg["dataset"]
        combo_id = make_combo_id(model, dataset)

        print(f"\n[{i}/{len(combinations)}] {combo_id}")
        print("-" * 40)

        if combo_id in existing_combos:
            print(f"  Skipping (already exists)")
            results["skipped"].append(combo_id)
            continue

        if args.dry_run:
            print(f"  Would extract: model={model}, dataset={dataset}")
            continue

        extraction_cfg = build_extraction_config(
            combo_cfg, defaults, output_root, manifest_path
        )

        try:
            run_extraction(extraction_cfg)
            results["success"].append(combo_id)
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            results["failed"].append((combo_id, str(e)))
        finally:
            if not args.no_clear_cache:
                clear_hf_cache()

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Successful: {len(results['success'])}")
    for c in results["success"]:
        print(f"  ✓ {c}")

    if results["skipped"]:
        print(f"\nSkipped: {len(results['skipped'])}")
        for c in results["skipped"]:
            print(f"  - {c}")

    if results["failed"]:
        print(f"\nFailed: {len(results['failed'])}")
        for c, err in results["failed"]:
            print(f"  ✗ {c}: {err}")


if __name__ == "__main__":
    main()
