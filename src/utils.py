"""
Core utilities for LT-MIA (Large-scale Transferable Membership Inference Attack).

This module provides:
- Data loading and preprocessing
- Model loading and LoRA configuration
- Per-token feature extraction
- MIA classifier architectures
- Training and evaluation utilities
- Manifest and metadata management
"""

import math
import random
import time
import copy
import os
import itertools
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import yaml
from tqdm.auto import tqdm


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility across all random number generators.

    Args:
        seed: The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get the best available compute device.

    Returns:
        torch.device for CUDA, MPS (Apple Silicon), or CPU in order of preference.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalize_dataset_name(dataset_name: str) -> str:
    """Normalize dataset name to a standard format.

    Args:
        dataset_name: Raw dataset name.

    Returns:
        Lowercased name with dashes and spaces replaced by underscores.
    """
    return dataset_name.lower().strip().replace("-", "_").replace(" ", "_")


def is_code_dataset(dataset_name: str) -> bool:
    """Check if a dataset contains code (for tokenizer configuration).

    Args:
        dataset_name: Name of the dataset.

    Returns:
        True if the dataset contains code samples.
    """
    dn = normalize_dataset_name(dataset_name)
    return dn in {"swallow_code", "tokyotech_swallow_code"}


def configure_tokenizer_for_dataset(tokenizer, dataset_name: str | None) -> None:
    """Configure tokenizer truncation behavior based on dataset type.

    For code datasets, keeps the last tokens (drops the beginning) to avoid
    inputs that are mostly import statements.

    Args:
        tokenizer: The tokenizer to configure.
        dataset_name: Name of the dataset being processed.
    """
    if dataset_name and is_code_dataset(dataset_name):
        tokenizer.truncation_side = "left"
    else:
        tokenizer.truncation_side = "right"


def load_data_splits(
    dataset_name: str,
    n_members: int = 15000,
    n_nonmembers: int = 15000,
    n_val: int = 100,
    seed: int = 42,
    streaming: bool = False,
    streaming_datasets: Optional[List[str]] = None,
    stream_max_samples: int = 200_000
) -> Tuple[List[str], List[str], List[str]]:
    """Load and split a dataset into members, non-members, and validation sets.

    Args:
        dataset_name: Name of the dataset to load.
        n_members: Number of member samples (used for fine-tuning).
        n_nonmembers: Number of non-member samples.
        n_val: Number of validation samples for fine-tuning.
        seed: Random seed for reproducible splits.
        streaming: Whether to use streaming mode for large datasets.
        streaming_datasets: List of dataset names that should always stream.
        stream_max_samples: Maximum samples to load when streaming.

    Returns:
        Tuple of (members, nonmembers, validation) text lists.

    Raises:
        ValueError: If the dataset name is unknown.
        RuntimeError: If there are insufficient samples.
    """
    from datasets import load_dataset

    dataset_name_lower = normalize_dataset_name(dataset_name)
    streaming_datasets = set(
        (streaming_datasets or [])
        + ["wikipedia", "wiki", "swallow_code", "swallowcode", "tokyotech_swallow_code", "arxiv", "arxiv_summarization"]
    )
    use_stream = streaming or dataset_name_lower in streaming_datasets

    def _example_to_text(ex: Dict[str, Any]) -> str:
        for k in ("text", "content", "code", "body", "document"):
            v = ex.get(k)
            if isinstance(v, str) and v.strip():
                return v
        strings = [v for v in ex.values() if isinstance(v, str) and v.strip()]
        return "\n".join(strings) if strings else ""

    if dataset_name_lower == "ag_news":
        ds = load_dataset("ag_news", streaming=use_stream)
        train_split = ds["train"]
        texts_iter = (x["text"] for x in train_split)
    elif dataset_name_lower == "xsum":
        ds = load_dataset("xsum", streaming=use_stream)
        train_split = ds["train"]
        texts_iter = (x["document"] for x in train_split)
    elif dataset_name_lower in {"wikitext", "wikitext103", "wikitext_103"}:
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", streaming=use_stream)
        train_split = ds["train"]
        texts_iter = (x["text"] for x in train_split)
    elif dataset_name_lower in {"wikipedia", "wiki"}:
        ds = load_dataset("wikimedia/wikipedia", "20231101.en", streaming=use_stream)
        train_split = ds["train"]
        texts_iter = (x["text"] for x in train_split)
    elif dataset_name_lower in {"news_category", "news_category_dataset"}:
        ds = load_dataset("heegyu/news-category-dataset", streaming=use_stream)
        train_split = ds["train"]
        texts_iter = (f"{x['headline']} {x['short_description']}" for x in train_split)
    elif dataset_name_lower in {"cnndm", "cnn_dailymail", "cnn_dm"}:
        ds = load_dataset("cnn_dailymail", "3.0.0", streaming=use_stream)
        train_split = ds["train"]
        texts_iter = (x["article"] for x in train_split)
    elif dataset_name_lower in {"swallow_code", "swallowcode", "tokyotech_swallow_code"}:
        train_split = load_dataset(
            "tokyotech-llm/swallow-code",
            "swallow-code",
            split="train",
            streaming=use_stream,
        )
        texts_iter = (_example_to_text(x) for x in train_split)
    elif dataset_name_lower in {"arxiv", "arxiv_summarization"}:
        ds = load_dataset("ccdv/arxiv-summarization", streaming=use_stream, trust_remote_code=True)
        train_split = ds["train"]
        texts_iter = (x["abstract"] for x in train_split)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if use_stream:
        texts = [t for t in itertools.islice(texts_iter, stream_max_samples) if t]
    else:
        texts = list(texts_iter)

    texts = [t for t in texts if t and len(t.strip()) > 50]

    total_needed = n_members + n_nonmembers + n_val
    if len(texts) < total_needed:
        raise RuntimeError(
            f"Insufficient samples in {dataset_name} for requested sizes "
            f"(needed {total_needed}, got {len(texts)})."
        )

    rng = random.Random(seed)
    indices = list(range(len(texts)))
    rng.shuffle(indices)
    texts_shuf = [texts[i] for i in indices]

    members = texts_shuf[:n_members]
    val = texts_shuf[n_members:n_members + n_val]
    nonmembers = texts_shuf[n_members + n_val:n_members + n_val + n_nonmembers]

    return members, nonmembers, val


def prepare_tokenizer_and_models(model_name: str = "gpt2", dataset_name: str | None = None):
    """Load tokenizer and paired reference/target models.

    Args:
        model_name: HuggingFace model name or path.
        dataset_name: Dataset name for tokenizer configuration.

    Returns:
        Tuple of (tokenizer, reference_model, target_model).
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

    is_mpt = "mpt" in model_name.lower()
    trust_remote = not is_mpt

    is_openelm = "openelm" in model_name.lower()
    is_stablelm = "stablelm" in model_name.lower()

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote)
    except ValueError as e:
        if "sentencepiece" in str(e) or "tiktoken" in str(e):
            if is_openelm:
                print(f"[Tokenizer] Using meta-llama/Llama-2-7b-hf tokenizer for {model_name}")
                tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
            else:
                raise
        else:
            raise

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    configure_tokenizer_for_dataset(tokenizer, dataset_name)

    model_kwargs = {"trust_remote_code": trust_remote}

    if is_stablelm:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote)
        if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
            config.pad_token_id = tokenizer.pad_token_id
        model_kwargs["config"] = config

    if is_openelm:
        model_kwargs["low_cpu_mem_usage"] = False
        model_kwargs["device_map"] = None
        model_kwargs["torch_dtype"] = None

    model_ref = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model_tgt = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    return tokenizer, model_ref, model_tgt


def apply_lora_if_enabled(model: nn.Module, lora_cfg: Dict[str, Any] | None):
    """Apply LoRA adapters to a model when configured.

    Args:
        model: The model to apply LoRA to.
        lora_cfg: LoRA configuration dict with keys: use_lora, r, alpha, dropout, target_modules.

    Returns:
        The model with LoRA adapters applied (or unchanged if not configured).

    Raises:
        RuntimeError: If LoRA is requested but peft is not installed.
    """
    if not lora_cfg or not lora_cfg.get("use_lora", False):
        return model
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except Exception as exc:
        raise RuntimeError(f"LoRA requested but peft is unavailable: {exc}")

    r = int(lora_cfg.get("r", 8))
    alpha = float(lora_cfg.get("alpha", 16))
    dropout = float(lora_cfg.get("dropout", 0.05))
    target_modules = lora_cfg.get("target_modules")

    if target_modules is None:
        model_type = getattr(getattr(model, "config", None), "model_type", "") or ""
        model_name = getattr(getattr(model, "config", None), "_name_or_path", "") or ""
        if "llama" in model_type:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        elif "gptj" in model_type or "gpt-j" in model_type:
            target_modules = [
                "attn.q_proj",
                "attn.k_proj",
                "attn.v_proj",
                "attn.out_proj",
                "mlp.fc_in",
                "mlp.fc_out",
            ]
        elif "gpt_neox" in model_type or "pythia" in model_type:
            target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
        elif "mamba" in model_type:
            target_modules = ["in_proj", "x_proj", "dt_proj"]
        elif "rwkv" in model_type:
            target_modules = ["attention.key", "attention.value", "attention.receptance", "attention.output"]
        elif "recurrentgemma" in model_type.lower() or "recurrentgemma" in model_name.lower():
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        elif "xlstm" in model_type.lower() or "xlstm" in model_name.lower():
            target_modules = ["up_proj", "down_proj"]
        elif "stripedhyena" in model_type.lower() or "stripedhyena" in model_name.lower() or "hyena" in model_type.lower():
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        elif "gemma" in model_type:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        else:
            target_modules = ["c_attn", "c_proj", "c_fc"]

    lcfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    return model


def batch_tokenize(tokenizer, texts: List[str], sequence_length: int = 128) -> Dict[str, torch.Tensor]:
    """Tokenize a batch of texts with padding and truncation.

    Args:
        tokenizer: The tokenizer to use.
        texts: List of text strings to tokenize.
        sequence_length: Target sequence length.

    Returns:
        Dict with 'input_ids' and 'attention_mask' tensors.
    """
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
    sequences: List[List[int]] = []
    masks: List[List[int]] = []

    for text in texts:
        max_len = sequence_length - 1 if eos_id is not None else sequence_length
        max_len = max(1, max_len)
        ids = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=max_len)
        if eos_id is not None and len(ids) < sequence_length:
            ids.append(eos_id)

        if not ids:
            continue

        ids = ids[:sequence_length]
        mask = [1] * len(ids)
        if len(ids) < sequence_length:
            pad_len = sequence_length - len(ids)
            ids = ids + [pad_id] * pad_len
            mask = mask + [0] * pad_len

        sequences.append(ids)
        masks.append(mask)

    if not sequences:
        return {
            "input_ids": torch.empty((0, sequence_length), dtype=torch.long),
            "attention_mask": torch.empty((0, sequence_length), dtype=torch.long),
        }

    return {
        "input_ids": torch.tensor(sequences, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
    }


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


@torch.no_grad()
def extract_per_token_features_both(
    model_tgt: nn.Module,
    model_ref: nn.Module,
    tokenizer,
    texts: List[str],
    device: torch.device,
    batch_size: int = 32,
    sequence_length: int = 128,
    k: int = 20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract per-token features comparing target (fine-tuned) and reference model outputs.

    Args:
        model_tgt: Fine-tuned target model.
        model_ref: Untrained reference model.
        tokenizer: Tokenizer for both models.
        texts: List of text samples to extract features from.
        device: Device to run inference on.
        batch_size: Batch size for inference.
        sequence_length: Maximum sequence length.
        k: Number of top/bottom logits to include in features.

    Returns:
        Tuple of (features, masks, token_ids) as numpy arrays.
        Features shape: (N, sequence_length, feature_dim) where feature_dim = 7*k + 14.
    """
    model_tgt.eval()
    model_ref.eval()
    feats_all: List[np.ndarray] = []
    masks_all: List[np.ndarray] = []
    token_ids_all: List[np.ndarray] = []

    enc_full = batch_tokenize(tokenizer, texts, sequence_length=sequence_length)
    input_ids_full = enc_full["input_ids"]
    attn_full = enc_full["attention_mask"]
    total = input_ids_full.size(0)

    if total == 0:
        feat_dim = 7 * k + 14
        return (
            np.empty((0, sequence_length, feat_dim), dtype=np.float32),
            np.empty((0, sequence_length), dtype=np.uint8),
            np.empty((0, sequence_length), dtype=np.int64),
        )

    for start in range(0, total, batch_size):
        end = start + batch_size
        enc = {
            "input_ids": input_ids_full[start:end].to(device),
            "attention_mask": attn_full[start:end].to(device),
        }

        input_ids = enc["input_ids"]
        attn = enc["attention_mask"]
        B, T = input_ids.shape
        attn_next = attn[:, 1:]

        logits_tgt = model_tgt(**enc).logits
        logits_tgt_shift = logits_tgt[:, :-1, :]
        next_ids = input_ids[:, 1:]
        gt_tgt = torch.gather(logits_tgt_shift, -1, next_ids.unsqueeze(-1)).squeeze(-1)
        top_tgt_val, top_tgt_idx = torch.topk(logits_tgt_shift, k=k, dim=-1, largest=True, sorted=True)
        bot_tgt_val, bot_tgt_idx = torch.topk(logits_tgt_shift, k=k, dim=-1, largest=False, sorted=True)

        logits_ref = model_ref(**enc).logits
        logits_ref_shift = logits_ref[:, :-1, :]
        gt_ref = torch.gather(logits_ref_shift, -1, next_ids.unsqueeze(-1)).squeeze(-1)
        top_ref_val, top_ref_idx = torch.topk(logits_ref_shift, k=k, dim=-1, largest=True, sorted=True)
        bot_ref_val, bot_ref_idx = torch.topk(logits_ref_shift, k=k, dim=-1, largest=False, sorted=True)

        gt_tgt_rank = (logits_tgt_shift > gt_tgt.unsqueeze(-1)).sum(dim=-1) + 1
        gt_ref_rank = (logits_ref_shift > gt_ref.unsqueeze(-1)).sum(dim=-1) + 1

        V = logits_ref_shift.size(-1)
        combined_idx = torch.cat([top_tgt_idx, bot_tgt_idx], dim=-1)
        base_combined_logits = torch.gather(logits_ref_shift, -1, combined_idx)
        cross_counts = torch.zeros_like(base_combined_logits, dtype=torch.int32)
        CHUNK = 2048
        for chunk_start in range(0, V, CHUNK):
            chunk_end = min(chunk_start + CHUNK, V)
            chunk = logits_ref_shift[:, :, chunk_start:chunk_end]
            cmp = chunk.unsqueeze(-1) > base_combined_logits.unsqueeze(-2)
            cross_counts += cmp.sum(dim=2, dtype=torch.int32)
            del chunk, cmp
        cross_rank_combined = cross_counts + 1
        base_top_logits = base_combined_logits[:, :, :k]
        base_bot_logits = base_combined_logits[:, :, k:]
        cross_rank_top = cross_rank_combined[:, :, :k]
        cross_rank_bot = cross_rank_combined[:, :, k:]

        V_tgt = logits_tgt_shift.size(-1)
        target_for_ref_top = torch.gather(logits_tgt_shift, -1, top_ref_idx)
        reverse_counts_top = torch.zeros_like(target_for_ref_top, dtype=torch.int32)
        for chunk_start in range(0, V_tgt, CHUNK):
            chunk_end = min(chunk_start + CHUNK, V_tgt)
            chunk_tgt = logits_tgt_shift[:, :, chunk_start:chunk_end]
            cmp_rev = chunk_tgt.unsqueeze(-1) > target_for_ref_top.unsqueeze(-2)
            reverse_counts_top += cmp_rev.sum(dim=2, dtype=torch.int32)
            del chunk_tgt, cmp_rev
        reverse_cross_rank_top = reverse_counts_top + 1

        log_V = math.log(V + 1)
        log_gt_tgt_rank = torch.log1p(gt_tgt_rank.float()) / log_V
        log_gt_ref_rank = torch.log1p(gt_ref_rank.float()) / log_V
        log_cross_rank_top = torch.log1p(cross_rank_top.float()) / log_V
        log_cross_rank_bot = torch.log1p(cross_rank_bot.float()) / log_V
        log_reverse_cross_rank_top = torch.log1p(reverse_cross_rank_top.float()) / log_V

        truth_pair = torch.stack([gt_tgt, gt_ref], dim=-1)
        truth_max = truth_pair.max(dim=-1, keepdim=True).values
        truth_pair = truth_pair - truth_max
        truth_tgt_logit = truth_pair[:, :, 0]
        truth_base_logit = truth_pair[:, :, 1]

        head_max = torch.maximum(top_tgt_val, base_top_logits)
        head_tgt_logits = top_tgt_val - head_max
        head_base_logits = base_top_logits - head_max

        tail_max = torch.maximum(bot_tgt_val, base_bot_logits)
        tail_tgt_logits = bot_tgt_val - tail_max
        tail_base_logits = base_bot_logits - tail_max

        truth_block = torch.stack([
            truth_tgt_logit,
            truth_base_logit,
            log_gt_tgt_rank,
            log_gt_ref_rank,
        ], dim=-1)

        head_block = torch.stack([
            head_tgt_logits,
            head_base_logits,
            log_cross_rank_top,
            log_reverse_cross_rank_top,
        ], dim=-1).reshape(B, -1, 4 * k)

        tail_block = torch.stack([
            tail_tgt_logits,
            tail_base_logits,
            log_cross_rank_bot,
        ], dim=-1).reshape(B, -1, 3 * k)

        blocks = [truth_block, head_block, tail_block]

        loss_tgt = F.cross_entropy(logits_tgt_shift.transpose(1, 2), next_ids, reduction='none')
        loss_ref = F.cross_entropy(logits_ref_shift.transpose(1, 2), next_ids, reduction='none')
        loss_diff = loss_tgt - loss_ref
        loss_block = torch.stack([loss_tgt, loss_ref, loss_diff], dim=-1)
        blocks.append(loss_block)

        valid = attn_next
        valid_sum = valid.sum(dim=1, keepdim=True).clamp_min(1)

        def masked_stats(x: torch.Tensor):
            mean = (x * valid).sum(dim=1, keepdim=True) / valid_sum
            var = ((x - mean) ** 2 * valid).sum(dim=1, keepdim=True) / valid_sum
            std = torch.sqrt(var + 1e-6)
            return mean.squeeze(1), std.squeeze(1)

        mean_tgt, std_tgt = masked_stats(loss_tgt)
        mean_ref, std_ref = masked_stats(loss_ref)
        mean_diff, std_diff = masked_stats(loss_diff)
        sum_llr = (loss_diff * valid).sum(dim=1)

        global_feats = torch.stack([
            mean_tgt, std_tgt,
            mean_ref, std_ref,
            mean_diff, std_diff,
            sum_llr,
        ], dim=-1)
        global_block = global_feats.unsqueeze(1).expand(-1, loss_tgt.size(1), -1)
        blocks.append(global_block)

        feat_steps = torch.cat(blocks, dim=-1)
        feat_steps = feat_steps * attn_next.unsqueeze(-1)
        zero_step = torch.zeros(B, 1, feat_steps.size(-1), device=feat_steps.device, dtype=feat_steps.dtype)
        feats_full = torch.cat([feat_steps, zero_step], dim=1)
        mask_full = torch.cat([
            attn_next,
            torch.zeros(B, 1, device=attn_next.device, dtype=attn_next.dtype)
        ], dim=1)

        feats_all.append(feats_full.detach().cpu().numpy())
        masks_all.append(mask_full.detach().cpu().to(torch.uint8).numpy())
        token_ids_all.append(input_ids_full[start:end].numpy())

    feats = np.concatenate(feats_all, axis=0)
    masks = np.concatenate(masks_all, axis=0)
    token_ids = np.concatenate(token_ids_all, axis=0)
    return feats, masks, token_ids


def finetune_model_on_texts(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    val_texts: List[str] | None,
    device: torch.device,
    epochs: int = 1,
    batch_size: int = 8,
    lr: float = 5e-5,
    sequence_length: int = 128,
    grad_accum_steps: int = 1
):
    """Fine-tune a language model on the given texts.

    Args:
        model: The model to fine-tune.
        tokenizer: Tokenizer for the model.
        texts: Training texts.
        val_texts: Validation texts (optional).
        device: Device to train on.
        epochs: Number of training epochs.
        batch_size: Effective batch size.
        lr: Learning rate.
        sequence_length: Maximum sequence length.
        grad_accum_steps: Gradient accumulation steps.
    """
    model.to(device)
    model.train()
    enc = batch_tokenize(tokenizer, texts, sequence_length=sequence_length)
    ds = LMDataset(enc)

    micro_batch = max(1, batch_size // grad_accum_steps)
    dl = DataLoader(ds, batch_size=micro_batch, shuffle=True, drop_last=False)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    total_steps = epochs * len(dl)
    print(f"[FT] Batches per epoch: {len(dl)}; micro_batch={micro_batch}, grad_accum={grad_accum_steps}; total steps: {total_steps}")

    best_val_loss = float('inf')
    best_state = None

    for ep in range(epochs):
        ep_loss = 0.0
        t0 = time.time()
        for step, batch in enumerate(dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / grad_accum_steps
            loss.backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dl):
                optim.step()
                optim.zero_grad(set_to_none=True)

            ep_loss += out.loss.item()
            if (step + 1) % 100 == 0:
                print(f"[FT] Epoch {ep+1} Step {step+1}/{len(dl)} Loss {ep_loss/(step+1):.4f}")
        dt = time.time() - t0

        val_loss = None
        if val_texts is not None and len(val_texts) > 0:
            with torch.no_grad():
                model.eval()
                val_enc = batch_tokenize(tokenizer, val_texts, sequence_length=sequence_length)
                val_ds = LMDataset(val_enc)
                val_dl = DataLoader(val_ds, batch_size=micro_batch, shuffle=False, drop_last=False)
                vloss_sum = 0.0
                vcount = 0
                for vb in val_dl:
                    vb = {k: v.to(device) for k, v in vb.items()}
                    vout = model(**vb)
                    vloss_sum += vout.loss.item()
                    vcount += 1
                val_loss = vloss_sum / max(1, vcount)
            model.train()

        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"[FT] Epoch {ep+1} done. train_loss={ep_loss/len(dl):.4f} "
            f"val_loss={(f'{val_loss:.4f}' if val_loss is not None else 'NA')} time={dt:.1f}s"
        )

    if best_state is not None:
        print(f"[FT] Restoring best model with val_loss: {best_val_loss:.4f}")
        model.load_state_dict(best_state)


class MaskedMeanPooling(nn.Module):
    """Mean pooling over valid (non-masked) tokens."""

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Pool features by taking mean over valid tokens.

        Args:
            x: Features of shape (B, T, D).
            mask: Attention mask of shape (B, T), 1 for valid tokens, 0 for padding.

        Returns:
            Pooled features of shape (B, D).
        """
        mask_expanded = mask.unsqueeze(-1).float()
        sum_x = (x * mask_expanded).sum(dim=1)
        count = mask_expanded.sum(dim=1).clamp(min=1)
        return sum_x / count


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer models."""

    def __init__(self, d_model: int, max_len: int = 512):
        """Initialize positional encoding.

        Args:
            d_model: Model dimension.
            max_len: Maximum sequence length.
        """
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        """Add positional encoding to input.

        Args:
            x: Input tensor of shape (B, T, D).

        Returns:
            Tensor with positional encoding added.
        """
        L = x.size(1)
        return x + self.pe[:, :L, :]


class TinyMIASequence(nn.Module):
    """Transformer-based MIA classifier operating on per-token features."""

    def __init__(
        self,
        d_in: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_ff: int = 256,
        dropout: float = 0.1
    ):
        """Initialize the transformer classifier.

        Args:
            d_in: Input feature dimension.
            d_model: Transformer model dimension.
            nhead: Number of attention heads.
            num_layers: Number of transformer layers.
            dim_ff: Feedforward dimension.
            dropout: Dropout rate.
        """
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            batch_first=True,
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos = SinusoidalPositionalEncoding(d_model=d_model, max_len=512)

        self.attn_pool = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )

        self.cls = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, feats: torch.Tensor, mask: torch.Tensor | None = None):
        """Forward pass.

        Args:
            feats: Input features of shape (B, T, D).
            mask: Optional attention mask of shape (B, T).

        Returns:
            Logits of shape (B,).
        """
        x = self.proj(feats.float())
        x = self.norm(x)
        x = self.dropout(x)
        x = self.pos(x)

        pad_mask = None
        if mask is not None:
            pad_mask = ~mask.bool()
        x = self.encoder(x, src_key_padding_mask=pad_mask)

        attn_weights = self.attn_pool(x).squeeze(-1)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(~mask.bool(), float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1).unsqueeze(-1)
        x = (x * attn_weights).sum(dim=1)

        return self.cls(x).squeeze(-1)


class LogisticRegressionMIA(nn.Module):
    """Simple logistic regression baseline that flattens the sequence."""

    def __init__(self, d_in: int, seq_len: int = 128, **kwargs):
        """Initialize logistic regression classifier.

        Args:
            d_in: Input feature dimension per token.
            seq_len: Sequence length.
        """
        super().__init__()
        self.seq_len = seq_len
        self.linear = nn.Linear(d_in * seq_len, 1)

    def forward(self, feats: torch.Tensor, mask: torch.Tensor | None = None):
        """Forward pass.

        Args:
            feats: Input features of shape (B, T, D).
            mask: Unused, kept for API compatibility.

        Returns:
            Logits of shape (B,).
        """
        B = feats.size(0)
        x = feats.float().view(B, -1)
        return self.linear(x).squeeze(-1)


class MLPMIA(nn.Module):
    """MLP classifier that flattens the sequence."""

    def __init__(
        self,
        d_in: int,
        seq_len: int = 128,
        hidden_dims: List[int] = None,
        dropout: float = 0.1,
        **kwargs
    ):
        """Initialize MLP classifier.

        Args:
            d_in: Input feature dimension per token.
            seq_len: Sequence length.
            hidden_dims: List of hidden layer dimensions.
            dropout: Dropout rate.
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]
        self.seq_len = seq_len
        input_dim = d_in * seq_len
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, feats: torch.Tensor, mask: torch.Tensor | None = None):
        """Forward pass.

        Args:
            feats: Input features of shape (B, T, D).
            mask: Unused, kept for API compatibility.

        Returns:
            Logits of shape (B,).
        """
        B = feats.size(0)
        x = feats.float().view(B, -1)
        return self.mlp(x).squeeze(-1)


class MeanMLPMIA(nn.Module):
    """MLP classifier that takes sequence mean as input.

    Unlike the regular MLP which flattens (B, T, D) -> (B, T*D),
    this pools to mean first: (B, T, D) -> mean -> (B, D) -> MLP.
    This tests whether per-token information matters.
    """

    def __init__(
        self,
        d_in: int,
        hidden_dims: List[int] = None,
        dropout: float = 0.1,
        **kwargs
    ):
        """Initialize mean-pooled MLP classifier.

        Args:
            d_in: Input feature dimension per token.
            hidden_dims: List of hidden layer dimensions.
            dropout: Dropout rate.
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.pool = MaskedMeanPooling()

        layers = []
        prev_dim = d_in
        for h in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, feats: torch.Tensor, mask: torch.Tensor | None = None):
        """Forward pass.

        Args:
            feats: Input features of shape (B, T, D).
            mask: Optional attention mask of shape (B, T).

        Returns:
            Logits of shape (B,).
        """
        if mask is None:
            mask = torch.ones(feats.shape[:2], device=feats.device)
        x = self.pool(feats.float(), mask)
        return self.mlp(x).squeeze(-1)


class PooledTransformerMIA(nn.Module):
    """Transformer that pools input features first (averages across sequence at input time).

    This ablates whether sequence-level processing adds value by averaging
    per-token features before any transformer processing.
    """

    def __init__(
        self,
        d_in: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_ff: int = 256,
        dropout: float = 0.1
    ):
        """Initialize pooled transformer classifier.

        Args:
            d_in: Input feature dimension.
            d_model: Model dimension.
            nhead: Number of attention heads (unused, kept for API compatibility).
            num_layers: Number of feedforward layers.
            dim_ff: Feedforward dimension.
            dropout: Dropout rate.
        """
        super().__init__()
        self.pool = MaskedMeanPooling()

        self.proj = nn.Linear(d_in, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout_layer = nn.Dropout(dropout)

        layers = []
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(d_model, dim_ff),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_ff, d_model),
                nn.LayerNorm(d_model),
            ])
        self.layers = nn.Sequential(*layers)

        self.cls = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, feats: torch.Tensor, mask: torch.Tensor | None = None):
        """Forward pass.

        Args:
            feats: Input features of shape (B, T, D).
            mask: Optional attention mask of shape (B, T).

        Returns:
            Logits of shape (B,).
        """
        if mask is None:
            mask = torch.ones(feats.shape[:2], device=feats.device)
        x = self.pool(feats.float(), mask)

        x = self.proj(x)
        x = self.norm(x)
        x = self.dropout_layer(x)
        x = self.layers(x)

        return self.cls(x).squeeze(-1)


def create_mia_model(architecture: str, d_in: int, seq_len: int = 128, **kwargs) -> nn.Module:
    """Factory function to create MIA model by architecture name.

    Args:
        architecture: One of 'transformer', 'pooled_transformer', 'mlp', 'mean_mlp', 'lr'.
        d_in: Input feature dimension.
        seq_len: Sequence length (needed for lr and mlp which flatten input).
        **kwargs: Additional arguments passed to model constructor.

    Returns:
        Instantiated MIA model.

    Raises:
        ValueError: If architecture is unknown.
    """
    transformer_keys = {"d_model", "nhead", "num_layers", "dim_ff", "dropout"}
    mlp_keys = {"hidden_dims", "dropout"}

    if architecture == "transformer":
        filtered = {k: v for k, v in kwargs.items() if k in transformer_keys}
        return TinyMIASequence(d_in=d_in, **filtered)
    elif architecture == "pooled_transformer":
        filtered = {k: v for k, v in kwargs.items() if k in transformer_keys}
        return PooledTransformerMIA(d_in=d_in, **filtered)
    elif architecture == "mlp":
        filtered = {k: v for k, v in kwargs.items() if k in mlp_keys}
        return MLPMIA(d_in=d_in, seq_len=seq_len, **filtered)
    elif architecture == "mean_mlp":
        filtered = {k: v for k, v in kwargs.items() if k in mlp_keys}
        return MeanMLPMIA(d_in=d_in, **filtered)
    elif architecture == "lr":
        return LogisticRegressionMIA(d_in=d_in, seq_len=seq_len)
    else:
        raise ValueError(f"Unknown architecture: {architecture}. Choose from: transformer, pooled_transformer, mlp, mean_mlp, lr")


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


def compute_curve_metrics(labels_np: np.ndarray, probs_np: np.ndarray):
    """Compute AUC, TPR@1%FPR, and TPR@0.1%FPR.

    Args:
        labels_np: Ground truth labels.
        probs_np: Predicted probabilities.

    Returns:
        Tuple of (auc, tpr_at_1pct_fpr, tpr_at_01pct_fpr) or (None, None, None) on error.
    """
    try:
        probs_np = np.nan_to_num(probs_np, nan=0.0, posinf=1.0, neginf=0.0)
        from sklearn.metrics import roc_auc_score, roc_curve

        auc = roc_auc_score(labels_np, probs_np)
        fpr, tpr, _ = roc_curve(labels_np, probs_np)

        def tpr_at(fpr_target: float):
            idx = np.searchsorted(fpr, fpr_target, side="right")
            if idx == 0:
                return float(tpr[0])
            if idx >= len(fpr):
                return float(tpr[-1])
            f0, f1 = fpr[idx - 1], fpr[idx]
            t0, t1 = tpr[idx - 1], tpr[idx]
            if f1 == f0:
                return float(max(t0, t1))
            return float(t0 + (fpr_target - f0) * (t1 - t0) / (f1 - f0))

        return auc, tpr_at(0.01), tpr_at(0.001)
    except Exception:
        return None, None, None


def train_mia_model(
    model: nn.Module,
    train_ds: Dataset,
    val_ds: Dataset,
    device: torch.device,
    epochs: int = 3,
    batch_size: int = 128,
    lr: float = 1e-3,
    label_smooth: float = 0.0,
    use_focal: bool = False,
    focal_gamma: float = 2.0,
    focal_alpha: float | None = 0.25,
    pairwise_loss: bool = False,
    pairwise_weight: float = 0.1,
    sampler: Optional[WeightedRandomSampler] = None,
    weight_decay: float = 0.01,
    lr_scheduler: str = "cosine",
    warmup_epochs: int = 0,
    min_lr: float = 1e-6,
    grad_clip: float | None = 1.0
):
    """Train the MIA classifier.

    Args:
        model: The MIA model to train.
        train_ds: Training dataset.
        val_ds: Validation dataset.
        device: Device to train on.
        epochs: Number of training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        label_smooth: Label smoothing factor.
        use_focal: Whether to use focal loss.
        focal_gamma: Focal loss gamma parameter.
        focal_alpha: Focal loss alpha parameter.
        pairwise_loss: Whether to use pairwise ranking loss.
        pairwise_weight: Weight for pairwise loss.
        sampler: Optional weighted sampler for balanced training.
        weight_decay: AdamW weight decay.
        lr_scheduler: Learning rate schedule type ("cosine", "linear", "none").
        warmup_epochs: Number of epochs for linear warmup.
        min_lr: Minimum learning rate for decay.
        grad_clip: Max gradient norm for clipping (None to disable).

    Returns:
        The trained model.
    """
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    if sampler is not None:
        train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, drop_last=False)
    else:
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    total_steps = epochs * len(train_dl)
    warmup_steps = warmup_epochs * len(train_dl)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        if lr_scheduler == "cosine":
            return max(min_lr / lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
        elif lr_scheduler == "linear":
            return max(min_lr / lr, 1.0 - progress)
        else:
            return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    best_auc = -1.0
    best_state = None

    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        t0 = time.time()
        steps = 0
        train_bar = tqdm(train_dl, desc=f"Train {ep+1}/{epochs}", leave=False)

        for feats, masks, labels in train_bar:
            feats = feats.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            logits = model(feats, masks)

            if use_focal:
                probs = torch.sigmoid(logits)
                ce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
                pt = torch.where(labels == 1, probs, 1 - probs)
                loss = ((1 - pt) ** focal_gamma) * ce
                if focal_alpha is not None:
                    alpha_t = labels * focal_alpha + (1 - labels) * (1 - focal_alpha)
                    loss = alpha_t * loss
                loss = loss.mean()
            else:
                targets = labels
                if label_smooth > 0:
                    targets = labels * (1 - label_smooth) + 0.5 * label_smooth
                loss = criterion(logits, targets)

            if pairwise_loss:
                pos = logits[labels == 1]
                neg = logits[labels == 0]
                if pos.numel() > 0 and neg.numel() > 0:
                    diff = pos.unsqueeze(1) - neg.unsqueeze(0)
                    pair_loss = F.relu(1.0 - diff).mean()
                    loss = loss + pairwise_weight * pair_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            scheduler.step()
            ep_loss += loss.item()
            steps += 1
            train_bar.set_postfix(loss=ep_loss / steps)

        dt = time.time() - t0

        with torch.no_grad():
            model.eval()
            all_logits, all_labels = [], []
            val_loss_sum = 0.0
            val_batches = 0
            val_bar = tqdm(val_dl, desc=f"Val {ep+1}/{epochs}", leave=False)

            for feats, masks, labels in val_bar:
                feats = feats.to(device)
                masks = masks.to(device)
                logits = model(feats, masks)
                targets = labels.to(device)

                if use_focal:
                    probs = torch.sigmoid(logits)
                    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
                    pt = torch.where(targets == 1, probs, 1 - probs)
                    vloss = ((1 - pt) ** focal_gamma) * ce
                    if focal_alpha is not None:
                        alpha_t = targets * focal_alpha + (1 - targets) * (1 - focal_alpha)
                        vloss = alpha_t * vloss
                    vloss = vloss.mean()
                else:
                    if label_smooth > 0:
                        targets = targets * (1 - label_smooth) + 0.5 * label_smooth
                    vloss = criterion(logits, targets)

                val_loss_sum += vloss.item()
                val_batches += 1
                all_logits.append(logits.detach().cpu())
                all_labels.append(labels)
                val_bar.set_postfix(loss=val_loss_sum / val_batches)

            all_logits = torch.cat(all_logits)
            all_labels = torch.cat(all_labels)
            probs = torch.sigmoid(all_logits)
            preds = (probs >= 0.5).float()
            acc = (preds == all_labels).float().mean().item()

            auc, tpr1, tpr01 = compute_curve_metrics(all_labels.numpy(), probs.numpy())

        if auc is not None and auc > best_auc:
            best_auc = auc
            best_state = copy.deepcopy(model.state_dict())

        current_lr = opt.param_groups[0]['lr']
        print(
            f"[MIA] Epoch {ep+1}: train_loss={ep_loss/len(train_dl):.4f} "
            f"val_loss={val_loss_sum/max(1,val_batches):.4f} val_acc={acc:.4f} "
            f"val_auc={auc if auc is not None else 'NA'} "
            f"val_tpr@1%={tpr1 if tpr1 is not None else 'NA'} "
            f"val_tpr@0.1%={tpr01 if tpr01 is not None else 'NA'} "
            f"lr={current_lr:.2e} (Best AUC: {best_auc:.4f}) time={dt:.1f}s"
        )

    if best_state is not None:
        print(f"Restoring best model with AUC: {best_auc:.4f}")
        model.load_state_dict(best_state)

    return model


def evaluate_mia_model(mia_model, mem_feats, mem_masks, non_feats, non_masks, device, batch_size=256):
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


def save_extraction_metadata(metadata: ExtractionMetadata, output_dir: Path):
    """Save extraction metadata to YAML.

    Args:
        metadata: The metadata to save.
        output_dir: Directory to save the metadata file.
    """
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
    meta_path = output_dir / "metadata.yaml"
    with open(meta_path, "r") as f:
        data = yaml.safe_load(f)
    return ExtractionMetadata(**data)


def save_features(
    output_dir: Path,
    members_feats: np.ndarray,
    members_masks: np.ndarray,
    nonmembers_feats: np.ndarray,
    nonmembers_masks: np.ndarray,
    members_token_ids: np.ndarray = None,
    nonmembers_token_ids: np.ndarray = None,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
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


def load_features_mmap(feature_dir: Path, split: str = "train") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load features with memory mapping for efficient access.

    Args:
        feature_dir: Directory containing feature files.
        split: Which split to load indices for ("train", "val", or "test").

    Returns:
        Tuple of (mem_feats, mem_masks, non_feats, non_masks, indices).
    """
    mem_feats = np.load(feature_dir / "members_feats.npy", mmap_mode='r')
    mem_masks = np.load(feature_dir / "members_masks.npy", mmap_mode='r')
    non_feats = np.load(feature_dir / "nonmembers_feats.npy", mmap_mode='r')
    non_masks = np.load(feature_dir / "nonmembers_masks.npy", mmap_mode='r')

    indices = np.load(feature_dir / f"{split}_indices.npy")

    return mem_feats, mem_masks, non_feats, non_masks, indices


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
    if not manifest_path.exists():
        return {}

    with open(manifest_path, "r") as f:
        data = yaml.safe_load(f) or {}

    entries = {}
    for combo_id, entry_data in data.get("combinations", {}).items():
        entries[combo_id] = ManifestEntry(combo_id=combo_id, **entry_data)

    return entries


def save_manifest(manifest_path: Path, entries: Dict[str, ManifestEntry]):
    """Save the feature manifest.

    Args:
        manifest_path: Path to save the manifest.
        entries: Dict mapping combo_id to ManifestEntry.
    """
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


def update_manifest(manifest_path: Path, entry: ManifestEntry):
    """Add or update a manifest entry.

    Args:
        manifest_path: Path to the manifest file.
        entry: The entry to add or update.
    """
    entries = load_manifest(manifest_path)
    entries[entry.combo_id] = entry
    save_manifest(manifest_path, entries)


class CombinedMIADataset(Dataset):
    """Dataset combining features from multiple model/dataset combinations."""

    def __init__(
        self,
        manifest_path: Path,
        split: str = "train",
        combinations: Optional[List[str]] = None,
        root_dir: Optional[Path] = None
    ):
        """Initialize combined dataset.

        Args:
            manifest_path: Path to the manifest file.
            split: Data split to use ("train", "val", or "test").
            combinations: Optional list of combo_ids to include.
            root_dir: Root directory for feature paths (defaults to manifest parent).
        """
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
