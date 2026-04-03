"""Per-token feature extraction for MIA."""

import math
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.tokenization import batch_tokenize


@torch.no_grad()
def extract_per_token_features_both(
    model_tgt: nn.Module,
    model_ref: nn.Module,
    tokenizer,
    texts: List[str],
    device: torch.device,
    batch_size: int = 32,
    sequence_length: int = 128,
    k: int = 20,
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
