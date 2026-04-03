"""Tokenization utilities."""

from typing import Dict, List

import torch

from .loaders import is_code_dataset


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


def batch_tokenize(
    tokenizer,
    texts: List[str],
    sequence_length: int = 128,
) -> Dict[str, torch.Tensor]:
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
