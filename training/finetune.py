"""Fine-tuning utilities for language models."""

import copy
import time
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..data.tokenization import batch_tokenize
from ..data.datasets import LMDataset


def finetune_model_on_texts(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    val_texts: Optional[List[str]],
    device: torch.device,
    epochs: int = 1,
    batch_size: int = 8,
    lr: float = 5e-5,
    sequence_length: int = 128,
    grad_accum_steps: int = 1,
) -> None:
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
