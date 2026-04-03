"""MIA model training utilities."""

import copy
import math
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

from ..evaluation.metrics import compute_curve_metrics


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
    focal_alpha: Optional[float] = 0.25,
    pairwise_loss: bool = False,
    pairwise_weight: float = 0.1,
    sampler: Optional[WeightedRandomSampler] = None,
    weight_decay: float = 0.01,
    lr_scheduler: str = "cosine",
    warmup_epochs: int = 0,
    min_lr: float = 1e-6,
    grad_clip: Optional[float] = 1.0,
) -> nn.Module:
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
