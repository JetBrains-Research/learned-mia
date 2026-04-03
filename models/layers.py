"""Neural network layers for MIA models."""

import math

import torch
import torch.nn as nn


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: Input tensor of shape (B, T, D).

        Returns:
            Tensor with positional encoding added.
        """
        L = x.size(1)
        return x + self.pe[:, :L, :]
