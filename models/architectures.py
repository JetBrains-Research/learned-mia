"""MIA classifier architectures."""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import MaskedMeanPooling, SinusoidalPositionalEncoding


class TinyMIASequence(nn.Module):
    """Transformer-based MIA classifier operating on per-token features."""

    def __init__(
        self,
        d_in: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_ff: int = 256,
        dropout: float = 0.1,
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

    def forward(self, feats: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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

    def forward(self, feats: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        **kwargs,
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

    def forward(self, feats: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        **kwargs,
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

    def forward(self, feats: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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
        dropout: float = 0.1,
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

    def forward(self, feats: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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
