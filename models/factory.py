"""Factory functions for creating MIA models."""

from typing import Dict, Type

import torch.nn as nn

from .architectures import (
    TinyMIASequence,
    PooledTransformerMIA,
    MLPMIA,
    MeanMLPMIA,
    LogisticRegressionMIA,
)


MIA_ARCHITECTURES: Dict[str, Type[nn.Module]] = {
    "transformer": TinyMIASequence,
    "pooled_transformer": PooledTransformerMIA,
    "mlp": MLPMIA,
    "mean_mlp": MeanMLPMIA,
    "lr": LogisticRegressionMIA,
}


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

    if architecture not in MIA_ARCHITECTURES:
        available = ", ".join(MIA_ARCHITECTURES.keys())
        raise ValueError(f"Unknown architecture: {architecture}. Choose from: {available}")

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
        raise ValueError(f"Unknown architecture: {architecture}")
