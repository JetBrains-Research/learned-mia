"""Device detection utilities."""

import torch


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
