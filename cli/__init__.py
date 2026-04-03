"""Command-line interface entry points for LT-MIA."""

from . import extract
from . import extract_all
from . import train
from . import evaluate

__all__ = ["extract", "extract_all", "train", "evaluate"]
