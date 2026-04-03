"""Dataset loading utilities with registry pattern."""

import itertools
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type


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


class DatasetLoader(ABC):
    """Abstract base class for dataset loaders."""

    requires_streaming: bool = False

    @abstractmethod
    def load(self, streaming: bool = False) -> Iterator[str]:
        """Load dataset and yield text samples.

        Args:
            streaming: Whether to use streaming mode.

        Yields:
            Text samples from the dataset.
        """
        pass


class DatasetRegistry:
    """Registry for dataset loaders."""

    _loaders: Dict[str, Type[DatasetLoader]] = {}

    @classmethod
    def register(cls, *names: str):
        """Decorator to register a dataset loader.

        Args:
            names: One or more names/aliases for the dataset.
        """
        def decorator(loader_cls: Type[DatasetLoader]):
            for name in names:
                cls._loaders[normalize_dataset_name(name)] = loader_cls
            return loader_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> DatasetLoader:
        """Get a dataset loader by name.

        Args:
            name: Dataset name.

        Returns:
            Instantiated dataset loader.

        Raises:
            ValueError: If dataset is not found.
        """
        normalized = normalize_dataset_name(name)
        if normalized not in cls._loaders:
            available = ", ".join(sorted(cls._loaders.keys()))
            raise ValueError(f"Unknown dataset: {name}. Available: {available}")
        return cls._loaders[normalized]()

    @classmethod
    def list_available(cls) -> List[str]:
        """List all available dataset names."""
        return sorted(cls._loaders.keys())


@DatasetRegistry.register("ag_news")
class AGNewsLoader(DatasetLoader):
    """Loader for AG News dataset."""

    def load(self, streaming: bool = False) -> Iterator[str]:
        from datasets import load_dataset
        ds = load_dataset("ag_news", streaming=streaming)
        return (x["text"] for x in ds["train"])


@DatasetRegistry.register("xsum")
class XSumLoader(DatasetLoader):
    """Loader for XSum dataset."""

    def load(self, streaming: bool = False) -> Iterator[str]:
        from datasets import load_dataset
        ds = load_dataset("xsum", streaming=streaming)
        return (x["document"] for x in ds["train"])


@DatasetRegistry.register("wikitext", "wikitext103", "wikitext_103")
class WikitextLoader(DatasetLoader):
    """Loader for Wikitext-103 dataset."""

    def load(self, streaming: bool = False) -> Iterator[str]:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", streaming=streaming)
        return (x["text"] for x in ds["train"])


@DatasetRegistry.register("wikipedia", "wiki")
class WikipediaLoader(DatasetLoader):
    """Loader for Wikipedia dataset."""

    requires_streaming = True

    def load(self, streaming: bool = False) -> Iterator[str]:
        from datasets import load_dataset
        ds = load_dataset("wikimedia/wikipedia", "20231101.en", streaming=streaming)
        return (x["text"] for x in ds["train"])


@DatasetRegistry.register("news_category", "news_category_dataset")
class NewsCategoryLoader(DatasetLoader):
    """Loader for News Category dataset."""

    def load(self, streaming: bool = False) -> Iterator[str]:
        from datasets import load_dataset
        ds = load_dataset("heegyu/news-category-dataset", streaming=streaming)
        return (f"{x['headline']} {x['short_description']}" for x in ds["train"])


@DatasetRegistry.register("cnndm", "cnn_dailymail", "cnn_dm")
class CNNDMLoader(DatasetLoader):
    """Loader for CNN/DailyMail dataset."""

    def load(self, streaming: bool = False) -> Iterator[str]:
        from datasets import load_dataset
        ds = load_dataset("cnn_dailymail", "3.0.0", streaming=streaming)
        return (x["article"] for x in ds["train"])


@DatasetRegistry.register("swallow_code", "swallowcode", "tokyotech_swallow_code")
class SwallowCodeLoader(DatasetLoader):
    """Loader for Swallow Code dataset."""

    requires_streaming = True

    def load(self, streaming: bool = False) -> Iterator[str]:
        from datasets import load_dataset

        def _example_to_text(ex: Dict[str, Any]) -> str:
            for k in ("text", "content", "code", "body", "document"):
                v = ex.get(k)
                if isinstance(v, str) and v.strip():
                    return v
            strings = [v for v in ex.values() if isinstance(v, str) and v.strip()]
            return "\n".join(strings) if strings else ""

        train_split = load_dataset(
            "tokyotech-llm/swallow-code",
            "swallow-code",
            split="train",
            streaming=streaming,
        )
        return (_example_to_text(x) for x in train_split)


@DatasetRegistry.register("arxiv", "arxiv_summarization")
class ArxivLoader(DatasetLoader):
    """Loader for Arxiv Summarization dataset."""

    requires_streaming = True

    def load(self, streaming: bool = False) -> Iterator[str]:
        from datasets import load_dataset
        ds = load_dataset("ccdv/arxiv-summarization", streaming=streaming, trust_remote_code=True)
        return (x["abstract"] for x in ds["train"])


# Set of datasets that should always stream
STREAMING_DATASETS = frozenset({
    "wikipedia", "wiki",
    "swallow_code", "swallowcode", "tokyotech_swallow_code",
    "arxiv", "arxiv_summarization",
})


def load_data_splits(
    dataset_name: str,
    n_members: int = 15000,
    n_nonmembers: int = 15000,
    n_val: int = 100,
    seed: int = 42,
    streaming: bool = False,
    streaming_datasets: Optional[List[str]] = None,
    stream_max_samples: int = 200_000,
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
    dataset_name_lower = normalize_dataset_name(dataset_name)
    streaming_set = set(streaming_datasets or []) | STREAMING_DATASETS
    use_stream = streaming or dataset_name_lower in streaming_set

    loader = DatasetRegistry.get(dataset_name)
    texts_iter = loader.load(streaming=use_stream)

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
