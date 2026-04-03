"""Model loading utilities."""

from typing import Any, Dict, Tuple

import torch.nn as nn

from ..data.tokenization import configure_tokenizer_for_dataset
from .lora import apply_lora


def prepare_tokenizer_and_models(
    model_name: str = "gpt2",
    dataset_name: str | None = None,
) -> Tuple[Any, nn.Module, nn.Module]:
    """Load tokenizer and paired reference/target models.

    Args:
        model_name: HuggingFace model name or path.
        dataset_name: Dataset name for tokenizer configuration.

    Returns:
        Tuple of (tokenizer, reference_model, target_model).
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

    is_mpt = "mpt" in model_name.lower()
    trust_remote = not is_mpt

    is_openelm = "openelm" in model_name.lower()
    is_stablelm = "stablelm" in model_name.lower()

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote)
    except ValueError as e:
        if "sentencepiece" in str(e) or "tiktoken" in str(e):
            if is_openelm:
                print(f"[Tokenizer] Using meta-llama/Llama-2-7b-hf tokenizer for {model_name}")
                tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
            else:
                raise
        else:
            raise

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    configure_tokenizer_for_dataset(tokenizer, dataset_name)

    model_kwargs = {"trust_remote_code": trust_remote}

    if is_stablelm:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote)
        if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
            config.pad_token_id = tokenizer.pad_token_id
        model_kwargs["config"] = config

    if is_openelm:
        model_kwargs["low_cpu_mem_usage"] = False
        model_kwargs["device_map"] = None
        model_kwargs["torch_dtype"] = None

    model_ref = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model_tgt = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    return tokenizer, model_ref, model_tgt


def apply_lora_if_enabled(model: nn.Module, lora_cfg: Dict[str, Any] | None) -> nn.Module:
    """Apply LoRA adapters to a model when configured.

    Args:
        model: The model to apply LoRA to.
        lora_cfg: LoRA configuration dict with keys: use_lora, r, alpha, dropout, target_modules.

    Returns:
        The model with LoRA adapters applied (or unchanged if not configured).

    Raises:
        RuntimeError: If LoRA is requested but peft is not installed.
    """
    if not lora_cfg or not lora_cfg.get("use_lora", False):
        return model

    return apply_lora(model, lora_cfg)
