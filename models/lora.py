"""LoRA (Low-Rank Adaptation) configuration and utilities."""

from typing import Any, Dict, List

import torch.nn as nn


LORA_TARGET_MODULES: Dict[str, List[str]] = {
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "gemma": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "recurrentgemma": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "gpt_neox": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    "pythia": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    "falcon": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    "bloom": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    "gptj": ["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out"],
    "gpt-j": ["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out"],
    "gpt2": ["c_attn", "c_proj", "c_fc"],
    "mamba": ["in_proj", "x_proj", "dt_proj"],
    "rwkv": ["attention.key", "attention.value", "attention.receptance", "attention.output"],
    "xlstm": ["up_proj", "down_proj"],
    "stripedhyena": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "hyena": ["q_proj", "k_proj", "v_proj", "o_proj"],
}


def detect_lora_targets(model: nn.Module) -> List[str]:
    """Auto-detect appropriate LoRA target modules for a model.

    Args:
        model: The model to detect targets for.

    Returns:
        List of module names to apply LoRA to.
    """
    config = getattr(model, "config", None)
    model_type = (getattr(config, "model_type", "") or "").lower() if config else ""
    model_name = (getattr(config, "_name_or_path", "") or "").lower() if config else ""

    for key, modules in LORA_TARGET_MODULES.items():
        if key in model_type or key in model_name:
            return modules

    return LORA_TARGET_MODULES["gpt2"]


def apply_lora(model: nn.Module, lora_cfg: Dict[str, Any]) -> nn.Module:
    """Apply LoRA adapters to a model.

    Args:
        model: The model to apply LoRA to.
        lora_cfg: LoRA configuration with keys: r, alpha, dropout, target_modules.

    Returns:
        The model with LoRA adapters applied.

    Raises:
        RuntimeError: If peft is not installed.
    """
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise RuntimeError(f"LoRA requested but peft is unavailable: {exc}")

    r = int(lora_cfg.get("r", 8))
    alpha = float(lora_cfg.get("alpha", 16))
    dropout = float(lora_cfg.get("dropout", 0.05))
    target_modules = lora_cfg.get("target_modules")

    if target_modules is None:
        target_modules = detect_lora_targets(model)

    lcfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    return model
