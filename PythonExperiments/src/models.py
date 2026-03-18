"""Model registry: paths, initializers, LoRA configs, and variant names.

The ``llm`` imports from jrcai_corekit are deferred (lazy) because importing
that package triggers torch/CUDA initialization.  Evaluation uses vLLM with
tensor parallelism which needs to *spawn* worker processes — that fails if
CUDA was already initialized in the parent process.  By deferring the import
we keep CUDA untouched until tuning actually needs those classes.
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional


def _ensure_corekit_on_path() -> None:
    """Add jrcai_corekit/llms_corekit to sys.path (same approach as notebooks)."""
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    corekit_path = os.path.join(project_dir, "jrcai_corekit", "llms_corekit")
    if corekit_path not in sys.path:
        sys.path.append(corekit_path)


def get_llm_imports():
    """Lazily import initializers and LoRAConfigRepository.

    Call this only when you actually need these classes (i.e. during tuning).
    """
    _ensure_corekit_on_path()
    from llm import (
        Llama3Initializer,
        Llama31Initializer,
        Qwen3Initializer,
        LoRAConfigRepository,
    )
    return Llama3Initializer, Llama31Initializer, Qwen3Initializer, LoRAConfigRepository


@dataclass
class ModelConfig:
    name: str               # Full model name, e.g. "AceGPT-7B"
    path: str               # Path to base model weights
    chat_path: str           # Path to chat/instruct variant weights
    # String identifiers resolved lazily via get_initializer() / get_lora_config()
    initializer_name: str    # e.g. "Llama3Initializer"
    lora_name: str           # e.g. "llama_3"
    # Names used in evaluation_results/ directories per variant
    results_names: dict = field(default_factory=dict)
    # If True, wrap eval inputs with the model's chat template for the chat variant.
    chat_apply_chat_template: bool = False

    def get_initializer(self):
        """Return the initializer class (imports llm lazily)."""
        Llama3Initializer, Llama31Initializer, Qwen3Initializer, _ = get_llm_imports()
        mapping = {
            "Llama3Initializer": Llama3Initializer,
            "Llama31Initializer": Llama31Initializer,
            "Qwen3Initializer": Qwen3Initializer,
        }
        return mapping[self.initializer_name]

    def get_lora_config(self):
        """Return the LoRA config factory method (imports llm lazily)."""
        _, _, _, LoRAConfigRepository = get_llm_imports()
        mapping = {
            "llama_3": LoRAConfigRepository.llama_3,
        }
        return mapping[self.lora_name]


MODELS: dict[str, ModelConfig] = {
    "AceGPT-v1-7B": ModelConfig(
        name="AceGPT-v1-7B",
        path="/raid_storage/shared_models/AceGPT-7B",
        chat_path="/raid_storage/shared_models/AceGPT-7B-chat",
        initializer_name="Llama3Initializer",
        lora_name="llama_3",
        results_names={
            "base": "AceGPT-v1-7B",
            "chat": "AceGPT-v1-7B-chat",
            "tuned": "AceGPT-v1-7B-tuned",
        },
    ),
    "AceGPT-v2-8B": ModelConfig(
        name="AceGPT-v2-8B",
        path="/raid_storage/shared_models/AceGPT-v2-8B",
        chat_path="/raid_storage/shared_models/AceGPT-v2-8B-Chat",
        initializer_name="Llama3Initializer",
        lora_name="llama_3",
        results_names={
            "base": "AceGPT-v2-8B",
            "chat": "AceGPT-v2-8B-Chat",
            "tuned": "AceGPT-v2-8B-tuned",
        },
    ),
    "Llama-3.1-8B": ModelConfig(
        name="Llama-3.1-8B",
        path="/raid_storage/shared_models/Meta-Llama-3.1-8B",
        chat_path="/raid_storage/shared_models/Meta-Llama-3.1-8B-Instruct",
        initializer_name="Llama31Initializer",
        lora_name="llama_3",
        results_names={
            "base": "Meta-Llama-3.1-8B",
            "chat": "Meta-Llama-3.1-8B-Instruct",
            "tuned": "Meta-Llama-3.1-8B-tuned",
        },
    ),
    "Qwen3-8B": ModelConfig(
        name="Qwen3-8B",
        path="/raid_storage/shared_models/Qwen3-8B-Base",
        chat_path="/raid_storage/shared_models/Qwen3-8B",
        initializer_name="Qwen3Initializer",
        lora_name="llama_3",
        results_names={
            "base": "Qwen3-8B",
            "chat": "Qwen3-8B-chat",
            "tuned": "Qwen3-8B-tuned",
        },
        chat_apply_chat_template=True,
    ),
}


# Reverse lookup: model folder name -> model key
MODEL_FOLDER_KEYS = {
    "AceGPT-v1-7B": "AceGPT-v1-7B",
    "AceGPT-v2-8B": "AceGPT-v2-8B",
    "Llama-3.1-8B": "Llama-3.1-8B",
    "Qwen3-8B": "Qwen3-8B",
}


def get_model_path(model_key: str, variant: str) -> str:
    """Get the model path for a given variant (base, chat, tuned)."""
    config = MODELS[model_key]
    if variant == "base":
        return config.path
    elif variant == "chat":
        return config.chat_path
    elif variant == "tuned":
        return config.path  # Base model + LoRA adapter loaded separately
    else:
        raise ValueError(f"Unknown variant: {variant}")


def get_results_dir_name(model_key: str, variant: str) -> str:
    """Get the model name used in evaluation_results/ directories.

    Explicit mapping to match existing directory names:
    - AceGPT-v1-7B: AceGPT-v1-7B, AceGPT-v1-7B-chat, AceGPT-v1-7B-tuned
    - AceGPT-v2-8B: AceGPT-v2-8B, AceGPT-v2-8B-Chat, AceGPT-v2-8B-tuned
    - Llama-3.1-8B: Meta-Llama-3.1-8B, Meta-Llama-3.1-8B-Instruct, Meta-Llama-3.1-8B-tuned
    - Qwen3-8B: Qwen3-8B, Qwen3-8B-chat, Qwen3-8B-tuned
    """
    return MODELS[model_key].results_names[variant]
