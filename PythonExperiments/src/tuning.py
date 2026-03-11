"""Shared tuning pipeline.

Replicates the flow from tune.ipynb notebooks:
1. Fetch prompts from PromptLab
2. Load HF dataset
3. Render training samples with round-robin prompt distribution
4. Train/test split (90/10)
5. Generate (prefix, suffix) tuples
6. Save samples to JSON
7. Load model and fine-tune with LoRA
"""

import json
import os
import random
import sys

import datasets
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from .experiments import EXPERIMENTS
from .models import MODELS
from .preprocessing import DATASET_FUNCTIONS
from .promptlab import fetch_prompts, filter_prompts, get_dataset_prompts


GLOBAL_SEED = 42


def run_tuning(task_name: str, model_key: str) -> None:
    """Run the full tuning pipeline for a task + model combination."""

    # ── Look up configs ──
    experiment = EXPERIMENTS[task_name]
    model_config = MODELS[model_key]
    dataset_name = experiment.primary_dataset
    task_fns = DATASET_FUNCTIONS[dataset_name]
    hf_dataset_path = experiment.hf_datasets[dataset_name]
    prompt_ids = experiment.prompt_ids[dataset_name]
    training_params = experiment.training_params

    print(f"[Tuning] Task: {task_name}")
    print(f"[Tuning] Dataset: {dataset_name} ({hf_dataset_path})")
    print(f"[Tuning] Model: {model_config.name} ({model_config.path})")

    # ── Fetch prompts ──
    all_prompts = fetch_prompts()
    filtered = filter_prompts(all_prompts)
    dataset_prompts = get_dataset_prompts(filtered, prompt_ids)

    # ── Load HF dataset ──
    print(f"[Tuning] Loading dataset: {hf_dataset_path}")
    hf_exp_dataset = datasets.load_dataset(hf_dataset_path)

    # Apply dataset remapping if needed (e.g. MT column remap)
    if task_fns.remap_dataset is not None:
        print("[Tuning] Applying dataset column remapping")
        hf_exp_dataset = task_fns.remap_dataset(hf_exp_dataset)

    print(f"[Tuning] Train samples: {len(hf_exp_dataset['train'])}")

    # ── Render training samples with round-robin prompt distribution ──
    step_size = len(hf_exp_dataset["train"]) / len(dataset_prompts)
    rendered_train_prompts_dataset = []

    for i, sample in enumerate(tqdm(hf_exp_dataset["train"], desc="Rendering templates")):
        prompt_idx = int(i / step_size)
        rendered_train_prompts_dataset.append(
            task_fns.apply_template(dataset_prompts[prompt_idx], sample, for_tuning=True)
        )

    print(f"[Tuning] Rendered {len(rendered_train_prompts_dataset)} training samples")

    # ── Train/test split ──
    random.seed(GLOBAL_SEED)
    train_samples, eval_samples = train_test_split(
        rendered_train_prompts_dataset,
        test_size=0.1,
        random_state=GLOBAL_SEED,
    )

    # ── Generate (prefix, suffix) tuples ──
    train_samples = list(map(task_fns.generate_tuple, train_samples))
    eval_samples = list(map(task_fns.generate_tuple, eval_samples))

    print(f"[Tuning] Train: {len(train_samples)} | Eval: {len(eval_samples)}")

    # ── Save samples to JSON ──
    # Determine the model folder name (AceGPT-v1-7B, AceGPT-v2-8B, Llama-3.1-8B, Qwen3-8B)
    model_folder = model_key
    samples_dir = f"Notebooks/Experiments/{task_name}/{dataset_name}/{model_folder}/samples_used_for_tuning"
    os.makedirs(samples_dir, exist_ok=True)

    with open(f"{samples_dir}/train.json", "w") as f:
        json.dump(train_samples, f, indent=4, ensure_ascii=False)
    with open(f"{samples_dir}/val.json", "w") as f:
        json.dump(eval_samples, f, indent=4, ensure_ascii=False)

    print(f"[Tuning] Saved samples to {samples_dir}/")

    # ── Load model ──
    from .models import get_llm_imports
    get_llm_imports()  # ensures sys.path is set up
    from llm import LLMLoader, train_llm

    initializer_cls = model_config.get_initializer()
    lora_config_fn = model_config.get_lora_config()

    print(f"[Tuning] Loading model: {model_config.path}")
    llm_loader = LLMLoader(
        model_config.path,
        llm_initializer=initializer_cls(),
    )
    model, tokenizer, generation_config = llm_loader()

    # ── Fine-tune ──
    output_dir = f"Notebooks/Experiments/{task_name}/{dataset_name}/tuned_models/{model_config.name}"
    print(f"[Tuning] Output: {output_dir}")
    print(f"[Tuning] Starting training with params: {training_params}")

    train_llm(
        model=model,
        tokenizer=tokenizer,
        train_samples=train_samples,
        eval_samples=eval_samples,
        peft_config=lora_config_fn(),
        output_dir=output_dir,
        **training_params,
    )

    print(f"[Tuning] Training complete. Adapter saved to {output_dir}")
    sys.exit(0)
