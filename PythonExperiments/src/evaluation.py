"""Shared evaluation pipeline.

Replicates the flow from evaluate/*.ipynb notebooks:
1. Fetch prompts from PromptLab
2. Load HF dataset, merge prompts with test samples
3. For each prompt:
   a. Create HF dataset (Parquet)
   b. Write YAML task config
   c. Run lm-eval simple_evaluate
   d. Save results JSON
"""

import json
import os
import sys

import datasets
from tqdm.auto import tqdm


from .experiments import EXPERIMENTS
from .models import MODELS, get_model_path, get_results_dir_name
from .preprocessing import DATASET_FUNCTIONS, get_yaml_template
from .promptlab import fetch_prompts, filter_prompts, get_dataset_prompts


# ── Helpers ──────────────────────────────────────────────────────────────────


def _resolve_adapter_path(task_name: str, experiment, model_config) -> str:
    """Return the LoRA adapter directory (always under the primary dataset)."""
    return (
        f"Notebooks/Experiments/{task_name}/{experiment.primary_dataset}"
        f"/tuned_models/{model_config.name}"
    )


MAX_MODEL_LEN_CAP = 8192


def _init_vllm(model_path: str, adapter_path: str | None):
    """Initialise the VLLM model used by lm-eval."""
    import torch
    from transformers import AutoConfig
    from lm_eval.models.vllm_causallms import VLLM

    tp_size = torch.cuda.device_count()

    config = AutoConfig.from_pretrained(model_path)
    model_max_len = getattr(config, "max_position_embeddings", None) or MAX_MODEL_LEN_CAP
    max_model_len = min(model_max_len, MAX_MODEL_LEN_CAP)
    print(f"[Eval] Initializing VLLM (tensor_parallel_size={tp_size}, max_model_len={max_model_len})")

    kwargs = dict(
        pretrained=model_path,
        trust_remote_code=True,
        tensor_parallel_size=tp_size,
        tokenizer=model_path,
        gpu_memory_utilization=0.9,
        max_model_len=max_model_len,
    )
    if adapter_path:
        kwargs["lora_local_path"] = adapter_path

    return VLLM(**kwargs)


def _load_and_merge_prompts(task_name, dataset_name, experiment, task_fns, prompt_ids):
    """Fetch prompts, load the HF dataset, and merge prompts with test samples."""
    promptlab_name = experiment.get_promptlab_name(dataset_name)
    hf_dataset_path = experiment.hf_datasets[dataset_name]

    # Fetch and filter prompts
    all_prompts = fetch_prompts()
    filtered = filter_prompts(all_prompts)
    prompts = get_dataset_prompts(filtered, prompt_ids)

    # Load HF dataset
    print(f"[Eval] Loading dataset: {hf_dataset_path}")
    hf_dataset = datasets.load_dataset(hf_dataset_path)

    if task_fns.remap_dataset is not None:
        print("[Eval] Applying dataset column remapping")
        hf_dataset = task_fns.remap_dataset(hf_dataset)

    test_split = hf_dataset["test"]
    print(f"[Eval] Test samples: {len(test_split)}")

    # Merge each prompt template with every test sample
    for prompt in prompts:
        prompt["merged_samples"] = [
            task_fns.apply_template(prompt, sample, for_tuning=False)
            for sample in tqdm(test_split, desc=f"Merging prompt {prompt['id']}")
        ]
        prompt["original_samples"] = list(test_split)

    return prompts, promptlab_name


def _save_parquet(hf_dataset, promptlab_name: str, prompt_id: int) -> None:
    """Write the prompt-specific HF dataset to Parquet."""
    parquet_dir = f"experimental_hf_datasets/{promptlab_name}/prompt_{prompt_id}"
    os.makedirs(parquet_dir, exist_ok=True)
    hf_dataset["test"].to_parquet(f"{parquet_dir}/data.parquet")


def _write_yaml_config(task_name: str, promptlab_name: str, prompt_id: int) -> str:
    """Write the lm-eval YAML task config and return the eval task name."""
    yaml_text = get_yaml_template(task_name).format(
        dataset_name=promptlab_name,
        prompt_id=prompt_id,
    )
    yaml_dir = f"eval_harness_extra_tasks/{promptlab_name}"
    os.makedirs(yaml_dir, exist_ok=True)
    with open(f"{yaml_dir}/prompt_{prompt_id}.yaml", "w") as f:
        f.write(yaml_text)

    return f"{promptlab_name}_prompt_{prompt_id}"


def _run_lm_eval(lm_obj, eval_task_name: str, promptlab_name: str) -> dict:
    """Run lm-eval simple_evaluate for a single task."""
    from lm_eval.tasks import TaskManager
    from lm_eval.evaluator import simple_evaluate

    task_manager = TaskManager(
        include_path=f"eval_harness_extra_tasks/{promptlab_name}"
    )
    return simple_evaluate(
        model=lm_obj,
        tasks=[eval_task_name],
        num_fewshot=0,
        task_manager=task_manager,
    )


def _json_default(o):
    """Try str(o) first; fall back to a placeholder."""
    try:
        return str(o)
    except Exception:
        return "<not serializable>"


def _save_results(results: dict, results_file: str) -> None:
    """Serialise evaluation results to JSON."""
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=4,
            default=_json_default,
        )


def _load_existing_results(results_file: str) -> dict | None:
    """Load previously saved results, or None if they don't exist / are empty."""
    if os.path.exists(results_file) and os.path.getsize(results_file) > 0:
        with open(results_file, "r") as f:
            return json.load(f)
    return None


# ── Main entry point ─────────────────────────────────────────────────────────


def run_evaluation(
    task_name: str,
    dataset_name: str,
    model_key: str,
    variant: str,
    force_re_evaluate: bool = False,
) -> None:
    """Run the full evaluation pipeline for a task + dataset + model + variant."""
    from lm_eval.utils import make_table

    # ── Resolve configs ──
    experiment = EXPERIMENTS[task_name]
    model_config = MODELS[model_key]
    task_fns = DATASET_FUNCTIONS[dataset_name]
    prompt_ids = experiment.prompt_ids[dataset_name]

    model_path = get_model_path(model_key, variant)
    results_model_name = get_results_dir_name(model_key, variant)
    adapter_path = (
        _resolve_adapter_path(task_name, experiment, model_config)
        if variant == "tuned"
        else None
    )

    # ── Log run info ──
    promptlab_name = experiment.get_promptlab_name(dataset_name)
    print(f"[Eval] Task: {task_name}")
    print(f"[Eval] Dataset: {dataset_name} (promptlab: {promptlab_name})")
    print(f"[Eval] Model: {results_model_name} ({model_path})")
    if adapter_path:
        print(f"[Eval] Adapter: {adapter_path}")

    # ── Check which prompts still need evaluation ──
    results_dir = f"evaluation_results/{results_model_name}/{task_name}/{promptlab_name}"
    all_results = {}
    pending_prompt_ids = []

    if not force_re_evaluate:
        for prompt_id in prompt_ids:
            results_file = f"{results_dir}/prompt_{prompt_id}.json"
            existing = _load_existing_results(results_file)
            if existing is not None:
                print(f"[Eval] Already done: prompt {prompt_id}")
                print(make_table(existing))
                all_results[f"{promptlab_name}_prompt_{prompt_id}"] = existing
            else:
                pending_prompt_ids.append(prompt_id)
    else:
        pending_prompt_ids = list(prompt_ids)

    if not pending_prompt_ids:
        print(f"[Eval] All {len(prompt_ids)} prompts already evaluated. Skipping.")
        sys.exit(0)

    print(f"[Eval] {len(pending_prompt_ids)}/{len(prompt_ids)} prompts need evaluation")

    # ── Fetch prompts & merge with test data ──
    prompts, promptlab_name = _load_and_merge_prompts(
        task_name, dataset_name, experiment, task_fns, pending_prompt_ids,
    )

    # ── Initialize model ──
    lm_obj = _init_vllm(model_path, adapter_path)

    # ── Evaluate each prompt ──
    print(f"[Eval] Evaluating {len(prompts)} prompts sequentially")

    for i, prompt in enumerate(prompts, 1):
        prompt_id = prompt["id"]
        results_file = f"{results_dir}/prompt_{prompt_id}.json"

        print("-" * 80)
        print(f"[Eval] Prompt {i}/{len(prompts)} (ID: {prompt_id})")
        print("-" * 80)

        # Prepare artifacts
        hf_dataset = task_fns.create_hf_dataset(prompt)
        _save_parquet(hf_dataset, promptlab_name, prompt_id)
        eval_task_name = _write_yaml_config(task_name, promptlab_name, prompt_id)

        # Run evaluation
        result = _run_lm_eval(lm_obj, eval_task_name, promptlab_name)
        print(make_table(result))

        # Save results
        _save_results(result, results_file)
        print(f"[Eval] Saved results for prompt {prompt_id}")
        all_results[f"{promptlab_name}_prompt_{prompt_id}"] = result

    print(
        f"\n[Eval] Done. Evaluated {len(prompts)} prompts "
        f"for {results_model_name}/{task_name}/{promptlab_name}"
    )
    sys.exit(0)
