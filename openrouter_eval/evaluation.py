"""Core evaluation pipeline for OpenRouter API models via lm-eval harness.

Reuses experiment configs, prompt fetching, and dataset preprocessing from
PythonExperiments/src/. Replaces the local model backends (HFLM/VLLM) with
lm-eval's openai-chat-completions backend pointing at OpenRouter.
"""

import json
import os
import sys

import datasets
from tqdm.auto import tqdm

from src.experiments import EXPERIMENTS
from src.promptlab import fetch_prompts, filter_prompts, get_dataset_prompts
from src.preprocessing import DATASET_FUNCTIONS

from .yaml_templates import write_yaml_config

# ── Constants ─────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_TASKS_DIR = os.path.join(SCRIPT_DIR, "eval_harness_tasks")
EVAL_RESULTS_DIR = os.path.join(SCRIPT_DIR, "evaluation_results")
CACHE_DIR = os.path.join(SCRIPT_DIR, "cache")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"


# ── Helpers ───────────────────────────────────────────────────────────────────


def setup_api_key():
    """Ensure OPENAI_API_KEY is set (required by lm-eval's openai backend)."""
    if os.environ.get("OPENAI_API_KEY"):
        return
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("Error: Set OPENROUTER_API_KEY or OPENAI_API_KEY environment variable.")
        sys.exit(1)
    os.environ["OPENAI_API_KEY"] = key


def results_dir_name(api_model: str) -> str:
    """e.g. 'google/gemini-3.1-pro-preview' -> 'gemini-3.1-pro-preview'."""
    return api_model.split("/")[-1]


def _load_and_merge_prompts(task_name, dataset_name, experiment, task_fns, prompt_ids):
    """Fetch prompts, load HF dataset, merge prompts with test samples."""
    promptlab_name = experiment.get_promptlab_name(dataset_name)
    hf_dataset_path = experiment.hf_datasets[dataset_name]

    all_prompts = fetch_prompts()
    filtered = filter_prompts(all_prompts)
    prompts = get_dataset_prompts(filtered, prompt_ids)

    print(f"[Eval] Loading dataset: {hf_dataset_path}")
    hf_dataset = datasets.load_dataset(hf_dataset_path)

    if task_fns.remap_dataset is not None:
        print("[Eval] Applying dataset column remapping")
        hf_dataset = task_fns.remap_dataset(hf_dataset)

    test_split = hf_dataset["test"]
    print(f"[Eval] Test samples: {len(test_split)}")

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


def _init_model(api_model: str):
    """Initialise lm-eval OpenAI chat completions model for OpenRouter."""
    from lm_eval.models.openai_completions import OpenAIChatCompletion

    print(f"[Eval] Initializing OpenRouter API model: {api_model}")
    return OpenAIChatCompletion(
        model=api_model,
        base_url=OPENROUTER_BASE_URL,
        num_concurrent=4,
        max_retries=10,
        timeout=300,
    )


def _run_lm_eval(lm_obj, eval_task_name: str, promptlab_name: str) -> dict:
    """Run lm-eval simple_evaluate for a single task."""
    from lm_eval.tasks import TaskManager
    from lm_eval.evaluator import simple_evaluate

    task_dir = os.path.join(EVAL_TASKS_DIR, promptlab_name)
    task_manager = TaskManager(include_path=task_dir)
    os.makedirs(CACHE_DIR, exist_ok=True)
    return simple_evaluate(
        model=lm_obj,
        tasks=[eval_task_name],
        num_fewshot=0,
        task_manager=task_manager,
        apply_chat_template=True,
        use_cache=os.path.join(CACHE_DIR, eval_task_name),
    )


def _json_default(o):
    try:
        return str(o)
    except Exception:
        return "<not serializable>"


def _save_results(results: dict, results_file: str) -> None:
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=4, default=_json_default)


def _load_existing_results(results_file: str) -> dict | None:
    if os.path.exists(results_file) and os.path.getsize(results_file) > 0:
        with open(results_file, "r") as f:
            return json.load(f)
    return None


# ── Main entry point ──────────────────────────────────────────────────────────


def clear_prompt_cache(promptlab_name: str, prompt_id: int) -> None:
    """Delete cached API response .db files for a single prompt."""
    import glob
    if not os.path.exists(CACHE_DIR):
        return
    pattern = os.path.join(CACHE_DIR, f"{promptlab_name}_prompt_{prompt_id}*")
    for f in glob.glob(pattern):
        os.remove(f)
        print(f"[Cache] Removed: {os.path.basename(f)}")


def run_evaluation(
    task_name: str,
    dataset_name: str,
    api_model: str,
    force_re_evaluate: bool = False,
    clear_cache: bool = False,
) -> None:
    """Run the full evaluation pipeline for a task + dataset via OpenRouter API."""
    from lm_eval.utils import make_table

    experiment = EXPERIMENTS[task_name]
    task_fns = DATASET_FUNCTIONS[dataset_name]
    prompt_ids = experiment.prompt_ids[dataset_name]

    model_name = results_dir_name(api_model)
    promptlab_name = experiment.get_promptlab_name(dataset_name)

    if clear_cache:
        for prompt_id in prompt_ids:
            clear_prompt_cache(promptlab_name, prompt_id)

    print(f"[Eval] Task: {task_name}")
    print(f"[Eval] Dataset: {dataset_name} (promptlab: {promptlab_name})")
    print(f"[Eval] API Model: {api_model} (results dir: {model_name})")

    # ── Check existing results ──
    results_dir = os.path.join(EVAL_RESULTS_DIR, model_name, task_name, promptlab_name)
    all_results = {}
    pending_prompt_ids = []

    if not force_re_evaluate:
        for prompt_id in prompt_ids:
            results_file = f"{results_dir}/prompt_{prompt_id}.json"
            existing = _load_existing_results(results_file)
            if existing is not None:
                print(f"[Eval] Already done: prompt {prompt_id}")
                all_results[f"{promptlab_name}_prompt_{prompt_id}"] = existing
            else:
                pending_prompt_ids.append(prompt_id)
    else:
        pending_prompt_ids = list(prompt_ids)

    if not pending_prompt_ids:
        print(f"[Eval] All {len(prompt_ids)} prompts already evaluated. Skipping.")
        return

    print(f"[Eval] {len(pending_prompt_ids)}/{len(prompt_ids)} prompts need evaluation")

    # ── Fetch prompts & merge with test data ──
    prompts, promptlab_name = _load_and_merge_prompts(
        task_name, dataset_name, experiment, task_fns, pending_prompt_ids,
    )

    # ── Initialize API model ──
    setup_api_key()
    lm_obj = _init_model(api_model)

    # ── Evaluate each prompt ──
    for i, prompt in enumerate(prompts, 1):
        prompt_id = prompt["id"]
        results_file = f"{results_dir}/prompt_{prompt_id}.json"

        print("-" * 80)
        print(f"[Eval] Prompt {i}/{len(prompts)} (ID: {prompt_id})")
        print("-" * 80)

        hf_dataset = task_fns.create_hf_dataset(prompt)
        _save_parquet(hf_dataset, promptlab_name, prompt_id)
        eval_task_name = write_yaml_config(
            task_name, promptlab_name, prompt_id, EVAL_TASKS_DIR,
        )

        result = _run_lm_eval(lm_obj, eval_task_name, promptlab_name)
        print(make_table(result))

        _save_results(result, results_file)
        print(f"[Eval] Saved: {results_file}")
        all_results[f"{promptlab_name}_prompt_{prompt_id}"] = result

    print(
        f"\n[Eval] Done. Evaluated {len(prompts)} prompts "
        f"for {model_name}/{task_name}/{promptlab_name}"
    )
