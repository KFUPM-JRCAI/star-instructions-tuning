#!/usr/bin/env python3
"""CLI entry point for OpenRouter API evaluation via lm-eval harness.

Usage:
    uv run python openrouter_eval/run_eval.py \
        --task summarization --dataset xlsum \
        --api-model google/gemini-3.1-pro-preview

    uv run python openrouter_eval/run_eval.py \
        --task dialect_identification --dataset all \
        --api-model google/gemini-3.1-pro-preview

Requires OPENROUTER_API_KEY (or OPENAI_API_KEY) environment variable.
"""

import os
import sys

import fire
from dotenv import load_dotenv

# ── Path setup ────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "PythonExperiments"))
os.chdir(PROJECT_ROOT)  # lm-eval YAML configs use relative dataset paths

from src.experiments import EXPERIMENTS


def main(
    task: str,
    dataset: str = "all",
    api_model: str = "google/gemini-3.1-pro-preview",
    force_re_evaluate: bool = False,
):
    """Evaluate an API model on a task dataset via OpenRouter.

    Args:
        task: Task name (e.g. dialect_identification, summarization).
        dataset: Dataset name or 'all' for all datasets in the task (default: all).
        api_model: OpenRouter model ID (default: google/gemini-3.1-pro-preview).
        force_re_evaluate: Re-evaluate even if results exist.
    """
    if task not in EXPERIMENTS:
        print(f"Error: Unknown task '{task}'. Available: {list(EXPERIMENTS.keys())}")
        sys.exit(1)

    experiment = EXPERIMENTS[task]
    all_datasets = experiment.get_all_datasets()

    if dataset == "all":
        datasets_to_eval = all_datasets
    elif dataset in all_datasets:
        datasets_to_eval = [dataset]
    else:
        print(f"Error: Unknown dataset '{dataset}' for task '{task}'.")
        print(f"Available: {all_datasets}")
        sys.exit(1)

    from openrouter_eval.evaluation import run_evaluation

    for dataset_name in datasets_to_eval:
        print(f"\n{'=' * 80}")
        print(f"Evaluating: {task}/{dataset_name} via {api_model}")
        print(f"{'=' * 80}\n")

        run_evaluation(
            task_name=task,
            dataset_name=dataset_name,
            api_model=api_model,
            force_re_evaluate=force_re_evaluate,
        )


if __name__ == "__main__":
    fire.Fire(main)
