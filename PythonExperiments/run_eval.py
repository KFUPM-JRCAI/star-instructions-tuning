#!/usr/bin/env python3
"""CLI entry point for evaluation.

Usage:
    uv run python PythonExperiments/run_eval.py --task dialect_identification --dataset AraBench_dev --model AceGPT-v1-7B --variant base
    uv run python PythonExperiments/run_eval.py --task dialect_identification --dataset all --model AceGPT-v1-7B --variant tuned --gpus 0
    uv run python PythonExperiments/run_eval.py --task summarization --dataset xlsum --model Llama-3.1-8B --variant chat
"""

import sys

import fire

# GPU setup MUST happen before any torch import
from src.gpu import setup_gpus
from src.experiments import EXPERIMENTS


def main(
    task: str,
    dataset: str,
    model: str,
    variant: str,
    gpus: str = None,
    force_re_evaluate: bool = False,
):
    """Evaluate a model on a task dataset using lm-evaluation-harness.

    Args:
        task: Task name (e.g. dialect_identification, summarization).
        dataset: Dataset name (e.g. AraBench_dev) or 'all' for all datasets.
        model: Model family (AceGPT-v1-7B, AceGPT-v2-8B, Llama-3.1-8B, Qwen3-8B).
        variant: Model variant to evaluate (base, chat, tuned).
        gpus: Comma-separated GPU IDs (default: all available).
        force_re_evaluate: Re-evaluate even if results already exist.
    """
    if task not in EXPERIMENTS:
        print(f"Error: Unknown task '{task}'. Available: {list(EXPERIMENTS.keys())}")
        sys.exit(1)

    if model not in ("AceGPT-v1-7B", "AceGPT-v2-8B", "Llama-3.1-8B", "Qwen3-8B"):
        print(f"Error: Unknown model '{model}'. Choose from: AceGPT-v1-7B, AceGPT-v2-8B, Llama-3.1-8B, Qwen3-8B")
        sys.exit(1)

    if variant not in ("base", "chat", "tuned"):
        print(f"Error: Unknown variant '{variant}'. Choose from: base, chat, tuned")
        sys.exit(1)

    # Validate dataset
    experiment = EXPERIMENTS[task]
    all_datasets = experiment.get_all_datasets()

    if dataset == "all":
        datasets_to_eval = all_datasets
    elif dataset in all_datasets:
        datasets_to_eval = [dataset]
    else:
        print(f"Error: Unknown dataset '{dataset}' for task '{task}'.")
        print(f"Available datasets: {all_datasets}")
        sys.exit(1)

    # Set up GPUs before importing torch-based code
    setup_gpus(gpus)

    # Now safe to import torch-dependent modules
    from src.evaluation import run_evaluation

    for dataset_name in datasets_to_eval:
        print(f"\n{'=' * 80}")
        print(f"Evaluating: {task}/{dataset_name}/{model}/{variant}")
        print(f"{'=' * 80}\n")

        run_evaluation(
            task_name=task,
            dataset_name=dataset_name,
            model_key=model,
            variant=variant,
            force_re_evaluate=force_re_evaluate,
        )


if __name__ == "__main__":
    fire.Fire(main)
