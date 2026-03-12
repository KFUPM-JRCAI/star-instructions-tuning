#!/usr/bin/env python3
"""CLI entry point for evaluation.

Usage:
    uv run python PythonExperiments/run_eval.py --task dialect_identification --dataset AraBench_dev --model AceGPT-v1-7B --variant base
    uv run python PythonExperiments/run_eval.py --task dialect_identification --model AceGPT-v1-7B --variant tuned --gpus 0
    uv run python PythonExperiments/run_eval.py --task summarization
    uv run python PythonExperiments/run_eval.py --task all
"""

import fire

# GPU setup MUST happen before any torch import
from src.gpu import setup_gpus
from src.experiments import EXPERIMENTS
from src.models import MODELS

ALL_MODELS = list(MODELS.keys())
ALL_VARIANTS = ["base", "chat", "tuned"]


def main(
    task: str = None,
    dataset: str = "all",
    model: str = "all",
    variant: str = "all",
    gpus: str = None,
    force_re_evaluate: bool = False,
    add_system_prompt: bool = False,
):
    """Evaluate a model on a task dataset using lm-evaluation-harness.

    Args:
        task: Task name (e.g. dialect_identification, summarization) or 'all'. Required.
        dataset: Dataset name (e.g. AraBench_dev) or 'all' for all datasets (default: all).
        model: Model family (AceGPT-v1-7B, AceGPT-v2-8B, Llama-3.1-8B, Qwen3-8B) or 'all' (default: all).
        variant: Model variant (base, chat, tuned) or 'all' (default: all).
        gpus: Comma-separated GPU IDs (default: all available).
        force_re_evaluate: Re-evaluate even if results already exist.
        add_system_prompt: Prepend global system prompt to each sample's input text.
    """
    if task is None:
        fire.Fire(main, command=["--help"])
        return

    # Resolve tasks
    if task == "all":
        tasks_to_eval = list(EXPERIMENTS.keys())
    elif task in EXPERIMENTS:
        tasks_to_eval = [task]
    else:
        raise ValueError(f"Unknown task '{task}'. Available: {list(EXPERIMENTS.keys())}")

    # Resolve models
    if model == "all":
        models_to_eval = ALL_MODELS
    elif model in MODELS:
        models_to_eval = [model]
    else:
        raise ValueError(f"Unknown model '{model}'. Choose from: {ALL_MODELS}")

    # Resolve variants
    if variant == "all":
        variants_to_eval = ALL_VARIANTS
    elif variant in ALL_VARIANTS:
        variants_to_eval = [variant]
    else:
        raise ValueError(f"Unknown variant '{variant}'. Choose from: {ALL_VARIANTS}")

    # Set up GPUs before importing torch-based code
    setup_gpus(gpus)

    # Now safe to import torch-dependent modules
    from src.evaluation import run_evaluation

    for task_name in tasks_to_eval:
        experiment = EXPERIMENTS[task_name]
        all_datasets = experiment.get_all_datasets()

        # Resolve datasets per task
        if dataset == "all":
            datasets_to_eval = all_datasets
        elif dataset in all_datasets:
            datasets_to_eval = [dataset]
        else:
            raise ValueError(f"Unknown dataset '{dataset}' for task '{task_name}'. Available: {all_datasets}")

        for dataset_name in datasets_to_eval:
            for model_key in models_to_eval:
                for variant_name in variants_to_eval:
                    print(f"\n{'=' * 80}")
                    print(f"Evaluating: {task_name}/{dataset_name}/{model_key}/{variant_name}")
                    print(f"{'=' * 80}\n")

                    run_evaluation(
                        task_name=task_name,
                        dataset_name=dataset_name,
                        model_key=model_key,
                        variant=variant_name,
                        force_re_evaluate=force_re_evaluate,
                        add_system_prompt=add_system_prompt,
                    )


if __name__ == "__main__":
    fire.Fire(main)
