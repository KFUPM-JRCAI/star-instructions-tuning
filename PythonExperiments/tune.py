#!/usr/bin/env python3
"""CLI entry point for fine-tuning.

Usage:
    uv run python PythonExperiments/tune.py --task dialect_identification --model AceGPT
    uv run python PythonExperiments/tune.py --task summarization --model Llama --gpus 0,1
"""

import argparse
import sys

# GPU setup MUST happen before any torch import
from src.gpu import setup_gpus
from src.experiments import EXPERIMENTS


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a model on a task's primary dataset."
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=list(EXPERIMENTS.keys()),
        help="Task name (primary dataset is auto-selected)",
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["AceGPT", "Llama", "Qwen"],
        help="Model family to fine-tune",
    )
    parser.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated GPU IDs (default: all available)",
    )

    args = parser.parse_args()

    # Set up GPUs before importing torch-based code
    setup_gpus(args.gpus)

    # Now safe to import torch-dependent modules
    from src.tuning import run_tuning

    run_tuning(task_name=args.task, model_key=args.model)


if __name__ == "__main__":
    main()
