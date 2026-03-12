"""Print evaluation results as a table for a given task or dataset, across all model variants.

Usage (from project root):
    # Show results for a specific dataset
    uv run python -m PythonExperiments.src.results_table --dataset AraBench_dev

    # Show results for an entire task (all datasets)
    uv run python -m PythonExperiments.src.results_table --task dialect_identification

    # Show per-prompt breakdown instead of averages
    uv run python -m PythonExperiments.src.results_table --task NLU --per-prompt

    # Choose a specific metric (acc, acc_norm, calculate_bleu, exact_match)
    uv run python -m PythonExperiments.src.results_table --dataset ArabicMMLU --metric acc_norm

    # Include OpenRouter API model results
    uv run python -m PythonExperiments.src.results_table --task NLU --include-openrouter
"""

import json
import os
from pathlib import Path

import fire
from tabulate import tabulate

from PythonExperiments.src.experiments import EXPERIMENTS
from PythonExperiments.src.models import MODELS


RESULTS_ROOT = Path("evaluation_results")
OPENROUTER_RESULTS_ROOT = Path("openrouter_eval/evaluation_results")

GENERATION_TASKS = {"machine_translation", "summarization"}

VARIANTS = ["base", "chat", "tuned"]


def _find_task_for_dataset(dataset: str) -> str:
    """Find which task a dataset belongs to."""
    for task_name, config in EXPERIMENTS.items():
        if dataset in config.get_all_datasets():
            return task_name
    raise ValueError(
        f"Dataset '{dataset}' not found in any experiment. "
        f"Available datasets: {[d for c in EXPERIMENTS.values() for d in c.get_all_datasets()]}"
    )


def _detect_metric(task_name: str) -> str:
    """Return the default metric key for a task."""
    if task_name in GENERATION_TASKS:
        return "calculate_bleu"
    return "acc"


def _read_result(result_path: Path, metric: str) -> float | None:
    """Read a single result JSON and extract the metric value."""
    if not result_path.exists():
        return None
    try:
        with open(result_path) as f:
            data = json.load(f)
        results = data.get("results", {})
        # Results dict has a single key like "AraBench_dev_prompt_14852"
        for task_results in results.values():
            metric_key = f"{metric},none"
            if metric_key in task_results:
                return task_results[metric_key]
    except (json.JSONDecodeError, KeyError):
        pass
    return None


def _collect_model_variants() -> list[tuple[str, str, str]]:
    """Return list of (model_key, variant, results_dir_name) for all models."""
    entries = []
    for model_key, model_config in MODELS.items():
        for variant in VARIANTS:
            results_dir = model_config.results_names.get(variant)
            if results_dir:
                entries.append((model_key, variant, results_dir))
    return entries


def _collect_openrouter_models() -> list[str]:
    """Return list of OpenRouter model names that have results."""
    if not OPENROUTER_RESULTS_ROOT.exists():
        return []
    return [d.name for d in OPENROUTER_RESULTS_ROOT.iterdir() if d.is_dir()]


def _get_scores(
    results_dir_name: str,
    task_name: str,
    dataset: str,
    prompt_ids: list[int],
    metric: str,
    root: Path = RESULTS_ROOT,
) -> list[float | None]:
    """Get metric scores for each prompt ID."""
    scores = []
    for pid in prompt_ids:
        path = root / results_dir_name / task_name / dataset / f"prompt_{pid}.json"
        scores.append(_read_result(path, metric))
    return scores


def main(
    task: str = None,
    dataset: str = None,
    metric: str = None,
    per_prompt: bool = True,
    include_openrouter: bool = True,
):
    """Print evaluation results as a table.

    Args:
        task: Task name (e.g. dialect_identification, NLU). Shows all datasets in the task.
        dataset: Dataset name (e.g. AraBench_dev, ArabicMMLU). Shows results for one dataset.
              Provide either --task or --dataset (not both).
        metric: Metric to display (acc, acc_norm, calculate_bleu, exact_match).
              Defaults based on task type.
        per_prompt: If True, show per-prompt scores instead of averages.
        include_openrouter: If True, also include OpenRouter API model results.
    """
    if task and dataset:
        # Both provided: validate dataset belongs to task
        config = EXPERIMENTS[task]
        if dataset not in config.get_all_datasets():
            raise ValueError(f"Dataset '{dataset}' is not part of task '{task}'")
        datasets_to_show = [dataset]
    elif dataset:
        task = _find_task_for_dataset(dataset)
        datasets_to_show = [dataset]
    elif task == "all":
        # Show all tasks
        for task_name in EXPERIMENTS:
            main(
                task=task_name,
                metric=metric,
                per_prompt=per_prompt,
                include_openrouter=include_openrouter,
            )
        return
    elif task:
        config = EXPERIMENTS[task]
        datasets_to_show = config.get_all_datasets()
    else:
        all_datasets = [d for c in EXPERIMENTS.values() for d in c.get_all_datasets()]
        print("Usage: uv run python -m PythonExperiments.src.results_table --task <task> [--dataset <dataset>]")
        print()
        print(f"Available tasks: {', '.join(EXPERIMENTS.keys())}")
        print(f"Available datasets: {', '.join(all_datasets)}")
        print()
        print("Use --task all to show results for all tasks.")
        return

    config = EXPERIMENTS[task]
    if metric is None:
        metric = _detect_metric(task)

    model_variants = _collect_model_variants()
    openrouter_models = _collect_openrouter_models() if include_openrouter else []

    # Determine the openrouter metric (API models use exact_match for classification)
    openrouter_metric = metric
    if metric in ("acc", "acc_norm"):
        openrouter_metric = "exact_match"

    for ds in datasets_to_show:
        prompt_ids = config.prompt_ids[ds]

        if per_prompt:
            _print_per_prompt_table(
                task, ds, prompt_ids, metric, model_variants,
                openrouter_models, openrouter_metric,
            )
        else:
            _print_summary_table(
                task, ds, prompt_ids, metric, model_variants,
                openrouter_models, openrouter_metric,
            )

        print()


def _format_score(value: float | None, metric: str) -> str:
    """Format a score for display."""
    if value is None:
        return "-"
    if metric in ("acc", "acc_norm", "exact_match"):
        return f"{value * 100:.2f}"
    return f"{value:.2f}"


def _print_summary_table(
    task: str,
    dataset: str,
    prompt_ids: list[int],
    metric: str,
    model_variants: list[tuple[str, str, str]],
    openrouter_models: list[str],
    openrouter_metric: str,
):
    """Print a table with one row per model, showing avg/min/max across prompts."""
    print(f"=== {task} / {dataset} (metric: {metric}) ===")

    headers = ["Model", "Variant", "Avg", "Min", "Max", "Prompts"]
    rows = []

    for model_key, variant, results_dir in model_variants:
        scores = _get_scores(results_dir, task, dataset, prompt_ids, metric)
        valid = [s for s in scores if s is not None]
        if not valid:
            rows.append([model_key, variant, "-", "-", "-", "0/" + str(len(prompt_ids))])
            continue
        avg = sum(valid) / len(valid)
        rows.append([
            model_key,
            variant,
            _format_score(avg, metric),
            _format_score(min(valid), metric),
            _format_score(max(valid), metric),
            f"{len(valid)}/{len(prompt_ids)}",
        ])

    for api_model in openrouter_models:
        scores = _get_scores(
            api_model, task, dataset, prompt_ids, openrouter_metric,
            root=OPENROUTER_RESULTS_ROOT,
        )
        valid = [s for s in scores if s is not None]
        if not valid:
            rows.append([api_model, "api", "-", "-", "-", "0/" + str(len(prompt_ids))])
            continue
        avg = sum(valid) / len(valid)
        rows.append([
            api_model,
            "api",
            _format_score(avg, openrouter_metric),
            _format_score(min(valid), openrouter_metric),
            _format_score(max(valid), openrouter_metric),
            f"{len(valid)}/{len(prompt_ids)}",
        ])

    print(tabulate(rows, headers=headers, tablefmt="simple"))


def _print_per_prompt_table(
    task: str,
    dataset: str,
    prompt_ids: list[int],
    metric: str,
    model_variants: list[tuple[str, str, str]],
    openrouter_models: list[str],
    openrouter_metric: str,
):
    """Print a table with one row per model, one column per prompt."""
    print(f"=== {task} / {dataset} (metric: {metric}) ===")

    headers = ["Model", "Variant"] + [str(pid) for pid in prompt_ids] + ["Avg"]
    rows = []

    for model_key, variant, results_dir in model_variants:
        scores = _get_scores(results_dir, task, dataset, prompt_ids, metric)
        valid = [s for s in scores if s is not None]
        avg = sum(valid) / len(valid) if valid else None
        row = [model_key, variant]
        row += [_format_score(s, metric) for s in scores]
        row.append(_format_score(avg, metric))
        rows.append(row)

    for api_model in openrouter_models:
        scores = _get_scores(
            api_model, task, dataset, prompt_ids, openrouter_metric,
            root=OPENROUTER_RESULTS_ROOT,
        )
        valid = [s for s in scores if s is not None]
        avg = sum(valid) / len(valid) if valid else None
        row = [api_model, "api"]
        row += [_format_score(s, openrouter_metric) for s in scores]
        row.append(_format_score(avg, openrouter_metric))
        rows.append(row)

    print(tabulate(rows, headers=headers, tablefmt="simple"))


if __name__ == "__main__":
    fire.Fire(main)
