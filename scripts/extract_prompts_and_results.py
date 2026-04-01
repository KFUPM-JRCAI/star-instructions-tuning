"""Extract prompt templates, per-prompt results tables, and generation sample outputs.

For each evaluation task/dataset, outputs:
  - The Jinja2 prompt templates (fetched from PromptLab API)
  - A per-prompt score table across all model variants
  - [Generation tasks only] Sampled source/reference/prediction comparisons

Usage (from project root):
    uv run python scripts/extract_prompts_and_results.py
"""

import json
import random
import re
import sys
from io import StringIO
from pathlib import Path

# Ensure project root is on sys.path so PythonExperiments is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from tabulate import tabulate
from tqdm.auto import tqdm

# Reuse results_table infrastructure
from PythonExperiments.src.experiments import EXPERIMENTS
from PythonExperiments.src.models import MODELS
from PythonExperiments.src.results_table import (
    GENERATION_TASKS,
    RESULTS_ROOT,
    VARIANTS,
    _collect_model_variants,
    _detect_metric,
    _format_score,
    _get_scores,
    _per_prompt_row,
)
from PythonExperiments.src.promptlab import fetch_prompts, filter_prompts

# Only these 3 models (ignore AceGPT-v1-7B)
SELECTED_MODELS = {"AceGPT-v2-8B", "Llama-3.1-8B", "Qwen3-8B"}

TASK_DISPLAY = {
    "NLU": "Natural Language Understanding (NLU)",
    "NLI": "Natural Language Inference (NLI)",
    "dialect_identification": "Dialect Identification",
    "sarcasm_detection": "Sarcasm Detection",
    "machine_translation": "Machine Translation",
    "summarization": "Summarization",
}

DATASET_DISPLAY = {
    "ArabicMMLU": "ArabicMMLU (MCQ)",
    "belebele": "Belebele",
    "ArEntail": "ArEntail",
    "ArabicTE": "ArabicTE",
    "AraBench_dev": "AraBench",
    "Arabic_Dialects_Dataset": "Arabic Dialects Dataset",
    "ArSarcasm_v2": "ArSarcasm-v2",
    "iSarcasmEval_task": "iSarcasmEval",
    "opus-100": "OPUS-100",
    "tatoeba_mt": "Tatoeba",
    "xlsum": "XLSUM",
    "AraSum": "AraSum",
}

TASK_ORDER = [
    "NLU",
    "NLI",
    "dialect_identification",
    "sarcasm_detection",
    "machine_translation",
    "summarization",
]

OUTPUT_FILE = Path("scripts/prompts_and_results_output.txt")
NUM_GENERATION_SAMPLES = 5
RANDOM_SEED = 42


def _get_selected_model_variants():
    """Return model variants filtered to SELECTED_MODELS only."""
    return [
        (model_key, variant, results_dir)
        for model_key, variant, results_dir in _collect_model_variants()
        if model_key in SELECTED_MODELS
    ]


def _fetch_prompt_templates():
    """Fetch all prompt templates from PromptLab API, keyed by ID."""
    all_prompts = fetch_prompts()
    filtered = filter_prompts(all_prompts)
    return {p["id"]: p["template"] for p in filtered}


def _write_prompt_templates(out, prompt_ids, templates):
    """Write numbered prompt templates for a dataset."""
    out.write("Prompt Templates:\n")
    for i, pid in enumerate(prompt_ids, 1):
        template = templates.get(pid, "<template not found>")
        out.write(f"\n  Prompt {i} (ID {pid}):\n")
        for line in template.strip().split("\n"):
            out.write(f"    {line}\n")
    out.write("\n")


def _write_results_table(out, task_name, dataset, prompt_ids, model_variants):
    """Write per-prompt results table for a dataset."""
    metric = _detect_metric(task_name)
    results_dataset_name = EXPERIMENTS[task_name].get_promptlab_name(dataset)

    metric_label = {
        "acc_norm": "Accuracy (Normalized)",
        "acc": "Accuracy",
        "calculate_bleu": "BLEU",
        "exact_match": "Exact Match",
    }.get(metric, metric)

    out.write(f"Results Table (metric: {metric_label}):\n")

    headers = ["Model", "Variant"] + [f"P{i+1}" for i in range(len(prompt_ids))] + ["Avg"]
    rows = []

    prev_model = None
    for model_key, variant, results_dir in model_variants:
        if prev_model is not None and model_key != prev_model:
            rows.append([""] * len(headers))
        prev_model = model_key
        scores = _get_scores(
            results_dir, task_name, results_dataset_name, prompt_ids, metric
        )
        rows.append(_per_prompt_row(model_key, variant, scores, metric))

    out.write(tabulate(rows, headers=headers, tablefmt="simple"))
    out.write("\n\n")


def _read_generation_samples(result_path, task_name):
    """Read samples from a generation result JSON. Returns list of (doc_id, doc, response, target).

    Applies the same preprocessing as the metrics files (first-line / collapse).
    """
    if not result_path.exists():
        return []
    try:
        with open(result_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, KeyError):
        return []

    samples_dict = data.get("samples", {})
    if not samples_dict:
        return []

    # Single key in samples dict
    task_key = next(iter(samples_dict))
    samples = samples_dict[task_key]

    results = []
    for s in samples:
        doc = s.get("doc", {})
        target = s.get("target", "")
        resps = s.get("filtered_resps", s.get("resps", []))
        response = ""
        if resps:
            resp = resps[0]
            if isinstance(resp, list):
                response = resp[0] if resp else ""
            else:
                response = resp
        cleaned = _clean_generation_output(str(response), task_name)
        results.append((s.get("doc_id", 0), doc, cleaned, str(target).strip()))

    return results


_THINK_REGEX = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)


def _clean_generation_output(response, task_name):
    """Apply the same preprocessing as the metrics files.

    Translation (opus-100, tatoeba_mt): strip <think> tags, take first line.
    Summarization (xlsum, AraSum): strip <think> tags, collapse newlines to spaces.
    """
    text = response.strip()
    text = _THINK_REGEX.sub("", text).strip()
    if not text:
        return ""
    if task_name == "machine_translation":
        text = text.splitlines()[0].strip()
    else:
        text = re.sub(r"\s+", " ", text.replace("\n", " "))
    return text


def _is_valid_output(response):
    """Check if a generation output is non-empty and not degenerate."""
    if not response or len(response) < 3:
        return False
    return True


def _get_source_text(doc, task_name):
    """Extract the source text from a document depending on task type."""
    if task_name == "machine_translation":
        return doc.get("en", "")
    elif task_name == "summarization":
        return doc.get("text", "")
    return ""


def _write_generation_samples(out, task_name, dataset, prompt_ids, model_variants):
    """Write sampled generation outputs for qualitative comparison."""
    results_dataset_name = EXPERIMENTS[task_name].get_promptlab_name(dataset)

    # Use the first prompt to find common valid sample indices across all models
    first_pid = prompt_ids[0]

    # Collect all model results for the first prompt to find common valid doc_ids
    all_model_samples = {}  # (model_key, variant) -> {doc_id: (doc, response, target)}
    for model_key, variant, results_dir in tqdm(
        model_variants, desc=f"Loading generation samples ({dataset})", leave=False,
    ):
        path = RESULTS_ROOT / results_dir / task_name / results_dataset_name / f"prompt_{first_pid}.json"
        samples = _read_generation_samples(path, task_name)
        sample_map = {}
        for doc_id, doc, response, target in samples:
            sample_map[doc_id] = (doc, response, target)
        all_model_samples[(model_key, variant)] = sample_map

    if not all_model_samples:
        out.write("  No generation samples available.\n\n")
        return

    # Find doc_ids where ALL models have valid (non-empty) outputs
    all_doc_ids = None
    for key, sample_map in all_model_samples.items():
        valid_ids = {
            doc_id for doc_id, (doc, response, target) in sample_map.items()
            if _is_valid_output(response)
        }
        if all_doc_ids is None:
            all_doc_ids = valid_ids
        else:
            all_doc_ids &= valid_ids

    if not all_doc_ids:
        out.write("  No samples with valid outputs from all models.\n\n")
        return

    # Sample randomly
    rng = random.Random(RANDOM_SEED)
    selected_ids = sorted(all_doc_ids)
    selected = rng.sample(selected_ids, min(NUM_GENERATION_SAMPLES, len(selected_ids)))

    out.write(f"Sample Outputs ({len(selected)} examples, prompt ID {first_pid}):\n\n")

    for idx, doc_id in enumerate(selected, 1):
        # Get source and reference from any model's data
        first_key = next(iter(all_model_samples))
        doc, _, target = all_model_samples[first_key][doc_id]
        source = _get_source_text(doc, task_name)

        out.write(f"  Example {idx} (doc_id={doc_id}):\n")

        # Truncate very long source texts
        source_display = source[:500] + ("..." if len(source) > 500 else "")
        out.write(f"    Source: {source_display}\n")
        out.write(f"    Reference: {target[:500]}{'...' if len(target) > 500 else ''}\n")

        for model_key, variant, results_dir in model_variants:
            _, response, _ = all_model_samples[(model_key, variant)].get(
                doc_id, ({}, "", "")
            )
            response_display = response[:500] + ("..." if len(response) > 500 else "")
            out.write(f"    {model_key} ({variant}): {response_display}\n")

        out.write("\n")


def main():
    print("Fetching prompt templates from PromptLab API...")
    templates = _fetch_prompt_templates()
    print(f"Loaded {len(templates)} prompt templates.")

    model_variants = _get_selected_model_variants()
    print(f"Using {len(model_variants)} model variants: "
          f"{[(m, v) for m, v, _ in model_variants]}")

    out = StringIO()
    out.write("=" * 80 + "\n")
    out.write("Evaluation Prompts, Results Tables, and Generation Samples\n")
    out.write("=" * 80 + "\n\n")

    all_datasets = [
        (task_name, dataset)
        for task_name in TASK_ORDER
        for dataset in EXPERIMENTS[task_name].get_all_datasets()
    ]

    pbar = tqdm(all_datasets, desc="Processing datasets")
    for task_name, dataset in pbar:
        config = EXPERIMENTS[task_name]
        task_label = TASK_DISPLAY.get(task_name, task_name)
        dataset_label = DATASET_DISPLAY.get(dataset, dataset)
        is_generation = task_name in GENERATION_TASKS
        prompt_ids = config.prompt_ids[dataset]

        pbar.set_postfix_str(f"{task_label} / {dataset_label}")

        # Write task header only for the first dataset in each task
        if dataset == config.get_all_datasets()[0]:
            out.write("#" * 80 + "\n")
            out.write(f"# Task: {task_label}\n")
            out.write("#" * 80 + "\n\n")

        out.write("-" * 60 + "\n")
        out.write(f"Dataset: {dataset_label}\n")
        out.write("-" * 60 + "\n\n")

        # 1. Prompt templates
        _write_prompt_templates(out, prompt_ids, templates)

        # 2. Results table
        _write_results_table(out, task_name, dataset, prompt_ids, model_variants)

        # 3. Generation samples (translation & summarization only)
        if is_generation:
            _write_generation_samples(
                out, task_name, dataset, prompt_ids, model_variants
            )

    content = out.getvalue()

    OUTPUT_FILE.write_text(content, encoding="utf-8")
    print(f"\nOutput written to {OUTPUT_FILE}")
    print(f"Total length: {len(content)} characters, {content.count(chr(10))} lines")


if __name__ == "__main__":
    main()
