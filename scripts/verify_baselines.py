"""
Verify baseline values for classification datasets used in scatter plots.

Computes two baselines per dataset:
  - Random baseline: mean(1 / num_choices) across all samples
    (handles variable choice counts, e.g. ArabicMMLU has 2/3/4/5-choice questions)
  - First choice bias: % of samples where the correct answer is the first option
    (i.e. accuracy if the model always picks option index 0)

Run from project root:
    uv run python Notebooks/results_visualization/verify_baselines.py

Sarcasm tasks are excluded because different prompts use different answer-choice
orderings (e.g. ['Non sarcastic', 'sarcastic'] vs ['True', 'False']), so the
"first choice" maps to different labels depending on the prompt. There is no
single first-choice-bias number that applies across all prompts.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

EVAL_HF_DATASETS_DIR = Path("experimental_hf_datasets")

# Each entry: (dataset_dir, list of prompt_ids to check)
# Using one prompt per dataset is sufficient since the label distribution and
# number of choices are the same across prompts for a given dataset.
# We use the first available prompt directory.
DATASETS = {
    "ArabicMMLU": {
        "task": "NLU (primary)",
        "prompt_ids": [14571, 14869, 14787, 14797, 14798],
    },
    "belebele": {
        "task": "NLU (secondary)",
        "prompt_ids": [14854, 14853, 14801, 14800, 14575],
    },
    "ArEntail": {
        "task": "NLI (primary)",
        "prompt_ids": [14581, 14816, 14818, 14819, 14820],
    },
    "ArabicTE": {
        "task": "NLI (secondary)",
        "prompt_ids": [14582, 14673, 14724, 14805, 14855],
    },
    "AraBench_dev": {
        "task": "Dialect ID (primary)",
        "prompt_ids": [14852, 14850, 14789, 14781, 14561],
    },
    "Arabic_Dialects_Dataset": {
        "task": "Dialect ID (secondary)",
        "prompt_ids": [14102, 14783, 14784, 14790, 14851],
    },
    # --- Sarcasm tasks excluded ---
    # ArSarcasm_v2 and iSarcasmEval_task_A are excluded because different
    # prompts use different answer-choice orderings:
    #
    # ArSarcasm_v2 (3,000 samples, label 0 = 72.6%, label 1 = 27.4%):
    #   - Prompts 14779, 14835, 14837, 14838: ['Non sarcastic', 'sarcastic']
    #     -> first choice = Non sarcastic -> first_choice_bias = 72.6%
    #   - Prompt 14802: ['True', 'False']
    #     -> "True" means sarcastic -> first choice = sarcastic -> first_choice_bias = 27.4%
    #
    # iSarcasmEval_task_A (1,400 samples, label 0 = 85.7%, label 1 = 14.3%):
    #   - Prompts 14602, 14780: ['Non sarcastic', 'sarcastic']
    #     -> first_choice_bias = 85.7%
    #   - Prompt 14859: ['false', 'true'] (false = not sarcastic)
    #     -> first_choice_bias = 85.7%
    #   - Prompt 14860: ['true', 'false'] (true = sarcastic)
    #     -> first_choice_bias = 14.3%
    #   - Prompt 14864: ['no', 'yes'] (no = not sarcastic)
    #     -> first_choice_bias = 85.7%
}


def compute_baselines(dataset_name: str, prompt_id: int) -> dict:
    """Compute random and first-choice-bias baselines for a dataset+prompt."""
    parquet_path = (
        EVAL_HF_DATASETS_DIR / dataset_name / f"prompt_{prompt_id}" / "data.parquet"
    )
    if not parquet_path.exists():
        return {"error": f"File not found: {parquet_path}"}

    df = pd.read_parquet(parquet_path)
    n_samples = len(df)

    # Number of choices per sample
    choice_lengths = df["choices"].apply(len)

    # Random baseline = mean(1/num_choices)
    random_baseline = (1.0 / choice_lengths).mean() * 100

    # First choice bias = fraction of samples where label == first choice
    first_choices = df["choices"].apply(lambda c: c[0])
    first_choice_correct = (df["label"] == first_choices).sum()
    first_choice_bias = first_choice_correct / n_samples * 100

    # Label distribution
    label_counts = df["label"].value_counts().sort_index()

    # Choice length distribution
    choice_len_dist = choice_lengths.value_counts().sort_index()

    return {
        "n_samples": n_samples,
        "random_baseline": random_baseline,
        "first_choice_bias": first_choice_bias,
        "label_counts": label_counts,
        "choice_len_dist": choice_len_dist,
        "first_choice_example": first_choices.iloc[0],
    }


def main():
    print("=" * 80)
    print("BASELINE VERIFICATION FOR CLASSIFICATION DATASETS")
    print("=" * 80)

    results_summary = {}

    for dataset_name, config in DATASETS.items():
        print(f"\n{'─' * 80}")
        print(f"  {dataset_name}  ({config['task']})")
        print(f"{'─' * 80}")

        # Use first available prompt
        prompt_id = config["prompt_ids"][0]
        result = compute_baselines(dataset_name, prompt_id)

        if "error" in result:
            print(f"  ERROR: {result['error']}")
            continue

        print(f"  Samples: {result['n_samples']}")
        print(f"  Prompt used: {prompt_id}")
        print(f"  First choice example: '{result['first_choice_example']}'")
        print()

        # Choice length distribution
        if result["choice_len_dist"].nunique() > 1 or len(result["choice_len_dist"]) > 1:
            print("  Choice length distribution (variable!):")
            for n_choices, count in result["choice_len_dist"].items():
                print(f"    {n_choices} choices: {count} samples")
            print()

        # Label distribution
        print("  Label distribution:")
        for label, count in result["label_counts"].items():
            pct = count / result["n_samples"] * 100
            print(f"    {label:>20s}: {count:>5d}  ({pct:5.1f}%)")
        print()

        # Baselines
        print(f"  Random baseline:     {result['random_baseline']:6.2f}%")
        print(f"  First choice bias:   {result['first_choice_bias']:6.2f}%")

        results_summary[dataset_name] = {
            "random": round(result["random_baseline"], 1),
            "first_choice": round(result["first_choice_bias"], 1),
        }

        # Cross-check with all prompts (in case choices differ)
        print(f"\n  Cross-checking all {len(config['prompt_ids'])} prompts...")
        for pid in config["prompt_ids"]:
            r = compute_baselines(dataset_name, pid)
            if "error" in r:
                print(f"    prompt_{pid}: MISSING")
                continue
            fc_example = r["first_choice_example"]
            match = "OK" if abs(r["random_baseline"] - result["random_baseline"]) < 0.01 else "DIFFERS!"
            print(
                f"    prompt_{pid}: random={r['random_baseline']:6.2f}%, "
                f"first_choice={r['first_choice_bias']:6.2f}% "
                f"(first='{fc_example}') [{match}]"
            )

    # Summary table
    print(f"\n\n{'=' * 80}")
    print("SUMMARY — copy into scatter_plot.ipynb")
    print(f"{'=' * 80}")
    print()
    print("dataset_baselines = {")
    for ds_name, vals in results_summary.items():
        print(f"    {ds_name!r:30s}: {{'random': {vals['random']}, 'first_choice': {vals['first_choice']}}},")
    print("    # Sarcasm excluded: prompt-dependent answer ordering (see comments above)")
    print("    'ArSarcasm':        {'random': 50.0, 'first_choice': None},")
    print("    'iSarcasmEval':     {'random': 50.0, 'first_choice': None},")
    print("}")

    # Final formatted table
    print(f"\n\n{'=' * 80}")
    print("FINAL SUMMARY TABLE")
    print(f"{'=' * 80}")
    print()
    header = f"{'Dataset':<30s} {'Task':<25s} {'Random (%)':<12s} {'First Choice (%)':<16s}"
    print(header)
    print("─" * len(header))
    for ds_name, config in DATASETS.items():
        vals = results_summary.get(ds_name)
        if vals is None:
            print(f"{ds_name:<30s} {config['task']:<25s} {'N/A':<12s} {'N/A':<16s}")
        else:
            print(f"{ds_name:<30s} {config['task']:<25s} {vals['random']:<12.1f} {vals['first_choice']:<16.1f}")
    # Sarcasm rows
    print(f"{'ArSarcasm_v2':<30s} {'Sarcasm (primary)':<25s} {'50.0':<12s} {'N/A (varies)':<16s}")
    print(f"{'iSarcasmEval_task_A':<30s} {'Sarcasm (secondary)':<25s} {'50.0':<12s} {'N/A (varies)':<16s}")
    print("─" * len(header))
    print()


if __name__ == "__main__":
    main()
