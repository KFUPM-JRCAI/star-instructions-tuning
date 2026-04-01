"""
Statistical analysis of per-prompt evaluation scores across models, tasks, and evaluation settings.

Analyses:
  1. Statistical significance of tuning gains (Wilcoxon signed-rank, Cohen's d)
  2. Prompt sensitivity (Coefficient of Variation, range)
  3. Intra-dataset vs intra-task correlation (Pearson, Spearman)
  4. Generalization gap (intra-dataset − intra-task) + Kruskal-Wallis
  5. Empty output analysis for generation tasks

Usage (from project root):
    uv run python scripts/statistical_analysis.py
"""

import json
import sys
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================================
# Configuration
# ============================================================================

EVAL_DIR = Path("evaluation_results")

# The three models requested for analysis (excluding AceGPT-v1)
MODELS = {
    "AceGPT-v2": {
        "base": "AceGPT-v2-8B",
        "chat": "AceGPT-v2-8B-Chat",
        "tuned": "AceGPT-v2-8B-tuned",
    },
    "LLaMA-3.1": {
        "base": "Meta-Llama-3.1-8B",
        "chat": "Meta-Llama-3.1-8B-Instruct",
        "tuned": "Meta-Llama-3.1-8B-tuned",
    },
    "Qwen3": {
        "base": "Qwen3-8B",
        "chat": "Qwen3-8B-chat",
        "tuned": "Qwen3-8B-tuned",
    },
}

# Task definitions: user-facing name -> (internal task name, primary dataset, secondary dataset)
# Primary = intra-dataset (trained & evaluated), Secondary = intra-task (cross-dataset evaluation)
TASKS = {
    "MCQ":           ("NLU",                     "ArabicMMLU",              "belebele"),
    "NLI":           ("NLI",                     "ArEntail",                "ArabicTE"),
    "Dialect":       ("dialect_identification",   "AraBench_dev",            "Arabic_Dialects_Dataset"),
    "Sarcasm":       ("sarcasm_detection",        "ArSarcasm_v2",            "iSarcasmEval_task_A"),
    "Translation":   ("machine_translation",      "opus-100",                "tatoeba_mt"),
    "Summarization": ("summarization",            "xlsum",                   "AraSum"),
}

# Prompt IDs per (internal_task, dataset) — matches experiments.py and result directories
PROMPT_IDS = {
    ("NLU", "ArabicMMLU"):                          [14571, 14869, 14787, 14797, 14798],
    ("NLU", "belebele"):                             [14854, 14853, 14801, 14800, 14575],
    ("NLI", "ArEntail"):                             [14581, 14816, 14818, 14819, 14820],
    ("NLI", "ArabicTE"):                             [14582, 14673, 14724, 14805, 14855],
    ("dialect_identification", "AraBench_dev"):       [14852, 14850, 14789, 14781, 14561],
    ("dialect_identification", "Arabic_Dialects_Dataset"): [14102, 14783, 14784, 14790, 14851],
    ("sarcasm_detection", "ArSarcasm_v2"):            [14779, 14802, 14835, 14837, 14838],
    ("sarcasm_detection", "iSarcasmEval_task_A"):     [14602, 14780, 14859, 14860, 14864],
    ("machine_translation", "opus-100"):              [14684, 14688, 14680, 14682, 14640],
    ("machine_translation", "tatoeba_mt"):            [14866, 14867, 14868, 14889, 14890],
    ("summarization", "xlsum"):                       [14871, 14803, 14856, 14858, 14668],
    ("summarization", "AraSum"):                      [14891, 14857, 14736, 14735, 14628],
}

# Classification tasks use acc_norm; generation tasks use calculate_bleu
GENERATION_TASKS = {"machine_translation", "summarization"}


# ============================================================================
# Data loading — preload all scores into a cache for speed
# ============================================================================

def get_metric(internal_task):
    """Return the metric key for a task."""
    if internal_task in GENERATION_TASKS:
        return "calculate_bleu"
    return "acc_norm"


def _read_score_from_json(path, metric):
    """Extract metric value from a single result JSON file."""
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        results = data.get("results", {})
        for task_results in results.values():
            key = f"{metric},none"
            if key in task_results:
                val = task_results[key]
                # Classification metrics are 0-1 scale -> convert to percentage
                if metric in ("acc", "acc_norm"):
                    return val * 100
                return val
    except (json.JSONDecodeError, KeyError):
        pass
    return None


# Global score cache: (model_dir, internal_task, dataset) -> list of 5 scores
_SCORE_CACHE = {}


def preload_all_scores():
    """Load all scores into cache upfront so analyses don't re-read files."""
    all_combos = []
    for model_name, variants in MODELS.items():
        for variant_label, variant_dir in variants.items():
            for task_label, (internal_task, primary_ds, secondary_ds) in TASKS.items():
                for dataset in [primary_ds, secondary_ds]:
                    all_combos.append((variant_dir, internal_task, dataset))

    for variant_dir, internal_task, dataset in tqdm(all_combos, desc="Loading scores"):
        metric = get_metric(internal_task)
        prompt_ids = PROMPT_IDS.get((internal_task, dataset), [])
        scores = []
        for pid in prompt_ids:
            path = EVAL_DIR / variant_dir / internal_task / dataset / f"prompt_{pid}.json"
            scores.append(_read_score_from_json(path, metric))
        _SCORE_CACHE[(variant_dir, internal_task, dataset)] = scores


def load_scores(model_dir, internal_task, dataset):
    """Load all 5 prompt scores for a model/task/dataset combination (from cache)."""
    return _SCORE_CACHE.get((model_dir, internal_task, dataset), [None] * 5)


def get_valid_scores(scores):
    """Filter out None values and return numpy array."""
    return np.array([s for s in scores if s is not None])


# ============================================================================
# Helpers: formatting
# ============================================================================

def fmt(val, decimals=2):
    """Format a number, or return 'N/A' if None/NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:.{decimals}f}"


def fmt_pval(p):
    """Format a p-value with significance stars."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "N/A"
    stars = ""
    if p < 0.001:
        stars = " ***"
    elif p < 0.01:
        stars = " **"
    elif p < 0.05:
        stars = " *"
    return f"{p:.4f}{stars}"


def cohens_d(a, b):
    """Compute Cohen's d effect size (b - a) / pooled_std."""
    if len(a) < 2 or len(b) < 2:
        return None
    pooled_std = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    if pooled_std == 0:
        return 0.0
    return (np.mean(b) - np.mean(a)) / pooled_std


def print_and_write(line, f):
    """Print to stdout and write to file."""
    print(line)
    f.write(line + "\n")


# ============================================================================
# Analysis 1: Statistical Significance of Tuning Gains
# ============================================================================

def tuning_significance(f):
    """Paired Wilcoxon signed-rank tests comparing base/chat vs tuned.

    For each model × task, reports p-values, Cohen's d, and mean ± std per variant.
    """
    print_and_write("=" * 100, f)
    print_and_write("ANALYSIS 1: STATISTICAL SIGNIFICANCE OF TUNING GAINS", f)
    print_and_write("=" * 100, f)
    print_and_write("", f)
    print_and_write(
        "For each model x task, paired Wilcoxon signed-rank tests compare (a) base vs tuned and", f
    )
    print_and_write(
        "(b) chat/instruct vs tuned across the 5 prompt scores. Cohen's d measures effect size.", f
    )
    print_and_write("Significance: * p<0.05, ** p<0.01, *** p<0.001", f)
    print_and_write("", f)

    for setting_label, dataset_idx in [("Intra-dataset (primary)", 1), ("Intra-task (secondary)", 2)]:
        print_and_write(f"\n--- {setting_label} ---", f)
        header = (
            f"{'Model':<14s} {'Task':<15s} "
            f"{'Base (mean+/-std)':<18s} {'Chat (mean+/-std)':<18s} {'Tuned (mean+/-std)':<18s} "
            f"{'Base->Tuned p':<16s} {'d':<8s} {'Chat->Tuned p':<16s} {'d':<8s}"
        )
        print_and_write(header, f)
        print_and_write("-" * len(header), f)

        for model_name, variants in MODELS.items():
            for task_label, (internal_task, primary_ds, secondary_ds) in TASKS.items():
                dataset = primary_ds if dataset_idx == 1 else secondary_ds

                base_scores = get_valid_scores(load_scores(variants["base"], internal_task, dataset))
                chat_scores = get_valid_scores(load_scores(variants["chat"], internal_task, dataset))
                tuned_scores = get_valid_scores(load_scores(variants["tuned"], internal_task, dataset))

                base_str = f"{np.mean(base_scores):.2f}+/-{np.std(base_scores):.2f}" if len(base_scores) >= 2 else "N/A"
                chat_str = f"{np.mean(chat_scores):.2f}+/-{np.std(chat_scores):.2f}" if len(chat_scores) >= 2 else "N/A"
                tuned_str = f"{np.mean(tuned_scores):.2f}+/-{np.std(tuned_scores):.2f}" if len(tuned_scores) >= 2 else "N/A"

                # Wilcoxon tests require matched pairs of length >= 5
                bt_p, bt_d = None, None
                ct_p, ct_d = None, None

                if len(base_scores) == 5 and len(tuned_scores) == 5:
                    try:
                        _, bt_p = stats.wilcoxon(base_scores, tuned_scores, alternative='two-sided')
                    except ValueError:
                        bt_p = None  # All differences are zero
                    bt_d = cohens_d(base_scores, tuned_scores)

                if len(chat_scores) == 5 and len(tuned_scores) == 5:
                    try:
                        _, ct_p = stats.wilcoxon(chat_scores, tuned_scores, alternative='two-sided')
                    except ValueError:
                        ct_p = None
                    ct_d = cohens_d(chat_scores, tuned_scores)

                line = (
                    f"{model_name:<14s} {task_label:<15s} "
                    f"{base_str:<18s} {chat_str:<18s} {tuned_str:<18s} "
                    f"{fmt_pval(bt_p):<16s} {fmt(bt_d):<8s} {fmt_pval(ct_p):<16s} {fmt(ct_d):<8s}"
                )
                print_and_write(line, f)

        print_and_write("", f)


# ============================================================================
# Analysis 2: Prompt Sensitivity (Coefficient of Variation)
# ============================================================================

def prompt_sensitivity(f):
    """Compute CV and range across the 5 prompt scores per model variant x task x setting."""
    print_and_write("\n" + "=" * 100, f)
    print_and_write("ANALYSIS 2: PROMPT SENSITIVITY (COEFFICIENT OF VARIATION)", f)
    print_and_write("=" * 100, f)
    print_and_write("", f)
    print_and_write("CV = std/mean x 100 (%). Range = max - min across 5 prompts.", f)
    print_and_write("Higher CV = more sensitive to prompt design.", f)
    print_and_write("", f)

    for setting_label, dataset_idx in [("Intra-dataset (primary)", 1), ("Intra-task (secondary)", 2)]:
        print_and_write(f"\n--- {setting_label} ---", f)
        header = (
            f"{'Model':<14s} {'Variant':<10s} {'Task':<15s} "
            f"{'Mean':<10s} {'Std':<10s} {'CV (%)':<10s} {'Min':<10s} {'Max':<10s} {'Range':<10s}"
        )
        print_and_write(header, f)
        print_and_write("-" * len(header), f)

        for model_name, variants in MODELS.items():
            for variant_label in ["base", "chat", "tuned"]:
                variant_dir = variants[variant_label]
                for task_label, (internal_task, primary_ds, secondary_ds) in TASKS.items():
                    dataset = primary_ds if dataset_idx == 1 else secondary_ds
                    scores = get_valid_scores(load_scores(variant_dir, internal_task, dataset))

                    if len(scores) < 2:
                        continue

                    mean = np.mean(scores)
                    std = np.std(scores, ddof=1)
                    cv = (std / mean * 100) if mean != 0 else float('inf')
                    rng = np.max(scores) - np.min(scores)

                    line = (
                        f"{model_name:<14s} {variant_label:<10s} {task_label:<15s} "
                        f"{mean:<10.2f} {std:<10.2f} {cv:<10.2f} "
                        f"{np.min(scores):<10.2f} {np.max(scores):<10.2f} {rng:<10.2f}"
                    )
                    print_and_write(line, f)

        print_and_write("", f)

    # Summary: average CV by variant type across all tasks and settings
    print_and_write("\n--- Summary: Average CV by Variant ---", f)
    cv_by_variant = defaultdict(list)
    for model_name, variants in MODELS.items():
        for variant_label in ["base", "chat", "tuned"]:
            variant_dir = variants[variant_label]
            for task_label, (internal_task, primary_ds, secondary_ds) in TASKS.items():
                for dataset in [primary_ds, secondary_ds]:
                    scores = get_valid_scores(load_scores(variant_dir, internal_task, dataset))
                    if len(scores) >= 2:
                        mean = np.mean(scores)
                        std = np.std(scores, ddof=1)
                        if mean != 0:
                            cv_by_variant[variant_label].append(std / mean * 100)

    header = f"{'Variant':<10s} {'Mean CV (%)':<15s} {'Median CV (%)':<15s} {'Max CV (%)':<15s} {'N':<5s}"
    print_and_write(header, f)
    print_and_write("-" * len(header), f)
    for v in ["base", "chat", "tuned"]:
        cvs = cv_by_variant[v]
        if cvs:
            line = f"{v:<10s} {np.mean(cvs):<15.2f} {np.median(cvs):<15.2f} {np.max(cvs):<15.2f} {len(cvs):<5d}"
            print_and_write(line, f)
    print_and_write("", f)


# ============================================================================
# Analysis 3: Intra-dataset vs Intra-task Correlation
# ============================================================================

def cross_dataset_correlation(f):
    """Pearson and Spearman correlations between intra-dataset and intra-task scores.

    High correlation means prompts that work well on the primary dataset also generalize.
    """
    print_and_write("\n" + "=" * 100, f)
    print_and_write("ANALYSIS 3: INTRA-DATASET vs INTRA-TASK CORRELATION", f)
    print_and_write("=" * 100, f)
    print_and_write("", f)
    print_and_write(
        "For each model variant x task, correlate the 5 prompt scores on the primary dataset", f
    )
    print_and_write(
        "(intra-dataset) with the 5 prompt scores on the secondary dataset (intra-task).", f
    )
    print_and_write(
        "High correlation means prompts that work well intra-dataset also work well intra-task.", f
    )
    print_and_write("", f)

    header = (
        f"{'Model':<14s} {'Variant':<10s} {'Task':<15s} "
        f"{'Pearson r':<12s} {'Pearson p':<14s} {'Spearman rho':<12s} {'Spearman p':<14s}"
    )
    print_and_write(header, f)
    print_and_write("-" * len(header), f)

    for model_name, variants in MODELS.items():
        for variant_label in ["base", "chat", "tuned"]:
            variant_dir = variants[variant_label]
            for task_label, (internal_task, primary_ds, secondary_ds) in TASKS.items():
                intra_ds = get_valid_scores(load_scores(variant_dir, internal_task, primary_ds))
                intra_task = get_valid_scores(load_scores(variant_dir, internal_task, secondary_ds))

                if len(intra_ds) != 5 or len(intra_task) != 5:
                    continue

                try:
                    pr, pp = stats.pearsonr(intra_ds, intra_task)
                except Exception:
                    pr, pp = None, None

                try:
                    sr, sp = stats.spearmanr(intra_ds, intra_task)
                except Exception:
                    sr, sp = None, None

                line = (
                    f"{model_name:<14s} {variant_label:<10s} {task_label:<15s} "
                    f"{fmt(pr):<12s} {fmt_pval(pp):<14s} {fmt(sr):<12s} {fmt_pval(sp):<14s}"
                )
                print_and_write(line, f)

    print_and_write("", f)


# ============================================================================
# Analysis 4: Generalization Gap
# ============================================================================

def generalization_gap(f):
    """Compute generalization gap = intra-dataset - intra-task, then Kruskal-Wallis tests."""
    print_and_write("\n" + "=" * 100, f)
    print_and_write("ANALYSIS 4: GENERALIZATION GAP", f)
    print_and_write("=" * 100, f)
    print_and_write("", f)
    print_and_write(
        "Gap = mean(intra-dataset scores) - mean(intra-task scores) per model x variant x task.", f
    )
    print_and_write(
        "Positive gap means the model degrades on the secondary (cross-dataset) evaluation.", f
    )
    print_and_write("", f)

    header = (
        f"{'Model':<14s} {'Variant':<10s} {'Task':<15s} "
        f"{'Intra-DS mean':<15s} {'Intra-Task mean':<16s} {'Gap mean':<12s} {'Gap std':<10s}"
    )
    print_and_write(header, f)
    print_and_write("-" * len(header), f)

    gaps_by_model = defaultdict(list)
    gaps_by_task = defaultdict(list)

    for model_name, variants in MODELS.items():
        for variant_label in ["base", "chat", "tuned"]:
            variant_dir = variants[variant_label]
            for task_label, (internal_task, primary_ds, secondary_ds) in TASKS.items():
                intra_ds = get_valid_scores(load_scores(variant_dir, internal_task, primary_ds))
                intra_task = get_valid_scores(load_scores(variant_dir, internal_task, secondary_ds))

                if len(intra_ds) < 2 or len(intra_task) < 2:
                    continue

                # Element-wise gaps if both have all 5 prompts
                if len(intra_ds) == 5 and len(intra_task) == 5:
                    prompt_gaps = intra_ds - intra_task
                    gap_mean = np.mean(prompt_gaps)
                    gap_std = np.std(prompt_gaps, ddof=1)
                else:
                    gap_mean = np.mean(intra_ds) - np.mean(intra_task)
                    gap_std = float('nan')

                gaps_by_model[model_name].append(gap_mean)
                gaps_by_task[task_label].append(gap_mean)

                line = (
                    f"{model_name:<14s} {variant_label:<10s} {task_label:<15s} "
                    f"{np.mean(intra_ds):<15.2f} {np.mean(intra_task):<16.2f} "
                    f"{gap_mean:<12.2f} {fmt(gap_std):<10s}"
                )
                print_and_write(line, f)

    # Kruskal-Wallis: does the gap differ across models?
    print_and_write("\n--- Kruskal-Wallis: Does the gap differ across models? ---", f)
    model_groups = [np.array(gaps_by_model[m]) for m in MODELS if len(gaps_by_model[m]) > 0]
    if len(model_groups) >= 2 and all(len(g) >= 2 for g in model_groups):
        try:
            h_stat, kw_p = stats.kruskal(*model_groups)
            print_and_write(f"H-statistic = {h_stat:.4f}, p-value = {fmt_pval(kw_p)}", f)
        except ValueError as e:
            print_and_write(f"Could not compute: {e}", f)
    else:
        print_and_write("Not enough data groups for Kruskal-Wallis test.", f)

    # Kruskal-Wallis: does the gap differ across tasks?
    print_and_write("\n--- Kruskal-Wallis: Does the gap differ across tasks? ---", f)
    task_groups = [np.array(gaps_by_task[t]) for t in TASKS if len(gaps_by_task[t]) > 0]
    if len(task_groups) >= 2 and all(len(g) >= 2 for g in task_groups):
        try:
            h_stat, kw_p = stats.kruskal(*task_groups)
            print_and_write(f"H-statistic = {h_stat:.4f}, p-value = {fmt_pval(kw_p)}", f)
        except ValueError as e:
            print_and_write(f"Could not compute: {e}", f)
    else:
        print_and_write("Not enough data groups for Kruskal-Wallis test.", f)

    print_and_write("", f)


# ============================================================================
# Analysis 5: Empty Output Analysis for Generation Tasks
# ============================================================================

def _load_samples(model_dir, internal_task, dataset, prompt_id):
    """Load the samples list for a specific prompt from its result JSON."""
    path = EVAL_DIR / model_dir / internal_task / dataset / f"prompt_{prompt_id}.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        task_key = f"{dataset}_prompt_{prompt_id}"
        return data.get("samples", {}).get(task_key, None)
    except (json.JSONDecodeError, KeyError):
        return None


def _is_empty_output(filtered_resps):
    """Check if a generation output is empty/whitespace/missing.

    For generation tasks, filtered_resps is a list with one string element,
    e.g. ['some generated text'] or [''].
    """
    if not filtered_resps:
        return True
    text = filtered_resps[0]
    if text is None:
        return True
    if not isinstance(text, str):
        return True
    return text.strip() == ""


def empty_output_analysis(f):
    """Analyze empty output rates for generation tasks (MT and summarization).

    Reports per-prompt empty rates, per-model/variant/task aggregates,
    and correlation between empty rate and BLEU score.
    """
    print_and_write("\n" + "=" * 100, f)
    print_and_write("ANALYSIS 5: EMPTY OUTPUT ANALYSIS FOR GENERATION TASKS", f)
    print_and_write("=" * 100, f)
    print_and_write("", f)
    print_and_write(
        "For generation tasks (Translation, Summarization), counts samples where the model", f
    )
    print_and_write(
        "produced empty or whitespace-only output. High empty rates depress BLEU scores.", f
    )
    print_and_write("", f)

    gen_tasks = {
        "Translation":   ("machine_translation", "opus-100",    "tatoeba_mt"),
        "Summarization": ("summarization",       "xlsum",       "AraSum"),
    }

    # Collect per-prompt data: (model, variant, task, dataset, prompt_id) -> (empty_rate, bleu)
    all_rows = []         # for the detailed table
    corr_data = defaultdict(lambda: {"empty_rates": [], "bleus": []})  # task_label -> lists

    # Build work items for tqdm
    work_items = []
    for model_name, variants in MODELS.items():
        for variant_label in ["base", "chat", "tuned"]:
            variant_dir = variants[variant_label]
            for task_label, (internal_task, primary_ds, secondary_ds) in gen_tasks.items():
                for dataset in [primary_ds, secondary_ds]:
                    prompt_ids = PROMPT_IDS.get((internal_task, dataset), [])
                    for pid in prompt_ids:
                        work_items.append((
                            model_name, variant_label, variant_dir,
                            task_label, internal_task, dataset, pid,
                        ))

    for item in tqdm(work_items, desc="Analyzing empty outputs"):
        model_name, variant_label, variant_dir, task_label, internal_task, dataset, pid = item

        samples = _load_samples(variant_dir, internal_task, dataset, pid)
        if samples is None:
            continue

        total = len(samples)
        empty_count = sum(1 for s in samples if _is_empty_output(s.get("filtered_resps")))
        empty_rate = (empty_count / total * 100) if total > 0 else None

        # Get BLEU for this prompt (already cached)
        bleu_scores = load_scores(variant_dir, internal_task, dataset)
        prompt_ids = PROMPT_IDS.get((internal_task, dataset), [])
        prompt_idx = prompt_ids.index(pid) if pid in prompt_ids else -1
        bleu = bleu_scores[prompt_idx] if 0 <= prompt_idx < len(bleu_scores) else None

        all_rows.append((
            model_name, variant_label, task_label, dataset, pid,
            total, empty_count, empty_rate, bleu,
        ))

        if empty_rate is not None and bleu is not None:
            corr_data[task_label]["empty_rates"].append(empty_rate)
            corr_data[task_label]["bleus"].append(bleu)

    # Part A: Detailed per-prompt table
    print_and_write("--- Per-Prompt Empty Output Rates ---", f)
    header = (
        f"{'Model':<14s} {'Variant':<10s} {'Task':<15s} {'Dataset':<28s} "
        f"{'Prompt':<8s} {'Total':<8s} {'Empty':<8s} {'Empty %':<10s} {'BLEU':<10s}"
    )
    print_and_write(header, f)
    print_and_write("-" * len(header), f)

    for row in all_rows:
        model_name, variant_label, task_label, dataset, pid, total, empty_count, empty_rate, bleu = row
        line = (
            f"{model_name:<14s} {variant_label:<10s} {task_label:<15s} {dataset:<28s} "
            f"{pid:<8d} {total:<8d} {empty_count:<8d} {fmt(empty_rate):<10s} {fmt(bleu):<10s}"
        )
        print_and_write(line, f)

    # Part B: Aggregate per model x variant x task
    print_and_write("", f)
    print_and_write("--- Aggregate Empty Rates (per model x variant x task) ---", f)
    header = (
        f"{'Model':<14s} {'Variant':<10s} {'Task':<15s} "
        f"{'Avg Empty %':<14s} {'Total Samples':<15s} {'Total Empty':<14s}"
    )
    print_and_write(header, f)
    print_and_write("-" * len(header), f)

    agg = defaultdict(lambda: {"empty_rates": [], "total_samples": 0, "total_empty": 0})
    for row in all_rows:
        model_name, variant_label, task_label, dataset, pid, total, empty_count, empty_rate, bleu = row
        key = (model_name, variant_label, task_label)
        if empty_rate is not None:
            agg[key]["empty_rates"].append(empty_rate)
        agg[key]["total_samples"] += total
        agg[key]["total_empty"] += empty_count

    for model_name, variants in MODELS.items():
        for variant_label in ["base", "chat", "tuned"]:
            for task_label in gen_tasks:
                key = (model_name, variant_label, task_label)
                data = agg[key]
                avg_empty = np.mean(data["empty_rates"]) if data["empty_rates"] else None
                line = (
                    f"{model_name:<14s} {variant_label:<10s} {task_label:<15s} "
                    f"{fmt(avg_empty):<14s} {data['total_samples']:<15d} {data['total_empty']:<14d}"
                )
                print_and_write(line, f)

    # Part C: Correlation between empty rate and BLEU
    print_and_write("", f)
    print_and_write("--- Correlation: Empty Rate vs BLEU Score ---", f)
    print_and_write("Computed across all model-variant-prompt combinations per task.", f)
    header = f"{'Task':<15s} {'N':<6s} {'Pearson r':<12s} {'Pearson p':<14s}"
    print_and_write(header, f)
    print_and_write("-" * len(header), f)

    for task_label in gen_tasks:
        rates = np.array(corr_data[task_label]["empty_rates"])
        bleus = np.array(corr_data[task_label]["bleus"])
        n = len(rates)
        if n >= 3:
            try:
                pr, pp = stats.pearsonr(rates, bleus)
            except Exception:
                pr, pp = None, None
        else:
            pr, pp = None, None
        line = f"{task_label:<15s} {n:<6d} {fmt(pr):<12s} {fmt_pval(pp):<14s}"
        print_and_write(line, f)

    print_and_write("", f)


# ============================================================================
# Main
# ============================================================================

def main():
    if not EVAL_DIR.exists():
        print(f"ERROR: Evaluation results directory not found: {EVAL_DIR}")
        print("Make sure to run this script from the project root.")
        sys.exit(1)

    output_path = Path("scripts/statistical_analysis_results.txt")
    print(f"Writing results to {output_path} and stdout...\n")

    # Preload all metric scores (fast — reads only results dicts, not samples)
    preload_all_scores()

    with open(output_path, "w") as f:
        print_and_write("STATISTICAL ANALYSIS OF EVALUATION RESULTS", f)
        print_and_write(f"Models: {', '.join(MODELS.keys())}", f)
        print_and_write(f"Tasks: {', '.join(TASKS.keys())}", f)
        print_and_write("", f)

        # Data availability check
        print_and_write("--- Data Availability Check ---", f)
        for model_name, variants in MODELS.items():
            for variant_label in ["base", "chat", "tuned"]:
                variant_dir = variants[variant_label]
                found = 0
                total = 0
                for task_label, (internal_task, primary_ds, secondary_ds) in TASKS.items():
                    for dataset in [primary_ds, secondary_ds]:
                        scores = load_scores(variant_dir, internal_task, dataset)
                        valid = [s for s in scores if s is not None]
                        found += len(valid)
                        total += len(scores)
                print_and_write(f"  {model_name} {variant_label}: {found}/{total} scores loaded", f)
        print_and_write("", f)

        # Run all five analyses
        tuning_significance(f)
        prompt_sensitivity(f)
        cross_dataset_correlation(f)
        generalization_gap(f)
        empty_output_analysis(f)

        print_and_write("\n" + "=" * 100, f)
        print_and_write("END OF ANALYSIS", f)
        print_and_write("=" * 100, f)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
