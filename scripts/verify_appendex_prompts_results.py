#!/usr/bin/env python3
"""
Verify that the numerical results reported in latex_paper.tex (copy your paper latex to this file) match the actual
evaluation result JSON files stored in evaluation_results/.

Overview
--------
The paper contains two types of results tables:

1. **Appendix per-prompt tables** (12 tables, one per dataset):
   Each table has 5 rows (one per prompt formulation) and 9 columns
   (3 models x 3 variants: Base, Chat/Instruct, Tuned). These are the raw
   per-prompt scores.

2. **Main results table** (Table 1, "tuning effectiveness"):
   Aggregated mean +/- std over the 5 prompts for each model-variant-task
   combination on the primary (intra-dataset) datasets only.

How LaTeX tables are parsed
---------------------------
The script locates each table by its \\label{} tag (e.g., "tab:appendix-arabicmmlu"),
then extracts the text between that label and \\end{tabular}. Within that block,
it uses \\midrule as a delimiter: lines after \\midrule containing '&' separators
are data rows. Each row is split on '&' -- the first column is the prompt name
(discarded), and the remaining 9 columns are the numerical values.

The "Average" row is identified by checking if the first column contains "Average"
or "textbf".

How prompt IDs are mapped to table rows
----------------------------------------
The LaTeX tables do NOT contain prompt IDs -- they just label prompts as
"1 (Compact)", "2 (Simple)", etc. The mapping from table row number to
Tajeeh prompt ID is hardcoded in the DATASETS config below. These orderings
come from the SELECTED_PROMPTS_IDS lists in the experiment notebooks and
PythonExperiments/src/experiments.py. Row 1 in the table corresponds to
prompt_ids[0], row 2 to prompt_ids[1], etc.

How JSON results are loaded
---------------------------
Each evaluation result lives at:
    evaluation_results/{model_variant}/{task}/{dataset}/prompt_{id}.json

The JSON structure is:
    {"results": {"<dataset>_prompt_<id>": {"acc,none": 0.48, "acc_norm,none": 0.48, ...}}}

The metric key and scaling factor depend on the task type:
- NLU (ArabicMMLU, belebele): "acc,none" x 100 -> accuracy %
- Classification (NLI, dialect, sarcasm): "acc_norm,none" x 100 -> normalized accuracy %
- Generation (translation, summarization): "calculate_bleu,none" x 1 -> BLEU score

Tolerance
---------
- Appendix tables: 0.02 (values reported to 2 decimal places)
- Main table: 0.1 (values reported to 1 decimal place, with rounding accumulation)

Usage
-----
    python scripts/verify_appendex_prompts_results.py
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "evaluation_results"
LATEX_FILE = PROJECT_ROOT / "latex_paper.tex"

# Rounding tolerance for comparing paper values vs JSON values.
# Appendix tables report 2 decimal places, so 0.02 covers rounding.
TOLERANCE = 0.02

# ---------------------------------------------------------------------------
# Model variant directories, ordered to match paper table columns:
#   AceGPT (Base, Chat, Tuned) | LLaMA (Base, Instruct, Tuned) | Qwen (Base, Chat, Tuned)
# ---------------------------------------------------------------------------
MODEL_VARIANTS = [
    "AceGPT-v2-8B",              # Col 1: AceGPT Base
    "AceGPT-v2-8B-Chat",         # Col 2: AceGPT Chat
    "AceGPT-v2-8B-tuned",        # Col 3: AceGPT Tuned
    "Meta-Llama-3.1-8B",         # Col 4: LLaMA Base
    "Meta-Llama-3.1-8B-Instruct",# Col 5: LLaMA Instruct
    "Meta-Llama-3.1-8B-tuned",   # Col 6: LLaMA Tuned
    "Qwen3-8B",                  # Col 7: Qwen Base
    "Qwen3-8B-chat",             # Col 8: Qwen Chat
    "Qwen3-8B-tuned",            # Col 9: Qwen Tuned
]

# ---------------------------------------------------------------------------
# Dataset configurations.
#
# Each entry maps a dataset name to:
#   - task:       subdirectory under evaluation_results/{model}/
#   - dataset:    subdirectory under evaluation_results/{model}/{task}/
#   - prompt_ids: ordered list of 5 Tajeeh prompt IDs matching paper row order
#   - metric:     the key to extract from the JSON results dict
#   - multiplier: scaling factor (100 for accuracy fractions, 1 for BLEU)
#
# Prompt ID orderings sourced from PythonExperiments/src/experiments.py
# ---------------------------------------------------------------------------
DATASETS = {
    # --- NLU (Natural Language Understanding) ---
    "ArabicMMLU": {
        "task": "NLU",
        "dataset": "ArabicMMLU",
        "prompt_ids": [14571, 14869, 14787, 14797, 14798],
        "metric": "acc,none",
        "multiplier": 100,
    },
    "belebele": {
        "task": "NLU",
        "dataset": "belebele",
        "prompt_ids": [14854, 14853, 14801, 14800, 14575],
        "metric": "acc,none",
        "multiplier": 100,
    },
    # --- NLI (Natural Language Inference) ---
    "ArEntail": {
        "task": "NLI",
        "dataset": "ArEntail",
        "prompt_ids": [14581, 14816, 14818, 14819, 14820],
        "metric": "acc_norm,none",
        "multiplier": 100,
    },
    "ArabicTE": {
        "task": "NLI",
        "dataset": "ArabicTE",
        "prompt_ids": [14582, 14673, 14724, 14805, 14855],
        "metric": "acc_norm,none",
        "multiplier": 100,
    },
    # --- Dialect Identification ---
    "AraBench_dev": {
        "task": "dialect_identification",
        "dataset": "AraBench_dev",
        "prompt_ids": [14852, 14850, 14789, 14781, 14561],
        "metric": "acc_norm,none",
        "multiplier": 100,
    },
    "Arabic_Dialects_Dataset": {
        "task": "dialect_identification",
        "dataset": "Arabic_Dialects_Dataset",
        "prompt_ids": [14102, 14783, 14784, 14790, 14851],
        "metric": "acc_norm,none",
        "multiplier": 100,
    },
    # --- Sarcasm Detection ---
    "ArSarcasm_v2": {
        "task": "sarcasm_detection",
        "dataset": "ArSarcasm_v2",
        "prompt_ids": [14779, 14802, 14835, 14837, 14838],
        "metric": "acc_norm,none",
        "multiplier": 100,
    },
    "iSarcasmEval_task_A": {
        "task": "sarcasm_detection",
        "dataset": "iSarcasmEval_task_A",
        "prompt_ids": [14602, 14780, 14859, 14860, 14864],
        "metric": "acc_norm,none",
        "multiplier": 100,
    },
    # --- Machine Translation (BLEU scores, already 0-100 scale) ---
    "opus-100": {
        "task": "machine_translation",
        "dataset": "opus-100",
        "prompt_ids": [14684, 14688, 14680, 14682, 14640],
        "metric": "calculate_bleu,none",
        "multiplier": 1,
    },
    "tatoeba_mt": {
        "task": "machine_translation",
        "dataset": "tatoeba_mt",
        "prompt_ids": [14866, 14867, 14868, 14889, 14890],
        "metric": "calculate_bleu,none",
        "multiplier": 1,
    },
    # --- Summarization (BLEU scores, already 0-100 scale) ---
    "xlsum": {
        "task": "summarization",
        "dataset": "xlsum",
        "prompt_ids": [14871, 14803, 14856, 14858, 14668],
        "metric": "calculate_bleu,none",
        "multiplier": 1,
    },
    "AraSum": {
        "task": "summarization",
        "dataset": "AraSum",
        "prompt_ids": [14891, 14857, 14736, 14735, 14628],
        "metric": "calculate_bleu,none",
        "multiplier": 1,
    },
}

# ---------------------------------------------------------------------------
# LaTeX \\label{} tags for each appendix table, used to locate them in the .tex
# ---------------------------------------------------------------------------
LATEX_TABLE_LABELS = {
    "ArabicMMLU": "tab:appendix-arabicmmlu",
    "belebele": "tab:appendix-belebele",
    "ArEntail": "tab:appendix-arentail",
    "ArabicTE": "tab:appendix-arabicte",
    "AraBench_dev": "tab:appendix-arabench",
    "Arabic_Dialects_Dataset": "tab:appendix-arabic-dialects",
    "ArSarcasm_v2": "tab:appendix-arsarcasm",
    "iSarcasmEval_task_A": "tab:appendix-isarcasmeval",
    "opus-100": "tab:appendix-opus",
    "tatoeba_mt": "tab:appendix-tatoeba",
    "xlsum": "tab:appendix-xlsum",
    "AraSum": "tab:appendix-arasum",
}


# ===========================================================================
# JSON loading helpers
# ===========================================================================

def load_result(model_variant, task, dataset, prompt_id):
    """Load a single lm-evaluation-harness result JSON file.

    Args:
        model_variant: Directory name under evaluation_results/ (e.g. "AceGPT-v2-8B")
        task: Task subdirectory (e.g. "NLU")
        dataset: Dataset subdirectory (e.g. "ArabicMMLU")
        prompt_id: Tajeeh prompt ID (e.g. 14571)

    Returns:
        (metrics_dict, json_path_str) where metrics_dict is the inner dict
        containing metric keys like "acc,none", or (None, path) if file missing.
    """
    json_path = RESULTS_DIR / model_variant / task / dataset / f"prompt_{prompt_id}.json"
    if not json_path.exists():
        return None, str(json_path)
    with open(json_path) as f:
        data = json.load(f)
    results = data.get("results", {})
    # The results dict has a single key like "ArabicMMLU_prompt_14571"
    # containing all metric values for that evaluation run.
    for _task_key, metrics in results.items():
        return metrics, str(json_path)
    return None, str(json_path)


def get_actual_values(model_variant, dataset_name):
    """Get all 5 prompt metric values for a given model variant and dataset.

    Reads the 5 JSON files (one per prompt ID) and extracts the appropriate
    metric, scaled by the dataset's multiplier.

    Args:
        model_variant: Directory name (e.g. "AceGPT-v2-8B-Chat")
        dataset_name: Key in DATASETS dict (e.g. "ArabicMMLU")

    Returns:
        List of 5 floats (or None for missing files), in prompt_ids order.
    """
    cfg = DATASETS[dataset_name]
    values = []
    for pid in cfg["prompt_ids"]:
        metrics, path = load_result(model_variant, cfg["task"], cfg["dataset"], pid)
        if metrics is None:
            values.append(None)
            continue
        metric_key = cfg["metric"]
        if metric_key not in metrics:
            print(f"  WARNING: metric '{metric_key}' not found in {path}")
            print(f"  Available keys: {list(metrics.keys())}")
            values.append(None)
            continue
        values.append(metrics[metric_key] * cfg["multiplier"])
    return values


# ===========================================================================
# LaTeX parsing
# ===========================================================================

def parse_latex_tables(latex_text):
    """Parse all 12 appendix per-prompt tables from the LaTeX source.

    Strategy:
        1. Find each table by its \\label{} tag (e.g. "tab:appendix-arabicmmlu")
        2. Extract text from the label to \\end{tabular}
        3. Use \\midrule as the delimiter to identify data rows
        4. Split each data row on '&' to extract the 9 numerical columns

    Args:
        latex_text: Full content of latex_paper.tex

    Returns:
        tables: dict mapping dataset_name -> list of 5 rows, each row is [9 floats]
        averages: dict mapping dataset_name -> [9 floats] for the Average row
    """
    tables = {}
    averages = {}

    for dataset_name, label in tqdm(LATEX_TABLE_LABELS.items(),
                                     desc="Parsing LaTeX tables",
                                     unit="table"):
        # Step 1: Locate the table by its \label{} tag
        label_pattern = re.escape(label)
        label_match = re.search(label_pattern, latex_text)
        if not label_match:
            print(f"  WARNING: Could not find table label '{label}' in LaTeX")
            continue

        # Step 2: Extract text from label to \end{tabular}
        start_pos = label_match.start()
        end_match = re.search(r'\\end\{tabular\}', latex_text[start_pos:])
        if not end_match:
            print(f"  WARNING: Could not find \\end{{tabular}} for {dataset_name}")
            continue
        table_text = latex_text[start_pos:start_pos + end_match.start()]

        # Step 3: Walk lines, toggling on \midrule to find data rows
        # Table structure:
        #   \midrule          <- first midrule: data rows start after this
        #   1 (Compact) & 48.94 & 56.31 & ... \\
        #   ...
        #   \midrule          <- second midrule: separates data from average
        #   \textbf{Average} & 42.35 & 46.01 & ... \\
        #   \bottomrule
        rows = []
        avg_row = None

        lines = table_text.split('\n')
        in_data = False
        for line in lines:
            line = line.strip()
            if '\\midrule' in line:
                in_data = True
                continue
            if '\\bottomrule' in line:
                break
            if not in_data or not line or line.startswith('%'):
                continue

            # Step 4: Parse rows that have '&' separators and end with '\\'
            if '&' in line and '\\\\' in line:
                line = line.replace('\\\\', '').strip()
                parts = [p.strip() for p in line.split('&')]

                # Distinguish average row from data rows by checking column 0
                if 'Average' in parts[0] or 'textbf' in parts[0]:
                    try:
                        avg_row = [float(p.strip()) for p in parts[1:]]
                    except ValueError as e:
                        print(f"  WARNING: Could not parse average row for {dataset_name}: {e}")
                else:
                    try:
                        nums = [float(p.strip()) for p in parts[1:]]
                        rows.append(nums)
                    except ValueError as e:
                        print(f"  WARNING: Could not parse data row for {dataset_name}: {e}")
                        print(f"    Line: {line}")

        # Sanity checks
        if len(rows) != 5:
            print(f"  WARNING: Expected 5 data rows for {dataset_name}, got {len(rows)}")
        if len(rows) > 0 and len(rows[0]) != 9:
            print(f"  WARNING: Expected 9 columns for {dataset_name}, got {len(rows[0])}")

        tables[dataset_name] = rows
        averages[dataset_name] = avg_row

    return tables, averages


def parse_main_table(latex_text):
    """Parse the main results table (Table 1: tuning effectiveness).

    This table uses a special format with \\multirow for model names and
    r@{$\\,\\pm\\,$}l columns for mean/std pairs. Each row contains:
        Model | Task | mean & std | mean & std | mean & std | d1 | d2

    Strategy:
        1. Find the table by "tab:tuning_effectiveness"
        2. Detect model name from \\multirow lines containing "AceGPT"/"LLaMA"/"Qwen"
        3. Identify task name (MCQ, NLI, etc.) from the row
        4. Extract all numeric values: first 6 are 3x (mean, std), last 2 are Cohen's d

    Args:
        latex_text: Full content of latex_paper.tex

    Returns:
        dict mapping (model_short, task_name) -> {
            "base": (mean, std),
            "chat": (mean, std),
            "tuned": (mean, std),
            "d_base_tuned": float,
            "d_chat_tuned": float,
        }
    """
    label_match = re.search(r'tab:tuning_effectiveness', latex_text)
    if not label_match:
        print("  WARNING: Could not find main results table (tab:tuning_effectiveness)")
        return {}

    start_pos = label_match.start()
    end_match = re.search(r'\\end\{tabular\}', latex_text[start_pos:])
    if not end_match:
        return {}
    table_text = latex_text[start_pos:start_pos + end_match.start()]

    results = {}
    current_model = None

    for line in table_text.split('\n'):
        line = line.strip()
        if not line or '\\midrule' in line or '\\toprule' in line or '\\bottomrule' in line:
            continue
        if 'textbf{Model}' in line or 'cmidrule' in line:
            continue

        # Detect model family from \multirow{6}{*}{AceGPT-v2} etc.
        if 'multirow' in line:
            if 'AceGPT' in line:
                current_model = 'AceGPT-v2'
            elif 'LLaMA' in line:
                current_model = 'LLaMA-3.1'
            elif 'Qwen' in line:
                current_model = 'Qwen3'
            # Strip the \multirow{...}{...} wrapper so we can parse the rest
            line = re.sub(r'\\multirow\{[^}]*\}\{[^}]*\}', '', line).strip()

        if '&' not in line or '\\\\' not in line:
            continue

        line = line.replace('\\\\', '').strip()
        parts = [p.strip() for p in line.split('&')]

        # Identify the task name from the text columns
        task_name = None
        for p in parts:
            p_clean = p.strip()
            if p_clean in ('MCQ', 'NLI', 'Dialect', 'Sarcasm', 'Translation', 'Summarization'):
                task_name = p_clean
                break

        if not task_name or not current_model:
            continue

        # Collect all numeric values from the row
        numeric_parts = []
        for p in parts:
            try:
                numeric_parts.append(float(p.strip()))
            except ValueError:
                pass

        # Expected: 6 values (3 mean/std pairs) + 2 Cohen's d = 8 total
        if len(numeric_parts) >= 8:
            results[(current_model, task_name)] = {
                "base": (numeric_parts[0], numeric_parts[1]),
                "chat": (numeric_parts[2], numeric_parts[3]),
                "tuned": (numeric_parts[4], numeric_parts[5]),
                "d_base_tuned": numeric_parts[6],
                "d_chat_tuned": numeric_parts[7],
            }

    return results


# ===========================================================================
# Comparison logic
# ===========================================================================

def compare_value(actual, paper, context, tolerance=TOLERANCE):
    """Compare a single actual value against its paper counterpart.

    Args:
        actual: Value computed from JSON file (or None if missing)
        paper: Value as reported in the LaTeX table
        context: Human-readable description for error messages
        tolerance: Maximum allowed absolute difference

    Returns:
        A mismatch description string, or None if values match.
    """
    if actual is None:
        return f"  MISSING: {context} - no JSON data found"

    diff = abs(actual - paper)
    if diff > tolerance:
        return f"  MISMATCH: {context}: paper={paper:.2f}, actual={actual:.2f}, diff={diff:.4f}"
    return None


# ===========================================================================
# Verification routines
# ===========================================================================

def verify_appendix_tables():
    """Verify all 12 appendix per-prompt tables (540 values + 108 averages).

    For each dataset:
        1. Parse the corresponding LaTeX table to get paper values
        2. Load all 5x9 JSON results (5 prompts x 9 model variants)
        3. Compare each value within tolerance
        4. Recompute averages from JSON and compare against paper averages

    Returns:
        (all_mismatches, total_checks, total_mismatches, total_missing)
    """
    print("=" * 80)
    print("VERIFYING APPENDIX PER-PROMPT TABLES")
    print("=" * 80)
    print(f"  Datasets: {len(DATASETS)}")
    print(f"  Prompts per dataset: 5")
    print(f"  Model variants: {len(MODEL_VARIANTS)}")
    print(f"  Expected checks: {len(DATASETS) * 5 * len(MODEL_VARIANTS)} per-prompt"
          f" + {len(DATASETS) * len(MODEL_VARIANTS)} averages")

    with open(LATEX_FILE) as f:
        latex_text = f.read()

    print(f"\nStep 1/3: Parsing LaTeX tables...")
    tables, averages = parse_latex_tables(latex_text)

    total_checks = 0
    total_mismatches = 0
    total_missing = 0
    all_mismatches = []

    print(f"\nStep 2/3: Loading JSON results and comparing per-prompt values...")
    for dataset_name in tqdm(DATASETS, desc="Verifying datasets", unit="dataset"):
        if dataset_name not in tables:
            tqdm.write(f"  {dataset_name}: TABLE NOT FOUND IN LATEX")
            continue

        paper_rows = tables[dataset_name]
        paper_avg = averages[dataset_name]
        cfg = DATASETS[dataset_name]

        # Build the actual values matrix: actual_matrix[prompt_idx][col_idx]
        # Each cell is the metric value from the JSON file.
        actual_matrix = []
        for prompt_idx in range(5):
            row = []
            for mv in MODEL_VARIANTS:
                values = get_actual_values(mv, dataset_name)
                row.append(values[prompt_idx])
            actual_matrix.append(row)

        # Compare per-prompt values (5 prompts x 9 variants = 45 checks per dataset)
        dataset_mismatches = []
        for prompt_idx in range(5):
            if prompt_idx >= len(paper_rows):
                tqdm.write(f"  WARNING: {dataset_name} paper has fewer than {prompt_idx + 1} rows")
                continue
            for col_idx, mv in enumerate(MODEL_VARIANTS):
                if col_idx >= len(paper_rows[prompt_idx]):
                    continue
                paper_val = paper_rows[prompt_idx][col_idx]
                actual_val = actual_matrix[prompt_idx][col_idx]
                context = (f"{dataset_name} prompt {prompt_idx+1} "
                           f"(ID {cfg['prompt_ids'][prompt_idx]}) {mv}")
                total_checks += 1

                result = compare_value(actual_val, paper_val, context)
                if result:
                    if "MISSING" in result:
                        total_missing += 1
                    else:
                        total_mismatches += 1
                    dataset_mismatches.append(result)

        # Compare averages (9 checks per dataset)
        if paper_avg:
            for col_idx, mv in enumerate(MODEL_VARIANTS):
                if col_idx >= len(paper_avg):
                    continue
                col_values = [actual_matrix[p][col_idx] for p in range(5)]
                if any(v is None for v in col_values):
                    tqdm.write(f"    SKIP avg for {mv}: missing values")
                    continue
                actual_avg = sum(col_values) / len(col_values)
                paper_avg_val = paper_avg[col_idx]
                context = f"{dataset_name} AVERAGE {mv}"
                total_checks += 1

                result = compare_value(actual_avg, paper_avg_val, context)
                if result:
                    if "MISSING" in result:
                        total_missing += 1
                    else:
                        total_mismatches += 1
                    dataset_mismatches.append(result)

        # Report per-dataset status
        if dataset_mismatches:
            tqdm.write(f"  {dataset_name}: {len(dataset_mismatches)} issue(s) found")
            for m in dataset_mismatches:
                tqdm.write(m)
            all_mismatches.extend(dataset_mismatches)
        else:
            n_values = len(paper_rows) * len(MODEL_VARIANTS)
            n_avgs = len(MODEL_VARIANTS) if paper_avg else 0
            tqdm.write(f"  {dataset_name}: OK ({n_values} values + {n_avgs} averages)")

    print(f"\nStep 3/3: Appendix tables summary")
    print(f"  Total checks: {total_checks}")
    print(f"  Mismatches:   {total_mismatches}")
    print(f"  Missing data: {total_missing}")
    print(f"  Passed:       {total_checks - total_mismatches - total_missing}")

    return all_mismatches, total_checks, total_mismatches, total_missing


def verify_main_table():
    """Verify the main results table (Table 1: mean +/- std).

    This table reports aggregated statistics (mean and standard deviation
    across 5 prompts) for each model-variant-task combination, but only
    for the primary (intra-dataset) datasets.

    The script recomputes mean and std from the 5 per-prompt JSON values
    and compares against the paper. Both population std (ddof=0) and
    sample std (ddof=1) are tried to accommodate different conventions.

    Returns:
        (all_mismatches, total_checks, total_mismatches, 0)
    """
    print(f"\n{'=' * 80}")
    print("VERIFYING MAIN RESULTS TABLE (Table 1: Tuning Effectiveness)")
    print(f"{'=' * 80}")

    with open(LATEX_FILE) as f:
        latex_text = f.read()

    print("  Parsing main table...")
    main_results = parse_main_table(latex_text)

    if not main_results:
        print("  Could not parse main results table")
        return [], 0, 0, 0

    print(f"  Found {len(main_results)} model-task entries")

    # The main table uses only the PRIMARY dataset for each task
    # (the one where training was done, i.e., the intra-dataset evaluation)
    task_to_primary = {
        "MCQ": "ArabicMMLU",
        "NLI": "ArEntail",
        "Dialect": "AraBench_dev",
        "Sarcasm": "ArSarcasm_v2",
        "Translation": "opus-100",
        "Summarization": "xlsum",
    }

    # Map short model names in the table to evaluation_results/ directory names
    model_to_variants = {
        "AceGPT-v2": ("AceGPT-v2-8B", "AceGPT-v2-8B-Chat", "AceGPT-v2-8B-tuned"),
        "LLaMA-3.1": ("Meta-Llama-3.1-8B", "Meta-Llama-3.1-8B-Instruct", "Meta-Llama-3.1-8B-tuned"),
        "Qwen3": ("Qwen3-8B", "Qwen3-8B-chat", "Qwen3-8B-tuned"),
    }

    total_checks = 0
    total_mismatches = 0
    all_mismatches = []

    # Use higher tolerance for the main table since values are rounded to fewer decimals
    main_tolerance = 0.1

    for (model_name, task_name), paper_data in tqdm(sorted(main_results.items()),
                                                     desc="Verifying main table",
                                                     unit="entry"):
        dataset_name = task_to_primary.get(task_name)
        if not dataset_name:
            tqdm.write(f"  WARNING: Unknown task '{task_name}'")
            continue

        base_mv, chat_mv, tuned_mv = model_to_variants[model_name]

        for variant_key, mv in [("base", base_mv), ("chat", chat_mv), ("tuned", tuned_mv)]:
            values = get_actual_values(mv, dataset_name)
            valid_values = [v for v in values if v is not None]

            if len(valid_values) != 5:
                tqdm.write(f"  WARNING: {model_name} {task_name} {variant_key}: "
                           f"only {len(valid_values)}/5 values")
                continue

            actual_mean = np.mean(valid_values)
            actual_std_pop = np.std(valid_values, ddof=0)   # population std

            paper_mean, paper_std = paper_data[variant_key]

            # --- Check mean ---
            context = f"{model_name} {task_name} {variant_key} MEAN"
            total_checks += 1
            result = compare_value(actual_mean, paper_mean, context, tolerance=main_tolerance)
            if result:
                total_mismatches += 1
                all_mismatches.append(result)
                tqdm.write(result)

            # --- Check std ---
            context = f"{model_name} {task_name} {variant_key} STD"
            total_checks += 1
            result = compare_value(actual_std_pop, paper_std, context, tolerance=main_tolerance)
            if result:
                # The paper might use sample std (ddof=1) instead of population std
                actual_std_sample = np.std(valid_values, ddof=1)
                result2 = compare_value(actual_std_sample, paper_std, context,
                                        tolerance=main_tolerance)
                if result2:
                    total_mismatches += 1
                    msg = result + f" (also tried sample std={actual_std_sample:.2f})"
                    all_mismatches.append(msg)
                    tqdm.write(msg)
                else:
                    tqdm.write(f"  NOTE: {context}: matches with ddof=1 (sample std)")

    print(f"\n  Main table summary")
    print(f"  Total checks: {total_checks}")
    print(f"  Mismatches:   {total_mismatches}")
    print(f"  Passed:       {total_checks - total_mismatches}")

    return all_mismatches, total_checks, total_mismatches, 0


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Paper Results Verification Script")
    print("=" * 80)
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Results dir:  {RESULTS_DIR}")
    print(f"  LaTeX file:   {LATEX_FILE}")
    print()

    # Phase 1: Verify appendix tables
    app_mismatches, app_checks, app_mm, app_missing = verify_appendix_tables()

    # Phase 2: Verify main table
    main_mismatches, main_checks, main_mm, _ = verify_main_table()

    # Overall summary
    total_checks = app_checks + main_checks
    total_mismatches = app_mm + main_mm
    total_missing = app_missing

    print(f"\n{'=' * 80}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 80}")
    print(f"  Total checks performed: {total_checks}")
    print(f"  Total mismatches:       {total_mismatches}")
    print(f"  Total missing data:     {total_missing}")
    print(f"  Total passed:           {total_checks - total_mismatches - total_missing}")

    if total_mismatches > 0:
        print(f"\n{'=' * 80}")
        print("ALL MISMATCHES:")
        print(f"{'=' * 80}")
        for m in app_mismatches + main_mismatches:
            print(m)

    if total_mismatches == 0 and total_missing == 0:
        print("\n  All values verified successfully!")

    sys.exit(1 if total_mismatches > 0 else 0)
