"""
Per-gold-class accuracy on Arabic Dialects Dataset (ADD) for tuned models,
aggregated as mean +/- std across the 5 ADD prompts.

Hypothesis (Reviewer Issue 4): models tuned on AraBench should achieve
substantially higher accuracy on ADD-gold classes whose label has an exact
match in the AraBench training taxonomy (Egyptian, MSA) than on classes
whose ADD label is a coarse grouping with no direct match (Levantine,
North African, Gulf). A large matched-vs-unmatched gap supports the
taxonomy-mismatch claim in Section 5.4.

Output: prints to stdout. Does not write any file.

Usage (from project root):
    python scripts/dialects_label_mismatch_analysis.py
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

EVAL_DIR = Path("evaluation_results")

MODELS_TUNED = {
    "AceGPT-v2": "AceGPT-v2-8B-tuned",
    "LLaMA-3.1": "Meta-Llama-3.1-8B-tuned",
    "Qwen3":     "Qwen3-8B-tuned",
}

# ADD prompt IDs (matching scripts/statistical_analysis.py)
PROMPT_IDS = [14102, 14783, 14784, 14790, 14851]

# Canonicalize prompt-specific surface strings to a common label space
LABEL_CANON = {
    "MSA":            "MSA",
    "msa":            "MSA",
    "Levant":         "Levantine",
    "Levantine":      "Levantine",
    "North Africa":   "North African",
    "North African":  "North African",
    "Egypt":          "Egyptian",
    "Egyptian":       "Egyptian",
    "GULF":           "Gulf",
    "Gulf":           "Gulf",
}

CANONICAL_CLASSES = ["MSA", "Levantine", "North African", "Egyptian", "Gulf"]
MATCHED = {"MSA", "Egyptian"}             # exact match in AraBench training
UNMATCHED = {"Levantine", "North African", "Gulf"}  # coarse groupings, no direct match


def canonicalize(label):
    if label is None:
        return None
    return LABEL_CANON.get(label.strip(), label.strip())


def analyze_prompt(model_dir, pid):
    """Compute per-canonical-class accuracy for one (model, prompt) cell.

    Returns a dict mapping each column name to a float percent, or None if
    the JSON is missing.
    """
    path = EVAL_DIR / model_dir / "dialect_identification" / "Arabic_Dialects_Dataset" / f"prompt_{pid}.json"
    if not path.exists():
        return None

    with open(path) as fp:
        data = json.load(fp)

    samples_dict = data.get("samples", {})
    if not samples_dict:
        return None
    task_key = next(iter(samples_dict))
    samples = samples_dict[task_key]

    correct = defaultdict(int)
    total = defaultdict(int)

    for s in samples:
        gold_raw = s.get("target")
        gold = canonicalize(gold_raw)
        if gold is None:
            continue

        choices = s.get("doc", {}).get("choices") or []
        resps = s.get("filtered_resps") or []
        if not choices or len(resps) != len(choices):
            continue

        logprobs = [r[0] if isinstance(r, list) else r for r in resps]
        pred_idx = int(np.argmax(logprobs))
        pred = canonicalize(choices[pred_idx])

        total[gold] += 1
        if pred == gold:
            correct[gold] += 1

    result = {}
    for cls in CANONICAL_CLASSES:
        if total[cls] > 0:
            result[cls] = correct[cls] / total[cls] * 100.0
        else:
            result[cls] = float("nan")

    matched_accs = [result[c] for c in MATCHED if not np.isnan(result[c])]
    unmatched_accs = [result[c] for c in UNMATCHED if not np.isnan(result[c])]
    result["Matched"] = float(np.mean(matched_accs)) if matched_accs else float("nan")
    result["Unmatched"] = float(np.mean(unmatched_accs)) if unmatched_accs else float("nan")

    # Harness-reported overall accuracy for this prompt (sanity check vs paper)
    res_dict = data.get("results", {}).get(task_key, {})
    overall = res_dict.get("acc_norm,none")
    if overall is None:
        overall = res_dict.get("acc,none")
    result["Overall"] = float(overall) * 100.0 if overall is not None else float("nan")

    # Also store per-class sample counts (for reporting context)
    result["_counts"] = {cls: total[cls] for cls in CANONICAL_CLASSES}
    return result


def aggregate(per_prompt_results, columns):
    """Across the 5 prompts, compute mean and sample std for each column."""
    out = {}
    for col in columns:
        vals = [r[col] for r in per_prompt_results if r is not None and not np.isnan(r[col])]
        if vals:
            out[col] = (float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
        else:
            out[col] = (float("nan"), float("nan"))
    return out


def fmt_cell(mean_std):
    m, s = mean_std
    if np.isnan(m):
        return "      N/A      "
    return f"{m:6.1f} +- {s:4.1f}"


def per_prompt_position_check():
    """For each ADD prompt, print the answer-choice ordering and the per-model
    Gulf-class recall. If Gulf is at the same position across all 5 prompts but
    has varying recall across models, the data is consistent with semantic
    preference rather than positional bias. If Gulf's position varies across
    prompts, we can correlate position with recall directly.
    """
    print("\n" + "=" * 110)
    print("POSITIONAL CHECK: Gulf-class recall vs. Gulf's position in answer_choices, per prompt")
    print("=" * 110)
    print(f"{'Prompt':<10s} {'Choices order':<60s} {'Gulf pos':>9s} "
          f"{'AceGPT %':>10s} {'LLaMA %':>10s} {'Qwen3 %':>10s}")
    print("-" * 110)

    for pid in PROMPT_IDS:
        choices_seen = None
        gulf_recalls = {}
        for model_label, model_dir in MODELS_TUNED.items():
            path = EVAL_DIR / model_dir / "dialect_identification" / "Arabic_Dialects_Dataset" / f"prompt_{pid}.json"
            if not path.exists():
                continue
            with open(path) as fp:
                data = json.load(fp)
            samples_dict = data.get("samples", {})
            if not samples_dict:
                continue
            task_key = next(iter(samples_dict))
            samples = samples_dict[task_key]

            # Capture choices ordering once (first sample, since constant per prompt)
            if choices_seen is None and samples:
                choices_seen = samples[0].get("doc", {}).get("choices") or []

            # Compute Gulf-recall: among gold==Gulf samples, fraction predicted Gulf
            gulf_correct = 0
            gulf_total = 0
            for s in samples:
                gold = canonicalize(s.get("target"))
                if gold != "Gulf":
                    continue
                gulf_total += 1
                resps = s.get("filtered_resps") or []
                choices = s.get("doc", {}).get("choices") or []
                if not resps or len(resps) != len(choices):
                    continue
                logprobs = [r[0] if isinstance(r, list) else r for r in resps]
                pred = canonicalize(choices[int(np.argmax(logprobs))])
                if pred == "Gulf":
                    gulf_correct += 1
            gulf_recalls[model_label] = (gulf_correct / gulf_total * 100.0) if gulf_total else float("nan")

        # Determine Gulf's position in canonical terms
        if choices_seen:
            canon_seen = [canonicalize(c) for c in choices_seen]
            gulf_idx = canon_seen.index("Gulf") if "Gulf" in canon_seen else -1
            gulf_pos_str = f"{gulf_idx + 1}/{len(canon_seen)}" if gulf_idx >= 0 else "N/A"
            order_str = ", ".join(choices_seen)
        else:
            gulf_pos_str = "N/A"
            order_str = "(no data)"

        ace = gulf_recalls.get("AceGPT-v2", float("nan"))
        lla = gulf_recalls.get("LLaMA-3.1", float("nan"))
        qwn = gulf_recalls.get("Qwen3", float("nan"))
        ace_s = f"{ace:8.1f}" if not np.isnan(ace) else "    N/A"
        lla_s = f"{lla:8.1f}" if not np.isnan(lla) else "    N/A"
        qwn_s = f"{qwn:8.1f}" if not np.isnan(qwn) else "    N/A"
        print(f"{pid:<10d} {order_str[:60]:<60s} {gulf_pos_str:>9s} "
              f"{ace_s:>10s} {lla_s:>10s} {qwn_s:>10s}")
    print()


def main():
    columns = CANONICAL_CLASSES + ["Matched", "Unmatched", "Overall"]
    print()
    print("=" * 110)
    print("PER-GOLD-CLASS ACCURACY ON ARABIC DIALECTS DATASET (ADD), TUNED MODELS")
    print(f"Mean +/- sample std (ddof=1) across {len(PROMPT_IDS)} ADD prompts.")
    print("'Matched' = mean of (Egyptian, MSA): labels with exact match in AraBench training.")
    print("'Unmatched' = mean of (Levantine, North African, Gulf): coarse groupings.")
    print("'Overall'  = harness-reported acc_norm (micro), for sanity check vs paper Table.")
    print("=" * 110)

    # Print per-class sample counts (constant across models/prompts)
    sample_counts = None
    for model_dir in MODELS_TUNED.values():
        for pid in PROMPT_IDS:
            r = analyze_prompt(model_dir, pid)
            if r is not None:
                sample_counts = r["_counts"]
                break
        if sample_counts is not None:
            break
    if sample_counts is not None:
        print("\nADD test set per-gold-class sample counts:")
        total_n = sum(sample_counts.values())
        for cls in CANONICAL_CLASSES:
            n = sample_counts[cls]
            pct = (n / total_n * 100.0) if total_n else 0.0
            print(f"  {cls:<14s} n = {n:5d} ({pct:5.1f}%)")
        print(f"  {'Total':<14s} n = {total_n:5d}")
    print()

    # Build the table
    header = f"{'Model':<11s}"
    for col in columns:
        header += f" | {col:<15s}"
    print(header)
    print("-" * len(header))

    for model_label, model_dir in MODELS_TUNED.items():
        per_prompt = []
        for pid in PROMPT_IDS:
            r = analyze_prompt(model_dir, pid)
            if r is not None:
                per_prompt.append(r)
        if not per_prompt:
            print(f"{model_label:<11s} (no data)")
            continue
        agg = aggregate(per_prompt, columns)
        row = f"{model_label:<11s}"
        for col in columns:
            row += f" | {fmt_cell(agg[col])}"
        print(row)

    per_prompt_position_check()

    print("=" * 110)
    print("END")
    print("=" * 110)


if __name__ == "__main__":
    main()
