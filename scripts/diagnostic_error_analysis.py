"""
Diagnostic Error Analysis for Arabic Instruction Tuning Paper
=============================================================
Extracts failure-mode evidence from lm-evaluation-harness result files

Analyses:
  1. First-choice positional bias in classification tasks
  2. Dialect label taxonomy mismatch (AraBench vs Arabic_Dialects_Dataset)
  3. Sarcasm prompt-level failure (bimodal iSarcasmEval behavior)
  4. Degenerate prediction analysis (models predicting same label for all samples)
  5. Qualitative case studies (concrete failure examples for the paper)
  6. Summary report (compact table for paper inclusion)

Usage:
  python scripts/diagnostic_error_analysis.py all
  python scripts/diagnostic_error_analysis.py 1        # positional_bias_report
  python scripts/diagnostic_error_analysis.py 5        # qualitative_case_studies
"""

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

# ── Configuration ──────────────────────────────────────────────────────────

EVAL_DIR = Path("evaluation_results")

# Models used in the paper (v2 only — drop v1 which is not in the paper)
MODELS_PAPER = {
    "AceGPT-v2": [
        "AceGPT-v2-8B",       # base
        "AceGPT-v2-8B-Chat",  # chat
        "AceGPT-v2-8B-tuned", # tuned
    ],
    "LLaMA-3.1": [
        "Meta-Llama-3.1-8B",          # base
        "Meta-Llama-3.1-8B-Instruct", # instruct
        "Meta-Llama-3.1-8B-tuned",    # tuned
    ],
    "Qwen3": [
        "Qwen3-8B",       # base
        "Qwen3-8B-chat",  # chat
        "Qwen3-8B-tuned", # tuned
    ],
}

ALL_MODELS = [m for variants in MODELS_PAPER.values() for m in variants]

VARIANT_LABEL = {}
for family, variants in MODELS_PAPER.items():
    VARIANT_LABEL[variants[0]] = "base"
    VARIANT_LABEL[variants[1]] = "chat"
    VARIANT_LABEL[variants[2]] = "tuned"

CLASSIFICATION_TASKS = {
    "dialect_identification": ["AraBench_dev", "Arabic_Dialects_Dataset"],
    "sarcasm_detection":      ["ArSarcasm_v2", "iSarcasmEval_task_A"],
    "NLI":                    ["ArEntail", "ArabicTE"],
    "NLU":                    ["ArabicMMLU", "belebele"],
}

PROMPT_IDS = {
    ("dialect_identification", "AraBench_dev"):          [14852, 14850, 14789, 14781, 14561],
    ("dialect_identification", "Arabic_Dialects_Dataset"): [14102, 14783, 14784, 14790, 14851],
    ("sarcasm_detection", "ArSarcasm_v2"):               [14779, 14802, 14835, 14837, 14838],
    ("sarcasm_detection", "iSarcasmEval_task_A"):        [14602, 14780, 14859, 14860, 14864],
    ("NLI", "ArEntail"):                                 [14581, 14816, 14818, 14819, 14820],
    ("NLI", "ArabicTE"):                                 [14582, 14673, 14724, 14805, 14855],
    ("NLU", "ArabicMMLU"):                               [14571, 14869, 14787, 14797, 14798],
    ("NLU", "belebele"):                                 [14854, 14853, 14801, 14800, 14575],
}

# Prompt labels matching the paper appendix (1-indexed)
PROMPT_LABELS = {
    ("dialect_identification", "AraBench_dev"):          ["Geographic", "Direct", "Expert", "Analytical", "Simple"],
    ("dialect_identification", "Arabic_Dialects_Dataset"): ["Uncertainty", "Linguistic", "Cultural", "Hierarchical", "Example"],
    ("sarcasm_detection", "ArSarcasm_v2"):               ["Definition", "Question", "Direct", "Steps", "Indicators"],
    ("sarcasm_detection", "iSarcasmEval_task_A"):        ["Prediction", "Comprehensive", "Direct", "Negative", "Question"],
    ("NLI", "ArEntail"):                                 ["Question", "Concise", "Steps", "Direct", "Game"],
    ("NLI", "ArabicTE"):                                 ["Question", "Contextual", "Structured", "Direct", "Instructional"],
    ("NLU", "ArabicMMLU"):                               ["Compact", "Simple", "Structured", "Choices", "Student"],
    ("NLU", "belebele"):                                 ["Structured", "Context", "Reading", "Compact", "Natural"],
}


# ── Helpers ────────────────────────────────────────────────────────────────

def load_result(model, task, dataset, prompt_id):
    """Load a single evaluation result JSON."""
    path = EVAL_DIR / model / task / dataset / f"prompt_{prompt_id}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def get_task_key(dataset, prompt_id):
    return f"{dataset}_prompt_{prompt_id}"


def predicted_choice_index(sample):
    """Return the index of the highest log-probability answer choice."""
    resps = sample.get("filtered_resps", [])
    if not resps:
        return None
    logprobs = []
    for r in resps:
        if isinstance(r, list) and len(r) >= 1:
            logprobs.append(r[0])
        elif isinstance(r, (int, float)):
            logprobs.append(r)
        else:
            return None
    return int(np.argmax(logprobs)) if logprobs else None


def logprob_margin(sample):
    """Return the gap between the top-1 and top-2 log-probabilities."""
    resps = sample.get("filtered_resps", [])
    logprobs = []
    for r in resps:
        if isinstance(r, list) and len(r) >= 1:
            logprobs.append(r[0])
        elif isinstance(r, (int, float)):
            logprobs.append(r)
    if len(logprobs) < 2:
        return None
    s = sorted(logprobs, reverse=True)
    return s[0] - s[1]


def get_accuracy(data, task_key):
    """Extract accuracy from result dict."""
    res = data.get("results", {}).get(task_key, {})
    acc = res.get("acc_norm,none") or res.get("acc,none")
    return acc * 100 if acc is not None else None


def get_samples(data, task_key):
    return data.get("samples", {}).get(task_key, [])


def truncate_arabic(text, max_len=80):
    """Truncate text for display, respecting word boundaries."""
    if not text or len(text) <= max_len:
        return text or ""
    return text[:max_len].rsplit(" ", 1)[0] + "..."


# ── Positional Bias Report ─────────────────────────────────────────────────

def positional_bias_report():
    """
    For each model-variant, measure how often the model selects the first
    answer choice (index 0). Compare to the expected rate (1/num_choices)
    and to the actual frequency of the correct answer being at index 0.
    """
    print("=" * 90)
    print("ANALYSIS 1: FIRST-CHOICE POSITIONAL BIAS IN CLASSIFICATION TASKS")
    print("=" * 90)
    print()

    for task, datasets in CLASSIFICATION_TASKS.items():
        for dataset in datasets:
            pids = PROMPT_IDS.get((task, dataset), [])
            print(f"── {task} / {dataset} ──")
            print(f"{'Model':<42s} {'Variant':<7s} {'Acc%':>6s} {'Pick[0]%':>8s} "
                  f"{'Gold[0]%':>8s} {'Bias':>6s}  Choice distribution")

            for model in ALL_MODELS:
                total = 0
                choice_counts = Counter()
                gold_at_0 = 0
                num_choices = 0
                accs = []

                for pid in pids:
                    data = load_result(model, task, dataset, pid)
                    if data is None:
                        continue
                    tk = get_task_key(dataset, pid)
                    acc = get_accuracy(data, tk)
                    if acc is not None:
                        accs.append(acc)

                    for s in get_samples(data, tk):
                        idx = predicted_choice_index(s)
                        choices = s.get("doc", {}).get("choices", [])
                        target = s.get("target", "")
                        if idx is None or not choices:
                            continue
                        num_choices = max(num_choices, len(choices))
                        choice_counts[idx] += 1
                        total += 1
                        # Check if gold is at position 0
                        if len(choices) > 0 and choices[0] == target:
                            gold_at_0 += 1

                if total == 0:
                    continue

                avg_acc = np.mean(accs) if accs else 0
                pick0 = choice_counts.get(0, 0) / total * 100
                gold0 = gold_at_0 / total * 100
                bias = pick0 - gold0  # positive = over-selecting first choice

                dist = " ".join(f"[{i}]={choice_counts.get(i,0)/total*100:4.1f}%"
                                for i in range(num_choices))
                variant = VARIANT_LABEL.get(model, "?")
                print(f"  {model:<40s} {variant:<7s} {avg_acc:6.1f} {pick0:8.1f} "
                      f"{gold0:8.1f} {bias:+6.1f}  {dist}")

            print()


# ── Dialect Taxonomy Confusion ─────────────────────────────────────────────

def dialect_taxonomy_confusion():
    """
    Show the answer-choice labels used per prompt for each dialect dataset,
    and build a confusion matrix showing which dialects get confused.
    This reveals whether the cross-dataset performance drop is caused by
    label-set mismatch (AraBench uses 6 classes, Arabic_Dialects uses 5).
    """
    print("=" * 90)
    print("ANALYSIS 2: DIALECT LABEL TAXONOMY & CONFUSION ANALYSIS")
    print("=" * 90)
    print()

    for dataset in ["AraBench_dev", "Arabic_Dialects_Dataset"]:
        pids = PROMPT_IDS.get(("dialect_identification", dataset), [])
        labels = PROMPT_LABELS.get(("dialect_identification", dataset), [])
        print(f"── {dataset} ──")

        # Show choice labels per prompt
        print("  Answer choices per prompt:")
        for i, pid in enumerate(pids):
            for m in ALL_MODELS:
                data = load_result(m, "dialect_identification", dataset, pid)
                if data is None:
                    continue
                tk = get_task_key(dataset, pid)
                samples = get_samples(data, tk)
                if samples:
                    choices = samples[0].get("doc", {}).get("choices", [])
                    lbl = labels[i] if i < len(labels) else "?"
                    print(f"    P{i+1} ({lbl}, id={pid}): {choices}")
                    break

        # Per-model confusion
        print()
        for model in ALL_MODELS:
            variant = VARIANT_LABEL.get(model, "?")
            confusion = defaultdict(Counter)  # gold -> pred -> count
            total = correct = 0

            for pid in pids:
                data = load_result(model, "dialect_identification", dataset, pid)
                if data is None:
                    continue
                tk = get_task_key(dataset, pid)
                for s in get_samples(data, tk):
                    choices = s.get("doc", {}).get("choices", [])
                    target = s.get("target", "")
                    idx = predicted_choice_index(s)
                    if idx is not None and idx < len(choices):
                        pred = choices[idx]
                        confusion[target][pred] += 1
                        total += 1
                        if pred == target:
                            correct += 1

            if total == 0:
                continue

            acc = correct / total * 100
            print(f"  {model} ({variant}) — overall: {acc:.1f}%")

            all_labels = sorted(confusion.keys())
            for gold in all_labels:
                preds = confusion[gold]
                n = sum(preds.values())
                top3 = preds.most_common(3)
                top_str = ", ".join(
                    f"{p}={c}({c/n*100:.0f}%)" for p, c in top3
                )
                gold_acc = preds.get(gold, 0) / n * 100
                print(f"    {gold:<20s} acc={gold_acc:5.1f}%  top preds: {top_str}")
            print()


# ── Sarcasm Bimodal Failure ────────────────────────────────────────────────

def sarcasm_bimodal_failure():
    """
    The iSarcasmEval dataset shows bimodal behavior: some prompts yield ~86%
    accuracy (always predicting majority class) while others yield ~14%.
    This analysis shows the per-prompt prediction distributions to explain
    why: prompts with inverted semantics ("is NOT sarcastic: true/false")
    flip the model's default toward the wrong class.
    """
    print("=" * 90)
    print("ANALYSIS 3: SARCASM DETECTION — BIMODAL PROMPT FAILURE")
    print("=" * 90)
    print()

    for dataset in ["ArSarcasm_v2", "iSarcasmEval_task_A"]:
        pids = PROMPT_IDS.get(("sarcasm_detection", dataset), [])
        labels = PROMPT_LABELS.get(("sarcasm_detection", dataset), [])
        print(f"── {dataset} ──")

        # Show choice labels per prompt
        print("  Answer choices per prompt:")
        for i, pid in enumerate(pids):
            for m in ALL_MODELS:
                data = load_result(m, "sarcasm_detection", dataset, pid)
                if data:
                    tk = get_task_key(dataset, pid)
                    samples = get_samples(data, tk)
                    if samples:
                        choices = samples[0].get("doc", {}).get("choices", [])
                        lbl = labels[i] if i < len(labels) else "?"
                        print(f"    P{i+1} ({lbl}, id={pid}): {choices}")
                        break

        # Per-model per-prompt accuracy table
        print()
        header_parts = [f"  {'Model':<40s} {'Var':<6s}"]
        for i in range(len(pids)):
            lbl = labels[i][:6] if i < len(labels) else f"P{i+1}"
            header_parts.append(f"{lbl:>7s}")
        header_parts += [f"{'AVG':>7s}", f"{'STD':>6s}"]
        print(" ".join(header_parts))

        for model in ALL_MODELS:
            variant = VARIANT_LABEL.get(model, "?")
            row = [f"  {model:<40s} {variant:<6s}"]
            accs = []
            for pid in pids:
                data = load_result(model, "sarcasm_detection", dataset, pid)
                if data is None:
                    row.append(f"{'N/A':>7s}")
                    continue
                tk = get_task_key(dataset, pid)
                acc = get_accuracy(data, tk)
                if acc is not None:
                    accs.append(acc)
                    row.append(f"{acc:7.1f}")
                else:
                    row.append(f"{'N/A':>7s}")
            avg = np.mean(accs) if accs else 0
            std = np.std(accs, ddof=1) if len(accs) > 1 else 0
            row += [f"{avg:7.1f}", f"{std:6.1f}"]
            print(" ".join(row))

        # For iSarcasmEval: show prediction distribution per prompt for one model family
        if dataset == "iSarcasmEval_task_A":
            print("\n  Prediction distribution per prompt (LLaMA-3.1 family):")
            for model in MODELS_PAPER["LLaMA-3.1"]:
                variant = VARIANT_LABEL.get(model, "?")
                print(f"\n    {model} ({variant}):")
                for i, pid in enumerate(pids):
                    data = load_result(model, "sarcasm_detection", dataset, pid)
                    if data is None:
                        continue
                    tk = get_task_key(dataset, pid)
                    acc = get_accuracy(data, tk)
                    samples = get_samples(data, tk)

                    pred_counts = Counter()
                    for s in samples:
                        idx = predicted_choice_index(s)
                        choices = s.get("doc", {}).get("choices", [])
                        if idx is not None and idx < len(choices):
                            pred_counts[choices[idx]] += 1

                    total = sum(pred_counts.values())
                    dist = ", ".join(
                        f"{k}={v}({v/total*100:.1f}%)" for k, v in pred_counts.most_common()
                    ) if total > 0 else "no predictions"
                    lbl = labels[i] if i < len(labels) else f"P{i+1}"
                    acc_str = f"{acc:.1f}%" if acc is not None else "N/A"
                    print(f"      P{i+1} ({lbl}): acc={acc_str:<7s} preds: {dist}")

        print()


# ── Degenerate Prediction Scan ─────────────────────────────────────────────

def degenerate_prediction_scan():
    """
    Detect cases where a model outputs the same label for (nearly) all
    samples — a sign it is not actually performing the task.
    Threshold: any single label predicted ≥ 90% of the time.
    """
    print("=" * 90)
    print("ANALYSIS 4: DEGENERATE PREDICTIONS (SINGLE-LABEL DOMINANCE)")
    print("=" * 90)
    print()

    print(f"  {'Task':<25s} {'Dataset':<30s} {'Model':<40s} {'Var':<6s} "
          f"{'Prompt':<12s} {'Acc%':>6s} {'DomLabel':<20s} {'Dom%':>6s}")
    print("  " + "-" * 160)

    for task, datasets in CLASSIFICATION_TASKS.items():
        for dataset in datasets:
            pids = PROMPT_IDS.get((task, dataset), [])
            labels = PROMPT_LABELS.get((task, dataset), [])

            for model in ALL_MODELS:
                variant = VARIANT_LABEL.get(model, "?")
                for i, pid in enumerate(pids):
                    data = load_result(model, task, dataset, pid)
                    if data is None:
                        continue
                    tk = get_task_key(dataset, pid)
                    acc = get_accuracy(data, tk)
                    samples = get_samples(data, tk)

                    pred_counts = Counter()
                    for s in samples:
                        idx = predicted_choice_index(s)
                        choices = s.get("doc", {}).get("choices", [])
                        if idx is not None and idx < len(choices):
                            pred_counts[choices[idx]] += 1

                    total = sum(pred_counts.values())
                    if total == 0:
                        continue

                    top_label, top_count = pred_counts.most_common(1)[0]
                    dom_pct = top_count / total * 100

                    if dom_pct >= 90.0:
                        lbl = labels[i] if i < len(labels) else f"P{i+1}"
                        acc_str = f"{acc:.1f}" if acc is not None else "N/A"
                        print(f"  {task:<25s} {dataset:<30s} {model:<40s} "
                              f"{variant:<6s} P{i+1}({lbl:<6s}) {acc_str:>6s} "
                              f"{top_label:<20s} {dom_pct:6.1f}")

    print()


# ── Qualitative Case Studies ───────────────────────────────────────────────

def qualitative_case_studies():
    """
    For each classification task, find the worst-performing prompt for a
    base/chat model and print concrete examples: the input text, gold label,
    predicted label, and log-prob distribution across choices.
    """
    print("=" * 90)
    print("ANALYSIS 5: QUALITATIVE CASE STUDIES — CONCRETE FAILURE EXAMPLES")
    print("=" * 90)
    print()

    NUM_EXAMPLES = 5  # examples per case study

    for task, datasets in CLASSIFICATION_TASKS.items():
        for dataset in datasets:
            pids = PROMPT_IDS.get((task, dataset), [])
            labels = PROMPT_LABELS.get((task, dataset), [])
            print(f"── {task} / {dataset} ──")

            # Find the worst (model, prompt) combination among base/chat variants
            worst_acc = 999
            worst_model = worst_pid = worst_pidx = None

            for model in ALL_MODELS:
                if VARIANT_LABEL.get(model) == "tuned":
                    continue  # skip tuned for failure analysis
                for i, pid in enumerate(pids):
                    data = load_result(model, task, dataset, pid)
                    if data is None:
                        continue
                    tk = get_task_key(dataset, pid)
                    acc = get_accuracy(data, tk)
                    if acc is not None and acc < worst_acc:
                        worst_acc = acc
                        worst_model = model
                        worst_pid = pid
                        worst_pidx = i

            if worst_model is None:
                print("  No data available.\n")
                continue

            data = load_result(worst_model, task, dataset, worst_pid)
            tk = get_task_key(dataset, worst_pid)
            samples = get_samples(data, tk)
            variant = VARIANT_LABEL.get(worst_model, "?")
            lbl = labels[worst_pidx] if worst_pidx < len(labels) else f"P{worst_pidx+1}"

            print(f"  Worst case: {worst_model} ({variant}), "
                  f"Prompt P{worst_pidx+1} ({lbl}, id={worst_pid}), acc={worst_acc:.1f}%")
            print()

            # Also find the BEST tuned model on same prompt for comparison
            best_tuned_model = None
            best_tuned_acc = -1
            for model in ALL_MODELS:
                if VARIANT_LABEL.get(model) != "tuned":
                    continue
                data_t = load_result(model, task, dataset, worst_pid)
                if data_t is None:
                    continue
                tk_t = get_task_key(dataset, worst_pid)
                acc_t = get_accuracy(data_t, tk_t)
                if acc_t is not None and acc_t > best_tuned_acc:
                    best_tuned_acc = acc_t
                    best_tuned_model = model

            if best_tuned_model:
                print(f"  Best tuned on same prompt: {best_tuned_model}, "
                      f"acc={best_tuned_acc:.1f}%")
                print()

            # Print concrete failure examples
            if not samples:
                print("  No samples found.\n")
                continue

            # Get misclassified samples
            failures = []
            for s in samples:
                choices = s.get("doc", {}).get("choices", [])
                target = s.get("target", "")
                idx = predicted_choice_index(s)
                if idx is None or idx >= len(choices):
                    continue
                pred = choices[idx]
                if pred != target:
                    # Get logprobs for each choice
                    resps = s.get("filtered_resps", [])
                    lps = []
                    for r in resps:
                        if isinstance(r, list) and r:
                            lps.append(r[0])
                        elif isinstance(r, (int, float)):
                            lps.append(r)
                        else:
                            lps.append(None)

                    # Get the text field — try common column names
                    doc = s.get("doc", {})
                    text = (doc.get("arabic") or doc.get("text") or
                            doc.get("tweet") or doc.get("premise") or
                            doc.get("Question") or doc.get("flores_passage") or
                            doc.get("Text") or "")

                    failures.append({
                        "text": text,
                        "gold": target,
                        "pred": pred,
                        "choices": choices,
                        "logprobs": lps,
                        "margin": logprob_margin(s),
                    })

            # Show a few diverse failures
            shown = 0
            for f in failures[:NUM_EXAMPLES]:
                shown += 1
                print(f"  Example {shown}:")
                print(f"    Text:    {truncate_arabic(f['text'], 120)}")
                print(f"    Gold:    {f['gold']}")
                print(f"    Pred:    {f['pred']}")
                lp_str = "  ".join(
                    f"{f['choices'][j]}={f['logprobs'][j]:.2f}"
                    if j < len(f['logprobs']) and f['logprobs'][j] is not None
                    else f"{f['choices'][j]}=?"
                    for j in range(len(f['choices']))
                )
                print(f"    LogProb: {lp_str}")
                if f["margin"] is not None:
                    print(f"    Margin:  {f['margin']:.3f}")
                print()

            # Categorize all failures
            n_total = len(samples)
            n_fail = len(failures)
            n_correct = n_total - n_fail

            # Failure categories
            positional_bias = sum(1 for f in failures
                                  if f["choices"].index(f["pred"]) == 0
                                  if f["pred"] in f["choices"])
            low_margin = sum(1 for f in failures
                             if f["margin"] is not None and f["margin"] < 0.5)
            degenerate = 0
            pred_counter = Counter(f["pred"] for f in failures)
            if pred_counter:
                top_pred, top_count = pred_counter.most_common(1)[0]
                if top_count / max(len(failures), 1) > 0.8:
                    degenerate = top_count

            print(f"  Failure breakdown ({n_fail}/{n_total} = {n_fail/max(n_total,1)*100:.1f}% errors):")
            print(f"    Predicted first choice:      {positional_bias}/{n_fail} "
                  f"({positional_bias/max(n_fail,1)*100:.1f}%)")
            print(f"    Low confidence (margin<0.5): {low_margin}/{n_fail} "
                  f"({low_margin/max(n_fail,1)*100:.1f}%)")
            if degenerate:
                print(f"    Degenerate (>{80}% same pred): {degenerate}/{n_fail} "
                      f"— always predicts '{top_pred}'")
            print()

            # If we have tuned data, show same examples with tuned model
            if best_tuned_model:
                data_t = load_result(best_tuned_model, task, dataset, worst_pid)
                if data_t:
                    tk_t = get_task_key(dataset, worst_pid)
                    samples_t = get_samples(data_t, tk_t)
                    # Build a lookup by text
                    tuned_preds = {}
                    for s in samples_t:
                        doc = s.get("doc", {})
                        text = (doc.get("arabic") or doc.get("text") or
                                doc.get("tweet") or doc.get("premise") or
                                doc.get("Question") or doc.get("flores_passage") or
                                doc.get("Text") or "")
                        idx = predicted_choice_index(s)
                        choices = s.get("doc", {}).get("choices", [])
                        if idx is not None and idx < len(choices):
                            tuned_preds[text[:80]] = choices[idx]

                    print(f"  Same examples with tuned model ({best_tuned_model}):")
                    for i, f in enumerate(failures[:NUM_EXAMPLES]):
                        key = f["text"][:80]
                        t_pred = tuned_preds.get(key, "?")
                        correct_now = "✓" if t_pred == f["gold"] else "✗"
                        print(f"    Ex {i+1}: gold={f['gold']:<15s} "
                              f"base_pred={f['pred']:<15s} "
                              f"tuned_pred={t_pred:<15s} {correct_now}")
                    print()

            print()


# ── Paper Summary Report ──────────────────────────────────────────────────

def paper_summary_report():
    """
    Compact summary tables suitable for paper inclusion.
    """
    print("=" * 90)
    print("ANALYSIS 6: SUMMARY REPORT FOR PAPER")
    print("=" * 90)
    print()

    # Table: Per-task, per-variant accuracy with prompt range
    print("── Accuracy Summary: mean (min–max) across 5 prompts ──")
    print()
    print(f"  {'Task':<20s} {'Dataset':<28s} ", end="")
    for family in MODELS_PAPER:
        print(f"{'base':>16s} {'chat':>16s} {'tuned':>16s} ", end="")
    print()
    print("  " + "-" * 170)

    for task, datasets in CLASSIFICATION_TASKS.items():
        for dataset in datasets:
            pids = PROMPT_IDS.get((task, dataset), [])
            row = f"  {task:<20s} {dataset:<28s} "

            for family, models in MODELS_PAPER.items():
                for model in models:
                    accs = []
                    for pid in pids:
                        data = load_result(model, task, dataset, pid)
                        if data is None:
                            continue
                        tk = get_task_key(dataset, pid)
                        acc = get_accuracy(data, tk)
                        if acc is not None:
                            accs.append(acc)

                    if accs:
                        avg = np.mean(accs)
                        lo = min(accs)
                        hi = max(accs)
                        row += f"{avg:5.1f}({lo:4.1f}-{hi:4.1f}) "
                    else:
                        row += f"{'N/A':>16s} "

            print(row)

    # Failure mode summary
    print()
    print("── Dominant Failure Modes by Task ──")
    print()

    failure_modes = {
        "MCQ (base/chat)": "Positional bias — models default to first answer choice "
                           "(pick rate ~30-50% vs expected 20-25%)",
        "NLI (base)": "Degenerate predictions — base models predict same class for "
                      "all samples (acc ≈ 50% = majority baseline)",
        "Dialect (base/chat)": "Positional bias + label ignorance — models pick first "
                               "choice regardless of content (acc ≈ 13-20% ≈ 1/6 random)",
        "Dialect (cross-dataset)": "Label taxonomy mismatch — AraBench uses 6 classes "
                                    "(MSA, Tunisian, Moroccan, Qatari, Egyptian, Lebanese), "
                                    "Arabic Dialects uses 5 classes (MSA, Levant, Gulf, "
                                    "Egypt, North Africa). Tuned models cannot map between them.",
        "Sarcasm (iSarcasmEval)": "Prompt semantics inversion — Prompts 4-5 use inverted "
                                   "phrasing ('is NOT sarcastic: true/false') causing models "
                                   "to flip predictions. 3 prompts → ~86% (majority class), "
                                   "2 prompts → ~14% (inverted majority class).",
    }

    for mode, explanation in failure_modes.items():
        print(f"  {mode}:")
        print(f"    {explanation}")
        print()


# ── Main ──────────────────────────────────────────────────────────────────

ANALYSES = {
    "1": ("First-Choice Positional Bias", positional_bias_report),
    "2": ("Dialect Label Taxonomy", dialect_taxonomy_confusion),
    "3": ("Sarcasm Bimodal Failure", sarcasm_bimodal_failure),
    "4": ("Degenerate Predictions", degenerate_prediction_scan),
    "5": ("Qualitative Case Studies", qualitative_case_studies),
    "6": ("Summary Report", paper_summary_report),
}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"

    if which == "all":
        for key in sorted(ANALYSES.keys()):
            name, fn = ANALYSES[key]
            print(f"\n{'#' * 90}")
            print(f"# Running Analysis {key}: {name}")
            print(f"{'#' * 90}\n")
            fn()
    elif which in ANALYSES:
        name, fn = ANALYSES[which]
        fn()
    else:
        print(f"Unknown analysis: {which}")
        print(f"Available: {', '.join(sorted(ANALYSES.keys()))}, all")
        sys.exit(1)