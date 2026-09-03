"""Compare strict ADD accuracy against the three choice-ablation variants for
the 3 LoRA-tuned models, aggregated across the 5 ADD prompts.

Sources read:
    strict              evaluation_results/{model}-tuned/dialect_identification/
                        Arabic_Dialects_Dataset/prompt_{id}.json
    explained_full      ablations/dialect_explained_choices/results/explained_full/
                        {model}-tuned/prompt_{id}.json
    explained_text_only ablations/dialect_explained_choices/results/explained_text_only/
                        {model}-tuned/prompt_{id}.json
    arabench_choices    ablations/dialect_explained_choices/results/arabench_choices/
                        {model}-tuned/prompt_{id}.json

Two table sections are printed:
    1. Overall acc_norm (paper-consistent: from lm-eval's `acc_norm,none`).
    2. Per-class breakdown (post-hoc from per-sample logprobs, raw argmax).
       For arabench_choices the per-class block uses the multi-correct
       `accept_choices` logic from the JSON to match the harness Overall.

Run from project root:
    uv run python ablations/dialect_explained_choices/analyze.py
    uv run python ablations/dialect_explained_choices/analyze.py --per-class-metric f1
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

METRIC_CHOICES = ("rec", "prec", "f1", "acc")
METRIC_DISPLAY = {"rec": "recall", "prec": "precision", "f1": "f1", "acc": "accuracy"}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STRICT_DIR = PROJECT_ROOT / "evaluation_results"
ABLATION_RESULTS = PROJECT_ROOT / "ablations" / "dialect_explained_choices" / "results"

PROMPT_IDS = [14102, 14783, 14784, 14790, 14851]

MODELS = {
    "AceGPT-v2": "AceGPT-v2-8B-tuned",
    "LLaMA-3.1": "Meta-Llama-3.1-8B-tuned",
    "Qwen3":     "Qwen3-8B-tuned",
}

CANONICAL_CLASSES = ["MSA", "Levantine", "North African", "Egyptian", "Gulf"]

# Canonicalisation maps. Strict / explained_text_only share the bare ADD labels.
# Explained_full uses the extended noun forms. Arabench_choices uses AraBench labels.
SHORT_TO_CANON = {
    "MSA":          "MSA",
    "Levant":       "Levantine",
    "North Africa": "North African",
    "Egypt":        "Egyptian",
    "GULF":         "Gulf",
}
EXTENDED_TO_CANON = {
    "MSA":                                       "MSA",
    "Levant (e.g., Lebanese)":                   "Levantine",
    "North Africa (e.g., Tunisian, Moroccan)":   "North African",
    "Egypt":                                     "Egyptian",
    "GULF (e.g., Qatari)":                       "Gulf",
}
ARABENCH_TO_CANON = {
    "MSA":      "MSA",
    "Lebanese": "Levantine",
    "Tunisian": "North African",
    "Moroccan": "North African",
    "Qatari":   "Gulf",
    "Egyptian": "Egyptian",
}

# Per source: (canon map for gold/pred, whether to use accept_choices for correctness).
SOURCE_SPEC = {
    "strict":              {"canon": SHORT_TO_CANON,    "multi_accept": False},
    "explained_full":      {"canon": EXTENDED_TO_CANON, "multi_accept": False},
    "explained_text_only": {"canon": SHORT_TO_CANON,    "multi_accept": False},
    "arabench_choices":    {"canon": ARABENCH_TO_CANON, "multi_accept": True},
}

# Display order for the rows of the main table.
SOURCE_ORDER = ["strict", "explained_full", "explained_text_only", "arabench_choices"]


def _result_path(source: str, model_dir: str, prompt_id: int) -> Path:
    if source == "strict":
        return (
            STRICT_DIR
            / model_dir
            / "dialect_identification"
            / "Arabic_Dialects_Dataset"
            / f"prompt_{prompt_id}.json"
        )
    return ABLATION_RESULTS / source / model_dir / f"prompt_{prompt_id}.json"


def _metric_from_counts(tp: int, fp: int, fn: int, tn: int, metric: str) -> float:
    """One-vs-rest per-class metric (%) from binary confusion counts."""
    if metric == "rec":
        denom = tp + fn
        return (tp / denom * 100.0) if denom else float("nan")
    if metric == "prec":
        denom = tp + fp
        return (tp / denom * 100.0) if denom else float("nan")
    if metric == "f1":
        p_denom = tp + fp
        r_denom = tp + fn
        if not p_denom or not r_denom:
            return float("nan")
        p = tp / p_denom
        r = tp / r_denom
        return (2 * p * r / (p + r) * 100.0) if (p + r) else float("nan")
    if metric == "acc":
        denom = tp + fp + fn + tn
        return ((tp + tn) / denom * 100.0) if denom else float("nan")
    raise ValueError(f"unknown metric: {metric}")


def _per_prompt_metrics(source: str, json_path: Path, metric: str) -> dict | None:
    """Returns dict with `_overall_acc_norm` (from harness) and per-class
    `metric` (post-hoc, one-vs-rest on canonical-class predictions), or None if
    the file is missing. Canonicalising both gold and prediction makes the
    multi-target arabench_choices case fall out of the same accounting: e.g.
    pred=`Tunisian` and pred=`Moroccan` both map to `North African`, which is
    equivalent to the old `pred_choice in accept_choices` rule when
    metric=recall."""
    if not json_path.exists():
        return None
    data = json.loads(json_path.read_text())

    samples_dict = data.get("samples", {})
    results_dict = data.get("results", {})
    if not samples_dict or not results_dict:
        return None
    task_key = next(iter(samples_dict))
    samples = samples_dict[task_key]

    spec = SOURCE_SPEC[source]
    canon_map = spec["canon"]
    multi_accept = spec["multi_accept"]

    out: dict = {}

    # Overall: read straight from harness.
    res = results_dict.get(task_key, {})
    acc_norm = res.get("acc_norm,none")
    out["_overall_acc_norm"] = float(acc_norm) * 100.0 if acc_norm is not None else float("nan")

    # Per-class one-vs-rest counters.
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    tn = defaultdict(int)

    for s in samples:
        # Gold coarse class. For multi-target tasks (arabench_choices) the
        # sample's `target` is a list of int indices, so derive the coarse
        # class from doc.accept_choices instead — every entry maps to the
        # same coarse class, so [0] is sufficient.
        if multi_accept:
            accept = s.get("doc", {}).get("accept_choices") or []
            gold_canon = canon_map.get(accept[0]) if accept else None
        else:
            gold_raw = s.get("target")
            gold_canon = canon_map.get(gold_raw.strip()) if isinstance(gold_raw, str) else None
        if gold_canon is None:
            continue

        choices = s.get("doc", {}).get("choices") or []
        resps = s.get("filtered_resps") or []
        if not choices or len(resps) != len(choices):
            continue
        logprobs = [r[0] if isinstance(r, list) else r for r in resps]
        pred_choice = choices[int(np.argmax(logprobs))]
        pred_canon = canon_map.get(pred_choice)

        for cls in CANONICAL_CLASSES:
            is_gold = (gold_canon == cls)
            is_pred = (pred_canon == cls)
            if is_gold and is_pred:
                tp[cls] += 1
            elif is_gold and not is_pred:
                fn[cls] += 1
            elif (not is_gold) and is_pred:
                fp[cls] += 1
            else:
                tn[cls] += 1

    for cls in CANONICAL_CLASSES:
        out[cls] = _metric_from_counts(tp[cls], fp[cls], fn[cls], tn[cls], metric)

    return out


def _aggregate(per_prompt: list[dict], col: str) -> tuple[float, float]:
    vals = [
        p[col] for p in per_prompt
        if p is not None and not np.isnan(p.get(col, float("nan")))
    ]
    if not vals:
        return float("nan"), float("nan")
    m = float(np.mean(vals))
    s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return m, s


def _gather(source: str, model_dir: str, metric: str) -> list[dict | None]:
    return [
        _per_prompt_metrics(source, _result_path(source, model_dir, pid), metric)
        for pid in PROMPT_IDS
    ]


def _fmt(m: float, s: float) -> str:
    if np.isnan(m):
        return "    N/A    "
    return f"{m:5.1f} +- {s:4.1f}"


def _fmt_delta(d: float) -> str:
    if np.isnan(d):
        return "   N/A   "
    return f"{d:+5.1f}    "


def main() -> None:
    parser = argparse.ArgumentParser(description="Choice-taxonomy ablation summary table.")
    parser.add_argument(
        "--per-class-metric",
        choices=METRIC_CHOICES,
        default="f1",
        help="Per-class diagnostic metric (default: f1).",
    )
    args = parser.parse_args()
    metric = args.per_class_metric
    metric_label = METRIC_DISPLAY[metric].upper()

    print("=" * 122)
    print("ABLATION: How does the dialect choice taxonomy affect cross-dataset (AraBench -> ADD) generalisation?")
    print("=" * 122)
    print(
        "Variants:\n"
        "  strict              ADD coarse labels {Levant, North Africa, Egypt, GULF, MSA}\n"
        "                      in both prompt body and choices (main paper).\n"
        "  explained_full      Prompt body AND choices/label extended with AraBench examples\n"
        "                      (e.g. 'GULF (e.g., Qatari)').\n"
        "  explained_text_only Body extended; choices/label remain bare ADD coarse labels.\n"
        "  arabench_choices    Body AND choices rewritten to the 6 AraBench labels\n"
        "                      {MSA, Lebanese, Tunisian, Moroccan, Qatari, Egyptian} so the\n"
        "                      prompt body and the scored options agree (Levant->Lebanese,\n"
        "                      North Africa->Tunisian/Moroccan, GULF->Qatari, Egypt->Egyptian,\n"
        "                      MSA->MSA). Tunisian and Moroccan are BOTH counted correct for\n"
        "                      North Africa via lm-eval's multiple_target mode (see\n"
        "                      eval_harness_tasks/arabench_choices/metrics.py).\n"
    )
    print(
        "Column semantics:\n"
        "  Overall (acc_norm)  Read directly from lm-eval's `acc_norm,none` field. Identical\n"
        "                      pipeline as the paper -> directly comparable across all variants.\n"
        f"  Per-class columns   One-vs-rest {METRIC_DISPLAY[metric]} (%) computed post-hoc from each sample's\n"
        "                      per-choice logprobs (raw argmax, no length normalisation), after\n"
        "                      canonicalising both gold and prediction to the 5 ADD classes.\n"
        "                      Switch via --per-class-metric {rec|prec|f1|acc}.\n"
    )

    # --- Main table: per source, per model -----------------------------------
    print("=" * 122)
    print(f"OVERALL acc_norm (harness) AND PER-CLASS {metric_label} (post-hoc), MEAN +- STD ACROSS 5 PROMPTS")
    print("=" * 122)
    header = (
        f"{'Model':<11s} | {'Source':<20s} | {'Overall':<13s} | "
        f"{'MSA':<13s} | {'Levantine':<13s} | {'N.African':<13s} | "
        f"{'Egyptian':<13s} | {'Gulf':<13s}"
    )
    print(header)
    print("-" * len(header))

    # Cache aggregates per (model_label, source, col) for the delta block below.
    cache: dict[tuple[str, str], list[dict | None]] = {}

    for model_label, model_dir in MODELS.items():
        for source in SOURCE_ORDER:
            per_prompt = _gather(source, model_dir, metric)
            cache[(model_label, source)] = per_prompt
            present = sum(p is not None for p in per_prompt)
            if present == 0:
                print(
                    f"{model_label:<11s} | {source:<20s} | "
                    f"(not yet run; 0/{len(PROMPT_IDS)} JSONs present)"
                )
                continue

            o_m, o_s = _aggregate(per_prompt, "_overall_acc_norm")
            cells = [_fmt(o_m, o_s)]
            for cls in CANONICAL_CLASSES:
                m, s = _aggregate(per_prompt, cls)
                cells.append(_fmt(m, s))
            print(
                f"{model_label:<11s} | {source:<20s} | "
                + " | ".join(c.ljust(13) for c in cells)
            )
        print()

    # --- Delta table: ablation variants minus strict -------------------------
    print("=" * 122)
    print("DELTA: ablation variant minus strict (positive = variant scores higher)")
    print("=" * 122)
    delta_header = (
        f"{'Model':<11s} | {'Variant':<20s} | {'Overall':<13s} | "
        f"{'MSA':<13s} | {'Levantine':<13s} | {'N.African':<13s} | "
        f"{'Egyptian':<13s} | {'Gulf':<13s}"
    )
    print(delta_header)
    print("-" * len(delta_header))

    for model_label in MODELS:
        strict = cache.get((model_label, "strict"))
        if strict is None or all(p is None for p in strict):
            continue
        for source in SOURCE_ORDER:
            if source == "strict":
                continue
            variant = cache.get((model_label, source))
            if variant is None or all(p is None for p in variant):
                continue
            cells = []
            for col in ["_overall_acc_norm"] + CANONICAL_CLASSES:
                sm, _ = _aggregate(strict, col)
                em, _ = _aggregate(variant, col)
                if np.isnan(sm) or np.isnan(em):
                    cells.append("   N/A   ")
                else:
                    cells.append(_fmt_delta(em - sm))
            print(
                f"{model_label:<11s} | {source:<20s} | "
                + " | ".join(c.ljust(13) for c in cells)
            )
        print()


if __name__ == "__main__":
    main()
