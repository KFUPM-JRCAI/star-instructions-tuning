"""Build modified ADD parquets + YAML configs for the dialect choice-ablation.

Three variants are supported, selected via --variant:

  explained_full       Rewrite both the prompt body AND the choices/label columns to the
                       extended forms (e.g. "GULF" -> "GULF (e.g., Qatari)"). This is
                       the original ablation: the model sees the hierarchy in the body
                       AND is scored against the extended strings.

  explained_text_only  Same body rewrites as explained_full, but choices/label remain
                       the bare ADD labels. Isolates the effect of seeing the hierarchy
                       in the prompt body vs. seeing it in the scored strings.

  arabench_choices     Leave the prompt body untouched (strict ADD form). Replace the
                       choices column with the 6 AraBench labels:
                           [MSA, Lebanese, Tunisian, Moroccan, Qatari, Egyptian]
                       Each row's `label` is set to a single canonical AraBench label per
                       coarse ADD class (the b1 convention; only a placeholder, real
                       scoring is done via the custom process_results). An extra
                       `accept_choices` column lists every AraBench label that should
                       count as correct for that row (Tunisian + Moroccan both map to
                       North Africa). Multi-correct acc/acc_norm is computed by the
                       custom `metrics.acc_norm_accept` referenced from the YAML; the
                       result JSON's `acc_norm,none` field is therefore directly
                       comparable to the other two variants and to the main paper.

Run from project root:
    uv run python ablations/dialect_explained_choices/build_data.py --variant all
    uv run python ablations/dialect_explained_choices/build_data.py --variant explained_full
    uv run python ablations/dialect_explained_choices/build_data.py --variant explained_text_only
    uv run python ablations/dialect_explained_choices/build_data.py --variant arabench_choices
"""
import argparse
import re
from pathlib import Path

import pandas as pd

# ----------------------------------------------------------------------------
# Paths and constants
# ----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ABLATION_ROOT = PROJECT_ROOT / "ablations" / "dialect_explained_choices"
SRC_DIR = PROJECT_ROOT / "experimental_hf_datasets" / "Arabic_Dialects_Dataset"
DATA_ROOT = ABLATION_ROOT / "data"
YAML_ROOT = ABLATION_ROOT / "eval_harness_tasks"

PROMPT_IDS = [14102, 14783, 14784, 14790, 14851]

# Prompt 14790 uses adjective-form labels in its rendered options list; the
# other four use noun-form labels matching the choices column.
ADJ_PROMPT_ID = 14790

VARIANTS = ("explained_full", "explained_text_only", "arabench_choices")

# ----------------------------------------------------------------------------
# Label mappings
# ----------------------------------------------------------------------------

# Noun-form extensions for explained_full and explained_text_only body rewrites,
# AND for explained_full's choices/label columns.
EXTENDED_NOUN = {
    "Levant":       "Levant (e.g., Lebanese)",
    "North Africa": "North Africa (e.g., Tunisian, Moroccan)",
    "Egypt":        "Egypt",
    "GULF":         "GULF (e.g., Qatari)",
    "MSA":          "MSA",
}

# Adjective-form extensions for prompt 14790's body only. MSA intentionally
# omitted: the prompt already inlines it as "MSA (Modern Standard Arabic)".
EXTENDED_ADJ = {
    "Levantine":     "Levantine (e.g., Lebanese)",
    "North African": "North African (e.g., Tunisian, Moroccan)",
    "Egyptian":      "Egyptian",
    "Gulf":          "Gulf (e.g., Qatari)",
}

# AraBench full 6-label choice set, in fixed display order for the choices column.
ARABENCH_CHOICES = ["MSA", "Lebanese", "Tunisian", "Moroccan", "Qatari", "Egyptian"]

# Body-rewrite mapping for arabench_choices: replace each ADD coarse label in the
# rendered prompt text with the matching AraBench label(s). North Africa expands
# to "Tunisian, Moroccan" because both AraBench labels map to that coarse class.
# Used for prompts 14102, 14783, 14784, 14851 (which use noun forms in the body).
ARABENCH_NOUN = {
    "Levant":       "Lebanese",
    "North Africa": "Tunisian, Moroccan",
    "Egypt":        "Egyptian",
    "GULF":         "Qatari",
    "MSA":          "MSA",
}

# Adjective-form body rewrite for prompt 14790 (which lists options as
# "Levantine, North African, Egyptian, or Gulf"). MSA intentionally omitted —
# same reason as EXTENDED_ADJ: the prompt already inlines "MSA (Modern Standard
# Arabic)" and we don't want to double-substitute it.
ARABENCH_ADJ = {
    "Levantine":     "Lebanese",
    "North African": "Tunisian, Moroccan",
    "Egyptian":      "Egyptian",
    "Gulf":          "Qatari",
}

# Coarse ADD label -> list of AraBench labels that should count as correct.
ADD_TO_ARABENCH_ACCEPT = {
    "MSA":          ["MSA"],
    "Levant":       ["Lebanese"],
    "North Africa": ["Tunisian", "Moroccan"],
    "Egypt":        ["Egyptian"],
    "GULF":         ["Qatari"],
}

# Coarse ADD label -> the single canonical AraBench label used for doc_to_target
# Real scoring is multi-correct via metrics.acc_norm_accept;
# this is only a placeholder so lm-eval's task initialisation does not complain.
ADD_TO_ARABENCH_CANONICAL = {
    "MSA":          "MSA",
    "Levant":       "Lebanese",
    "North Africa": "Tunisian",
    "Egypt":        "Egyptian",
    "GULF":         "Qatari",
}


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _extend_text(text: str, mapping: dict[str, str]) -> str:
    """Replace each short label with its extended version, using word
    boundaries to avoid partial matches (e.g. avoid matching 'Levant' inside
    'Levantine')."""
    for short, long in mapping.items():
        text = re.sub(rf"\b{re.escape(short)}\b", long, text)
    return text


def _read_source(prompt_id: int) -> pd.DataFrame:
    src_path = SRC_DIR / f"prompt_{prompt_id}" / "data.parquet"
    if not src_path.exists():
        raise FileNotFoundError(f"Source parquet missing: {src_path}")
    df = pd.read_parquet(src_path)
    expected_cols = {"text", "label", "choices"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Parquet {src_path} missing columns: {missing}")
    return df


def _write_parquet(df: pd.DataFrame, variant: str, prompt_id: int) -> Path:
    out_dir = DATA_ROOT / variant / f"prompt_{prompt_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "data.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


def _write_yaml(text: str, variant: str, prompt_id: int) -> Path:
    yaml_dir = YAML_ROOT / variant
    yaml_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = yaml_dir / f"prompt_{prompt_id}.yaml"
    yaml_path.write_text(text)
    return yaml_path


def _standard_yaml(task_name: str, dataset_path: str) -> str:
    """YAML for variants that use lm-eval's built-in multiple_choice scoring."""
    return (
        f"task: {task_name}\n"
        f"dataset_path: {dataset_path}\n"
        f"output_type: multiple_choice\n"
        f"test_split: train\n"
        f"doc_to_text: text\n"
        f"doc_to_choice: choices\n"
        f"doc_to_target: label\n"
        f"metric_list:\n"
        f"  - metric: acc\n"
        f"    aggregation: mean\n"
        f"    higher_is_better: true\n"
        f"  - metric: acc_norm\n"
        f"    aggregation: mean\n"
        f"    higher_is_better: true\n"
        f"metadata:\n"
        f"  version: 1.0\n"
    )


def _accept_yaml(task_name: str, dataset_path: str) -> str:
    """YAML for arabench_choices.

    `doc_to_target` calls `metrics.target_indices(doc)`, which returns a list
    of indices into `choices` derived from each row's `accept_choices` column.
    Returning a list makes lm-eval auto-activate `multiple_target` mode (see
    https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.11/lm_eval/api/task.py#L822-L826
    ), so the built-in multiple_choice scoring counts the row correct if argmax
    matches *any* index in the list. Tunisian and Moroccan both count correct
    for North Africa; everything else stays single-target.
    """
    return (
        f"task: {task_name}\n"
        f"dataset_path: {dataset_path}\n"
        f"output_type: multiple_choice\n"
        f"test_split: train\n"
        f"doc_to_text: text\n"
        f"doc_to_choice: choices\n"
        f"doc_to_target: !function metrics.target_indices\n"
        f"metric_list:\n"
        f"  - metric: acc\n"
        f"    aggregation: mean\n"
        f"    higher_is_better: true\n"
        f"  - metric: acc_norm\n"
        f"    aggregation: mean\n"
        f"    higher_is_better: true\n"
        f"metadata:\n"
        f"  version: 1.0\n"
    )


# ----------------------------------------------------------------------------
# Per-variant builders
# ----------------------------------------------------------------------------

def build_explained_full(prompt_id: int) -> None:
    df = _read_source(prompt_id)

    text_mapping = EXTENDED_ADJ if prompt_id == ADJ_PROMPT_ID else EXTENDED_NOUN
    df["text"] = df["text"].apply(lambda t: _extend_text(t, text_mapping))
    df["choices"] = df["choices"].apply(lambda cs: [EXTENDED_NOUN[c] for c in cs])
    df["label"] = df["label"].map(EXTENDED_NOUN)

    sample = df.iloc[0]
    if sample["label"] not in list(sample["choices"]):
        raise ValueError(
            f"explained_full / prompt_{prompt_id}: label {sample['label']!r} not in "
            f"choices {list(sample['choices'])!r}."
        )

    parquet = _write_parquet(df, "explained_full", prompt_id)
    yaml = _write_yaml(
        _standard_yaml(
            task_name=f"Arabic_Dialects_Dataset_explained_full_prompt_{prompt_id}",
            dataset_path=f"ablations/dialect_explained_choices/data/explained_full/prompt_{prompt_id}",
        ),
        "explained_full", prompt_id,
    )
    print(f"  wrote {parquet}  (rows={len(df)})")
    print(f"  wrote {yaml}")
    print(f"    text head: {df['text'].iloc[0][:200]!r}...")


def build_explained_text_only(prompt_id: int) -> None:
    df = _read_source(prompt_id)

    text_mapping = EXTENDED_ADJ if prompt_id == ADJ_PROMPT_ID else EXTENDED_NOUN
    df["text"] = df["text"].apply(lambda t: _extend_text(t, text_mapping))
    # choices and label intentionally left as the bare ADD strings.

    sample = df.iloc[0]
    if sample["label"] not in list(sample["choices"]):
        raise ValueError(
            f"explained_text_only / prompt_{prompt_id}: label {sample['label']!r} "
            f"not in choices {list(sample['choices'])!r}."
        )

    parquet = _write_parquet(df, "explained_text_only", prompt_id)
    yaml = _write_yaml(
        _standard_yaml(
            task_name=f"Arabic_Dialects_Dataset_explained_text_only_prompt_{prompt_id}",
            dataset_path=f"ablations/dialect_explained_choices/data/explained_text_only/prompt_{prompt_id}",
        ),
        "explained_text_only", prompt_id,
    )
    print(f"  wrote {parquet}  (rows={len(df)})")
    print(f"  wrote {yaml}")
    print(f"    text head: {df['text'].iloc[0][:200]!r}...")
    print(f"    choices: {list(df['choices'].iloc[0])}")


def build_arabench_choices(prompt_id: int) -> None:
    df = _read_source(prompt_id)

    # Rewrite ADD coarse labels in the rendered body to the matching AraBench
    # label(s) so the body and the scored choices agree on what the model is
    # supposed to pick from. Otherwise the model would be prompted with one
    # option list and scored over a different one.
    text_mapping = ARABENCH_ADJ if prompt_id == ADJ_PROMPT_ID else ARABENCH_NOUN
    df["text"] = df["text"].apply(lambda t: _extend_text(t, text_mapping))

    add_labels = df["label"].tolist()
    unknown = set(add_labels) - set(ADD_TO_ARABENCH_CANONICAL)
    if unknown:
        raise ValueError(
            f"arabench_choices / prompt_{prompt_id}: unknown ADD labels {unknown!r}"
        )

    df["choices"] = [ARABENCH_CHOICES for _ in range(len(df))]
    df["label"] = [ADD_TO_ARABENCH_CANONICAL[lbl] for lbl in add_labels]
    df["accept_choices"] = [ADD_TO_ARABENCH_ACCEPT[lbl] for lbl in add_labels]

    # Sanity: every accept_choice must be inside choices (metrics.target_indices
    # would otherwise raise ValueError at eval time).
    sample = df.iloc[0]
    if not set(sample["accept_choices"]).issubset(set(sample["choices"])):
        raise ValueError(
            f"arabench_choices / prompt_{prompt_id}: accept_choices "
            f"{list(sample['accept_choices'])!r} not subset of choices "
            f"{list(sample['choices'])!r}."
        )

    parquet = _write_parquet(df, "arabench_choices", prompt_id)
    yaml = _write_yaml(
        _accept_yaml(
            task_name=f"Arabic_Dialects_Dataset_arabench_choices_prompt_{prompt_id}",
            dataset_path=f"ablations/dialect_explained_choices/data/arabench_choices/prompt_{prompt_id}",
        ),
        "arabench_choices", prompt_id,
    )
    print(f"  wrote {parquet}  (rows={len(df)})")
    print(f"  wrote {yaml}")
    print(f"    text head: {df['text'].iloc[0][:200]!r}...")
    print(f"    choices: {ARABENCH_CHOICES}")


VARIANT_BUILDERS = {
    "explained_full":      build_explained_full,
    "explained_text_only": build_explained_text_only,
    "arabench_choices":    build_arabench_choices,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--variant",
        choices=(*VARIANTS, "all"),
        default="all",
        help="Which variant to build (default: all).",
    )
    args = parser.parse_args()

    variants = list(VARIANTS) if args.variant == "all" else [args.variant]

    print(f"Source: {SRC_DIR}")
    print(f"Data root: {DATA_ROOT}")
    print(f"YAML root: {YAML_ROOT}")
    print(f"Variants: {variants}\n")

    for variant in variants:
        builder = VARIANT_BUILDERS[variant]
        print(f"==================== {variant} ====================")
        for pid in PROMPT_IDS:
            print(f"--- prompt_{pid} ---")
            builder(pid)
            print()


if __name__ == "__main__":
    main()
