# Dialect Identification — Choice-Taxonomy Ablation

Side ablation responding to a reviewer comment on cross-dataset generalisation
in dialect ID: AraBench-tuned models evaluated on the Arabic Dialects Dataset
(ADD) showed a 52–62 point generalisation gap. The reviewer hypothesised this
is partly due to incompatible label taxonomies — AraBench has fine-grained
labels (MSA / Tunisian / Moroccan / Qatari / Egyptian / Lebanese) while ADD
has coarse labels (MSA / Levant / North Africa / Egypt / GULF) — and
recommended a label-mapping analysis.

## What this ablation does

This ablation evaluates the 3 LoRA-tuned models (AceGPT-v2-8B,
Meta-Llama-3.1-8B, Qwen3-8B) on ADD with **the same five prompts as the main
paper**, but varies how the taxonomy is exposed at inference time. No
retraining; same 9,992-row ADD test set; same `multiple_choice` loglikelihood
backbone.

Three variants are tested, isolating where in the prompt the hierarchy is
exposed:

| Variant | Prompt body | Choices column | Label column | Scoring |
|---|---|---|---|---|
| `explained_full` | extended (`GULF (e.g., Qatari)` etc.) | extended | extended | lm-eval built-in `acc_norm` |
| `explained_text_only` | extended | bare ADD (`Levant`, `GULF`, …) | bare ADD | lm-eval built-in `acc_norm` |
| `arabench_choices` | ADD labels rewritten to AraBench labels (`Levant` -> `Lebanese`, `North Africa` -> `Tunisian, Moroccan`, etc.) so the body lists the same 6 labels that are scored | 6 AraBench labels: `[MSA, Lebanese, Tunisian, Moroccan, Qatari, Egyptian]` | one canonical AraBench label per coarse class (b1) | multi-correct via `doc_to_target: !function metrics.target_indices` (Tunisian + Moroccan both count for North Africa) |

`explained_full` tests whether full transparency in body and choices closes
the gap. `explained_text_only` isolates whether seeing the hierarchy in the
body alone (without changing what is scored) is enough. `arabench_choices`
asks whether scoring directly over AraBench labels (with post-mapping to
coarse for evaluation) recovers the tuned representations.

For `arabench_choices` the multi-correct scoring rides on **lm-eval's
built-in** `multiple_target` mode. The YAML's
`doc_to_target: !function metrics.target_indices` returns a list of indices
into `choices` derived from each row's `accept_choices` column (see
`eval_harness_tasks/arabench_choices/metrics.py` — four lines). When the
first sample's `doc_to_target` returns a list, lm-eval auto-activates
`multiple_target` mode and the built-in `multiple_choice` `acc`/`acc_norm`
scoring counts a row correct if argmax matches *any* index in the gold list.
The result JSON's `acc_norm,none` field is therefore directly comparable
across all four sources (strict + three variants) and to the main paper.

## What this ablation is NOT

- It is **not** a replacement for the strict ADD eval. The original 27–38%
  numbers stay in Table 6 of the paper; this is a side analysis.
- It is **not** done on the chat/instruct or base variants. Only the LoRA-
  tuned checkpoints are tested, since the question is whether AraBench-aligned
  representations are unlocked by changing the choice taxonomy.

## Layout

```
ablations/dialect_explained_choices/
├── build_data.py                # generate parquets + YAMLs (--variant all)
├── data/
│   ├── explained_full/prompt_{id}/data.parquet
│   ├── explained_text_only/prompt_{id}/data.parquet
│   └── arabench_choices/prompt_{id}/data.parquet   (extra `accept_choices` col)
├── eval_harness_tasks/
│   ├── explained_full/prompt_{id}.yaml
│   ├── explained_text_only/prompt_{id}.yaml
│   └── arabench_choices/
│       ├── prompt_{id}.yaml                        # doc_to_target: !function metrics.target_indices
│       └── metrics.py                              # 4-line target_indices(doc) -> list[int]
├── run_eval.py                  # --variant all|<name> --model all|<key>
├── results/{variant}/{model_dir}/prompt_{id}.json  # eval JSON outputs (gitignored)
├── analyze.py                   # 4-row table: strict + 3 variants
├── explore.ipynb                # eyeball one sample per (variant, prompt) + sanity checks
├── slurm/
│   ├── run.sbatch               # SLURM submission wrapper
│   └── logs/                    # SLURM stdout/stderr (gitignored)
└── README.md
```

## How to run

From project root:

```bash
# 1. Build all variants' parquets and YAMLs (fast).
uv run python ablations/dialect_explained_choices/build_data.py --variant all

# 2. Eyeball one rendered sample per (variant, prompt) and run sanity checks
#    by opening explore.ipynb in VS Code and running all cells.

# 3. Run the eval (heavy: ~30–60 min/model on 1–2 A100s).
#    Submit to SLURM (all variants × all models):
sbatch ablations/dialect_explained_choices/slurm/run.sbatch
#    Or restrict to one variant / one model:
sbatch ablations/dialect_explained_choices/slurm/run.sbatch --variant arabench_choices
sbatch ablations/dialect_explained_choices/slurm/run.sbatch --variant explained_text_only --model Qwen3-8B
#    Or run directly:
uv run python ablations/dialect_explained_choices/run_eval.py --variant all

# 4. Print the comparison table:
uv run python ablations/dialect_explained_choices/analyze.py
```

`run_eval.py` skips any `(variant, model, prompt)` triple whose result JSON
already exists; pass `--force` to re-evaluate.

## Reading `analyze.py` output

- **Overall (acc_norm)** is read directly from lm-eval's `acc_norm,none` for
  every row, so all four sources (strict + three variants) come from the same
  pipeline the main paper uses. This is the paper-table-style number.
- **Per-class columns** are a diagnostic post-hoc breakdown derived from each
  sample's per-choice logprobs (raw argmax, no length normalisation). For
  `arabench_choices` the per-class block applies the multi-correct
  `accept_choices` rule so it matches the Overall.
