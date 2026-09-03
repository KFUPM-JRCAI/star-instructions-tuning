#!/bin/bash
#SBATCH --job-name=dialect-choice-ablation
#SBATCH --partition=A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --output=ablations/dialect_explained_choices/slurm/logs/run_%j.out
#SBATCH --error=ablations/dialect_explained_choices/slurm/logs/run_%j.err

# Run the dialect choice-ablation eval on all 3 LoRA-tuned models for all
# three variants:
#   explained_full, explained_text_only, arabench_choices
#
# Submit with:
#   sbatch ablations/dialect_explained_choices/slurm/run.sbatch
#
# Pass-through args go to run_eval.py, e.g. restrict to one variant or one model:
#   sbatch ablations/dialect_explained_choices/slurm/run.sbatch --variant arabench_choices
#   sbatch ablations/dialect_explained_choices/slurm/run.sbatch --variant explained_text_only --model Qwen3-8B
#
# Or run interactively (after `salloc --partition=A100 --gres=gpu:2`):
#   bash ablations/dialect_explained_choices/slurm/run.sbatch
#
# Skips prompts whose result JSON already exists; pass --force to re-evaluate.

set -euo pipefail

PROJECT_DIR="/raid_storage/SLURM/home/slurm_majedalshaibani/Projects/instructions-tuning"
cd "$PROJECT_DIR"

mkdir -p ablations/dialect_explained_choices/slurm/logs

# Build all variants' parquets/YAMLs if any are missing (idempotent and fast).
NEED_BUILD=0
for variant in explained_full explained_text_only arabench_choices; do
    if [[ ! -f "ablations/dialect_explained_choices/data/${variant}/prompt_14102/data.parquet" ]]; then
        NEED_BUILD=1
        break
    fi
done
if [[ "$NEED_BUILD" -eq 1 ]]; then
    echo "[run.sbatch] Building parquets and YAMLs for all variants..."
    uv run python ablations/dialect_explained_choices/build_data.py --variant all
fi

echo "[run.sbatch] Running eval (args: $*)..."
uv run python ablations/dialect_explained_choices/run_eval.py "$@"

echo "[run.sbatch] Aggregating results..."
uv run python ablations/dialect_explained_choices/analyze.py
