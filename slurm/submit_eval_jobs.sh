#!/bin/bash
# Submit a SLURM job that runs all evaluations.
# Usage: bash slurm/submit_eval_jobs.sh [--dry-run] [--add-system-prompt]
#
# --dry-run is forwarded to run_evals.sh (preview only, no execution).
# --add-system-prompt is forwarded to run_evals.sh → run_eval.py.
# For interactive use: salloc, then bash slurm/scripts/run_evals.sh [--dry-run] [--add-system-prompt]

set -euo pipefail

PROJECT_DIR="/raid_storage/SLURM/home/slurm_majedalshaibani/Projects/instructions-tuning"

mkdir -p "$PROJECT_DIR/slurm/logs"
log_prefix="$PROJECT_DIR/slurm/logs/eval_all"

# Collect flags to forward to run_evals.sh
SCRIPT_ARGS=()
DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true; SCRIPT_ARGS+=("$arg") ;;
        --add-system-prompt) SCRIPT_ARGS+=("$arg") ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

if [[ "$DRY_RUN" == true ]]; then
    echo "--- DRY RUN: previewing via run_evals.sh --dry-run ---"
    bash "$PROJECT_DIR/slurm/scripts/run_evals.sh" "${SCRIPT_ARGS[@]}"
    exit 0
fi

sbatch \
    --job-name="eval-all" \
    --partition=A100 \
    --nodes=1 \
    --ntasks=1 \
    --gres=gpu:2 \
    --output="${log_prefix}_%j.out" \
    --error="${log_prefix}_%j.err" \
    "$PROJECT_DIR/slurm/scripts/run_evals.sh" "${SCRIPT_ARGS[@]}"

echo "Submitted SLURM eval job."
