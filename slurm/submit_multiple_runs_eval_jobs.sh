#!/bin/bash
# Submit a SLURM job that runs multiple evaluation runs for classification tasks.
# Usage: bash slurm/submit_multiple_runs_eval_jobs.sh [--runs N] [--dry-run] [--add-system-prompt]
#
# --runs N sets the number of runs per prompt (default: 10).
# --dry-run is forwarded to run_multiple_runs_evals.sh (preview only, no execution).
# --add-system-prompt is forwarded to run_multiple_runs_evals.sh -> run_eval.py.
# For interactive use: salloc, then bash slurm/scripts/run_multiple_runs_evals.sh [--runs N] [--dry-run]

set -euo pipefail

PROJECT_DIR="/raid_storage/SLURM/home/slurm_majedalshaibani/Projects/instructions-tuning"

mkdir -p "$PROJECT_DIR/slurm/logs"
log_prefix="$PROJECT_DIR/slurm/logs/eval_multiple_runs"

# Collect flags to forward to run_multiple_runs_evals.sh
SCRIPT_ARGS=()
DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; SCRIPT_ARGS+=("$1"); shift ;;
        --add-system-prompt) SCRIPT_ARGS+=("$1"); shift ;;
        --runs) SCRIPT_ARGS+=("$1" "$2"); shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ "$DRY_RUN" == true ]]; then
    echo "--- DRY RUN: previewing via run_multiple_runs_evals.sh --dry-run ---"
    bash "$PROJECT_DIR/slurm/scripts/run_multiple_runs_evals.sh" "${SCRIPT_ARGS[@]}"
    exit 0
fi

sbatch \
    --job-name="eval-multiple-runs" \
    --partition=A100 \
    --nodes=1 \
    --ntasks=1 \
    --gres=gpu:2 \
    --output="${log_prefix}_%j.out" \
    --error="${log_prefix}_%j.err" \
    "$PROJECT_DIR/slurm/scripts/run_multiple_runs_evals.sh" "${SCRIPT_ARGS[@]}"

echo "Submitted SLURM multiple-runs eval job."
