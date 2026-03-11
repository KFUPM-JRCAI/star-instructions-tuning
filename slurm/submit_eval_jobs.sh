#!/bin/bash
# Submit a SLURM job that runs all evaluations.
# Usage: bash slurm/submit_eval_jobs.sh [--dry-run]
#
# --dry-run is forwarded to run_evals.sh (preview only, no execution).
# For interactive use: salloc, then bash slurm/scripts/run_evals.sh [--dry-run]

set -euo pipefail

PROJECT_DIR="/raid_storage/SLURM/home/slurm_majedalshaibani/Projects/instructions-tuning"

mkdir -p "$PROJECT_DIR/slurm/logs"
log_prefix="$PROJECT_DIR/slurm/logs/eval_all"

# Forward --dry-run to the script if provided
SCRIPT_ARGS=""
if [[ "${1:-}" == "--dry-run" ]]; then
    echo "--- DRY RUN: previewing via run_evals.sh --dry-run ---"
    bash "$PROJECT_DIR/slurm/scripts/run_evals.sh" --dry-run
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
    "$PROJECT_DIR/slurm/scripts/run_evals.sh"

echo "Submitted SLURM eval job."
