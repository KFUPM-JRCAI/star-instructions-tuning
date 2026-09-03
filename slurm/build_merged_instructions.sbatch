#!/bin/bash
#SBATCH --job-name=build-merged-instructions
#SBATCH --partition=A6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --output=slurm/logs/build_merged_instructions_%j.out
#SBATCH --error=slurm/logs/build_merged_instructions_%j.err

set -euo pipefail

PROJECT_DIR="/raid_storage/SLURM/home/slurm_majedalshaibani/Projects/star-instructions-tuning"
cd "$PROJECT_DIR"

mkdir -p slurm/logs

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"

# Activate the virtual environment
source .venv/bin/activate

NOTEBOOK="Notebooks/push_to_hf/build_merged_instructions.ipynb"

# Run the notebook with papermill (output overwrites the original)
papermill \
    "$NOTEBOOK" \
    "$NOTEBOOK" \
    --cwd "$PROJECT_DIR" \
    --log-output \
    --log-level INFO \
    --progress-bar

echo "Finished at: $(date)"