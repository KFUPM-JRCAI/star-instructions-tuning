#!/bin/bash
# Submit SLURM tuning jobs for any tune.ipynb that hasn't produced a tuned model yet.
# Usage: bash slurm/submit_tuning_jobs.sh [--dry-run]

set -euo pipefail

PROJECT_DIR="/raid_storage/SLURM/home/slurm_majedalshaibani/Projects/instructions-tuning"
DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "--- DRY RUN mode (no jobs will be submitted) ---"
fi

# Map model folder name -> MODEL_NAME used in tuned_models output dir
declare -A MODEL_NAME_MAP
MODEL_NAME_MAP["AceGPT"]="AceGPT-7B"
MODEL_NAME_MAP["Llama"]="Meta-Llama-3.1-8B"
MODEL_NAME_MAP["Qwen"]="Qwen3-8B"

mkdir -p "$PROJECT_DIR/slurm/logs"

submitted=0
skipped=0

while IFS= read -r notebook; do
    model_dir=$(dirname "$notebook")
    model_folder=$(basename "$model_dir")
    dataset_dir=$(dirname "$model_dir")
    dataset_name=$(basename "$dataset_dir")
    task_name=$(basename "$(dirname "$dataset_dir")")

    model_name="${MODEL_NAME_MAP[$model_folder]:-}"
    if [[ -z "$model_name" ]]; then
        echo "WARNING: Unknown model folder '$model_folder', skipping: $notebook"
        continue
    fi

    tuned_model_dir="$dataset_dir/tuned_models/$model_name"

    if [[ -f "$tuned_model_dir/adapter_config.json" ]]; then
        echo "SKIP (already done): $task_name/$dataset_name/$model_folder -> $tuned_model_dir"
        ((skipped++)) || true
        continue
    fi

    job_name="tune-${task_name}-${dataset_name}-${model_folder}"
    log_prefix="$PROJECT_DIR/slurm/logs/${task_name}_${dataset_name}_${model_folder}"
    rel_notebook="${notebook#$PROJECT_DIR/}"

    echo "SUBMIT: $task_name/$dataset_name/$model_folder -> $tuned_model_dir"

    if [[ "$DRY_RUN" == true ]]; then
        ((submitted++)) || true
        continue
    fi

    # The notebook calls exit() as its last cell to free GPU memory.
    # papermill exits non-zero on SystemExit, so we check adapter_config.json for real success.
    sbatch \
        --job-name="$job_name" \
        --partition=A100 \
        --nodes=1 \
        --ntasks=1 \
        --gres=gpu:4 \
        --output="${log_prefix}_%j.out" \
        --error="${log_prefix}_%j.err" \
        --wrap="
cd '$PROJECT_DIR'

echo 'Job ID: \$SLURM_JOB_ID'
echo 'Node: \$SLURM_NODELIST'
echo 'Started at: \$(date)'

uv run papermill \\
    '$rel_notebook' \\
    '$rel_notebook' \\
    --cwd '$PROJECT_DIR' \\
    --log-output \\
    --log-level INFO \\
    --progress-bar || true   # exit() in last cell causes non-zero; model is already saved

if [[ -f '$tuned_model_dir/adapter_config.json' ]]; then
    echo 'SUCCESS: adapter saved at $tuned_model_dir'
else
    echo 'ERROR: adapter_config.json not found at $tuned_model_dir — training may have failed'
    exit 1
fi

echo 'Finished at: \$(date)'
"
    ((submitted++)) || true

done < <(find "$PROJECT_DIR/Notebooks/Experiments" -name "tune.ipynb" | sort)

echo ""
echo "Done. Submitted: $submitted | Skipped (already done): $skipped"
