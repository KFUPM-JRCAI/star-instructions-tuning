#!/bin/bash
# Submit SLURM tuning jobs for each task/model that hasn't produced a tuned model yet.
# Uses PythonExperiments/tune.py instead of papermill.
# Usage: bash slurm/submit_tuning_jobs.sh [--dry-run]

set -euo pipefail

PROJECT_DIR="/raid_storage/SLURM/home/slurm_majedalshaibani/Projects/instructions-tuning"
DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "--- DRY RUN mode (no jobs will be submitted) ---"
fi

# Tasks and their primary datasets
declare -A TASK_PRIMARY_DATASET
TASK_PRIMARY_DATASET["dialect_identification"]="AraBench_dev"
TASK_PRIMARY_DATASET["machine_translation"]="opus-100"
TASK_PRIMARY_DATASET["NLI"]="ArEntail"
TASK_PRIMARY_DATASET["NLU"]="ArabicMMLU"
TASK_PRIMARY_DATASET["sarcasm_detection"]="ArSarcasm_v2"
TASK_PRIMARY_DATASET["summarization"]="xlsum"

MODEL_FOLDERS=("AceGPT-v1-7B" "AceGPT-v2-8B" "Llama-3.1-8B" "Qwen3-8B")

mkdir -p "$PROJECT_DIR/slurm/logs"

submitted=0
skipped=0

for task_name in "${!TASK_PRIMARY_DATASET[@]}"; do
    dataset_name="${TASK_PRIMARY_DATASET[$task_name]}"

    for model_folder in "${MODEL_FOLDERS[@]}"; do
        tuned_model_dir="$PROJECT_DIR/tuned_models/$task_name/$dataset_name/$model_folder"

        if [[ -f "$tuned_model_dir/adapter_config.json" ]]; then
            echo "SKIP (already done): $task_name/$model_folder -> $tuned_model_dir"
            ((skipped++)) || true
            continue
        fi

        job_name="tune-${task_name}-${dataset_name}-${model_folder}"
        log_prefix="$PROJECT_DIR/slurm/logs/${task_name}_${dataset_name}_${model_folder}"

        echo "SUBMIT: $task_name/$dataset_name/$model_folder"

        if [[ "$DRY_RUN" == true ]]; then
            ((submitted++)) || true
            continue
        fi

        # tune.py calls sys.exit(0) after training, so we check adapter_config.json for real success.
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

uv run python PythonExperiments/tune.py \\
    --task '$task_name' \\
    --model '$model_folder' || true

if [[ -f '$tuned_model_dir/adapter_config.json' ]]; then
    echo 'SUCCESS: adapter saved at $tuned_model_dir'
else
    echo 'ERROR: adapter_config.json not found at $tuned_model_dir — training may have failed'
    exit 1
fi

echo 'Finished at: \$(date)'
"
        ((submitted++)) || true
    done
done

echo ""
echo "Done. Submitted: $submitted | Skipped (already done): $skipped"
