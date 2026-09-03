#!/bin/bash
# Run all pending tuning jobs sequentially.
# Usage: bash slurm/scripts/run_tuning.sh [--dry-run]

set -euo pipefail

PROJECT_DIR="/raid_storage/SLURM/home/slurm_majedalshaibani/Projects/star-instructions-tuning"
DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "--- DRY RUN mode (nothing will be executed) ---"
fi

cd "$PROJECT_DIR"

# Map model folder name -> MODEL_NAME used in tuned_models output dir
declare -A MODEL_NAME_MAP
MODEL_NAME_MAP["AceGPT-v1-7B"]="AceGPT-v1-7B"
MODEL_NAME_MAP["AceGPT-v2-8B"]="AceGPT-v2-8B"
MODEL_NAME_MAP["Llama-3.1-8B"]="Meta-Llama-3.1-8B"
MODEL_NAME_MAP["Qwen3-8B"]="Qwen3-8B"

# Tasks and their primary datasets (tuning only happens on primary)
declare -A TASK_PRIMARY_DATASET
TASK_PRIMARY_DATASET["dialect_identification"]="AraBench_dev"
TASK_PRIMARY_DATASET["machine_translation"]="opus-100"
TASK_PRIMARY_DATASET["NLI"]="ArEntail"
TASK_PRIMARY_DATASET["NLU"]="ArabicMMLU"
TASK_PRIMARY_DATASET["sarcasm_detection"]="ArSarcasm_v2"
TASK_PRIMARY_DATASET["summarization"]="xlsum"

MODEL_FOLDERS=("AceGPT-v1-7B" "AceGPT-v2-8B" "Llama-3.1-8B" "Qwen3-8B")

# Collect pending tuning jobs
tuning_commands=()
skipped=0

for task_name in "${!TASK_PRIMARY_DATASET[@]}"; do
    dataset_name="${TASK_PRIMARY_DATASET[$task_name]}"

    for model_folder in "${MODEL_FOLDERS[@]}"; do
        model_name="${MODEL_NAME_MAP[$model_folder]}"
        tuned_model_dir="$PROJECT_DIR/Notebooks/Experiments/$task_name/$dataset_name/tuned_models/$model_name"

        if [[ -f "$tuned_model_dir/adapter_config.json" ]]; then
            echo "SKIP (already done): $task_name / $model_folder"
            ((skipped++)) || true
            continue
        fi

        tuning_commands+=("$task_name|$model_folder|$tuned_model_dir")
    done
done

total=${#tuning_commands[@]}
echo ""
echo "Tuning jobs to run: $total | Skipped (already done): $skipped"
echo ""

if [[ "$total" -eq 0 ]]; then
    echo "Nothing to run."
    exit 0
fi

# Print all pending jobs
for cmd in "${tuning_commands[@]}"; do
    IFS='|' read -r t m _ <<< "$cmd"
    echo "  $t / $m"
done
echo ""

if [[ "$DRY_RUN" == true ]]; then
    echo "--- DRY RUN complete ---"
    exit 0
fi

# Run tuning jobs
echo "Started at: $(date)"
echo ""

counter=0
for cmd in "${tuning_commands[@]}"; do
    IFS='|' read -r task_name model_folder tuned_model_dir <<< "$cmd"
    counter=$((counter + 1))
    echo "========================================"
    echo "[$counter/$total] $task_name / $model_folder"
    echo "Started at: $(date)"
    echo "========================================"

    uv run python PythonExperiments/tune.py \
        --task "$task_name" \
        --model "$model_folder" || true

    if [[ -f "$tuned_model_dir/adapter_config.json" ]]; then
        echo "SUCCESS: adapter saved at $tuned_model_dir"
    else
        echo "ERROR: adapter_config.json not found — training may have failed"
    fi

    echo "Finished at: $(date)"
    echo ""
done

echo "All done. Ran $counter tuning jobs."
echo "Finished at: $(date)"
