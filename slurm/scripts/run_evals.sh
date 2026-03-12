#!/bin/bash
# Run all evaluations sequentially.
# Usage: bash slurm/scripts/run_evals.sh [--dry-run]

set -euo pipefail

PROJECT_DIR="/raid_storage/SLURM/home/slurm_majedalshaibani/Projects/instructions-tuning"
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

# Task -> primary dataset (needed for adapter path check)
declare -A TASK_PRIMARY_DATASET
TASK_PRIMARY_DATASET["dialect_identification"]="AraBench_dev"
TASK_PRIMARY_DATASET["machine_translation"]="opus-100"
TASK_PRIMARY_DATASET["NLI"]="ArEntail"
TASK_PRIMARY_DATASET["NLU"]="ArabicMMLU"
TASK_PRIMARY_DATASET["sarcasm_detection"]="ArSarcasm_v2"
TASK_PRIMARY_DATASET["summarization"]="xlsum"

MODEL_FOLDERS=("AceGPT-v1-7B" "AceGPT-v2-8B" "Llama-3.1-8B" "Qwen3-8B")
VARIANTS=("base" "chat" "tuned")

# Collect all eval commands, skipping tuned variants without adapters
eval_commands=()
skipped_no_adapter=0

for task_name in "${!TASK_PRIMARY_DATASET[@]}"; do
    primary="${TASK_PRIMARY_DATASET[$task_name]}"

    for model_folder in "${MODEL_FOLDERS[@]}"; do
        model_name="${MODEL_NAME_MAP[$model_folder]}"

        for variant in "${VARIANTS[@]}"; do
            if [[ "$variant" == "tuned" ]]; then
                adapter_dir="$PROJECT_DIR/Notebooks/Experiments/$task_name/$primary/tuned_models/$model_name"
                if [[ ! -f "$adapter_dir/adapter_config.json" ]]; then
                    echo "SKIP (adapter not found): $task_name/$model_folder/tuned"
                    ((skipped_no_adapter++)) || true
                    continue
                fi
            fi

            eval_commands+=("$task_name|$model_folder|$variant")
        done
    done
done

total=${#eval_commands[@]}
echo ""
echo "Evaluations to run: $total | Skipped (no adapter): $skipped_no_adapter"
echo ""

if [[ "$total" -eq 0 ]]; then
    echo "Nothing to run."
    exit 0
fi

# Print all evaluations
for cmd in "${eval_commands[@]}"; do
    IFS='|' read -r t m v <<< "$cmd"
    echo "  $t / $m / $v"
done
echo ""

if [[ "$DRY_RUN" == true ]]; then
    echo "--- DRY RUN complete ---"
    exit 0
fi

# Run evaluations
echo "Started at: $(date)"
echo ""

counter=0
for cmd in "${eval_commands[@]}"; do
    IFS='|' read -r task_name model_folder variant <<< "$cmd"
    counter=$((counter + 1))
    echo "========================================"
    echo "[$counter/$total] $task_name / $model_folder / $variant (all datasets)"
    echo "Started at: $(date)"
    echo "========================================"

    uv run python PythonExperiments/run_eval.py \
        --task "$task_name" \
        --model "$model_folder" \
        --variant "$variant" || { echo "WARN: evaluation exited non-zero"; }

    echo "Finished at: $(date)"
    echo ""
done

echo "All done. Ran $counter evaluations."
echo "Finished at: $(date)"
