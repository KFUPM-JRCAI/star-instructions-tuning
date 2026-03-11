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
MODEL_NAME_MAP["AceGPT"]="AceGPT-7B"
MODEL_NAME_MAP["Llama"]="Meta-Llama-3.1-8B"
MODEL_NAME_MAP["Qwen"]="Qwen3-8B"

# Tasks, primary datasets, and secondary datasets
declare -A TASK_PRIMARY_DATASET
TASK_PRIMARY_DATASET["dialect_identification"]="AraBench_dev"
TASK_PRIMARY_DATASET["machine_translation"]="opus-100"
TASK_PRIMARY_DATASET["NLI"]="ArEntail"
TASK_PRIMARY_DATASET["NLU"]="ArabicMMLU"
TASK_PRIMARY_DATASET["sarcasm_detection"]="ArSarcasm_v2"
TASK_PRIMARY_DATASET["summarization"]="xlsum"

declare -A TASK_SECONDARY_DATASETS
TASK_SECONDARY_DATASETS["dialect_identification"]="Arabic_Dialects_Dataset"
TASK_SECONDARY_DATASETS["machine_translation"]="tatoeba_mt"
TASK_SECONDARY_DATASETS["NLI"]="ArabicTE"
TASK_SECONDARY_DATASETS["NLU"]="belebele"
TASK_SECONDARY_DATASETS["sarcasm_detection"]="iSarcasmEval_task"
TASK_SECONDARY_DATASETS["summarization"]="AraSum"

MODEL_FOLDERS=("AceGPT" "Llama" "Qwen")
VARIANTS=("base" "chat" "tuned")

# Collect all eval commands
eval_commands=()
skipped_no_adapter=0

for task_name in "${!TASK_PRIMARY_DATASET[@]}"; do
    primary="${TASK_PRIMARY_DATASET[$task_name]}"
    secondary="${TASK_SECONDARY_DATASETS[$task_name]:-}"
    all_datasets=("$primary")
    if [[ -n "$secondary" ]]; then
        all_datasets+=("$secondary")
    fi

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

            for dataset_name in "${all_datasets[@]}"; do
                eval_commands+=("$task_name|$dataset_name|$model_folder|$variant")
            done
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
    IFS='|' read -r t d m v <<< "$cmd"
    echo "  $t / $d / $m / $v"
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
    IFS='|' read -r task_name dataset_name model_folder variant <<< "$cmd"
    counter=$((counter + 1))
    echo "========================================"
    echo "[$counter/$total] $task_name / $dataset_name / $model_folder / $variant"
    echo "Started at: $(date)"
    echo "========================================"

    uv run python PythonExperiments/run_eval.py \
        --task "$task_name" \
        --dataset "$dataset_name" \
        --model "$model_folder" \
        --variant "$variant" || { echo "WARN: evaluation exited non-zero (may be expected from sys.exit)"; }

    echo "Finished at: $(date)"
    echo ""
done

echo "All done. Ran $counter evaluations."
echo "Finished at: $(date)"
