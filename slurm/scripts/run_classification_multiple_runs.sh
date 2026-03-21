#!/bin/bash
# Run multiple evaluation runs for classification tasks only (loglikelihood / multiple_choice).
# Usage: bash slurm/scripts/run_multiple_runs_evals.sh [--runs N] [--dry-run] [--add-system-prompt]

set -euo pipefail

PROJECT_DIR="/raid_storage/SLURM/home/slurm_majedalshaibani/Projects/instructions-tuning"
DRY_RUN=false
ADD_SYSTEM_PROMPT=false
RUNS=10

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; echo "--- DRY RUN mode (nothing will be executed) ---"; shift ;;
        --add-system-prompt) ADD_SYSTEM_PROMPT=true; echo "--- System prompt ENABLED ---"; shift ;;
        --runs) RUNS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "Runs per prompt: $RUNS"

SYSTEM_PROMPT_FLAG=""
if [[ "$ADD_SYSTEM_PROMPT" == true ]]; then
    SYSTEM_PROMPT_FLAG="--add_system_prompt"
fi

cd "$PROJECT_DIR"

# Map model folder name -> MODEL_NAME used in tuned_models output dir
declare -A MODEL_NAME_MAP
MODEL_NAME_MAP["AceGPT-v2-8B"]="AceGPT-v2-8B"
MODEL_NAME_MAP["Llama-3.1-8B"]="Meta-Llama-3.1-8B"
MODEL_NAME_MAP["Qwen3-8B"]="Qwen3-8B"

# Classification tasks only (loglikelihood / multiple_choice)
declare -A TASK_PRIMARY_DATASET
TASK_PRIMARY_DATASET["dialect_identification"]="AraBench_dev"
TASK_PRIMARY_DATASET["NLI"]="ArEntail"
TASK_PRIMARY_DATASET["NLU"]="ArabicMMLU"
TASK_PRIMARY_DATASET["sarcasm_detection"]="ArSarcasm_v2"

MODEL_FOLDERS=("AceGPT-v2-8B" "Llama-3.1-8B" "Qwen3-8B")
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
                adapter_dir="$PROJECT_DIR/tuned_models/$task_name/$primary/$model_folder"
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
echo "Evaluations to run: $total (x${RUNS} runs each) | Skipped (no adapter): $skipped_no_adapter"
echo ""

if [[ "$total" -eq 0 ]]; then
    echo "Nothing to run."
    exit 0
fi

# Print all evaluations
for cmd in "${eval_commands[@]}"; do
    IFS='|' read -r t m v <<< "$cmd"
    echo "  $t / $m / $v (${RUNS} runs)"
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
    echo "[$counter/$total] $task_name / $model_folder / $variant (all datasets, ${RUNS} runs)"
    echo "Started at: $(date)"
    echo "========================================"

    uv run python PythonExperiments/run_eval.py \
        --task "$task_name" \
        --model "$model_folder" \
        --variant "$variant" \
        --runs "$RUNS" $SYSTEM_PROMPT_FLAG || { echo "WARN: evaluation exited non-zero"; }

    echo "Finished at: $(date)"
    echo ""
done

echo "All done. Ran $counter evaluations (x${RUNS} runs each)."
echo "Finished at: $(date)"
