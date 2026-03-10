#!/bin/bash
# Submit a single SLURM job that runs all evaluate/*.ipynb notebooks sequentially.
# For *-tuned* notebooks, skips if no tuned model adapter exists under the task directory.
# Usage: bash slurm/submit_eval_jobs.sh [--dry-run]

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

# Collect all notebooks to evaluate
notebooks=()
skipped_no_adapter=0
skipped_other=0

while IFS= read -r notebook; do
    # Path: .../Notebooks/Experiments/{task}/{dataset}/{model_folder}/evaluate/{variant}.ipynb
    eval_dir=$(dirname "$notebook")            # .../evaluate
    model_dir=$(dirname "$eval_dir")           # .../{model_folder}
    model_folder=$(basename "$model_dir")      # e.g. Qwen, Llama, AceGPT
    dataset_dir=$(dirname "$model_dir")        # .../{dataset}
    dataset_name=$(basename "$dataset_dir")    # e.g. ArEntail
    task_dir=$(dirname "$dataset_dir")
    task_name=$(basename "$task_dir")          # e.g. NLI

    variant=$(basename "$notebook" .ipynb)      # e.g. 8B-tuned, 8B-chat, 8B

    model_name="${MODEL_NAME_MAP[$model_folder]:-}"
    if [[ -z "$model_name" ]]; then
        echo "WARNING: Unknown model folder '$model_folder', skipping: $notebook"
        ((skipped_other++)) || true
        continue
    fi

    # For tuned variants, check if the adapter exists anywhere under the task directory
    if [[ "$variant" == *tuned* ]]; then
        adapter_found=false
        while IFS= read -r adapter; do
            adapter_found=true
            break
        done < <(find "$task_dir" -path "*/tuned_models/$model_name/adapter_config.json" 2>/dev/null)

        if [[ "$adapter_found" == false ]]; then
            echo "SKIP (adapter not found): $task_name/$dataset_name/$model_folder/$variant"
            ((skipped_no_adapter++)) || true
            continue
        fi
    fi

    echo "INCLUDE: $task_name/$dataset_name/$model_folder/$variant"
    notebooks+=("$notebook")

done < <(find "$PROJECT_DIR/Notebooks/Experiments" -path "*/evaluate/*.ipynb" \
    ! -path "*/cross_tasks_tuning/*" \
    ! -name "*cross-tasks*" | sort)

total=${#notebooks[@]}
echo ""
echo "Notebooks to run: $total | Skipped (no adapter): $skipped_no_adapter | Skipped (other): $skipped_other"

if [[ "$total" -eq 0 ]]; then
    echo "Nothing to submit."
    exit 0
fi

if [[ "$DRY_RUN" == true ]]; then
    echo "--- DRY RUN complete, no job submitted ---"
    exit 0
fi

# Build the list of notebook paths as a newline-separated string for the job script
notebook_list=""
for nb in "${notebooks[@]}"; do
    rel="${nb#$PROJECT_DIR/}"
    notebook_list+="$rel"$'\n'
done

log_prefix="$PROJECT_DIR/slurm/logs/eval_all"

sbatch \
    --job-name="eval-all" \
    --partition=A100 \
    --nodes=1 \
    --ntasks=1 \
    --gres=gpu:1 \
    --output="${log_prefix}_%j.out" \
    --error="${log_prefix}_%j.err" \
    --wrap="
cd '$PROJECT_DIR'

echo 'Job ID: \$SLURM_JOB_ID'
echo 'Node: \$SLURM_NODELIST'
echo 'Started at: \$(date)'
echo 'Total notebooks: $total'
echo ''

counter=0
failed=0

while IFS= read -r rel_notebook; do
    [[ -z \"\$rel_notebook\" ]] && continue
    ((counter++)) || true
    echo \"========================================\"
    echo \"[\$counter/$total] \$rel_notebook\"
    echo \"Started at: \$(date)\"
    echo \"========================================\"

    uv run papermill \\
        \"\$rel_notebook\" \\
        \"\$rel_notebook\" \\
        --cwd '$PROJECT_DIR' \\
        --log-output \\
        --log-level INFO \\
        --progress-bar || { echo \"WARN: \$rel_notebook exited non-zero (may be expected from exit())\"; }

    echo \"Finished \$rel_notebook at: \$(date)\"
    echo ''
done <<'NOTEBOOK_LIST'
${notebook_list}NOTEBOOK_LIST

echo ''
echo \"All done. Ran \$counter notebooks.\"
echo 'Finished at: \$(date)'
"

echo "Submitted single SLURM job with $total notebooks."
