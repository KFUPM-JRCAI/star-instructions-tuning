"""YAML task config templates for lm-eval harness with API models.

Classification tasks use generate_until + exact_match instead of
multiple_choice + acc/acc_norm, since API models don't support
the loglikelihood calls that multiple_choice requires.

Generation tasks (MT, summarization) use the same generate_until
output type as the local model pipeline.
"""

import os
import shutil


CLASSIFICATION_TASKS = {"dialect_identification", "NLI", "NLU", "sarcasm_detection"}

# ── YAML templates ────────────────────────────────────────────────────────────

CLASSIFICATION_YAML = """\
task: {dataset_name}_prompt_{prompt_id}
dataset_path: experimental_hf_datasets/{dataset_name}/prompt_{prompt_id}
output_type: generate_until
test_split: train
doc_to_text: "{{{{text}}}}\\nThe answer is:"
doc_to_target: label
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
generation_kwargs:
  max_new_tokens: 50
  until:
    - "\\n"
metadata:
  version: 1.0"""

MT_YAML = """\
task: {dataset_name}_prompt_{prompt_id}
dataset_path: experimental_hf_datasets/{dataset_name}/prompt_{prompt_id}
output_type: generate_until
test_split: train
doc_to_text: en
doc_to_target: ar
metric_list:
  - metric: !function metrics.calculate_bleu
    aggregation: !function metrics.calculate_bleu_agg
    higher_is_better: true
generation_kwargs:
  max_new_tokens: 512
  until:
    - "<|endoftext|>"
metadata:
  version: 1.0"""

SUMMARIZATION_YAML = """\
task: {dataset_name}_prompt_{prompt_id}
dataset_path: experimental_hf_datasets/{dataset_name}/prompt_{prompt_id}
output_type: generate_until
test_split: train
doc_to_text: text
doc_to_target: summary
metric_list:
  - metric: !function metrics.calculate_bleu
    aggregation: !function metrics.calculate_bleu_agg
    higher_is_better: true
generation_kwargs:
  max_new_tokens: 512
  until:
    - "<|endoftext|>"
metadata:
  version: 1.0"""


def get_yaml_template(task_name: str) -> str:
    if task_name in CLASSIFICATION_TASKS:
        return CLASSIFICATION_YAML
    elif task_name == "machine_translation":
        return MT_YAML
    elif task_name == "summarization":
        return SUMMARIZATION_YAML
    else:
        raise ValueError(f"Unknown task: {task_name}")


def ensure_metrics_file(eval_tasks_dir: str, promptlab_name: str) -> None:
    """Copy metrics.py into the task directory for generation tasks."""
    dst_dir = os.path.join(eval_tasks_dir, promptlab_name)
    dst = os.path.join(dst_dir, "metrics.py")
    if os.path.exists(dst):
        return
    for candidate in os.listdir("eval_harness_extra_tasks"):
        src = os.path.join("eval_harness_extra_tasks", candidate, "metrics.py")
        if os.path.exists(src):
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy2(src, dst)
            return
    raise FileNotFoundError("No metrics.py found in eval_harness_extra_tasks/")


def write_yaml_config(
    task_name: str,
    promptlab_name: str,
    prompt_id: int,
    eval_tasks_dir: str,
) -> str:
    """Write lm-eval YAML task config and return the eval task name."""
    yaml_text = get_yaml_template(task_name).format(
        dataset_name=promptlab_name,
        prompt_id=prompt_id,
    )
    yaml_dir = os.path.join(eval_tasks_dir, promptlab_name)
    os.makedirs(yaml_dir, exist_ok=True)
    with open(os.path.join(yaml_dir, f"prompt_{prompt_id}.yaml"), "w") as f:
        f.write(yaml_text)

    if task_name not in CLASSIFICATION_TASKS:
        ensure_metrics_file(eval_tasks_dir, promptlab_name)

    return f"{promptlab_name}_prompt_{prompt_id}"
