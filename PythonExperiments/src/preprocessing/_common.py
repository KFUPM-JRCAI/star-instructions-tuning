"""Shared types and YAML templates for the preprocessing package."""

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class DatasetFunctions:
    apply_template: Callable
    create_hf_dataset: Callable
    preprocess_template: Optional[Callable] = None  # only needed for tuning (primary datasets)
    generate_tuple: Optional[Callable] = None  # only needed for tuning (primary datasets)
    remap_dataset: Optional[Callable] = None


# =============================================================================
# YAML templates for evaluation harness
# =============================================================================

MULTIPLE_CHOICE_YAML = """\
task: {dataset_name}_prompt_{prompt_id}
dataset_path: experimental_hf_datasets/{dataset_name}/prompt_{prompt_id}
output_type: multiple_choice
test_split: train
doc_to_text: text
doc_to_choice: choices
doc_to_target: label
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: True
  - metric: acc_norm
    aggregation: mean
    higher_is_better: true
metadata:
  version: 1.0"""

GENERATE_UNTIL_MT_YAML = """\
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
    - </s>
    - <|eot_id|>
    - <|end_of_text|>
    - <|endoftext|>
    - <|im_end|>
metadata:
  version: 1.0"""

GENERATE_UNTIL_SUMMARIZATION_YAML = """\
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
    - </s>
    - <|eot_id|>
    - <|end_of_text|>
    - <|endoftext|>
    - <|im_end|>
metadata:
  version: 1.0"""


def get_yaml_template(task_name: str) -> str:
    if task_name in ("dialect_identification", "NLI", "NLU", "sarcasm_detection"):
        return MULTIPLE_CHOICE_YAML
    elif task_name == "machine_translation":
        return GENERATE_UNTIL_MT_YAML
    elif task_name == "summarization":
        return GENERATE_UNTIL_SUMMARIZATION_YAML
    else:
        raise ValueError(f"Unknown task: {task_name}")
