"""Preprocessing for NLI datasets: ArEntail (primary), ArabicTE (secondary).

ArEntail appends '\\nThe answer is:' to the prefix in both generate_tuple and create_hf_dataset.
ArabicTE does NOT append '\\nThe answer is:' in create_hf_dataset.
"""

import datasets
from datasets import DatasetDict
from jinja2 import Environment, StrictUndefined


# Used by both ArEntail and ArabicTE
def preprocess_template(template: str) -> str:
    prefix, suffix = template.split("|||")
    return f"{prefix.strip()}\n{suffix.strip()}"


# Used by ArEntail only (ArabicTE is eval-only, no tuning)
def arentail_generate_tuple(sample: str) -> tuple[str, str]:
    sample_lines = sample.splitlines()
    prefix = "\n".join(sample_lines[:-1])
    prefix = prefix.strip()
    prefix += "\nThe answer is:"
    suffix = sample_lines[-1].strip()
    suffix = f" {suffix}"
    return prefix, suffix


# Used by both ArEntail and ArabicTE
def apply_template(
    prompt_template: dict, sample: dict, for_tuning: bool = False
) -> str:
    template = prompt_template["template"]
    if for_tuning:
        template = preprocess_template(template)
    sample["answer_choices"] = prompt_template["answer_choices"]
    env = Environment(undefined=StrictUndefined)
    if not for_tuning and "|||" not in template:
        raise ValueError("No ||| divider")
    tmpl = env.from_string(template)
    return tmpl.render(**sample)


# -- ArEntail specific --


def arentail_create_hf_dataset(dataset_prompt: dict) -> DatasetDict:
    """Appends '\\nThe answer is:' to the prefix."""
    texts, labels, choices = [], [], []
    for merged_sample in dataset_prompt["merged_samples"]:
        prefix = merged_sample.split("|||")[0].strip()
        prefix += "\nThe answer is:"
        output = merged_sample.split("|||")[1].strip()
        texts.append(prefix)
        labels.append(output)
        choices.append(dataset_prompt["answer_choices"])
    return DatasetDict(
        {
            "test": datasets.Dataset.from_dict(
                {"text": texts, "label": labels, "choices": choices}
            )
        }
    )


# -- ArabicTE specific (eval only) --


def arabic_te_create_hf_dataset(dataset_prompt: dict) -> DatasetDict:
    """No '\\nThe answer is:' appended."""
    texts, labels, choices = [], [], []
    for merged_sample in dataset_prompt["merged_samples"]:
        prefix = merged_sample.split("|||")[0].strip()
        output = merged_sample.split("|||")[1].strip()
        texts.append(prefix)
        labels.append(output)
        choices.append(dataset_prompt["answer_choices"])
    return DatasetDict(
        {
            "test": datasets.Dataset.from_dict(
                {"text": texts, "label": labels, "choices": choices}
            )
        }
    )
