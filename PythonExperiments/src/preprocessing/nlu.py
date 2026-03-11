"""Preprocessing for NLU datasets: ArabicMMLU (primary), belebele (secondary).

ArabicMMLU uses dynamic answer_choices filtered from original samples.
belebele uses static answer_choices (no dynamic filtering).
Both remove newlines from output with .replace('\\n', '').
"""

import datasets
from datasets import DatasetDict
from jinja2 import Environment, StrictUndefined


# Used by ArabicMMLU only (belebele is eval-only, no tuning)
def preprocess_template(template: str) -> str:
    prefix, suffix = template.split("|||")
    return f"{prefix.strip()}\n{suffix.strip()}"


# Used by ArabicMMLU only (belebele is eval-only, no tuning)
def generate_tuple(sample: str) -> tuple[str, str]:
    sample_lines = sample.splitlines()
    prefix = "\n".join(sample_lines[:-1])
    prefix = prefix.strip()
    suffix = sample_lines[-1].strip()
    suffix = f" {suffix}"
    return prefix, suffix


# Used by both ArabicMMLU and belebele
def apply_template(
    prompt_template: dict, sample: dict, for_tuning: bool = False
) -> str:
    template = prompt_template["template"]
    if for_tuning:
        template = preprocess_template(template)
    # Dynamic answer_choices: only include choices that have non-empty values
    sample["answer_choices"] = []
    for choice in prompt_template["answer_choices"]:
        if choice in sample and sample[choice]:
            if isinstance(sample[choice], str) and sample[choice].strip():
                sample["answer_choices"].append(choice)
            else:
                sample[choice] = None
        else:
            sample[choice] = None
    env = Environment(undefined=StrictUndefined)
    if not for_tuning and "|||" not in template:
        raise ValueError("No ||| divider")
    tmpl = env.from_string(template)
    return tmpl.render(**sample)


# -- ArabicMMLU specific --


def arabic_mmlu_create_hf_dataset(dataset_prompt: dict) -> DatasetDict:
    """Dynamic answer_choices from original samples, newline removal."""
    texts, labels, choices = [], [], []
    for i, merged_sample in enumerate(dataset_prompt["merged_samples"]):
        prefix = merged_sample.split("|||")[0].strip()
        output = merged_sample.split("|||")[1].replace("\n", "").strip()
        # Use the original sample's dynamic answer_choices
        original_sample = dataset_prompt["original_samples"][i]
        example_choices = []
        for choice in dataset_prompt["answer_choices"]:
            if choice in original_sample and original_sample[choice]:
                if isinstance(original_sample[choice], str) and original_sample[choice].strip():
                    example_choices.append(choice)
        texts.append(prefix)
        labels.append(output)
        choices.append(example_choices)
    return DatasetDict(
        {
            "test": datasets.Dataset.from_dict(
                {"text": texts, "label": labels, "choices": choices}
            )
        }
    )


# -- belebele specific (eval only) --


def belebele_create_hf_dataset(dataset_prompt: dict) -> DatasetDict:
    """Static answer_choices (no dynamic filtering), newline removal."""
    texts, labels, choices = [], [], []
    for merged_sample in dataset_prompt["merged_samples"]:
        prefix = merged_sample.split("|||")[0].strip()
        output = merged_sample.split("|||")[1].replace("\n", "").strip()
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
