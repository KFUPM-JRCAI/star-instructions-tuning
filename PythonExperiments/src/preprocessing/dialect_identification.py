"""Preprocessing for dialect identification datasets: AraBench_dev, Arabic_Dialects_Dataset.

Both datasets use identical preprocessing (with \\xa0 removal).
"""

import datasets
from datasets import DatasetDict
from jinja2 import Environment, StrictUndefined


# Used by both AraBench_dev and Arabic_Dialects_Dataset
def preprocess_template(template: str) -> str:
    prefix, suffix = template.split("|||")
    prefix = prefix.replace("\xa0", "")
    return f"{prefix.strip()}\n{suffix.strip()}"


# Used by both AraBench_dev and Arabic_Dialects_Dataset
def generate_tuple(sample: str) -> tuple[str, str]:
    sample_lines = sample.splitlines()
    prefix = "\n".join(sample_lines[:-1])
    prefix = prefix.replace("\xa0", "")
    prefix = prefix.strip()
    suffix = sample_lines[-1].strip()
    suffix = f" {suffix}"
    return prefix, suffix


# Used by both AraBench_dev and Arabic_Dialects_Dataset
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


# Used by both AraBench_dev and Arabic_Dialects_Dataset
def create_hf_dataset(dataset_prompt: dict) -> DatasetDict:
    texts, labels, choices = [], [], []
    for merged_sample in dataset_prompt["merged_samples"]:
        prefix = merged_sample.split("|||")[0].replace("\xa0", "").strip()
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
