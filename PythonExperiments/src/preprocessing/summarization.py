"""Preprocessing for summarization datasets: xlsum, AraSum.

Both datasets use identical preprocessing (with \\xa0 removal).
"""

import datasets
from datasets import DatasetDict
from jinja2 import Environment, StrictUndefined


# Used by both xlsum and AraSum
def preprocess_template(template: str) -> str:
    prefix, suffix = template.split("|||")
    prefix = prefix.replace("\xa0", "")
    return f"{prefix.strip()}|||{suffix.strip()}"


# Used by both xlsum and AraSum
def generate_tuple(sample: str) -> tuple[str, str]:
    prefix, suffix = sample.split("|||")
    prefix = prefix.strip()
    prefix = prefix.replace("\xa0", "")
    suffix = suffix.strip()
    suffix = f" {suffix}"
    return prefix, suffix


# Used by both xlsum and AraSum
def apply_template(
    prompt_template: dict, sample: dict, for_tuning: bool = False
) -> str:
    template = prompt_template["template"]
    if for_tuning:
        template = preprocess_template(template)
    env = Environment(undefined=StrictUndefined)
    if not for_tuning and "|||" not in template:
        raise ValueError("No ||| divider")
    tmpl = env.from_string(template)
    return tmpl.render(**sample)


# Used by both xlsum and AraSum
def create_hf_dataset(dataset_prompt: dict) -> DatasetDict:
    texts, summaries = [], []
    for merged_sample in dataset_prompt["merged_samples"]:
        prefix = merged_sample.split("|||")[0].strip()
        prefix = prefix.replace("\xa0", "")
        output = merged_sample.split("|||")[1].strip()
        texts.append(prefix)
        summaries.append(output)
    return DatasetDict(
        {"test": datasets.Dataset.from_dict({"text": texts, "summary": summaries})}
    )
