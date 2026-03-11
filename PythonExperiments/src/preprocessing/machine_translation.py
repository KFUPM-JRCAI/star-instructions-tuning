"""Preprocessing for machine translation datasets: opus-100 (primary), tatoeba_mt (secondary).

opus-100 appends '\\nTranslation:' and requires column remapping (en/ar -> translation dict).
tatoeba_mt does NOT append '\\nTranslation:' and needs no column remapping.
"""

import datasets
from datasets import DatasetDict
from jinja2 import Environment, StrictUndefined


# Used by both opus-100 and tatoeba_mt
def preprocess_template(template: str) -> str:
    prefix, suffix = template.split("|||")
    return f"{prefix.strip()}\n{suffix.strip()}"


# Used by both opus-100 and tatoeba_mt
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


# -- opus-100 specific --


def opus100_remap_dataset(hf_dataset):
    """Remap en/ar columns to nested translation dict."""
    return hf_dataset.map(
        lambda ex: {"translation": {"en": ex["en"], "ar": ex["ar"]}},
        remove_columns=["en", "ar"],
    )


def opus100_generate_tuple(sample: str) -> tuple[str, str]:
    sample_lines = sample.splitlines()
    prefix = "\n".join(sample_lines[:-1])
    prefix += "\nTranslation:"
    prefix = prefix.strip()
    suffix = sample_lines[-1].strip()
    suffix = f" {suffix}"
    return prefix, suffix


def opus100_create_hf_dataset(dataset_prompt: dict) -> DatasetDict:
    en_texts, ar_texts = [], []
    for merged_sample in dataset_prompt["merged_samples"]:
        prefix = merged_sample.split("|||")[0].strip()
        prefix += "\nTranslation:"
        output = merged_sample.split("|||")[1].strip()
        en_texts.append(prefix)
        ar_texts.append(output)
    return DatasetDict(
        {"test": datasets.Dataset.from_dict({"en": en_texts, "ar": ar_texts})}
    )


# -- tatoeba_mt specific (eval only, no tuning) --


def tatoeba_mt_create_hf_dataset(dataset_prompt: dict) -> DatasetDict:
    en_texts, ar_texts = [], []
    for merged_sample in dataset_prompt["merged_samples"]:
        prefix = merged_sample.split("|||")[0].strip()
        output = merged_sample.split("|||")[1].strip()
        en_texts.append(prefix)
        ar_texts.append(output)
    return DatasetDict(
        {"test": datasets.Dataset.from_dict({"en": en_texts, "ar": ar_texts})}
    )
