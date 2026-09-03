---
pretty_name: STAR Instructions
language:
  - ar
multilinguality:
  - monolingual
size_categories:
  - 10M<n<100M
source_datasets:
  - extended
task_categories:
  - text-generation
  - text-classification
  - token-classification
  - translation
  - summarization
  - multiple-choice
  - question-answering
  - sentence-similarity
tags:
  - arabic
  - instruction-tuning
  - instruction-following
  - prompts
dataset_info:
  - config_name: default
    features:
      - name: tags
        sequence: string
      - name: dataset_name
        dtype: string
      - name: dataset_subset
        dtype: string
      - name: answer_choices
        sequence: string
      - name: text_direction
        dtype: string
      - name: prompter_id
        dtype: string
      - name: split_name
        dtype: string
      - name: full_instruction
        dtype: string
      - name: instruction_input
        dtype: string
      - name: instruction_output
        dtype: string
      - name: instruction_template
        dtype: string
      - name: instruction_template_id
        dtype: int64
      - name: instruction_name
        dtype: string
      - name: instruction_tasks
        sequence: string
---

# STAR Instructions

STAR Instructions is a large-scale Arabic instruction-tuning dataset built by rendering the 355 STAR Jinja2 prompt templates against their 87 source datasets, covering 27 raw task labels (20 tasks after merging closely related categories, as reported in the paper). The underlying templates were authored collaboratively using [PromptLab](https://aclanthology.org/2026.eacl-demo.18/). This dataset and the experiments built on it are described in [STAR: instruction tuning for Arabic across tasks, datasets, and models](https://link.springer.com/article/10.1007/s10579-026-09942-8).

For the raw templates and experiment design details, see the companion dataset: [STAR Templates](https://huggingface.co/datasets/KFUPM-JRCAI/star-dataset-templates).

📦 **Code:** the tuning and evaluation pipelines built on this dataset is accessible at [github.com/KFUPM-JRCAI/star-instructions-tuning](https://github.com/KFUPM-JRCAI/star-instructions-tuning).

## Overview

Each record in this dataset is created by rendering a Jinja2 prompt template against a real data sample, then splitting the result on `|||` to separate the instruction input from the expected output. The dataset preserves both the rendered instruction and the original template metadata.

**Tasks covered:** dialect identification, summarization, sentiment analysis, stance detection, multiple choice, sarcasm detection, machine translation, NLI, question answering, diacritization, review classification, offensive language detection, emotion classification, commonsense validation, topic classification, math solving, era classification, yes/no question answering, claim verification, theme classification, targeted sentiment analysis, text classification, meter classification, named entity recognition, semantic similarity, part-of-speech tagging, and semantic question similarity.

The six tasks used in the paper's fine-tuning experiments (dialect identification, machine translation, NLI, multiple choice, sarcasm detection, summarization) are a subset of these; filter by `instruction_template_id` against the `experimental` subset of the templates dataset to isolate them.

**Size:** 46,263,378 rendered instructions in a single `train` split.

## Usage

```python
from datasets import load_dataset

ds = load_dataset("KFUPM-JRCAI/star-dataset-instructions")

# Filter by task
summarization = ds.filter(lambda x: 'summarization' in x['instruction_tasks'])

# Filter by dataset
xlsum = ds.filter(lambda x: x['dataset_name'] == 'GEM/xlsum')

# Access instruction parts
sample = ds['train'][0]
print(sample['instruction_input'])   # The prompt/question
print(sample['instruction_output'])  # The expected answer
print(sample['instruction_template']) # The raw Jinja2 template
```

## How Merging Works

1. Templates are fetched from the PromptLab API and filtered to approved status
2. Each template's referenced HuggingFace dataset is downloaded
3. For every (template, data sample, split) combination:
   - The Jinja2 template is rendered with the sample's fields as variables
   - The rendered text is split on `|||` into input and output parts
   - The record is stored with both the rendered instruction and template metadata
4. Templates that produce invalid splits (multiple `|||` occurrences) are skipped

The exact pipeline is a notebook in the repository: [`build_merged_instructions.ipynb`](https://github.com/KFUPM-JRCAI/star-instructions-tuning/blob/main/star_dataset/push_to_hf/build_merged_instructions.ipynb) renders every approved template against its source dataset and writes the merged Arrow dataset, and [`push_merged_instructions.ipynb`](https://github.com/KFUPM-JRCAI/star-instructions-tuning/blob/main/star_dataset/push_to_hf/push_merged_instructions.ipynb) uploads the result here. Read them to see exactly how any record in this dataset was produced, or to rebuild it yourself.

## Schema

| Column | Type | Description |
|---|---|---|
| `instruction_template_id` | int | ID of the template used |
| `instruction_name` | str | Human-readable template name |
| `instruction_template` | str | Raw Jinja2 template with `\|\|\|` separator |
| `instruction_input` | str | Rendered instruction text (before `\|\|\|`) |
| `instruction_output` | str | Expected output text (after `\|\|\|`) |
| `full_instruction` | str | Complete rendered text including `\|\|\|` |
| `instruction_tasks` | list[str] | NLP task(s) |
| `dataset_name` | str | Source HuggingFace dataset ID |
| `dataset_subset` | str | Dataset configuration/subset name |
| `answer_choices` | list[str] | Valid answers (classification tasks) |
| `text_direction` | str | Text direction (`ltr` or `rtl`) |
| `tags` | list[str] | Optional tags |
| `prompter_id` | str | Anonymized prompter identifier (a, b, c, ...) |
| `split_name` | str | Source data split (`train`, `test`, or `validation`) |

## Connection to STAR Templates

This dataset is derived from [STAR Templates](https://huggingface.co/datasets/KFUPM-JRCAI/star-dataset-templates). Each record here corresponds to a template from that dataset applied to a specific data sample. The `instruction_template_id` column can be used to join with the templates dataset for experiment-level metadata (e.g., which prompts were selected for tuning vs. evaluation).

## Paper

This dataset accompanies **[STAR: instruction tuning for Arabic across tasks, datasets, and models](https://link.springer.com/article/10.1007/s10579-026-09942-8)**, published in *Language Resources and Evaluation* (2026), which describes how the templates were collected and the experiments they were used for.

Code for the fine-tuning and evaluation experiments: [github.com/KFUPM-JRCAI/star-instructions-tuning](https://github.com/KFUPM-JRCAI/star-instructions-tuning).

If you use this dataset, please cite:

```bibtex
@article{alshaibani2026star,
  title   = {STAR: instruction tuning for Arabic across tasks, datasets, and models},
  author  = {Al-Shaibani, Maged S. and Alyafeai, Zaid and Ahmad, Irfan},
  journal = {Language Resources and Evaluation},
  volume  = {60},
  number  = {4},
  pages   = {69},
  year    = {2026},
  doi     = {10.1007/s10579-026-09942-8},
  url     = {https://link.springer.com/article/10.1007/s10579-026-09942-8}
}
```
