---
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

A large-scale Arabic instruction-following dataset created by merging Jinja2 prompt templates with actual data samples across 6 NLP tasks.

For the raw templates and experiment design details, see the companion dataset: [star-templates](KFUPM-JRCAI/star-templates).

## Overview

Each record in this dataset is created by rendering a Jinja2 prompt template against a real data sample, then splitting the result on `|||` to separate the instruction input from the expected output. The dataset preserves both the rendered instruction and the original template metadata.

**Tasks covered (6):** Dialect Identification, Machine Translation, NLI, NLU, Sarcasm Detection, Summarization.

## Usage

```python
from datasets import load_dataset

ds = load_dataset("KFUPM-JRCAI/star-instructions")

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

## Relationship to STAR Templates

This dataset is derived from [star-templates](KFUPM-JRCAI/star-templates). Each record here corresponds to a template from that dataset applied to a specific data sample. The `instruction_template_id` column can be used to join with the templates dataset for experiment-level metadata (e.g., which prompts were selected for tuning vs. evaluation).
