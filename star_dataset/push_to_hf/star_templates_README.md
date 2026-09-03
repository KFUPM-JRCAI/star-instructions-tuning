---
pretty_name: STAR Templates
language:
  - ar
multilinguality:
  - monolingual
size_categories:
  - n<1K
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
  - prompt-templates
dataset_info:
  - config_name: all
    features:
      - name: template_id
        dtype: int64
      - name: template_name
        dtype: string
      - name: template
        dtype: string
      - name: tasks
        sequence: string
      - name: dataset_name
        dtype: string
      - name: dataset_subset
        dtype: string
      - name: answer_choices
        sequence: string
      - name: text_direction
        dtype: string
      - name: tags
        sequence: string
      - name: prompter_id
        dtype: string
  - config_name: experimental
    features:
      - name: template_id
        dtype: int64
      - name: template_name
        dtype: string
      - name: template
        dtype: string
      - name: tasks
        sequence: string
      - name: dataset_name
        dtype: string
      - name: dataset_subset
        dtype: string
      - name: answer_choices
        sequence: string
      - name: text_direction
        dtype: string
      - name: tags
        sequence: string
      - name: prompter_id
        dtype: string
      - name: experiment_task
        dtype: string
      - name: dataset_role
        dtype: string
      - name: paired_dataset
        dtype: string
    splits:
      - name: tuning
      - name: evaluation
configs:
  - config_name: all
    data_files:
      - split: train
        path: all/train-*
  - config_name: experimental
    data_files:
      - split: tuning
        path: experimental/tuning-*
      - split: evaluation
        path: experimental/evaluation-*
---

# STAR Templates

STAR Templates is a curated collection of 355 Jinja2 instruction templates for Arabic NLP tasks, spanning 27 tasks across 87 source datasets, contributed by 7 prompters. The templates were authored collaboratively on [PromptLab](https://aclanthology.org/2026.eacl-demo.18/). This dataset and the experiments built on it are described in [STAR: instruction tuning for Arabic across tasks, datasets, and models](https://link.springer.com/article/10.1007/s10579-026-09942-8).

For these templates rendered against datasets samples, see the companion dataset: [STAR Instructions](https://huggingface.co/datasets/KFUPM-JRCAI/star-dataset-instructions).

📦 **Code:** the tuning and evaluation pipelines that use these templates live at [github.com/KFUPM-JRCAI/star-instructions-tuning](https://github.com/KFUPM-JRCAI/star-instructions-tuning).

## Overview

This dataset contains prompt templates designed for fine-tuning and evaluating large language models on Arabic NLP tasks. Each template is a Jinja2 string with a `|||` separator that divides the instruction input from the expected output.

**Coverage:** the `all` catalog spans **27 tasks** across **87 source datasets**. The best represented are dialect identification, summarization, sentiment analysis, stance detection, multiple choice, sarcasm detection, machine translation, NLI, question answering, and diacritization. The long tail includes offensive language detection, emotion and topic classification, commonsense validation, poetry meter and era classification, named entity recognition, part-of-speech tagging, and math solving.

The `experimental` subset described below is a deliberately narrow 6-task slice of this catalog, not the whole of it.

## Subsets

### `all`

The full catalog of all 355 approved templates from the PromptLab platform. Users can filter by `tasks`, `dataset_name`, or any other column.

```python
from datasets import load_dataset
templates = load_dataset("KFUPM-JRCAI/star-dataset-templates", "all")
```

### `experimental`

A curated set of 60 templates (5 per dataset, across 12 datasets) that were specifically selected for our fine-tuning and evaluation experiments. This subset has two splits:

- **`tuning`** (30 templates): Prompts from primary datasets, used for fine-tuning models and for intra-dataset evaluation (evaluating on the same dataset used for training).
- **`evaluation`** (30 templates): Prompts from secondary datasets, used for intra-task evaluation (evaluating on a different dataset within the same task to test generalization).

```python
from datasets import load_dataset

# Load tuning prompts (from primary datasets)
tuning = load_dataset("KFUPM-JRCAI/star-dataset-templates", "experimental", split="tuning")

# Load evaluation prompts (from secondary datasets)
evaluation = load_dataset("KFUPM-JRCAI/star-dataset-templates", "experimental", split="evaluation")
```

## Experiment Setup

Our experiments evaluate how different instruction prompts affect LLM performance on Arabic NLP tasks. The paper fine-tunes three model families - [AceGPT-v2-8B](https://huggingface.co/FreedomIntelligence/AceGPT-v2-8B), [Meta-Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B), and [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B-Base) - and compares three variants per model: base, chat/instruct, and LoRA fine-tuned. (The [accompanying repository](https://github.com/KFUPM-JRCAI/star-instructions-tuning) additionally carries results for an earlier AceGPT-v1-7B.)

Each task has two datasets — a **primary** dataset (used for fine-tuning) and a **secondary** dataset (used for cross-dataset evaluation):

| Task | Primary Dataset | Secondary Dataset |
|---|---|---|
| Dialect Identification | arbml/AraBench_dev | arbml/Arabic_Dialects_Dataset |
| Machine Translation | Helsinki-NLP/opus-100 | Helsinki-NLP/tatoeba_mt |
| NLI | arbml/ArEntail | arbml/ArabicTE |
| NLU | arbml/ArabicMMLU | facebook/belebele |
| Sarcasm Detection | arbml/ArSarcasm_v2 | arbml/iSarcasmEval_task_A |
| Summarization | GEM/xlsum | arbml/AraSum |

### Intra-Dataset Experiment

The model is fine-tuned on a primary dataset using 5 selected prompts, then evaluated on the **same dataset** using the same 5 prompts. This measures how different instruction formulations affect performance when training and evaluation data share the same distribution. The `tuning` split of the `experimental` subset contains these prompts.

### Intra-Task Experiment

The model fine-tuned on a primary dataset is evaluated on a **different dataset within the same task** (e.g., trained on xlsum, evaluated on AraSum — both summarization). This measures cross-dataset generalization within a task. The `evaluation` split of the `experimental` subset contains the prompts used for this cross-dataset evaluation.

## Template Format

Templates use [Jinja2](https://jinja.palletsprojects.com/) syntax with `{{variable}}` placeholders. The `|||` separator divides the input (left) from the expected output (right). See the below example:

```
You are an expert Arabic text summarizer. The following article:
{{article}}
can be summarized as:
|||
{{summary}}
```

For classification tasks, templates may include `answer_choices`:

```
Consider this text: {{arabic}}. The dialect is: ||| {{answer_choices[label]}}
```

## Dataset Schema

### `all` subset

| Column | Type | Description |
|---|---|---|
| `template_id` | int | Unique template identifier |
| `template_name` | str | Human-readable template name |
| `template` | str | Raw Jinja2 template with `\|\|\|` separator |
| `tasks` | list[str] | NLP task(s) this template targets |
| `dataset_name` | str | HuggingFace dataset ID (e.g., `arbml/AraSum`) |
| `dataset_subset` | str | Dataset configuration/subset name |
| `answer_choices` | list[str] | Valid answer options (for classification tasks) |
| `text_direction` | str | Text direction (`ltr` or `rtl`) |
| `tags` | list[str] | Optional tags |
| `prompter_id` | str | Anonymized prompter identifier (a, b, c, ...) |

### `experimental` subset (additional columns)

| Column | Type | Description |
|---|---|---|
| `experiment_task` | str | Task name for this experiment |
| `dataset_role` | str | `primary` (tuning dataset) or `secondary` (cross-eval dataset) |
| `paired_dataset` | str | The other dataset in the experiment pair |

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
