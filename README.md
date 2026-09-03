# STAR: Instruction Tuning for Arabic Across Tasks, Datasets, and Models

A systematic study of how different instruction prompts impact LLM fine-tuning and evaluation across 6 Arabic NLP tasks, 4 model families, and 12 datasets. This project fine-tunes models with LoRA, evaluates 3 model variants per family (base, chat, tuned), and measures per-prompt performance to quantify the effect of instruction formulation.

> **KFUPM - Joint Research Center for AI (JRCAI)**

## Key Results

- **STAR datasets** published on HuggingFace: [`KFUPM-JRCAI/star-instructions`](https://huggingface.co/datasets/KFUPM-JRCAI/star-instructions) and [`KFUPM-JRCAI/star-templates`](https://huggingface.co/datasets/KFUPM-JRCAI/star-templates)
- Emprical evaluation on a selection of Start dataset covering:
    - **6 NLP tasks** evaluated: dialect identification, machine translation, NLI, NLU, sarcasm detection, summarization
    - **3 model families** fine-tuned and compared: AceGPT-v2-8B, Meta-Llama-3.1-8B, Qwen3-8B.
    - **60 curated prompt templates** (5 per dataset) from the [PromptLab/PromptLab](https://PromptLab.up.railway.app) platform
    - **Cross-dataset generalization** tested via intra-task evaluation on held-out secondary datasets

## Tasks and Datasets

Each task has a **primary dataset** (used for fine-tuning + intra-dataset evaluation) and a **secondary dataset** (used for cross-dataset evaluation to test generalization):

| Task | Primary Dataset | Secondary Dataset | Eval Type |
|------|----------------|-------------------|-----------|
| Dialect Identification | [AraBench_dev](https://huggingface.co/datasets/arbml/AraBench_dev) | [Arabic_Dialects_Dataset](https://huggingface.co/datasets/arbml/Arabic_Dialects_Dataset) | Classification |
| Machine Translation | [opus-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100) | [tatoeba_mt](https://huggingface.co/datasets/Helsinki-NLP/tatoeba_mt) | Generation |
| NLI | [ArEntail](https://huggingface.co/datasets/arbml/ArEntail) | [ArabicTE](https://huggingface.co/datasets/arbml/ArabicTE) | Classification |
| NLU | [ArabicMMLU](https://huggingface.co/datasets/arbml/ArabicMMLU) | [Belebele](https://huggingface.co/datasets/facebook/belebele) | Classification |
| Sarcasm Detection | [ArSarcasm_v2](https://huggingface.co/datasets/arbml/ArSarcasm_v2) | [iSarcasmEval](https://huggingface.co/datasets/arbml/iSarcasmEval_task_A) | Classification |
| Summarization | [xlsum](https://huggingface.co/datasets/GEM/xlsum) | [AraSum](https://huggingface.co/datasets/arbml/AraSum) | Generation |

## Models

Each model family is evaluated in three variants: **base**, **chat/instruct**, and **LoRA fine-tuned**.

| Model | Base | Chat/Instruct | Tuned |
|-------|------|---------------|-------|
| AceGPT-v2 | AceGPT-v2-8B | AceGPT-v2-8B-Chat | AceGPT-v2-8B-tuned |
| Meta Llama 3.1 | Meta-Llama-3.1-8B | Meta-Llama-3.1-8B-Instruct | Meta-Llama-3.1-8B-tuned |
| Qwen 3 | Qwen3-8B | Qwen3-8B-chat | Qwen3-8B-tuned |

## Prompt System

Prompts are Jinja2 templates fetched from the [PromptLab](https://promptlab.up.railway.app) platform. Each template uses `|||` as a separator between instruction input and expected output:

```
Classify the dialect of the following Arabic text: {{ sentence }} ||| {{ dialect }}
```

For each dataset, **5 approved prompts** are selected. During training, prompts are distributed evenly across samples (each chunk of training data gets a different prompt), allowing the model to learn from diverse instruction formulations.

## Project Structure

```
star-instructions-tuning/
├── PythonExperiments/          # CLI-based training & evaluation pipelines
│   ├── tune.py                 # Fine-tuning entry point (argparse)
│   ├── run_eval.py             # Evaluation entry point (fire)
│   └── src/
│       ├── experiments.py      # Task configs, prompt IDs, training params
│       ├── models.py           # Model registry & variant mapping
│       ├── tuning.py           # LoRA fine-tuning pipeline
│       ├── evaluation.py       # Eval-harness integration pipeline
│       ├── promptlab.py        # Tajeeh API client
│       └── preprocessing/      # Per-dataset preprocessing functions
├── Notebooks/
│   ├── Experiments/            # Jupyter-based experiments (per task/dataset/model)
│   └── results_visualization/  # Box plots, scatter plots, bar charts
├── openrouter_eval/            # API-based evaluation (GPT-4, Gemini, etc.)
├── star_dataset/               # STAR instruction dataset for HuggingFace
├── slurm/                      # SLURM job submission scripts
├── evaluation_results/         # Evaluation outputs (JSON per prompt per model)
├── tuned_models/               # LoRA adapter weights
├── eval_harness_extra_tasks/   # YAML task configs for lm-evaluation-harness
├── experimental_hf_datasets/   # Pre-built HuggingFace datasets (Parquet)
├── scripts/                    # Analysis & verification utilities
├── jrcai_corekit/              # Internal LLM utilities library (gitignored)
└── pyproject.toml              # Project metadata & dependencies (uv)
```

## Setup

**Requirements:** Python 3.10-3.12, [uv](https://docs.astral.sh/uv/) package manager, CUDA-capable GPUs.

```bash
git clone https://github.com/KFUPM-JRCAI/star-instructions-tuning.git
cd star-instructions-tuning
uv sync
```

> **Tip:** If `uv sync` fails with download timeouts (e.g., large packages like `ray` timing out through the Nexus proxy), increase the HTTP timeout:
> ```bash
> UV_HTTP_TIMEOUT=300 uv sync
> ```

## Usage

### Fine-Tuning

```bash
# Fine-tune AceGPT on dialect identification using GPUs 0 and 1
uv run python PythonExperiments/tune.py --task dialect_identification --model AceGPT --gpus 0,1

# Fine-tune Llama on summarization
uv run python PythonExperiments/tune.py --task summarization --model Llama --gpus 0,1
```

**Model keys:** `AceGPT`, `Llama`, `Qwen`

### Evaluation

```bash
# Evaluate the chat variant of AceGPT on dialect identification (all datasets)
uv run python PythonExperiments/run_eval.py --task dialect_identification --dataset all --model AceGPT --variant chat

# Evaluate the tuned variant of Llama on xlsum summarization
uv run python PythonExperiments/run_eval.py --task summarization --dataset xlsum --model Llama --variant tuned
```

**Variants:** `base`, `chat`, `tuned`

### API-Based Evaluation (OpenRouter)

Evaluate hosted models without local GPUs:

```bash
export OPENROUTER_API_KEY=your_key_here

uv run python openrouter_eval/run_eval.py \
    --task summarization --dataset xlsum \
    --api-model google/gemini-3.1-pro-preview
```

### SLURM (HPC Cluster)

```bash
# Submit all pending tuning jobs (4 GPUs each)
bash slurm/submit_tuning_jobs.sh

# Submit evaluation orchestrator (2 GPUs)
bash slurm/submit_eval_jobs.sh

# Preview without submitting
bash slurm/submit_tuning_jobs.sh --dry-run
```

## STAR Datasets

The project publishes two companion datasets on HuggingFace:

- [**KFUPM-JRCAI/star-templates**](https://huggingface.co/datasets/KFUPM-JRCAI/star-templates) - Curated Jinja2 instruction templates for Arabic NLP tasks
- [**KFUPM-JRCAI/star-instructions**](https://huggingface.co/datasets/KFUPM-JRCAI/star-instructions) - Large-scale Arabic instruction-following dataset created by merging templates with data samples

```python
from datasets import load_dataset

# Load all approved templates
templates = load_dataset("KFUPM-JRCAI/star-templates", "all")

# Load the experimental subset used in our study
tuning_prompts = load_dataset("KFUPM-JRCAI/star-templates", "experimental", split="tuning")
eval_prompts = load_dataset("KFUPM-JRCAI/star-templates", "experimental", split="evaluation")

# Load the full instruction dataset
instructions = load_dataset("KFUPM-JRCAI/star-instructions")
```

## Experiment Design

### Training

- **LoRA fine-tuning** with task-specific hyperparameters (see table below)
- **Causal LM masking**: prompt tokens get `-100` labels (ignored in loss), only output tokens contribute to training
- **Early stopping** to prevent overfitting

| Parameter | Classification Tasks | Generation Tasks |
|-----------|---------------------|------------------|
| Learning rate | 2.5e-4 | 2.5e-4 |
| Epochs | 10 | 10 |
| Train batch size | 16 | 1 |
| Early stopping patience | 5 | 20 |
| Eval steps | 250 | 1000 |
| Precision | bf16 | bf16 |

### Evaluation

Evaluation uses [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (v0.4.11):

- **Classification tasks**: `multiple_choice` output type with `acc` / `acc_norm` metrics (via HFLM backend)
- **Generation tasks**: `generate_until` output type with BLEU (via vLLM backend)
- **API models**: `generate_until` with `exact_match` (via `openai-chat-completions` backend)
- **5 prompts per dataset**: each evaluated independently to measure prompt sensitivity

### Intra-Dataset vs. Intra-Task Evaluation

- **Intra-dataset**: Train and evaluate on the same dataset (e.g., train on xlsum, evaluate on xlsum) - measures how instruction formulation affects performance on familiar data distributions
- **Intra-task**: Train on a primary dataset, evaluate on a secondary dataset within the same task (e.g., train on xlsum, evaluate on AraSum) - measures cross-dataset generalization

## Internal Dependencies

- [**jrcai_corekit**](https://github.com/MagedSaeed/llms-corekit) (by Eng. Raed Mughaus) - Internal LLM utilities for training, evaluation, and inference. Installed as an editable dependency.

## Note on Git History Rewrite

The `evaluation_results/` directory was removed from Git LFS tracking and the repository history was rewritten to reduce LFS storage usage. If you had cloned this repo before this change, you will need to re-clone it:

```bash
git clone https://github.com/KFUPM-JRCAI/star-instructions-tuning.git
```