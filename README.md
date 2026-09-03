# STAR: Instruction Tuning for Arabic Across Tasks, Datasets, and Models

Code and resources for **STAR** (*in**S**truction **T**uning for **AR**abic*) - a curated collection of Arabic NLP instruction templates - and for the systematic study of instruction tuning built on top of it.

> **KFUPM - Joint Research Center for AI (JRCAI)**

📄 **Paper:** [STAR: instruction tuning for Arabic across tasks, datasets, and models](https://link.springer.com/article/10.1007/s10579-026-09942-8) - *Language Resources and Evaluation* (2026)

🤗 **HuggingFace collection:** [STAR: Arabic instructions tuning](https://huggingface.co/collections/KFUPM-JRCAI/star-arabic-instructions-tuning) - the STAR templates and instructions datasets alongside the experimental dataset copies used in this work

## Overview

Instruction tuning has become a standard method to bridge the gap between an LLM's next-token objective and users' convenience of instruction-following behaviour. Arabic - morphologically rich, dialectally diverse, and under-resourced - has lagged behind on both the large-scale prompt collections and the systematic evidence needed to investigate whether these methods transfer and generalize.

This work approaches this domain with two complementary contributions:

1. **STAR**, a publicly available dataset of **355 Jinja2 instruction templates** spanning **87 HuggingFace datasets** and **20 NLP tasks**, contributed by 7 prompt designers, together with a rendered variant of **over 46 million instruction-output pairs** attained by merging the prompts with the datasets instances. The templates were collaboratively authored and peer-reviewed through the [PromptLab](https://promptlab.up.railway.app) platform ([see the promptlab paper published as an EACL demo paper](https://aclanthology.org/2026.eacl-demo.18/)).
2. **A systematic evaluation** using a 60-template subset of STAR (5 templates per dataset across 12 datasets). Three 8B-parameter LLMs spanning the spectrum of Arabic support - LLaMA 3.1-8B (general multilingual), AceGPT-v2-8B (Arabic-focused), and Qwen3-8B (recent multilingual with strong Arabic support) - are each evaluated in three variants (base, provider instruction-tuned, and our LoRA-tuned) over six Arabic NLP tasks, under **intra-dataset** (in-distribution) and **intra-task** (cross-dataset) evaluation. This yields **540 evaluations** (180 per LLM: 6 tasks x 2 datasets x 5 prompts x 3 variants).

## Released Artifacts

- **STAR templates** - [`KFUPM-JRCAI/star-dataset-templates`](https://huggingface.co/datasets/KFUPM-JRCAI/star-dataset-templates): 355 templates over 87 datasets and 20 tasks, with the 60-template experimental subset as its own config
- **STAR instructions** - [`KFUPM-JRCAI/star-dataset-instructions`](https://huggingface.co/datasets/KFUPM-JRCAI/star-dataset-instructions): 46M+ rendered instruction-output pairs
- **Experimental datasets** - the 12 preprocessed datasets the experiments actually consume, gathered with the two datasets above in the [STAR: Arabic instructions tuning](https://huggingface.co/collections/KFUPM-JRCAI/star-arabic-instructions-tuning) collection
- **LoRA tuning code** - tuned weights for all 24 task x model runs (6 tasks x the 4 registered models), committed under [`tuned_models/`](tuned_models/)
- **Evaluation task configs** - the generated lm-evaluation-harness YAML for each dataset x prompt, under [`eval_harness_extra_tasks/`](eval_harness_extra_tasks/)
- **This codebase** - the full tuning and evaluation pipelines behind the 540 evaluations reported in the paper

## Tasks and Datasets

Each task has a **primary dataset** (train + test splits, used for fine-tuning and intra-dataset evaluation) and a **secondary dataset** (test only, used for intra-task / cross-dataset evaluation). Datasets were shuffled and capped at **30k training** and **10k test** samples before use.

Each row links the original public dataset, and below it (↳) the **experimental version** - the preprocessed, split, and subsampled copy actually consumed by the tuning and evaluation pipelines in this repository. All of them are gathered in the [STAR: Arabic instructions tuning](https://huggingface.co/collections/KFUPM-JRCAI/star-arabic-instructions-tuning) collection on HuggingFace.

| Task | Primary Dataset | Secondary Dataset | Eval Type |
|------|----------------|-------------------|-----------|
| Dialect Identification | [AraBench_dev](https://huggingface.co/datasets/arbml/AraBench_dev)<br>↳ [`arabench_dev_experimental`](https://huggingface.co/datasets/KFUPM-JRCAI/arabench_dev_experimental) | [Arabic_Dialects_Dataset](https://huggingface.co/datasets/arbml/Arabic_Dialects_Dataset)<br>↳ [`arabic_dialects_dataset_experimental`](https://huggingface.co/datasets/KFUPM-JRCAI/arabic_dialects_dataset_experimental) | Classification |
| NLI | [ArEntail](https://huggingface.co/datasets/arbml/ArEntail)<br>↳ [`ArEntail_experimental`](https://huggingface.co/datasets/KFUPM-JRCAI/ArEntail_experimental) | [ArabicTE](https://huggingface.co/datasets/arbml/ArabicTE)<br>↳ [`ArabicTE_experimental`](https://huggingface.co/datasets/KFUPM-JRCAI/ArabicTE_experimental) | Classification |
| NLU (MCQ) | [ArabicMMLU](https://huggingface.co/datasets/arbml/ArabicMMLU)<br>↳ [`ArabicMMLU_experimental`](https://huggingface.co/datasets/KFUPM-JRCAI/ArabicMMLU_experimental) | [Belebele](https://huggingface.co/datasets/facebook/belebele)<br>↳ [`belebele_experimental`](https://huggingface.co/datasets/KFUPM-JRCAI/belebele_experimental) | Classification |
| Sarcasm Detection | [ArSarcasm_v2](https://huggingface.co/datasets/arbml/ArSarcasm_v2)<br>↳ [`ArSarcasm_v2_experimental`](https://huggingface.co/datasets/KFUPM-JRCAI/ArSarcasm_v2_experimental) | [iSarcasmEval](https://huggingface.co/datasets/arbml/iSarcasmEval_task_A)<br>↳ [`iSarcasmEval_task_A_experimental`](https://huggingface.co/datasets/KFUPM-JRCAI/iSarcasmEval_task_A_experimental) | Classification |
| Machine Translation (en→ar) | [opus-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100)<br>↳ [`opus-100_ar_en_experimental`](https://huggingface.co/datasets/KFUPM-JRCAI/opus-100_ar_en_experimental) | [tatoeba_mt](https://huggingface.co/datasets/Helsinki-NLP/tatoeba_mt)<br>↳ [`tatoeba_mt_ara_eng_experimental`](https://huggingface.co/datasets/KFUPM-JRCAI/tatoeba_mt_ara_eng_experimental) | Generation |
| Summarization | [xlsum](https://huggingface.co/datasets/GEM/xlsum)<br>↳ [`xlsum_arabic_experimental`](https://huggingface.co/datasets/KFUPM-JRCAI/xlsum_arabic_experimental) | [AraSum](https://huggingface.co/datasets/arbml/AraSum)<br>↳ [`AraSum_arabic_experimental`](https://huggingface.co/datasets/KFUPM-JRCAI/AraSum_arabic_experimental) | Generation |

## Models

Each model family is evaluated in three variants: **base**, **chat/instruct**, and **LoRA fine-tuned**.

| Model | Base | Chat/Instruct | Tuned |
|-------|------|---------------|-------|
| AceGPT-v2 | [AceGPT-v2-8B](https://huggingface.co/FreedomIntelligence/AceGPT-v2-8B) | [AceGPT-v2-8B-Chat](https://huggingface.co/FreedomIntelligence/AceGPT-v2-8B-Chat) | AceGPT-v2-8B-tuned |
| Meta Llama 3.1 | [Meta-Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) | [Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | Meta-Llama-3.1-8B-tuned |
| Qwen 3 | [Qwen3-8B-Base](https://huggingface.co/Qwen/Qwen3-8B-Base) | [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | Qwen3-8B-tuned |

The **base** and **chat/instruct** columns link to the upstream HuggingFace checkpoints; the **tuned** column is produced in this repository - LoRA adapters are committed under `tuned_models/`, and the variant name is the directory key used under `evaluation_results/` (generated locally, not tracked in Git).

## Prompts

Prompts are Jinja2 templates authored, peer-reviewed, and served through the [PromptLab](https://promptlab.up.railway.app) platform ([paper](https://aclanthology.org/2026.eacl-demo.18/) | [code](https://github.com/KFUPM-JRCAI/PromptLab)), which this project uses as its prompt-curation backend. Each template uses `|||` as a separator between instruction input and expected output.

For example, a dialect-identification template looks like:

```
Classify the dialect of the following Arabic text: {{ sentence }} ||| {{ dialect }}
```

For each dataset, **5 approved templates** are selected, chosen to maximise stylistic diversity across broad categories, like:

- **Direct** - straightforward task instructions
- **Structured** - step-by-step instructions, possibly with reasoning (CoT-style)
- **Context-Rich** - instructions carrying domain knowledge and cultural context
- **Role-based** - prompts positioning the model as a domain expert

## Project Structure

```
star-instructions-tuning/
├── PythonExperiments/          # CLI-based training & evaluation pipelines (run from project root)
│   ├── tune.py                 # Fine-tuning entry point (argparse)
│   ├── run_eval.py             # Evaluation entry point (fire)
│   ├── playground/             # One-off exploratory scripts
│   └── src/
│       ├── experiments.py      # Task configs: prompt IDs, HF datasets, training params
│       ├── models.py           # Model registry: weight paths, initializers, variant names
│       ├── tuning.py           # LoRA fine-tuning pipeline
│       ├── evaluation.py       # lm-evaluation-harness integration pipeline
│       ├── promptlab.py        # PromptLab API client
│       ├── results_table.py    # Results aggregation into tables
│       ├── gpu.py              # CUDA_VISIBLE_DEVICES setup (must run before torch import)
│       └── preprocessing/      # Per-dataset preprocessing + eval-harness YAML templates
├── Notebooks/
│   ├── Experiments/            # Jupyter-based experiments (per task/dataset/model)
│   └── results_visualization/  # Box plots, scatter plots, stacked bar charts
├── openrouter_eval/            # API-based evaluation via OpenRouter (exploratory, unmaintained)
├── star_dataset/               # STAR dataset tooling
│   ├── analyze_star.py         # Dataset statistics & figures
│   └── push_to_hf/             # Build & publish the templates / instructions datasets
├── ablations/                  # Self-contained ablation studies (e.g. dialect choice taxonomy)
├── slurm/                      # SLURM submission scripts + job runners
├── scripts/                    # Analysis, statistics & verification utilities
├── eval_harness_extra_tasks/   # Generated YAML task configs for lm-evaluation-harness
├── tuned_models/               # LoRA adapter weights (one dir per task/model)
├── lm_eval/                    # Legacy ArabicMMLU-only eval setup (superseded)
├── pyproject.toml              # Project metadata & dependencies (uv)
└── uv.lock                     # Pinned dependency lockfile

# Generated at runtime and not tracked in Git:
#   experimental_hf_datasets/   # Parquet datasets built per prompt for the eval harness
#   evaluation_results/         # Evaluation outputs (one JSON per prompt per model)
#   jrcai_corekit/              # Internal LLM utilities library (separate repo)
```

## Setup

**Requirements**

- Python **>=3.10, <3.13** (declared in `pyproject.toml`; `.python-version` pins 3.12)
- [uv](https://docs.astral.sh/uv/) is recommended as a package manager
- CUDA-capable GPUs - tuning jobs are configured for 4 GPUs, evaluation for 2 (vLLM tensor parallelism)

```bash
git clone https://github.com/KFUPM-JRCAI/star-instructions-tuning.git
cd star-instructions-tuning
uv sync
```

**Setup details worth considering:**

- **Run everything from the project root.** The generated eval-harness YAML configs point at relative `dataset_path`s (`experimental_hf_datasets/...`), so scripts and notebooks break if launched from a subdirectory. You can also configure notebooks to run from the project root.
- **`jrcai_corekit` is a separate, gitignored dependency.** It is declared in `pyproject.toml` as an editable local source (`llms-corekit = { path = "jrcai_corekit", editable = true }`), so clone [llms-corekit](https://github.com/MagedSaeed/llms-corekit) into `jrcai_corekit/` at the project root before `uv sync`.
- **The evaluation harness is pinned** to `lm-evaluation-harness@v0.4.11`.
- **Model weights are resolved from local paths**, not the Hub. Update the `path` / `chat_path` fields in [`PythonExperiments/src/models.py`](PythonExperiments/src/models.py) to point to them from the hub or at your own checkpoint directories.
- **GPU config must be set before torch is imported** (`CUDA_VISIBLE_DEVICES`, `TOKENIZERS_PARALLELISM=false`). The CLI entry points handle this via `src/gpu.py`; notebooks set it in their first cell.
- **`pyproject.toml` declares an internal JRCAI package index** in addition to PyPI. Outside the JRCAI network those extra `[[tool.uv.index]]` entries can be removed - PyPI is the default index and resolves everything.
- **API keys:** pushing datasets to the Hub needs a `huggingface-cli login` token; the exploratory OpenRouter path additionally reads `OPENROUTER_API_KEY` (or `OPENAI_API_KEY`) from the environment or a local `.env`.

## Usage

### Fine-Tuning

```bash
# Fine-tune AceGPT-v2 on dialect identification using GPUs 0 and 1
uv run python PythonExperiments/tune.py --task dialect_identification --model AceGPT-v2-8B --gpus 0,1

# Fine-tune Llama 3.1 on summarization
uv run python PythonExperiments/tune.py --task summarization --model Llama-3.1-8B --gpus 0,1
```

- **Model keys:** `AceGPT-v1-7B`, `AceGPT-v2-8B`, `Llama-3.1-8B`, `Qwen3-8B`
- **Tasks:** `dialect_identification`, `machine_translation`, `NLI`, `NLU`, `sarcasm_detection`, `summarization`

### Evaluation

```bash
# Evaluate the chat variant of AceGPT-v2 on dialect identification (all datasets)
uv run python PythonExperiments/run_eval.py --task dialect_identification --dataset all --model AceGPT-v2-8B --variant chat

# Evaluate the tuned variant of Llama 3.1 on xlsum summarization
uv run python PythonExperiments/run_eval.py --task summarization --dataset xlsum --model Llama-3.1-8B --variant tuned
```

**Variants:** `base`, `chat`, `tuned`. `--dataset`, `--model` and `--variant` all accept `all` (the default), which expands over the whole matrix. Existing result files are skipped unless re-evaluation is forced.

### SLURM (HPC Cluster)

```bash
# Submit all pending tuning jobs (4 GPUs each)
bash slurm/submit_tuning_jobs.sh

# Submit evaluation orchestrator (2 GPUs)
bash slurm/submit_eval_jobs.sh

# Preview without submitting
bash slurm/submit_tuning_jobs.sh --dry-run
```

> **Note on `openrouter_eval/`:** the repository also contains a path for evaluating hosted API models (OpenRouter) instead of local weights. It was exploratory, is not part of the paper's results, and has not been kept current with the rest of the pipeline - expect it to need updates and maintenance before use.

## STAR Datasets

Everything published for this work is collected in the [**STAR: Arabic instructions tuning**](https://huggingface.co/collections/KFUPM-JRCAI/star-arabic-instructions-tuning) collection on HuggingFace. At its centre are two companion datasets:

- [**KFUPM-JRCAI/star-dataset-templates**](https://huggingface.co/datasets/KFUPM-JRCAI/star-dataset-templates) - 355 curated, peer-reviewed Jinja2 instruction templates covering 87 HuggingFace datasets across 20 Arabic NLP tasks (27 raw task labels before merging related categories). The `experimental` config holds the 60-template subset used in the paper, split into `tuning` and `evaluation`.
- [**KFUPM-JRCAI/star-dataset-instructions**](https://huggingface.co/datasets/KFUPM-JRCAI/star-dataset-instructions) - Over 46 million rendered instruction-output pairs, produced by applying the templates to their source datasets.

```python
from datasets import load_dataset

# Load all approved templates
templates = load_dataset("KFUPM-JRCAI/star-dataset-templates", "all")

# Load the experimental subset used in our study
tuning_prompts = load_dataset("KFUPM-JRCAI/star-dataset-templates", "experimental", split="tuning")
eval_prompts = load_dataset("KFUPM-JRCAI/star-dataset-templates", "experimental", split="evaluation")

# Load the full instruction dataset
instructions = load_dataset("KFUPM-JRCAI/star-dataset-instructions")
```

The collection also contains the per-dataset **experimental copies**: all 12 datasets from the table above, plus four left over from exploratory work that the paper does not report - `emotone_ar_experimental` and `Emotional_Tone_experimental` (emotion classification), `opus_infopankki_ar_en_experimental` (an alternative machine-translation set), and `ANERcorp_experimental` (named entity recognition, predating this project).

## Experiment Design

### Training

- **LoRA fine-tuning** (`r=16`, `alpha=32`, `dropout=0.05`, targeting `q_proj` / `v_proj`) - 18 runs total (6 tasks x 3 LLMs).
- **90% / 10% train/validation split**, fixed seed `42`.
- **Causal LM masking**: prompt tokens get `-100` labels (ignored in loss), only output tokens contribute to training

### Evaluation

Evaluation uses [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (v0.4.11), zero-shot, on NVIDIA A100 GPUs with a maximum context length of 8,192 tokens:

- **Answer-choice tasks** (MCQ/NLU, NLI, dialect identification, sarcasm detection): `multiple_choice` output type, scoring the log-likelihood of each choice conditioned on the prompt (via the HFLM backend). **Length-normalized accuracy (`acc_norm`)** is reported, to account for answer choices of differing token length.
- **Generation tasks** (machine translation, summarization): `generate_until` with greedy decoding and a 512-token generation cap (via the vLLM backend), scored with **corpus-level BLEU** via SacreBLEU. Outputs are normalized before scoring: any `<think>...</think>` block is stripped (mainly Qwen3), translations are truncated to their first line, and summaries have newlines and repeated whitespace collapsed to single spaces. On the answer-choice path the HFLM backend is additionally initialised with `enable_thinking=False`.
- **5 prompts per dataset**: each evaluated independently to measure prompt sensitivity.

### Intra-Dataset vs. Intra-Task Evaluation

- **Intra-dataset**: Train and evaluate on the same dataset (e.g., train on xlsum, evaluate on xlsum). The tuned model sees the *same* five prompts it was tuned on, so this measures in-distribution performance and prompt sensitivity.
- **Intra-task**: Train on a primary dataset, evaluate on a secondary dataset within the same task (e.g., train on xlsum, evaluate on AraSum). The tuned model here faces *both* a new data distribution and unseen prompt templates, so this measures cross-dataset generalization under instruction shift.

## Citation

If you use STAR datasets, templates, or this codebase in your research, please cite:

```bibtex
@article{al2026star,
  title={STAR: instruction tuning for Arabic across tasks, datasets, and models},
  author={Al-Shaibani, Maged S and Alyafeai, Zaid and Ahmad, Irfan},
  journal={Language Resources and Evaluation},
  volume={60},
  number={4},
  pages={69},
  year={2026},
  publisher={Springer}
}
```

The prompt templates were created and reviewed with **PromptLab**, which has its own paper:

```bibtex
@inproceedings{al-shaibani-etal-2026-promptlab,
    title = "{P}rompt{L}ab: A Collaborative Platform for Prompt Engineering and Dataset Curation",
    author = "Al-shaibani, Maged S. and Alyafeai, Zaid and Refai, Dania and
      Alomari, Nawaf and Ashraf, Ahmed and Alheraki, Mais and
      Alturki, Mustafa and Luqman, Hamzah and Ahmad, Irfan",
    booktitle = "Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 3: System Demonstrations)",
    month = mar,
    year = "2026",
    address = "Rabat, Morocco",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.eacl-demo.18/",
    doi = "10.18653/v1/2026.eacl-demo.18",
    pages = "225--260"
}
```

## Internal Dependencies

- [**jrcai_corekit**](https://github.com/MagedSaeed/llms-corekit) (cloned from the work done by Eng. Raed Mughaus) - Internal LLM utilities for training, evaluation, and inference. Installed as an editable dependency.