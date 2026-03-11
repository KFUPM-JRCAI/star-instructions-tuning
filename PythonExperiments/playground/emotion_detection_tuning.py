"""
Emotion detection fine-tuning script.

Fine-tunes JAIS-13B on the emotone_ar dataset for emotion classification using
a single prompt template from the promptlab API and LoRA.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

import sys
import random

import torch
import requests
import datasets
from tqdm.auto import tqdm
from jinja2 import Environment, StrictUndefined
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer

from dotenv import load_dotenv

# Add corekit to path
sys.path.append('/raid_storage/SLURM/home/slurm_majedalshaibani/Projects/instructions-tuning/jrcai_corekit/llms_corekit')

from llm import train_llm, LLMLoader, JAISInitializer, LoRAConfigRepository
from llm.text_generator import TextGenerator

# ============================================================
# Load environment variables
# ============================================================

load_dotenv()

# ============================================================
# Fetch prompts from promptlab API
# ============================================================

prompts = None
tries = 10
for i in tqdm(range(tries)):
    api_response = requests.get(url='https://promptlab.up.railway.app/api/prompt/list?project_secret_key=6Wirj')
    if api_response.ok:
        prompts = api_response.json()
        break
if not prompts:
    raise Exception('Failed to fetch prompts')

print(f"Total prompts fetched: {len(prompts)}")

# Filter to approved prompts
filtered_prompts = list(filter(lambda prompt: prompt['status'] == 'APPROVED', prompts))
print(f"Approved prompts: {len(filtered_prompts)}")

# ============================================================
# Get emotone_ar dataset prompts
# ============================================================

dataset_prompts = list(
    filter(
        lambda prompt: 'emotone_ar' in prompt['dataset_name'],
        filtered_prompts,
    )
)
print(f"emotone_ar prompts: {len(dataset_prompts)}")

# ============================================================
# Download the dataset
# ============================================================

emotone_ar_experimental = datasets.load_dataset('MagedSaeed/emotone_ar_experimental')
print(emotone_ar_experimental)

# ============================================================
# Template processing functions
# ============================================================

def preprocess_template(template):
    # remove punc at the end
    prefix, suffix = template.split('|||')
    prefix = prefix.strip()
    suffix = suffix.strip()
    if prefix.endswith('.'):
        prefix = prefix[:-1]
    return f'{prefix} {suffix}'


def apply_template(prompt_template, sample):
    template = prompt_template['template']
    template = preprocess_template(template)
    sample['answer_choices'] = prompt_template['answer_choices']
    env = Environment(undefined=StrictUndefined)
    template = env.from_string(template)
    rendered_template = template.render(**sample)
    return rendered_template

# ============================================================
# Render training prompts using one example prompt template
# ============================================================

example_prompt_template = dataset_prompts[4]
print(apply_template(example_prompt_template, emotone_ar_experimental['train'][2]))

rendered_train_prompts_dataset = list(
    map(
        lambda sample: apply_template(example_prompt_template, sample),
        tqdm(emotone_ar_experimental['train'].select(range(min(len(emotone_ar_experimental['train']), 10_000)))),
    )
)
print(f"Rendered training prompts: {len(rendered_train_prompts_dataset)}")

# ============================================================
# Fine-tune the LLM
# ============================================================

GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)

MODEL_PATH = '/hdd/shared_models/jais-13b'
TOKENIZER_PATH = MODEL_PATH

llm_loader = LLMLoader(
    MODEL_PATH,
    llm_initializer=JAISInitializer(),
)

model, tokenizer, generation_config = llm_loader()

# Split into train/eval and create (prefix, suffix) tuples
train_samples, eval_samples = train_test_split(rendered_train_prompts_dataset, test_size=0.1, random_state=GLOBAL_SEED)


def generate_tuple(sample):
    sample_words = sample.split(' ')
    prefix, suffix = ' '.join(sample_words[:-1]), f' {sample_words[-1]}'
    return prefix, suffix


train_samples = list(map(generate_tuple, train_samples))
eval_samples = list(map(generate_tuple, eval_samples))
print(f"Train samples: {len(train_samples)}, Eval samples: {len(eval_samples)}")

train_llm(
    model=model,
    tokenizer=tokenizer,
    train_samples=train_samples,
    eval_samples=eval_samples,
    peft_config=LoRAConfigRepository.jais_v1(),
    learning_rate=2.5e-4,
    epochs_count=10,
    train_batch_size=16,
    eval_batch_size=16,
    output_dir='./tuned_models/jais-13b/'
)
