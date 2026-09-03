"""
Emotion detection evaluation script.

Evaluates JAIS-13B (with optional LoRA adapter) on the emotone_ar test set
by computing per-option log-probabilities and selecting the most likely emotion.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import torch
import requests
import datasets
from tqdm.auto import tqdm
from jinja2 import Environment, StrictUndefined
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

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

emotone_ar_experimental = datasets.load_dataset('KFUPM-JRCAI/emotone_ar_experimental')
print(emotone_ar_experimental)

# ============================================================
# Template processing
# ============================================================

def apply_template(prompt_template, sample):
    template = prompt_template['template']
    sample['answer_choices'] = prompt_template['answer_choices']
    env = Environment(undefined=StrictUndefined)
    if "|||" not in template:
        raise ValueError("No ||| dividor")
    template = env.from_string(template)
    rendered_template = template.render(**sample)
    return rendered_template

# ============================================================
# Render test prompts
# ============================================================

example_prompt_template = dataset_prompts[4]
print(apply_template(example_prompt_template, emotone_ar_experimental['train'][2]))

rendered_test_prompts_dataset = list(
    map(lambda sample: apply_template(example_prompt_template, sample), emotone_ar_experimental['test'])
)
print(f"Rendered test prompts: {len(rendered_test_prompts_dataset)}")

# ============================================================
# Load model and tokenizer
# ============================================================

MODEL_PATH = '/hdd/shared_models/jais-13b'
TUNED_MODEL_PATH = './tuned_models/jais-13b'
TOKENIZER_PATH = MODEL_PATH


def load_model_and_tokenizer(model_name=MODEL_PATH, tokenizer_name=TOKENIZER_PATH):
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    return model, tokenizer


model, tokenizer = load_model_and_tokenizer()

# Load tuned adapter if available
if TUNED_MODEL_PATH:
    print('loading tuned model')
    model = PeftModel.from_pretrained(model, TUNED_MODEL_PATH)

# ============================================================
# Check tokenizer space handling
# ============================================================

leftspace_counter = 0
rightspace_counter = 0
for vocab, vocab_id in tqdm(tokenizer.get_vocab().items()):
    if vocab.startswith('\u0120'):
        leftspace_counter += 1
    elif vocab.endswith('\u0120'):
        rightspace_counter += 1
print(f"Left-space tokens: {leftspace_counter}, Right-space tokens: {rightspace_counter}")

# ============================================================
# Probability computation
# ============================================================

def compute_probability(model, tokenizer, prompt, option):
    prompt = prompt.strip()
    prompt = prompt.strip('.')
    prefix_inputs = tokenizer(prompt, add_special_tokens=True, return_tensors='pt').to('cuda')
    suffix = f' {option}'  # important: space before the option token
    suffix_inputs = tokenizer(suffix, add_special_tokens=False, return_tensors='pt').to('cuda')
    model.eval()
    with torch.no_grad():
        output_token_id = suffix_inputs['input_ids'][:, 0]
        outputs = model(**prefix_inputs)
        next_token_logits = outputs.logits.log_softmax(dim=-1)[0, -1]
        next_token_probs = next_token_logits[output_token_id]
    return next_token_probs.item()

# ============================================================
# Evaluate
# ============================================================

print(f"Answer choices: {example_prompt_template['answer_choices']}")


def evaluate_llm(model, tokenizer, dataset):
    correct_predictions = 0
    for prompt in tqdm(dataset):
        prompt, expected_output = prompt.split("|||")
        prompt = prompt.strip()
        expected_output = expected_output.strip()
        probabilities = []
        options = example_prompt_template['answer_choices']
        for option in options:
            prob = compute_probability(model, tokenizer, prompt, option)
            if prob is None:
                continue
            probabilities.append(prob)
        print('probs:', probabilities)
        predicted_index = probabilities.index(max(probabilities))
        predicted_output = options[predicted_index].strip()
        print('predicted output:', predicted_output, 'expected output:', expected_output)
        print('-' * 80)
        if predicted_output == expected_output:
            correct_predictions += 1
    accuracy = correct_predictions / len(dataset)
    return accuracy


accuracy = evaluate_llm(model, tokenizer, dataset=rendered_test_prompts_dataset)
print(f"Accuracy: {accuracy:.4f}")
