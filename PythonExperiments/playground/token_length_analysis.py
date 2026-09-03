"""
Token length analysis script.

Computes token-length statistics (mean, median, P90, P95, P99, max) for
input and output columns of summarization and machine-translation datasets
across multiple tokenizers (AceGPT-v1-7B, AceGPT-v2-8B, Llama-3.1-8B, Qwen3-8B).
"""

import numpy as np
import pandas as pd
import datasets
from tqdm.auto import tqdm
from transformers import AutoTokenizer

# ============================================================
# Configuration
# ============================================================

MODEL_PATHS = {
    "AceGPT-v1-7B": "/raid_storage/shared_models/AceGPT-7B",
    "AceGPT-v2-8B": "/raid_storage/shared_models/AceGPT-v2-8B",
    "Llama-3.1-8B": "/raid_storage/shared_models/Meta-Llama-3.1-8B",
    "Qwen3-8B": "/raid_storage/shared_models/Qwen3-8B-Base",
}

DATASETS_CONFIG = {
    "dialect_identification": {
        "AraBench_dev": {
            "hf_path": "KFUPM-JRCAI/arabench_dev_experimental",
            "input_col": "arabic",
            "splits": ["train", "test"],
        },
        "Arabic_Dialects_Dataset": {
            "hf_path": "KFUPM-JRCAI/arabic_dialects_dataset_experimental",
            "input_col": "Text",
            "splits": ["test"],
        },
    },
    "summarization": {
        "xlsum": {
            "hf_path": "KFUPM-JRCAI/xlsum_arabic_experimental",
            "input_col": "text",
            "output_col": "target",
            "splits": ["train", "test"],
        },
        "AraSum": {
            "hf_path": "KFUPM-JRCAI/AraSum_arabic_experimental",
            "input_col": "article",
            "output_col": "summary",
            "splits": ["train", "test"],
        },
    },
    "machine_translation": {
        "opus-100": {
            "hf_path": "KFUPM-JRCAI/opus-100_ar_en_experimental",
            "input_col": "en",
            "output_col": "ar",
            "splits": ["train", "test"],
        },
        "tatoeba_mt": {
            "hf_path": "KFUPM-JRCAI/tatoeba_mt_ara_eng_experimental",
            "input_col": "sourceString",
            "output_col": "targetString",
            "splits": ["train", "test"],
        },
    },
}

# ============================================================
# Load tokenizers
# ============================================================

tokenizers = {}
for name, path in MODEL_PATHS.items():
    print(f"Loading tokenizer: {name}")
    tokenizers[name] = AutoTokenizer.from_pretrained(path)
print("All tokenizers loaded.")

# ============================================================
# Load datasets and compute token lengths
# ============================================================

results = []

for task, ds_configs in DATASETS_CONFIG.items():
    for ds_name, ds_info in ds_configs.items():
        print(f"Loading {task}/{ds_name}...")
        hf_ds = datasets.load_dataset(ds_info["hf_path"])

        for split in ds_info["splits"]:
            if split not in hf_ds:
                print(f"  Skipping split '{split}' (not found)")
                continue

            split_ds = hf_ds[split]
            print(f"  {split}: {len(split_ds)} samples")

            sides = [("input", ds_info["input_col"])]
            if "output_col" in ds_info:
                sides.append(("output", ds_info["output_col"]))
            for side, col in tqdm(
                sides,
                desc=f"  {split} sides",
            ):
                texts = split_ds[col]
                for model_name, tokenizer in tokenizers.items():
                    token_lengths = [
                        len(tokenizer.encode(text))
                        for text in tqdm(texts, desc=f"    {side}/{model_name}", leave=False)
                    ]
                    results.append({
                        "task": task,
                        "dataset": ds_name,
                        "split": split,
                        "side": side,
                        "model": model_name,
                        "token_lengths": token_lengths,
                    })

print("Done.")

# ============================================================
# Statistics: Mean, Median, P90, P95, P99
# ============================================================

stats_rows = []
for r in results:
    lengths = np.array(r["token_lengths"])
    stats_rows.append({
        "Task": r["task"],
        "Dataset": r["dataset"],
        "Split": r["split"],
        "Side": r["side"],
        "Tokenizer": r["model"],
        "Count": len(lengths),
        "Mean": round(np.mean(lengths), 1),
        "Median": round(np.median(lengths), 1),
        "P90": round(np.percentile(lengths, 90), 1),
        "P95": round(np.percentile(lengths, 95), 1),
        "P99": round(np.percentile(lengths, 99), 1),
        "Max": int(np.max(lengths)),
    })

stats_df = pd.DataFrame(stats_rows)

# Per-side stats
for side in ["input", "output"]:
    print(f"\n=== {side.upper()} Stats ===")
    print(stats_df[stats_df["Side"] == side].to_string())

# Per-task stats
for task in ["dialect_identification", "summarization", "machine_translation"]:
    for side in ["input", "output"]:
        task_side = [r for r in results if r["task"] == task and r["side"] == side]
        if not task_side:
            continue
        all_lengths = np.concatenate([r["token_lengths"] for r in task_side])
        print(f"\n=== {task.upper()} / {side.upper()} (all datasets, splits, tokenizers combined) ===")
        print(f"  Total samples (tokenized): {len(all_lengths)}")
        print(f"  Mean:   {np.mean(all_lengths):.1f}")
        print(f"  Median: {np.median(all_lengths):.1f}")
        print(f"  P90:    {np.percentile(all_lengths, 90):.1f}")
        print(f"  P95:    {np.percentile(all_lengths, 95):.1f}")
        print(f"  P99:    {np.percentile(all_lengths, 99):.1f}")
        print(f"  Max:    {int(np.max(all_lengths))}")