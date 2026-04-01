"""
Analyze the STAR dataset (star-templates + star-instructions) to produce
statistics and visualizations for the paper's STAR Dataset section.

Follows the patterns from star_dataset/explore.ipynb
for task merging, prompter anonymization, and plot styling.

Outputs (all saved to star_dataset/analysis_output/):
  - star_templates_per_task.pdf
  - star_contributions_per_prompter.pdf
  - star_template_length_distribution.pdf
  - analysis_output.txt (full console output)
"""

import io
import os
import sys
from collections import Counter, defaultdict

import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from tqdm.auto import tqdm

SCRIPT_DIR = os.path.dirname(__file__)
FIGURES_DIR = os.path.join(SCRIPT_DIR, "analysis_output")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "analysis_output", "analysis_output.txt")
os.makedirs(FIGURES_DIR, exist_ok=True)


class TeeOutput:
    """Write to both console and a file simultaneously."""

    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.file = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def close(self):
        self.file.close()


tee = TeeOutput(OUTPUT_FILE)
sys.stdout = tee

STAR_INSTRUCTIONS_DIR = "star_dataset/push_to_hf/built_datasets/star-instructions"

# ── Configuration (from explore.ipynb) ────────────────────────────────────────

PROMPTER_ALIASES = {
    "zaid": ["zaid", "zaid1"],
    "ahmed": ["ahmed", "ahmed6"],
    "irfan": ["irfan", "irfan9"],
}

TASK_ALIASES = {
    "sentiment analysis": [
        "sentiment analysis",
        "sentiment Analysis (open-domain targeted)",
        "emotion classification",
    ],
    "question answering": ["question answering", "question answering (yes/no)"],
    "semantic similarity": ["semantic similarity", "semantic question similarity"],
    "topic classification": [
        "topic classification",
        "text classification",
        "theme classification",
    ],
    "stance detection": ["stance detection", "claim verification"],
}

MIN_PROMPTS_TASK = 5  # tasks with fewer prompts are grouped into "Others"

# ── 1. Fetch and filter prompts ──────────────────────────────────────────────

PROMPTLAB_API_URL = (
    "https://promptlab.up.railway.app/api/prompt/list?project_secret_key=6Wirj"
)

print("Fetching prompts from PromptLab API...")
response = requests.get(PROMPTLAB_API_URL)
response.raise_for_status()
all_prompts = response.json()
print(f"  Total prompts fetched: {len(all_prompts)}")

df = pd.DataFrame(all_prompts)
df["task_name"] = df["task"].apply(
    lambda t: t["name"] if isinstance(t, dict) else t
)

# Filter to approved LTR prompts
approved = df[(df["status"] == "APPROVED") & (df["text_direction"] == "ltr")].copy()
print(f"  Approved LTR prompts: {len(approved)}")

# ── 2. Apply prompter merging & anonymization ────────────────────────────────

alias_map = {v: k for k, variants in PROMPTER_ALIASES.items() for v in variants}
approved["prompter"] = approved["created_by"].map(lambda x: alias_map.get(x, x))

# Anonymize: rank by count, label as "Prompter 1", "Prompter 2", ...
prompter_counts = approved["prompter"].value_counts()
prompter_label_map = {
    name: f"Prompter {i + 1}" for i, name in enumerate(prompter_counts.index)
}
approved["prompter_label"] = approved["prompter"].map(prompter_label_map)

print(f"  Unique prompters (after merging): {approved['prompter'].nunique()}")

# ── 3. Apply task merging ─────────────────────────────────────────────────────

task_alias_map = {v: k for k, variants in TASK_ALIASES.items() for v in variants}
approved["task_merged"] = approved["task_name"].map(
    lambda x: task_alias_map.get(x, x)
)


def group_others(series, min_count):
    """Group categories with fewer than min_count into 'Others'."""
    counts = series.value_counts()
    small = counts[counts < min_count].index
    return series.replace({name: "Others" for name in small})


approved["task_grouped"] = group_others(approved["task_merged"], MIN_PROMPTS_TASK)

# Build the canonical set of "Others" tasks — reuse everywhere for consistency
others_tasks = set(
    approved.loc[approved["task_grouped"] == "Others", "task_merged"].unique()
)

print("\nTask merges applied:")
for canonical, variants in TASK_ALIASES.items():
    merged = [v for v in variants if v != canonical]
    if merged:
        print(f"  {merged} -> \"{canonical}\"")

# ── 4. Summary statistics ────────────────────────────────────────────────────

unique_datasets = approved["dataset_name"].nunique()
unique_tasks_merged = approved["task_merged"].nunique()

print("\n" + "=" * 60)
print("STAR TEMPLATES — SUMMARY STATISTICS")
print("=" * 60)
print(f"  Total prompts (all statuses):    {len(df)}")
print(f"  Approved LTR prompts:            {len(approved)}")
print(f"  Unique tasks (after merging):    {unique_tasks_merged}")
print(f"  Unique tasks (before merging):   {approved['task_name'].nunique()}")
print(f"  Unique datasets:                 {unique_datasets}")
print(f"  Unique prompters:                {approved['prompter'].nunique()}")

# ── 5. Plot style ─────────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid", font_scale=1.05)

# ── 6. Templates per task (bar chart) ────────────────────────────────────────

task_counts = approved["task_grouped"].value_counts().sort_values()

print(f"\n{'Task':<45} {'Templates':>9}")
print("-" * 56)
for task in task_counts.sort_values(ascending=False).index:
    print(f"  {task:<43} {task_counts[task]:>9}")

fig, ax = plt.subplots(figsize=(9, 7))
sns.barplot(
    x=task_counts.values,
    y=task_counts.index,
    hue=task_counts.index,
    palette="crest",
    legend=False,
    ax=ax,
    edgecolor="white",
    linewidth=0.5,
)
for i, v in enumerate(task_counts.values):
    ax.text(
        v + 0.4, i, str(v), va="center", fontsize=8.5,
        fontweight="semibold", color="#555555",
    )
ax.set_xlabel("Number of Prompts", fontsize=11, color="#333333", labelpad=10)
ax.set_ylabel("")
ax.set_title(
    "Distribution of Prompts Across Tasks",
    fontsize=14, fontweight="bold", color="#222222", pad=15,
)
ax.tick_params(axis="y", labelsize=10, colors="#444444")
ax.tick_params(axis="x", labelsize=9, colors="#666666")
ax.set_xlim(0, task_counts.max() * 1.14)
sns.despine(left=True, bottom=True)
ax.grid(axis="x", linestyle="--", alpha=0.3)
ax.set_axisbelow(True)
plt.tight_layout()
path = os.path.join(FIGURES_DIR, "star_templates_per_task.pdf")
fig.savefig(path, bbox_inches="tight", dpi=300)
plt.close()
print(f"\nSaved: {path}")

# ── 7. Contributions per prompter (bar chart) ────────────────────────────────

# Relabel "Prompter 1" etc. to just "1", "2", ...
prompter_counts_numeric = approved["prompter_label"].str.replace(
    "Prompter ", "", regex=False,
).value_counts().sort_values()

print(f"\n{'Prompter':<15} {'Count':>5}")
print("-" * 22)
for name in prompter_counts_numeric.sort_values(ascending=False).index:
    print(f"  {name:<13} {prompter_counts_numeric[name]:>5}")

fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(
    x=prompter_counts_numeric.values,
    y=prompter_counts_numeric.index,
    hue=prompter_counts_numeric.index,
    palette="crest",
    legend=False,
    ax=ax,
    edgecolor="white",
    linewidth=0.5,
)
for i, v in enumerate(prompter_counts_numeric.values):
    ax.text(
        v + 0.5, i, str(v), va="center", fontsize=8.5,
        fontweight="semibold", color="#555555",
    )
ax.set_xlabel("Number of Prompts", fontsize=11, color="#333333", labelpad=10)
ax.set_ylabel("Prompter", fontsize=11, color="#333333", labelpad=10)
ax.set_title(
    "Distribution of Prompts Across Prompters",
    fontsize=14, fontweight="bold", color="#222222", pad=15,
)
ax.tick_params(axis="y", labelsize=10, colors="#444444")
ax.tick_params(axis="x", labelsize=9, colors="#666666")
ax.set_xlim(0, prompter_counts_numeric.max() * 1.14)
sns.despine(left=True, bottom=True)
ax.grid(axis="x", linestyle="--", alpha=0.3)
ax.set_axisbelow(True)
plt.tight_layout()
path = os.path.join(FIGURES_DIR, "star_contributions_per_prompter.pdf")
fig.savefig(path, bbox_inches="tight", dpi=300)
plt.close()
print(f"Saved: {path}")

# ── 8. Template length distribution — words only (histogram) ─────────────────

template_lengths = approved["template"].apply(lambda t: len(t.split()))

print(
    f"\nTemplate length (words): min={template_lengths.min()}, "
    f"max={template_lengths.max()}, mean={template_lengths.mean():.1f}, "
    f"median={template_lengths.median():.1f}"
)

fig, ax = plt.subplots(figsize=(7, 4))
sns.histplot(
    template_lengths, bins=30, color="#55A868", edgecolor="white",
    alpha=0.8, ax=ax,
)
ax.axvline(
    template_lengths.median(), color="red", linestyle="--",
    label=f"Median: {template_lengths.median():.0f}",
)
ax.set_xlabel("Number of Words", fontsize=11, color="#333333", labelpad=10)
ax.set_ylabel("Count", fontsize=11, color="#333333", labelpad=10)
ax.set_title(
    "Template Length Distribution",
    fontsize=14, fontweight="bold", color="#222222", pad=15,
)
ax.legend()
sns.despine(left=True, bottom=True)
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.set_axisbelow(True)
plt.tight_layout()
path = os.path.join(FIGURES_DIR, "star_template_length_distribution.pdf")
fig.savefig(path, bbox_inches="tight", dpi=300)
plt.close()
print(f"Saved: {path}")

# ── 9. Merged samples per task (from star-instructions) ──────────────────────

print("\n" + "=" * 60)
print("STAR INSTRUCTIONS — MERGED SAMPLES PER TASK")
print("=" * 60)

if os.path.exists(STAR_INSTRUCTIONS_DIR):
    print(f"Loading star-instructions from {STAR_INSTRUCTIONS_DIR}...")
    star_instructions = datasets.Dataset.load_from_disk(STAR_INSTRUCTIONS_DIR)
    print(f"  Total merged instructions: {len(star_instructions):,}")

    # Count samples per task using Arrow -> pandas (zero-copy where possible)
    print("  Selecting columns via Arrow and converting to pandas...")
    arrow_table = star_instructions.data.select(
        ["instruction_tasks", "instruction_template_id"]
    )
    tasks_df = arrow_table.to_pandas()
    tasks_df.rename(
        columns={"instruction_template_id": "template_id"}, inplace=True,
    )
    del arrow_table

    # Explode the list column so each task gets its own row
    print("  Exploding task lists...")
    tasks_df = tasks_df.explode("instruction_tasks")

    # Apply task alias merging + same "Others" grouping as the templates table
    tasks_df["task"] = tasks_df["instruction_tasks"].map(
        lambda t: task_alias_map.get(t, t)
    )
    tasks_df["task"] = tasks_df["task"].map(
        lambda t: "Others" if t in others_tasks else t
    )

    # Aggregate: sample counts and unique template counts per task
    sample_counts = tasks_df.groupby("task").size().rename("samples")
    template_counts = (
        tasks_df.groupby("task")["template_id"].nunique().rename("templates")
    )
    stats = pd.concat([template_counts, sample_counts], axis=1).sort_values(
        "samples", ascending=False,
    )

    # Move "Others" to the end if present
    if "Others" in stats.index:
        others_row = stats.loc[["Others"]]
        stats = pd.concat([stats.drop("Others"), others_row])

    print(f"\n{'Task':<45} {'Templates':>9} {'Samples':>14}")
    print("-" * 70)
    for task, row in stats.iterrows():
        print(f"  {task:<43} {row['templates']:>9} {row['samples']:>14,}")
    print("-" * 70)
    print(
        f"  {'Total':<43} "
        f"{tasks_df['template_id'].nunique():>9} "
        f"{len(tasks_df):>14,}"
    )
else:
    print(
        f"  WARNING: {STAR_INSTRUCTIONS_DIR} not found. "
        f"Skipping merged samples analysis."
    )
    print("  Run build_merged_instructions.ipynb first.")

print("\nAnalysis complete.")
print(f"Output saved to: {OUTPUT_FILE}")

sys.stdout = tee.terminal
tee.close()
