"""Migrate tuned model adapters from legacy Notebooks/Experiments/ to tuned_models/.

Copies LoRA adapters from:
    Notebooks/Experiments/{task}/{dataset}/tuned_models/{legacy_name}/
to:
    tuned_models/{task}/{dataset}/{new_name}/

Legacy model names are mapped to the new CLI naming convention defined in
PythonExperiments/src/models.py.

Usage:
    uv run python scripts/migrate_tuned_models.py              # execute migration
    uv run python scripts/migrate_tuned_models.py --dry-run    # preview only
"""

import argparse
import shutil
from pathlib import Path

# Legacy model folder name -> new CLI model key
LEGACY_TO_NEW = {
    "AceGPT-7B": "AceGPT-v1-7B",
    "Meta-Llama-3.1-8B": "Llama-3.1-8B",
    "Qwen3-8B": "Qwen3-8B",
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEGACY_BASE = PROJECT_ROOT / "Notebooks" / "Experiments"
NEW_BASE = PROJECT_ROOT / "tuned_models"


def discover_adapters():
    """Find all legacy adapter directories that contain adapter_config.json."""
    adapters = []
    for adapter_config in LEGACY_BASE.rglob("tuned_models/*/adapter_config.json"):
        adapter_dir = adapter_config.parent
        legacy_model_name = adapter_dir.name

        if legacy_model_name not in LEGACY_TO_NEW:
            print(f"  WARN: Unknown model name '{legacy_model_name}' at {adapter_dir}, skipping")
            continue

        # Parse task and dataset from path:
        # .../Notebooks/Experiments/{task}/{dataset}/tuned_models/{model}/
        tuned_models_dir = adapter_dir.parent  # tuned_models/
        dataset_dir = tuned_models_dir.parent
        task_dir = dataset_dir.parent

        adapters.append({
            "source": adapter_dir,
            "task": task_dir.name,
            "dataset": dataset_dir.name,
            "legacy_name": legacy_model_name,
            "new_name": LEGACY_TO_NEW[legacy_model_name],
        })

    return sorted(adapters, key=lambda a: (a["task"], a["dataset"], a["new_name"]))


def migrate(dry_run: bool = False):
    adapters = discover_adapters()

    if not adapters:
        print("No legacy adapters found.")
        return

    print(f"Found {len(adapters)} adapter(s) to migrate:\n")

    copied = 0
    skipped = 0

    for entry in adapters:
        dest = NEW_BASE / entry["task"] / entry["dataset"] / entry["new_name"]
        label = f"  {entry['task']}/{entry['dataset']}/{entry['legacy_name']} -> {entry['new_name']}"

        if dest.exists():
            print(f"  SKIP (already exists): {dest.relative_to(PROJECT_ROOT)}")
            skipped += 1
            continue

        print(label)
        if not dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(entry["source"], dest)
            copied += 1
        else:
            copied += 1

    print(f"\nDone. {'Would copy' if dry_run else 'Copied'}: {copied} | Skipped: {skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate tuned models to new directory structure")
    parser.add_argument("--dry-run", action="store_true", help="Preview without copying")
    args = parser.parse_args()
    migrate(dry_run=args.dry_run)
