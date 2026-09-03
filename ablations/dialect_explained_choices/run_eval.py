"""Run lm-eval on the dialect choice-ablation parquets for the 3 LoRA-tuned models.

Three variants are supported (see build_data.py for the spec):
    explained_full       prompt body + choices both extended.
    explained_text_only  body extended, choices stay bare ADD.
    arabench_choices     bare body, 6 AraBench choices, multi-correct via metrics.py.

Outputs per (variant, model, prompt) JSON files.

Run from project root, after build_data.py:
    uv run python ablations/dialect_explained_choices/run_eval.py --variant all
    uv run python ablations/dialect_explained_choices/run_eval.py --variant explained_full
    uv run python ablations/dialect_explained_choices/run_eval.py --variant arabench_choices --model AceGPT-v2-8B
    uv run python ablations/dialect_explained_choices/run_eval.py --variant all --gpus 0,1
"""
import json
import os
import sys
from pathlib import Path

import fire

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ABLATION_ROOT = PROJECT_ROOT / "ablations" / "dialect_explained_choices"
DATA_ROOT = ABLATION_ROOT / "data"
YAML_ROOT = ABLATION_ROOT / "eval_harness_tasks"
RESULTS_ROOT = ABLATION_ROOT / "results"

PROMPT_IDS = [14102, 14783, 14784, 14790, 14851]

VARIANTS = ("explained_full", "explained_text_only", "arabench_choices")

# Task-name template matches what build_data.py writes into each YAML.
TASK_NAME_TEMPLATES = {
    "explained_full":      "Arabic_Dialects_Dataset_explained_full_prompt_{pid}",
    "explained_text_only": "Arabic_Dialects_Dataset_explained_text_only_prompt_{pid}",
    "arabench_choices":    "Arabic_Dialects_Dataset_arabench_choices_prompt_{pid}",
}

# (model_key) -> (base_model_path, adapter_path, results_dir_name)
MODELS = {
    "AceGPT-v2-8B": (
        "/raid_storage/shared_models/AceGPT-v2-8B",
        "tuned_models/dialect_identification/AraBench_dev/AceGPT-v2-8B",
        "AceGPT-v2-8B-tuned",
    ),
    "Llama-3.1-8B": (
        "/raid_storage/shared_models/Meta-Llama-3.1-8B",
        "tuned_models/dialect_identification/AraBench_dev/Llama-3.1-8B",
        "Meta-Llama-3.1-8B-tuned",
    ),
    "Qwen3-8B": (
        "/raid_storage/shared_models/Qwen3-8B-Base",
        "tuned_models/dialect_identification/AraBench_dev/Qwen3-8B",
        "Qwen3-8B-tuned",
    ),
}

MAX_MODEL_LEN_CAP = 8192
# Matches PythonExperiments/src/experiments.py:48-50 for the
# Arabic_Dialects_Dataset eval batch size (per-experiment override).
BATCH_SIZE = 16


def _setup_gpus(gpus: str | None) -> None:
    """Configure GPU visibility before any torch import. Mirrors
    PythonExperiments/src/gpu.py:setup_gpus."""
    if gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus)
        print(f"[GPU] Using GPUs: {gpus} (from --gpus)")
    elif "CUDA_VISIBLE_DEVICES" in os.environ:
        print(f"[GPU] Using GPUs: {os.environ['CUDA_VISIBLE_DEVICES']} (from CUDA_VISIBLE_DEVICES)")
    elif "SLURM_GPUS_ON_NODE" in os.environ:
        gpu_count = int(os.environ["SLURM_GPUS_ON_NODE"])
        gpu_ids = ",".join(str(i) for i in range(gpu_count))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        print(f"[GPU] Using GPUs: {gpu_ids} (from SLURM, {gpu_count} GPUs)")
    elif "SLURM_JOB_GPUS" in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_JOB_GPUS"]
        print(f"[GPU] Using GPUs: {os.environ['SLURM_JOB_GPUS']} (from SLURM_JOB_GPUS)")
    else:
        print("[GPU] Using all available GPUs (no restriction set)")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _init_hflm(model_path: str, adapter_path: str):
    from transformers import AutoConfig
    from lm_eval.models.huggingface import HFLM

    config = AutoConfig.from_pretrained(model_path)
    model_max_len = getattr(config, "max_position_embeddings", None) or MAX_MODEL_LEN_CAP
    max_length = min(model_max_len, MAX_MODEL_LEN_CAP)

    print(f"[Eval] HFLM (model={model_path}, peft={adapter_path}, "
          f"max_length={max_length}, batch_size={BATCH_SIZE})")
    return HFLM(
        pretrained=model_path,
        peft=adapter_path,
        trust_remote_code=True,
        parallelize=True,
        device_map="auto",
        tokenizer=model_path,
        batch_size=BATCH_SIZE,
        max_length=max_length,
        enable_thinking=False,
    )


def _run_one(lm, variant: str, prompt_id: int) -> dict:
    from lm_eval.tasks import TaskManager
    from lm_eval.evaluator import simple_evaluate

    task_name = TASK_NAME_TEMPLATES[variant].format(pid=prompt_id)
    # include_path scoped to this variant's YAML dir so lm-eval finds
    # metrics.py next to the YAML for arabench_choices.
    task_manager = TaskManager(include_path=str(YAML_ROOT / variant))
    return simple_evaluate(
        model=lm,
        tasks=[task_name],
        num_fewshot=0,
        task_manager=task_manager,
        apply_chat_template=False,
    )


def _json_default(o):
    try:
        return str(o)
    except Exception:
        return "<not serializable>"


def _save(result: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=4, default=_json_default)
    )


def _verify_artifacts(variants: list[str]) -> None:
    missing = []
    for variant in variants:
        for pid in PROMPT_IDS:
            if not (DATA_ROOT / variant / f"prompt_{pid}" / "data.parquet").exists():
                missing.append(f"data/{variant}/prompt_{pid}/data.parquet")
            if not (YAML_ROOT / variant / f"prompt_{pid}.yaml").exists():
                missing.append(f"eval_harness_tasks/{variant}/prompt_{pid}.yaml")
        if variant == "arabench_choices":
            if not (YAML_ROOT / variant / "metrics.py").exists():
                missing.append(f"eval_harness_tasks/{variant}/metrics.py")
    if missing:
        raise FileNotFoundError(
            "Missing build artifacts. Run build_data.py first. Missing:\n  - "
            + "\n  - ".join(missing)
        )


def main(
    variant: str = "all",
    model: str = "all",
    gpus: str | None = None,
    force: bool = False,
):
    """Run the dialect choice-ablation eval.

    Args:
        variant: 'all' or one of {explained_full, explained_text_only, arabench_choices}.
        model: Model key (AceGPT-v2-8B, Llama-3.1-8B, Qwen3-8B) or 'all'.
        gpus: Comma-separated GPU IDs (default: all visible).
        force: Re-evaluate even if results JSON already exists.
    """
    if variant != "all" and variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}. Choose from: {list(VARIANTS)} or 'all'.")
    if model != "all" and model not in MODELS:
        raise ValueError(f"Unknown model {model!r}. Choose from: {list(MODELS)} or 'all'.")

    variants_to_run = list(VARIANTS) if variant == "all" else [variant]
    models_to_run = list(MODELS) if model == "all" else [model]

    _verify_artifacts(variants_to_run)
    _setup_gpus(gpus)

    # Group work by model so we only load each LM once even when running
    # multiple variants. Within a model, we iterate variants then prompts.
    for model_key in models_to_run:
        model_path, adapter_path, results_dir_name = MODELS[model_key]

        # Build the pending (variant, prompt_id) list for this model.
        pending: list[tuple[str, int]] = []
        for v in variants_to_run:
            out_dir = RESULTS_ROOT / v / results_dir_name
            for prompt_id in PROMPT_IDS:
                out_file = out_dir / f"prompt_{prompt_id}.json"
                if not force and out_file.exists() and out_file.stat().st_size > 0:
                    print(f"[Eval] Skip (already done): {v}/{results_dir_name}/prompt_{prompt_id}.json")
                else:
                    pending.append((v, prompt_id))

        if not pending:
            print(f"[Eval] {model_key}: nothing to do.")
            continue

        if not Path(adapter_path).exists():
            print(f"[Eval] WARN: adapter missing for {model_key}: {adapter_path}", file=sys.stderr)
            continue

        print(f"\n{'=' * 80}")
        print(f"[Eval] Model: {model_key}  ({len(pending)} (variant, prompt) pairs pending)")
        print(f"{'=' * 80}\n")

        lm = _init_hflm(model_path, adapter_path)

        for v, prompt_id in pending:
            print(f"\n--- {model_key} / {v} / prompt_{prompt_id} ---")
            result = _run_one(lm, v, prompt_id)
            _save(result, RESULTS_ROOT / v / results_dir_name / f"prompt_{prompt_id}.json")
            print(f"[Eval] Saved {v}/{results_dir_name}/prompt_{prompt_id}.json")

        # Free GPU memory between models.
        del lm
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    print("\n[Eval] All done.")


if __name__ == "__main__":
    fire.Fire(main)
