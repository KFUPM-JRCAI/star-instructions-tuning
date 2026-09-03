"""Experiment configurations: tasks, datasets, prompt IDs, and training parameters."""

from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    task_name: str
    primary_dataset: str
    secondary_datasets: list[str]
    # dataset_name -> HuggingFace dataset path
    hf_datasets: dict[str, str]
    # dataset_name -> list of selected prompt IDs
    prompt_ids: dict[str, list[int]]
    # Training hyperparameters for this task
    training_params: dict = field(default_factory=dict)
    # dataset_name -> promptlab dataset name (when it differs from folder name)
    promptlab_dataset_names: dict[str, str] = field(default_factory=dict)
    # dataset_name -> HFLM eval batch size (classification tasks only)
    eval_batch_sizes: dict[str, int] = field(default_factory=dict)

    def get_eval_batch_size(self, dataset_name: str, default: int = 40) -> int:
        """Get HFLM batch size for a dataset (classification tasks only)."""
        return self.eval_batch_sizes.get(dataset_name, default)

    def get_promptlab_name(self, dataset_name: str) -> str:
        """Get the PromptLab dataset name (may differ from folder name)."""
        return self.promptlab_dataset_names.get(dataset_name, dataset_name)

    def get_all_datasets(self) -> list[str]:
        """Return all dataset names (primary + secondary)."""
        return [self.primary_dataset] + self.secondary_datasets


EXPERIMENTS: dict[str, ExperimentConfig] = {
    "dialect_identification": ExperimentConfig(
        task_name="dialect_identification",
        primary_dataset="AraBench_dev",
        secondary_datasets=["Arabic_Dialects_Dataset"],
        hf_datasets={
            "AraBench_dev": "KFUPM-JRCAI/arabench_dev_experimental",
            "Arabic_Dialects_Dataset": "KFUPM-JRCAI/arabic_dialects_dataset_experimental",
        },
        prompt_ids={
            "AraBench_dev": [14852, 14850, 14789, 14781, 14561],
            "Arabic_Dialects_Dataset": [14102, 14783, 14784, 14790, 14851],
        },
        eval_batch_sizes={
            "Arabic_Dialects_Dataset": 16,
        },
        training_params={
            "learning_rate": 2.5e-4,
            "epochs_count": 10,
            "train_batch_size": 16,
            "eval_batch_size": 16,
        },
    ),
    "machine_translation": ExperimentConfig(
        task_name="machine_translation",
        primary_dataset="opus-100",
        secondary_datasets=["tatoeba_mt"],
        hf_datasets={
            "opus-100": "KFUPM-JRCAI/opus-100_ar_en_experimental",
            "tatoeba_mt": "KFUPM-JRCAI/tatoeba_mt_ara_eng_experimental",
        },
        prompt_ids={
            "opus-100": [14684, 14688, 14680, 14682, 14640],
            "tatoeba_mt": [14866, 14867, 14868, 14889, 14890],
        },
        eval_batch_sizes={
            "opus-100": 32,
            "tatoeba_mt": 32,
        },
        training_params={
            "learning_rate": 2.5e-4,
            "epochs_count": 10,
            "train_batch_size": 4,
            "eval_batch_size": 4,
            "early_stopping_patience": 10,
            "eval_steps": 500,
        },
    ),
    "NLI": ExperimentConfig(
        task_name="NLI",
        primary_dataset="ArEntail",
        secondary_datasets=["ArabicTE"],
        hf_datasets={
            "ArEntail": "KFUPM-JRCAI/ArEntail_experimental",
            "ArabicTE": "KFUPM-JRCAI/ArabicTE_experimental",
        },
        prompt_ids={
            "ArEntail": [14581, 14816, 14818, 14819, 14820],
            "ArabicTE": [14582, 14673, 14724, 14805, 14855],
        },
        training_params={
            "learning_rate": 2.5e-4,
            "epochs_count": 10,
            "train_batch_size": 16,
            "eval_batch_size": 16,
        },
    ),
    "NLU": ExperimentConfig(
        task_name="NLU",
        primary_dataset="ArabicMMLU",
        secondary_datasets=["belebele"],
        hf_datasets={
            "ArabicMMLU": "KFUPM-JRCAI/ArabicMMLU_experimental",
            "belebele": "KFUPM-JRCAI/belebele_experimental",
        },
        prompt_ids={
            "ArabicMMLU": [14571, 14869, 14787, 14797, 14798],
            "belebele": [14854, 14853, 14801, 14800, 14575],
        },
        training_params={
            "learning_rate": 2.5e-4,
            "epochs_count": 10,
            "train_batch_size": 16,
            "eval_batch_size": 16,
        },
    ),
    "sarcasm_detection": ExperimentConfig(
        task_name="sarcasm_detection",
        primary_dataset="ArSarcasm_v2",
        secondary_datasets=["iSarcasmEval_task"],
        hf_datasets={
            "ArSarcasm_v2": "KFUPM-JRCAI/ArSarcasm_v2_experimental",
            "iSarcasmEval_task": "KFUPM-JRCAI/iSarcasmEval_task_A_experimental",
        },
        prompt_ids={
            "ArSarcasm_v2": [14779, 14802, 14835, 14837, 14838],
            "iSarcasmEval_task": [14602, 14780, 14859, 14860, 14864],
        },
        promptlab_dataset_names={
            "iSarcasmEval_task": "iSarcasmEval_task_A",
        },
        training_params={
            "learning_rate": 2.5e-4,
            "epochs_count": 10,
            "train_batch_size": 16,
            "eval_batch_size": 16,
        },
    ),
    "summarization": ExperimentConfig(
        task_name="summarization",
        primary_dataset="xlsum",
        secondary_datasets=["AraSum"],
        hf_datasets={
            "xlsum": "KFUPM-JRCAI/xlsum_arabic_experimental",
            "AraSum": "KFUPM-JRCAI/AraSum_arabic_experimental",
        },
        prompt_ids={
            "xlsum": [14871, 14803, 14856, 14858, 14668],
            "AraSum": [14891, 14857, 14736, 14735, 14628],
        },
        eval_batch_sizes={
            "xlsum": 16,
            "AraSum": 16,
        },
        training_params={
            "learning_rate": 2.5e-4,
            "epochs_count": 10,
            "train_batch_size": 1,
            "eval_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "early_stopping_patience": 20,
            "eval_steps": 500,
        },
    ),
}
