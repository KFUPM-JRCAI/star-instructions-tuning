"""Per-dataset preprocessing functions.

Each dataset has its own set of functions, copied exactly from the experiment notebooks.
Functions are registered in DATASET_FUNCTIONS for lookup by dataset name.

Key difference between tuning and evaluation:
- Tuning: apply_template calls preprocess_template first, then renders via Jinja2
- Evaluation: apply_template renders the raw template (keeps ||| separator intact)
  Then create_hf_dataset splits on ||| to separate input/output.
"""

from . import (
    dialect_identification,
    machine_translation,
    nli,
    nlu,
    sarcasm_detection,
    summarization,
)
from ._common import DatasetFunctions, get_yaml_template

DATASET_FUNCTIONS: dict[str, DatasetFunctions] = {
    # Dialect Identification
    "AraBench_dev": DatasetFunctions(
        preprocess_template=dialect_identification.preprocess_template,
        generate_tuple=dialect_identification.generate_tuple,
        apply_template=dialect_identification.apply_template,
        create_hf_dataset=dialect_identification.create_hf_dataset,
    ),
    "Arabic_Dialects_Dataset": DatasetFunctions(  # eval only
        apply_template=dialect_identification.apply_template,
        create_hf_dataset=dialect_identification.create_hf_dataset,
    ),
    # NLI
    "ArEntail": DatasetFunctions(
        preprocess_template=nli.preprocess_template,
        generate_tuple=nli.arentail_generate_tuple,
        apply_template=nli.apply_template,
        create_hf_dataset=nli.arentail_create_hf_dataset,
    ),
    "ArabicTE": DatasetFunctions(  # eval only
        apply_template=nli.apply_template,
        create_hf_dataset=nli.arabic_te_create_hf_dataset,
    ),
    # NLU
    "ArabicMMLU": DatasetFunctions(
        preprocess_template=nlu.preprocess_template,
        generate_tuple=nlu.generate_tuple,
        apply_template=nlu.arabic_mmlu_apply_template,
        create_hf_dataset=nlu.arabic_mmlu_create_hf_dataset,
    ),
    "belebele": DatasetFunctions(  # eval only
        apply_template=nlu.belebele_apply_template,
        create_hf_dataset=nlu.belebele_create_hf_dataset,
    ),
    # Sarcasm Detection
    "ArSarcasm_v2": DatasetFunctions(
        preprocess_template=sarcasm_detection.preprocess_template,
        generate_tuple=sarcasm_detection.generate_tuple,
        apply_template=sarcasm_detection.apply_template,
        create_hf_dataset=sarcasm_detection.create_hf_dataset,
    ),
    "iSarcasmEval_task": DatasetFunctions(  # eval only
        apply_template=sarcasm_detection.apply_template,
        create_hf_dataset=sarcasm_detection.create_hf_dataset,
    ),
    # Machine Translation
    "opus-100": DatasetFunctions(
        preprocess_template=machine_translation.preprocess_template,
        generate_tuple=machine_translation.opus100_generate_tuple,
        apply_template=machine_translation.apply_template,
        create_hf_dataset=machine_translation.opus100_create_hf_dataset,
        remap_dataset=machine_translation.opus100_remap_dataset,
    ),
    "tatoeba_mt": DatasetFunctions(  # eval only
        apply_template=machine_translation.apply_template,
        create_hf_dataset=machine_translation.tatoeba_mt_create_hf_dataset,
    ),
    # Summarization
    "xlsum": DatasetFunctions(
        preprocess_template=summarization.preprocess_template,
        generate_tuple=summarization.generate_tuple,
        apply_template=summarization.apply_template,
        create_hf_dataset=summarization.create_hf_dataset,
    ),
    "AraSum": DatasetFunctions(  # eval only
        apply_template=summarization.apply_template,
        create_hf_dataset=summarization.create_hf_dataset,
    ),
}

__all__ = ["DATASET_FUNCTIONS", "DatasetFunctions", "get_yaml_template"]
