"""GPU detection and CUDA_VISIBLE_DEVICES setup.

MUST be called before importing torch or any torch-based libraries.
"""

import os


def setup_gpus(gpus: str | None = None) -> None:
    """Configure GPU visibility.

    Priority:
    1. Explicit --gpus argument (e.g. "0,1")
    2. Already-set CUDA_VISIBLE_DEVICES env var
    3. SLURM-allocated GPUs (SLURM_GPUS_ON_NODE or SLURM_JOB_GPUS)
    4. All GPUs on host (no restriction)
    """
    if gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
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
