"""PromptLab API client for fetching and filtering prompts."""

import requests
from tqdm.auto import tqdm


PROMPTLAB_API_URL = "https://promptlab.up.railway.app/api/prompt/list?project_secret_key=6Wirj"


def fetch_prompts(max_retries: int = 10) -> list[dict]:
    """Fetch all prompts from the PromptLab API with retry logic."""
    prompts = None
    for i in tqdm(range(max_retries), desc="Fetching prompts"):
        api_response = requests.get(url=PROMPTLAB_API_URL)
        if api_response.ok:
            prompts = api_response.json()
            break
    if not prompts:
        raise Exception(f"Failed to fetch prompts after {max_retries} retries")
    print(f"[PromptLab] Fetched {len(prompts)} total prompts")
    return prompts


def filter_prompts(all_prompts: list[dict]) -> list[dict]:
    """Filter prompts: APPROVED status + ltr text direction."""
    filtered = [
        p for p in all_prompts
        if p["status"] == "APPROVED" and p["text_direction"].lower() == "ltr"
    ]
    print(f"[PromptLab] {len(filtered)} prompts after filtering (APPROVED + ltr)")
    return filtered


def get_dataset_prompts(
    filtered_prompts: list[dict], selected_ids: list[int]
) -> list[dict]:
    """Select specific prompts by ID from the filtered set."""
    dataset_prompts = [p for p in filtered_prompts if p["id"] in selected_ids]
    if len(dataset_prompts) != len(selected_ids):
        found_ids = {p["id"] for p in dataset_prompts}
        missing = set(selected_ids) - found_ids
        raise ValueError(f"Missing prompt IDs: {missing}")
    print(f"[PromptLab] Selected {len(dataset_prompts)} prompts: {selected_ids}")
    return dataset_prompts
