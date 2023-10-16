import gzip
import json
import os
import urllib.request
from typing import Dict

BASE_PATH = os.path.join("./", "objaverse")


def load_object_paths() -> Dict[str, str]:
    """Load the object paths from the dataset.

    The object paths specify the location of where the object is located
    in the Hugging Face repo.

    Returns:
        A dictionary mapping the uid to the object path.
    """
    object_paths_file = "object-paths.json.gz"
    local_path = os.path.join(BASE_PATH, object_paths_file)
    if not os.path.exists(local_path):
        hf_url = f"https://huggingface.co/datasets/allenai/objaverse/resolve/main/{object_paths_file}"
        # wget the file and put it in local_path
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        urllib.request.urlretrieve(hf_url, local_path)
    with gzip.open(local_path, "rb") as f:
        object_paths = json.load(f)
    return object_paths


def download_to_memory(uid: str, 
                       object_path: str):
    hf_url = (
        f"https://huggingface.co/datasets/allenai/objaverse/resolve/main/{object_path}"
    )
    # wget the file and put it in local_path
    os.makedirs(os.path.dirname(tmp_local_path), exist_ok=True)
    urllib.request.urlretrieve(hf_url, tmp_local_path)