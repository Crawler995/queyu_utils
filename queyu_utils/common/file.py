import os
import json
from typing import Any


def ensure_dir(file_path: str):
    """Ensure the existence of the directory of :attr:`file_path` to prevent some `'No such file or directory'` errors.

    Args:
        file_path: Target file path.
    """
    if not os.path.exists(file_path):
        os.makedirs(file_path)


def read_json(json_file_path) -> Any:
    """Read JSON file more conveniently.

    Args:
        json_file_path: Target JSON file path.

    Returns:
        JSON object.
    """
    with open(json_file_path, 'r') as f:
        return json.loads(f.read())
