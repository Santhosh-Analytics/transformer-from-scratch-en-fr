from __future__ import annotations

import urllib.request
from pathlib import Path
from dataclasses import dataclass

from src.logger import logger
from src.exception import DataIngestionError
from src.utils.common import read_yaml, ensure_dir

BASE_URL = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw"

SPLITS: dict[str, dict[str, str]] = {
    "train": {
        "en": f"{BASE_URL}/train.en.gz",
        "fr": f"{BASE_URL}/train.fr.gz",
    },
    "val": {
        "en": f"{BASE_URL}/val.en.gz",
        "fr": f"{BASE_URL}/val.fr.gz",
    },
    "test": {
        "en": f"{BASE_URL}/test_2016_flickr.en.gz",
        "fr": f"{BASE_URL}/test_2016_flickr.fr.gz",
    },
}


print(type(Path))
