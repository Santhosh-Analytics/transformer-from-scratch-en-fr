"""
utils/common.py
───────────────
Shared utility helpers used across all pipeline components.
  - read_yaml        : load config.yaml → SimpleNamespace (dot-access)
  - save_json        : persist any dict as a .json artifact
  - load_json        : reload a saved .json artifact
  - ensure_dir       : create a directory (and parents) if absent
  - count_parameters : report trainable parameter count for any nn.Module
  - epoch_timer      : context-manager that logs elapsed time per epoch
"""

import json
import time
import contextlib
from pathlib import Path
from types import SimpleNamespace

import yaml
import torch.nn as nn

from src.logger import logger
from src.exception import ConfigError


# ──────────────────────────────────────────────────────────────
# YAML  →  nested SimpleNamespace  (dot-access everywhere)
# ──────────────────────────────────────────────────────────────
def _dict_to_namespace(d: dict) -> SimpleNamespace:
    """Recursively convert a dict to a SimpleNamespace for dot-access."""
    ns = SimpleNamespace()
    for key, value in d.items():
        setattr(
            ns, key, _dict_to_namespace(value) if isinstance(value, dict) else value
        )
    return ns


def read_yaml(path: str | Path) -> SimpleNamespace:
    """
    Load a YAML config file and return a dot-accessible namespace.

    Usage
    -----
    cfg = read_yaml("config/config.yaml")
    print(cfg.model.d_model)   # 256
    print(cfg.training.epochs) # 20
    """
    path = Path(path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise ConfigError(f"config.yaml must be a YAML mapping, got {type(raw)}")
        config = _dict_to_namespace(raw)
        logger.info(f"Config loaded from: {path}")
        return config
    except FileNotFoundError:
        raise ConfigError(f"Config file not found: {path}")
    except yaml.YAMLError as e:
        raise ConfigError(f"YAML parse error in {path}: {e}")


# ──────────────────────────────────────────────────────────────
# JSON  save / load  (for vocab, metrics, etc.)
# ──────────────────────────────────────────────────────────────
def save_json(data: dict, path: str | Path) -> None:
    """Persist a dictionary as a JSON file, creating parent dirs if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved JSON  →  {path}")


def load_json(path: str | Path) -> dict:
    """Load a JSON file and return as a dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded JSON ←  {path}")
    return data


# ──────────────────────────────────────────────────────────────
# Directory helper
# ──────────────────────────────────────────────────────────────
def ensure_dir(path: str | Path) -> Path:
    """Create directory (and all parents) if it doesn't exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ──────────────────────────────────────────────────────────────
# Model parameter counter
# ──────────────────────────────────────────────────────────────
def count_parameters(model: nn.Module) -> int:
    """
    Return the number of *trainable* parameters in an nn.Module.
    Logs a human-readable summary automatically.

    Usage
    -----
    n = count_parameters(transformer_model)
    # logs:  "Trainable parameters: 14,582,016"
    """
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {total:,}")
    return total


# ──────────────────────────────────────────────────────────────
# Epoch timer  (context manager)
# ──────────────────────────────────────────────────────────────
@contextlib.contextmanager
def epoch_timer(epoch: int):
    """
    Context manager that logs the wall-clock time for one training epoch.

    Usage
    -----
    with epoch_timer(epoch=1):
        train_one_epoch(...)
    # logs:  "Epoch 1 completed in 45.3s"
    """
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    logger.info(f"Epoch {epoch} completed in {elapsed:.1f}s")


# ──────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg = read_yaml("config/config.yaml")
    print(cfg.model.d_model)  # 256
    print(cfg.training.epochs)  # 20
    print(cfg.data.src_lang)  # en

    ensure_dir("artifacts/test_dir")

    sample = {"word": 0, "hello": 1}
    save_json(sample, "artifacts/test_dir/sample_vocab.json")
    loaded = load_json("artifacts/test_dir/sample_vocab.json")
    print(loaded)  # {'word': 0, 'hello': 1}
