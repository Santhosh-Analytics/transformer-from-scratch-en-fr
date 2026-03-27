"""
data_ingestion.py
─────────────────
Responsibility: Download the Multi30k dataset and persist the raw
English / French sentence pairs as plain .txt files under artifacts/data/raw/.

What is Multi30k?
  ~30,000 English–French/German sentence pairs drawn from image captions.
  Standard benchmark for Neural Machine Translation (NMT) research.
  Small enough to train on a free Kaggle/Colab GPU in minutes.

Splits produced
  train.en  /  train.fr
  val.en    /  val.fr
  test.en   /  test.fr

torchtext's Multi30k helper is deprecated in newer versions, so we fetch
the dataset directly from the authoritative GitHub mirror used by the
community, which is robust and version-stable.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from dataclasses import dataclass

from src.logger import logger
from src.exception import DataIngestionError
from src.utils.common import read_yaml, ensure_dir


# ──────────────────────────────────────────────────────────────
# Raw file URLs  (official Multi30k GitHub mirror)
# ──────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────
# DataIngestionConfig  — typed container for all path settings
# ──────────────────────────────────────────────────────────────
@dataclass
class DataIngestionConfig:
    raw_dir: Path
    src_lang: str
    tgt_lang: str


# ──────────────────────────────────────────────────────────────
# DataIngestion
# ──────────────────────────────────────────────────────────────
class DataIngestion:
    """
    Downloads the Multi30k dataset splits and saves them as raw .txt files.

    Directory layout after run()
    ────────────────────────────
    artifacts/data/raw/
        train.en   train.fr
        val.en     val.fr
        test.en    test.fr
    """

    def __init__(self, config: DataIngestionConfig):
        self.config = config
        ensure_dir(self.config.raw_dir)
        logger.info("DataIngestion initialised.")
        logger.info(f"  raw_dir  : {self.config.raw_dir}")
        logger.info(f"  src_lang : {self.config.src_lang}")
        logger.info(f"  tgt_lang : {self.config.tgt_lang}")

    # ── private helpers ───────────────────────────────────────
    def _dest_path(self, split: str, lang: str) -> Path:
        return self.config.raw_dir / f"{split}.{lang}"

    def _already_downloaded(self) -> bool:
        """Return True only if every expected file already exists."""
        for split in SPLITS:
            for lang in (self.config.src_lang, self.config.tgt_lang):
                if not self._dest_path(split, lang).exists():
                    return False
        return True

    def _download_and_decompress(self, url: str, dest: Path) -> None:
        """
        Download a .gz file from `url` and write the decompressed text to `dest`.
        Uses only the standard library — no extra dependencies.
        """
        import gzip
        import io

        logger.info(f"  Downloading {url}")
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                compressed = response.read()
            with gzip.open(io.BytesIO(compressed), "rt", encoding="utf-8") as gz:
                text = gz.read()
            dest.write_text(text, encoding="utf-8")
            line_count = text.count("\n")
            logger.info(f"  Saved → {dest}  ({line_count:,} lines)")
        except Exception as e:
            raise DataIngestionError(f"Failed to download/decompress {url}: {e}") from e

    # ── public API ────────────────────────────────────────────
    def run(self) -> dict[str, dict[str, Path]]:
        """
        Download all splits.  Skips download if files already exist.

        Returns
        -------
        paths : dict  {split: {lang: Path}}
            e.g. paths["train"]["en"] → Path("artifacts/data/raw/train.en")
        """
        if self._already_downloaded():
            logger.info("Raw data already present — skipping download.")
        else:
            logger.info("Starting Multi30k download …")
            for split, langs in SPLITS.items():
                for lang, url in langs.items():
                    # only download the two languages we care about
                    if lang not in (self.config.src_lang, self.config.tgt_lang):
                        continue
                    dest = self._dest_path(split, lang)
                    self._download_and_decompress(url, dest)
            logger.info("Download complete.")

        # build and return path manifest
        paths: dict[str, dict[str, Path]] = {}
        for split in SPLITS:
            paths[split] = {
                self.config.src_lang: self._dest_path(split, self.config.src_lang),
                self.config.tgt_lang: self._dest_path(split, self.config.tgt_lang),
            }

        self._log_summary(paths)
        return paths

    def _log_summary(self, paths: dict[str, dict[str, Path]]) -> None:
        logger.info("── Data Ingestion Summary ──────────────────────")
        for split, langs in paths.items():
            for lang, path in langs.items():
                if path.exists():
                    lines = path.read_text(encoding="utf-8").count("\n")
                    logger.info(f"  {split:5s}.{lang}  →  {path}  ({lines:,} lines)")
                else:
                    logger.warning(f"  {split:5s}.{lang}  →  MISSING: {path}")
        print()
        logger.info("_" * 90)


# ──────────────────────────────────────────────────────────────
# Factory  —  build from config.yaml
# ──────────────────────────────────────────────────────────────
def build_data_ingestion(config_path: str = "config/config.yaml") -> DataIngestion:
    """Convenience factory: read config.yaml → return a ready DataIngestion."""
    cfg = read_yaml(config_path)
    ingestion_config = DataIngestionConfig(
        raw_dir=Path(cfg.data.raw_dir),
        src_lang=cfg.data.src_lang,
        tgt_lang=cfg.data.tgt_lang,
    )
    return DataIngestion(ingestion_config)


# ──────────────────────────────────────────────────────────────
# Smoke test  —  run this file directly to verify download
# python -m src.components.data_ingestion
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ingestion = build_data_ingestion()
    paths = ingestion.run()

    # Quick sanity check — print first 3 sentence pairs from train
    src = paths["train"]["en"].read_text(encoding="utf-8").splitlines()
    tgt = paths["train"]["fr"].read_text(encoding="utf-8").splitlines()
    print("\nSample sentence pairs (train):")
    print("_" * 80)
    for en, fr in zip(src[:3], tgt[:3]):
        print(f"  EN: {en}")
        print(f"  FR: {fr}")
        print()
