"""
data_preprocessing.py
─────────────────────
Responsibility: Turn raw .txt sentence pairs into PyTorch DataLoaders
ready for the Transformer training loop.

Pipeline
────────
1. Load raw .txt files produced by data_ingestion.py
2. Tokenise every sentence using spaCy
   - English  →  en_core_web_sm
   - French   →  fr_core_news_sm
3. Build vocabulary for each language
   - Special tokens: <pad>=0  <sos>=1  <eos>=2  <unk>=3
   - Drop words that appear fewer than min_freq times
4. Numericalize  (token strings → integer IDs)
5. Wrap in a TranslationDataset  (torch.utils.data.Dataset)
6. Collate with dynamic padding  (pad all seqs in a batch to same length)
7. Return train / val / test DataLoaders
8. Persist vocabularies to artifacts/vocab/ as JSON

Why these special tokens?
  <pad>  sentences in a batch must be the same length → pad shorter ones
  <sos>  decoder needs a "start decoding" signal at inference time
  <eos>  model learns when to stop generating
  <unk>  words below min_freq are replaced with this at train time
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass, field
from collections import Counter
from typing import Iterator

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import spacy

from src.logger import logger
from src.exception import PreprocessingError
from src.utils.common import read_yaml, ensure_dir, save_json


# ──────────────────────────────────────────────────────────────
# Constants — special token strings and their fixed indices
# ──────────────────────────────────────────────────────────────
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────
@dataclass
class PreprocessingConfig:
    raw_dir: Path
    processed_dir: Path
    vocab_dir: Path
    src_lang: str
    tgt_lang: str
    max_seq_len: int
    min_freq: int
    batch_size: int
    pin_memory: bool = False


# ──────────────────────────────────────────────────────────────
# Vocabulary
# ──────────────────────────────────────────────────────────────
class Vocabulary:
    """
    Bidirectional mapping between tokens (str) and integer indices.

    Special tokens are always inserted first at fixed positions:
        <pad>=0   <sos>=1   <eos>=2   <unk>=3

    Attributes
    ----------
    stoi : dict[str, int]   string → index
    itos : dict[int, str]   index  → string
    """

    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.stoi: dict[str, int] = {}
        self.itos: dict[int, str] = {}
        self._insert_special_tokens()

    def _insert_special_tokens(self) -> None:
        for idx, tok in enumerate(SPECIAL_TOKENS):
            self.stoi[tok] = idx
            self.itos[idx] = tok

    def build(self, token_lists: list[list[str]]) -> None:
        """
        Build vocab from a list of already-tokenised sentences.

        Parameters
        ----------
        token_lists : list of token lists
            e.g. [["two", "dogs", "run"], ["a", "cat", "sits"], ...]
        """
        counter: Counter = Counter()
        for tokens in token_lists:
            counter.update(tokens)

        for token, freq in counter.items():
            if freq >= self.min_freq and token not in self.stoi:
                idx = len(self.stoi)
                self.stoi[token] = idx
                self.itos[idx] = token

        logger.info(
            f"Vocabulary built — {len(self.stoi):,} tokens (min_freq={self.min_freq})"
        )

    def numericalize(self, tokens: list[str]) -> list[int]:
        """Convert a token list to an integer ID list (unk for missing tokens)."""
        return [self.stoi.get(t, UNK_IDX) for t in tokens]

    def to_dict(self) -> dict:
        """Serialise for JSON persistence."""
        return {"min_freq": self.min_freq, "stoi": self.stoi}

    @classmethod
    def from_dict(cls, data: dict) -> "Vocabulary":
        """Deserialise from a JSON-loaded dict."""
        vocab = cls(min_freq=data["min_freq"])
        vocab.stoi = {k: int(v) for k, v in data["stoi"].items()}
        vocab.itos = {int(v): k for k, v in data["stoi"].items()}
        return vocab

    def __len__(self) -> int:
        return len(self.stoi)


# ──────────────────────────────────────────────────────────────
# Tokeniser  (thin spaCy wrapper)
# ──────────────────────────────────────────────────────────────
class Tokeniser:
    """
    Wraps a spaCy pipeline to produce lowercase token lists.

    Parameters
    ----------
    spacy_model : str
        e.g. "en_core_web_sm" or "fr_core_news_sm"
    """

    def __init__(self, spacy_model: str):
        try:
            # disable unused components for speed — we only need the tokeniser
            self.nlp = spacy.load(spacy_model, disable=["parser", "ner", "tagger"])
            logger.info(f"Loaded spaCy model: {spacy_model}")
        except OSError:
            raise PreprocessingError(
                f"spaCy model '{spacy_model}' not found. "
                f"Run: python -m spacy download {spacy_model}"
            )

    def tokenise(self, sentence: str) -> list[str]:
        """Lowercase and tokenise a single sentence."""
        return [tok.text.lower() for tok in self.nlp(sentence.strip())]

    def tokenise_batch(self, sentences: list[str]) -> list[list[str]]:
        """Tokenise a list of sentences using spaCy's pipe (faster than a loop)."""
        return [
            [tok.text.lower() for tok in doc]
            for doc in self.nlp.pipe(sentences, batch_size=512)
        ]


# ──────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────
class TranslationDataset(Dataset):
    """
    Holds numericalized source/target pairs as tensors.

    Each item is a (src_tensor, tgt_tensor) pair where:
      src_tensor  —  [src_len]   integer IDs, no special tokens
                                 (special tokens added in collate)
      tgt_tensor  —  [tgt_len]   integer IDs, no special tokens

    Special tokens (<sos>, <eos>) are added by the collate function
    so we have full control over what the encoder vs decoder sees.
    """

    def __init__(
        self,
        src_token_lists: list[list[str]],
        tgt_token_lists: list[list[str]],
        src_vocab: Vocabulary,
        tgt_vocab: Vocabulary,
        max_seq_len: int,
    ):
        assert len(src_token_lists) == len(tgt_token_lists), (
            "Source and target must have the same number of sentences."
        )
        self.max_seq_len = max_seq_len
        self.pairs: list[tuple[Tensor, Tensor]] = []

        skipped = 0
        for src_tokens, tgt_tokens in zip(src_token_lists, tgt_token_lists):
            # filter out pairs that exceed max_seq_len
            if len(src_tokens) > max_seq_len or len(tgt_tokens) > max_seq_len:
                skipped += 1
                continue
            src_ids = src_vocab.numericalize(src_tokens)
            tgt_ids = tgt_vocab.numericalize(tgt_tokens)
            self.pairs.append(
                (
                    torch.tensor(src_ids, dtype=torch.long),
                    torch.tensor(tgt_ids, dtype=torch.long),
                )
            )

        if skipped:
            logger.warning(
                f"Skipped {skipped:,} pairs exceeding max_seq_len={max_seq_len}"
            )
        logger.info(f"Dataset ready — {len(self.pairs):,} pairs")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.pairs[idx]


# ──────────────────────────────────────────────────────────────
# Collate function  (dynamic padding + special token insertion)
# ──────────────────────────────────────────────────────────────
class Collator:
    """
    Called by DataLoader to assemble a list of (src, tgt) pairs into
    padded batch tensors.

    What this does per batch
    ────────────────────────
    src  →  pad all sequences to the longest src in the batch
            shape: [batch_size, src_len]

    tgt  →  prepend <sos>, append <eos>, then pad
            shape: [batch_size, tgt_len + 2]

    Why add <sos>/<eos> here and not in the Dataset?
    Because padding happens per-batch (dynamic length), and it's
    cleaner to handle all sequence manipulation in one place.
    """

    def __init__(self):
        self.pad_idx = PAD_IDX
        self.sos_idx = SOS_IDX
        self.eos_idx = EOS_IDX

    def __call__(self, batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
        src_batch, tgt_batch = zip(*batch)

        # source: just pad  (encoder doesn't need <sos>/<eos>)
        src_padded = pad_sequence(
            src_batch,
            batch_first=True,
            padding_value=self.pad_idx,
        )  # [B, src_len]

        # target: wrap with <sos> ... <eos>, then pad
        tgt_wrapped = [
            torch.cat(
                [
                    torch.tensor([self.sos_idx]),
                    tgt,
                    torch.tensor([self.eos_idx]),
                ]
            )
            for tgt in tgt_batch
        ]
        tgt_padded = pad_sequence(
            tgt_wrapped,
            batch_first=True,
            padding_value=self.pad_idx,
        )  # [B, tgt_len + 2]

        return src_padded, tgt_padded


# ──────────────────────────────────────────────────────────────
# DataPreprocessing  —  main orchestrator
# ──────────────────────────────────────────────────────────────
class DataPreprocessing:
    """
    Orchestrates the full preprocessing pipeline:
      load raw text → tokenise → build vocab → dataset → DataLoader

    Outputs
    ───────
    - artifacts/vocab/src_vocab.json
    - artifacts/vocab/tgt_vocab.json
    - train / val / test DataLoaders (returned from run())
    """

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        ensure_dir(config.processed_dir)
        ensure_dir(config.vocab_dir)

        # spaCy model names derived from language codes
        spacy_models = {"en": "en_core_web_sm", "fr": "fr_core_news_sm"}
        src_model = spacy_models.get(config.src_lang)
        tgt_model = spacy_models.get(config.tgt_lang)

        if not src_model or not tgt_model:
            raise PreprocessingError(
                f"No spaCy model mapping for langs: {config.src_lang}, {config.tgt_lang}"
            )

        self.src_tokeniser = Tokeniser(src_model)
        self.tgt_tokeniser = Tokeniser(tgt_model)
        logger.info("DataPreprocessing initialised.")

    # ── private helpers ───────────────────────────────────────
    def _load_sentences(self, split: str, lang: str) -> list[str]:
        path = self.config.raw_dir / f"{split}.{lang}"
        if not path.exists():
            raise PreprocessingError(f"Raw file not found: {path}")
        sentences = path.read_text(encoding="utf-8").splitlines()
        logger.info(f"Loaded {len(sentences):,} sentences from {path}")
        return sentences

    def _tokenise_split(self, split: str) -> tuple[list[list[str]], list[list[str]]]:
        src_sents = self._load_sentences(split, self.config.src_lang)
        tgt_sents = self._load_sentences(split, self.config.tgt_lang)
        logger.info(f"Tokenising {split} split …")
        src_tokens = self.src_tokeniser.tokenise_batch(src_sents)
        tgt_tokens = self.tgt_tokeniser.tokenise_batch(tgt_sents)
        return src_tokens, tgt_tokens

    def _build_vocabs(
        self,
        train_src_tokens: list[list[str]],
        train_tgt_tokens: list[list[str]],
    ) -> tuple[Vocabulary, Vocabulary]:
        """Build and persist vocabularies from training data only."""
        logger.info("Building vocabularies …")

        src_vocab = Vocabulary(min_freq=self.config.min_freq)
        src_vocab.build(train_src_tokens)

        tgt_vocab = Vocabulary(min_freq=self.config.min_freq)
        tgt_vocab.build(train_tgt_tokens)

        # persist
        save_json(src_vocab.to_dict(), self.config.vocab_dir / "src_vocab.json")
        save_json(tgt_vocab.to_dict(), self.config.vocab_dir / "tgt_vocab.json")

        logger.info(f"src vocab size: {len(src_vocab):,}")
        logger.info(f"tgt vocab size: {len(tgt_vocab):,}")
        return src_vocab, tgt_vocab

    def _make_loader(
        self,
        src_tokens: list[list[str]],
        tgt_tokens: list[list[str]],
        src_vocab: Vocabulary,
        tgt_vocab: Vocabulary,
        shuffle: bool,
    ) -> DataLoader:
        dataset = TranslationDataset(
            src_tokens,
            tgt_tokens,
            src_vocab,
            tgt_vocab,
            self.config.max_seq_len,
        )
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            collate_fn=Collator(),
            num_workers=0,  # 0 is safest on all platforms
            pin_memory=self.config.pin_memory,  # speeds up GPU transfer
        )

    # ── public API ────────────────────────────────────────────
    def run(self) -> tuple[DataLoader, DataLoader, DataLoader, Vocabulary, Vocabulary]:
        """
        Run the full preprocessing pipeline.

        Returns
        -------
        train_loader, val_loader, test_loader, src_vocab, tgt_vocab
        """
        # 1. tokenise all splits
        logger.info("── Tokenising splits ───────────────────────────")
        train_src, train_tgt = self._tokenise_split("train")
        val_src, val_tgt = self._tokenise_split("val")
        test_src, test_tgt = self._tokenise_split("test")

        # 2. build vocabs from train only (never peek at val/test)
        logger.info("── Building vocabularies ────────────────────────")
        src_vocab, tgt_vocab = self._build_vocabs(train_src, train_tgt)

        # 3. build DataLoaders
        logger.info("── Building DataLoaders ─────────────────────────")
        train_loader = self._make_loader(
            train_src, train_tgt, src_vocab, tgt_vocab, shuffle=True
        )
        val_loader = self._make_loader(
            val_src, val_tgt, src_vocab, tgt_vocab, shuffle=False
        )
        test_loader = self._make_loader(
            test_src, test_tgt, src_vocab, tgt_vocab, shuffle=False
        )

        self._log_summary(train_loader, val_loader, test_loader, src_vocab, tgt_vocab)
        return train_loader, val_loader, test_loader, src_vocab, tgt_vocab

    def _log_summary(self, train, val, test, src_vocab, tgt_vocab) -> None:
        logger.info("── Preprocessing Summary ────────────────────────")
        logger.info(f"  train batches : {len(train):,}")
        logger.info(f"  val   batches : {len(val):,}")
        logger.info(f"  test  batches : {len(test):,}")
        logger.info(f"  src vocab     : {len(src_vocab):,} tokens")
        logger.info(f"  tgt vocab     : {len(tgt_vocab):,} tokens")
        logger.info(f"  batch size    : {self.config.batch_size}")
        logger.info(f"  max_seq_len   : {self.config.max_seq_len}")
        logger.info("─────────────────────────────────────────────────")


# ──────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────
def build_data_preprocessing(
    config_path: str = "config/config.yaml",
) -> DataPreprocessing:
    cfg = read_yaml(config_path)
    pp_config = PreprocessingConfig(
        raw_dir=Path(cfg.data.raw_dir),
        processed_dir=Path(cfg.data.processed_dir),
        vocab_dir=Path(cfg.data.vocab_dir),
        src_lang=cfg.data.src_lang,
        tgt_lang=cfg.data.tgt_lang,
        max_seq_len=cfg.data.max_seq_len,
        min_freq=cfg.data.min_freq,
        batch_size=cfg.training.batch_size,
        pin_memory=cfg.training.pin_memory,
    )
    return DataPreprocessing(pp_config)


# ──────────────────────────────────────────────────────────────
# Smoke test
# python -m src.components.data_preprocessing
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pp = build_data_preprocessing()
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = pp.run()

    # inspect one batch
    src_batch, tgt_batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  src : {src_batch.shape}   (batch_size × src_seq_len)")
    print(f"  tgt : {tgt_batch.shape}   (batch_size × tgt_seq_len incl <sos>/<eos>)")

    # decode first sentence back to tokens to verify
    print(f"\nFirst source sentence (decoded):")
    print(
        " ".join(src_vocab.itos[i.item()] for i in src_batch[0] if i.item() != PAD_IDX)
    )
    print(f"First target sentence (decoded):")
    print(
        " ".join(tgt_vocab.itos[i.item()] for i in tgt_batch[0] if i.item() != PAD_IDX)
    )
