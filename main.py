"""
main.py
───────
Full pipeline entry point for the Transformer MT project.

Usage
─────
# Full pipeline: ingest → preprocess → train → evaluate
python main.py

# Skip ingestion if data already downloaded
python main.py --skip-ingest

# Resume training from checkpoint
python main.py --skip-ingest --resume

# Evaluate only (no training)
python main.py --skip-ingest --skip-train

# Translate custom sentences only
python main.py --skip-ingest --skip-train --skip-eval

Pipeline Stages
───────────────
1. DataIngestion      download Multi30k raw .txt files
2. DataPreprocessing  tokenise, build vocab, create DataLoaders
3. ModelTrainer       train Transformer with Noam LR + early stopping
4. ModelEvaluator     BLEU score on test set + sample translations
5. Translate          interactive custom sentence demo
"""

from __future__ import annotations

import argparse
import sys

import torch

from src.logger import logger
from src.utils.common import read_yaml
from src.components.data_ingestion import build_data_ingestion
from src.components.data_preprocessing import build_data_preprocessing
from src.components.model import build_transformer
from src.components.model_trainer import build_trainer
from src.components.model_evaluation import build_evaluator, translate


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transformer EN→FR Machine Translation Pipeline"
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip data download (use if raw data already exists)",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training (use if checkpoint already exists)",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from existing checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config file (default: config/config.yaml)",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────
# Pipeline stages
# ──────────────────────────────────────────────────────────────
def stage_ingest(config_path: str) -> None:
    logger.info("━━ Stage 1 / 4 : Data Ingestion ━━━━━━━━━━━━━━━")
    ingestion = build_data_ingestion(config_path)
    ingestion.run()


def stage_preprocess(config_path: str):
    logger.info("━━ Stage 2 / 4 : Data Preprocessing ━━━━━━━━━━━")
    pp = build_data_preprocessing(config_path)
    return pp.run()  # train_loader, val_loader, test_loader, src_vocab, tgt_vocab


def stage_train(
    train_loader,
    val_loader,
    src_vocab,
    tgt_vocab,
    config_path: str,
    resume: bool,
) -> None:
    logger.info("━━ Stage 3 / 4 : Model Training ━━━━━━━━━━━━━━━")
    cfg = read_yaml(config_path)
    model = build_transformer(len(src_vocab), len(tgt_vocab), config_path)
    trainer = build_trainer(model, train_loader, val_loader, config_path)

    start_epoch = 1
    if resume:
        checkpoint = f"{cfg.training.save_dir}/{cfg.training.model_name}"
        start_epoch = trainer.load_checkpoint(checkpoint)

    history = trainer.run(start_epoch=start_epoch)

    # print loss curve summary
    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>10}  {'LR':>12}")
    print("─" * 46)
    for i, (tl, vl, lr) in enumerate(
        zip(history["train_loss"], history["val_loss"], history["lr"]),
        start=start_epoch,
    ):
        print(f"{i:>6}  {tl:>10.4f}  {vl:>10.4f}  {lr:>12.2e}")


def stage_evaluate(
    test_loader,
    src_vocab,
    tgt_vocab,
    config_path: str,
) -> dict:
    logger.info("━━ Stage 4 / 4 : Evaluation ━━━━━━━━━━━━━━━━━━━")
    cfg = read_yaml(config_path)
    model = build_transformer(len(src_vocab), len(tgt_vocab), config_path)
    evaluator = build_evaluator(model, test_loader, src_vocab, tgt_vocab, config_path)
    checkpoint = f"{cfg.training.save_dir}/{cfg.training.model_name}"
    return evaluator.run(checkpoint)


def stage_translate(src_vocab, tgt_vocab, config_path: str) -> None:
    """Interactive translation demo with a few hardcoded examples."""
    logger.info("━━ Custom Translations ━━━━━━━━━━━━━━━━━━━━━━━━")
    cfg = read_yaml(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load best checkpoint into a fresh model
    from src.components.model_trainer import ModelTrainer

    model = build_transformer(len(src_vocab), len(tgt_vocab), config_path)
    checkpoint_path = f"{cfg.training.save_dir}/{cfg.training.model_name}"

    import torch as _torch

    ckpt = _torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)

    sentences = [
        "A dog is running in the park.",
        "Two children are playing with a ball.",
        "A woman is reading a book.",
        "The man is riding a bicycle.",
        "A group of people are sitting on the grass.",
        "A young girl is swimming in a pool.",
    ]

    print("\n── Custom sentence translations ──────────────────")
    for sent in sentences:
        fr = translate(sent, model, src_vocab, tgt_vocab, device)
        print(f"  EN : {sent}")
        print(f"  FR : {fr}")
        print()


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    logger.info("══════════════════════════════════════════════")
    logger.info("   Transformer EN→FR  —  Full Pipeline")
    logger.info("══════════════════════════════════════════════")
    logger.info(f"  config       : {args.config}")
    logger.info(f"  skip_ingest  : {args.skip_ingest}")
    logger.info(f"  skip_train   : {args.skip_train}")
    logger.info(f"  skip_eval    : {args.skip_eval}")
    logger.info(f"  resume       : {args.resume}")
    logger.info(f"  device       : {'cuda' if torch.cuda.is_available() else 'cpu'}")
    logger.info("══════════════════════════════════════════════")

    # ── stage 1: ingest ───────────────────────────────────────
    if not args.skip_ingest:
        stage_ingest(args.config)
    else:
        logger.info("Skipping data ingestion.")

    # ── stage 2: preprocess (always runs — builds DataLoaders) ─
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = stage_preprocess(
        args.config
    )

    # ── stage 3: train ────────────────────────────────────────
    if not args.skip_train:
        stage_train(
            train_loader,
            val_loader,
            src_vocab,
            tgt_vocab,
            args.config,
            args.resume,
        )
    else:
        logger.info("Skipping training.")

    # ── stage 4: evaluate ─────────────────────────────────────
    if not args.skip_eval:
        scores = stage_evaluate(test_loader, src_vocab, tgt_vocab, args.config)
        print(f"\n{'─' * 40}")
        print(f"  Final BLEU-4 : {scores['bleu']}")
        print(f"{'─' * 40}\n")
    else:
        logger.info("Skipping evaluation.")

    # ── stage 5: translate demo ───────────────────────────────
    stage_translate(src_vocab, tgt_vocab, args.config)

    logger.info("══ Pipeline complete ══════════════════════════")


if __name__ == "__main__":
    main()
