"""
model_trainer.py
────────────────
Responsibility: Train the Transformer model end-to-end.

What this file contains
───────────────────────
1. NoamScheduler      — warmup + decay LR schedule from Vaswani et al. 2017
2. EarlyStopping      — stops training when val loss stops improving
3. ModelTrainer       — full training loop with:
                          - teacher forcing
                          - gradient clipping
                          - checkpoint saving (best model only)
                          - per-epoch train/val loss logging

Key concepts implemented here
──────────────────────────────
Teacher Forcing
  During training, the decoder receives the GROUND TRUTH target tokens
  as input at each step, not its own previous predictions.
  e.g. to predict  "chat  mange  du  poisson"
  decoder input  : <sos> chat  mange  du
  decoder target : chat  mange  du   poisson  <eos>
  This makes training stable and fast. Without it, early errors
  compound and the model never learns.

Gradient Clipping
  Clips the global norm of all gradients to clip_grad_norm.
  Prevents exploding gradients which are common in deep networks.
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

Label Smoothing
  Instead of a hard one-hot target [0,0,1,0,...], we use soft targets
  [ε/V, ε/V, 1-ε+(ε/V), ε/V, ...] where ε=0.1 (label_smoothing=0.1).
  This prevents the model from becoming overconfident and improves
  generalisation. Built into PyTorch's CrossEntropyLoss.
"""

from __future__ import annotations

import math
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import Adam

from src.logger import logger
from src.exception import ModelTrainingError, CheckpointError
from src.utils.common import read_yaml, ensure_dir, epoch_timer, count_parameters
from src.components.model import Transformer, build_transformer
from src.components.data_preprocessing import (
    build_data_preprocessing,
    Vocabulary,
    PAD_IDX,
)


# ══════════════════════════════════════════════════════════════
# 1. Noam Learning Rate Scheduler
# ══════════════════════════════════════════════════════════════
class NoamScheduler:
    """
    Learning rate schedule from "Attention Is All You Need" (Section 5.3)

    Formula
    ───────
    lrate = d_model^(-0.5) × min(step^(-0.5), step × warmup^(-1.5))

    Behaviour
    ─────────
    - Steps 1 → warmup_steps : LR increases linearly
    - Steps > warmup_steps   : LR decays as 1/sqrt(step)

    Why this works for Transformers
    ────────────────────────────────
    At step 1, weights are Xavier-initialised (random).
    Gradients are noisy and unreliable.
    Starting with a tiny LR and ramping up slowly lets the model
    find a stable loss basin before taking large steps.
    After warmup, the slow decay keeps learning going without
    overshooting minima.

    Usage
    ─────
    scheduler = NoamScheduler(optimizer, d_model=256, warmup_steps=4000)
    # call once per STEP (not per epoch) after optimizer.step()
    scheduler.step()
    """

    def __init__(self, optimizer: Adam, d_model: int, warmup_steps: int):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self._step = 0
        self._rate = 0.0

    def _compute_lr(self) -> float:
        """Noam formula — returns the LR for the current step."""
        step = max(self._step, 1)  # avoid division by zero at step 0
        return self.d_model ** (-0.5) * min(
            step ** (-0.5), step * self.warmup_steps ** (-1.5)
        )

    def step(self) -> None:
        """Advance one step and update the optimizer's learning rate."""
        self._step += 1
        self._rate = self._compute_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self._rate

    @property
    def current_lr(self) -> float:
        return self._rate

    def state_dict(self) -> dict:
        return {"step": self._step, "rate": self._rate}

    def load_state_dict(self, state: dict) -> None:
        self._step = state["step"]
        self._rate = state["rate"]


# ══════════════════════════════════════════════════════════════
# 2. Early Stopping
# ══════════════════════════════════════════════════════════════
class EarlyStopping:
    """
    Stops training when validation loss has not improved for
    `patience` consecutive epochs.

    Also tracks whether the current epoch produced the best model
    so the trainer knows when to save a checkpoint.
    """

    def __init__(self, patience: int = 5):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False
        self.improved = False

    def __call__(self, val_loss: float) -> None:
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.improved = True
            logger.info(f"  Val loss improved → {val_loss:.4f}")
        else:
            self.counter += 1
            self.improved = False
            logger.info(
                f"  Val loss did not improve. Patience: {self.counter}/{self.patience}"
            )
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info("  Early stopping triggered.")


# ══════════════════════════════════════════════════════════════
# 3. Config
# ══════════════════════════════════════════════════════════════
@dataclass
class TrainerConfig:
    epochs: int
    learning_rate: float
    clip_grad_norm: float
    warmup_steps: int
    save_dir: Path
    model_name: str
    early_stopping_patience: int
    d_model: int
    device: torch.device


# ══════════════════════════════════════════════════════════════
# 4. ModelTrainer
# ══════════════════════════════════════════════════════════════
class ModelTrainer:
    """
    Orchestrates the full training loop.

    Per-epoch flow
    ──────────────
    1. train_epoch()  — forward pass, loss, backward, clip, step
    2. val_epoch()    — forward pass only, no gradients
    3. EarlyStopping  — check if val loss improved
    4. Checkpoint     — save model if val loss is best so far
    5. Log            — epoch summary to logger

    Teacher Forcing (implemented here)
    ───────────────────────────────────
    tgt input  = tgt[:, :-1]   → all tokens EXCEPT the last  (<sos> … last-1)
    tgt target = tgt[:, 1:]    → all tokens EXCEPT the first (first+1 … <eos>)

    Example for "chat mange du poisson":
      tgt full   : <sos> chat  mange  du    poisson <eos>
      tgt input  : <sos> chat  mange  du    poisson       ← fed to decoder
      tgt target : chat  mange  du    poisson <eos>        ← what we predict

    The model sees <sos> and must predict "chat".
    The model sees <sos> chat and must predict "mange". Etc.
    """

    def __init__(
        self,
        model: Transformer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainerConfig,
    ):
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        ensure_dir(config.save_dir)

        # Adam with β1=0.9, β2=0.98, ε=1e-9  (paper Section 5.3)
        # lr=0 because NoamScheduler sets it on the first step()
        self.optimizer = Adam(
            model.parameters(),
            lr=0,
            betas=(0.9, 0.98),
            eps=1e-9,
        )

        self.scheduler = NoamScheduler(
            self.optimizer,
            d_model=config.d_model,
            warmup_steps=config.warmup_steps,
        )

        # CrossEntropyLoss with:
        #   ignore_index = PAD_IDX  → don't penalise padding tokens
        #   label_smoothing = 0.1   → soft targets, reduces overconfidence
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=PAD_IDX,
            label_smoothing=0.1,
        )

        self.early_stopping = EarlyStopping(config.early_stopping_patience)

        logger.info("ModelTrainer initialised.")
        logger.info(f"  device          : {config.device}")
        logger.info(f"  epochs          : {config.epochs}")
        logger.info(f"  warmup_steps    : {config.warmup_steps}")
        logger.info(f"  clip_grad_norm  : {config.clip_grad_norm}")
        logger.info(f"  save_dir        : {config.save_dir}")

    # ── private: one training epoch ──────────────────────────
    def _train_epoch(self) -> float:
        """
        Run one full pass over the training set.
        Returns average loss per token.
        """
        self.model.train()
        total_loss = 0.0
        total_tokens = 0

        for src, tgt in self.train_loader:
            src = src.to(self.config.device)  # [B, src_len]
            tgt = tgt.to(self.config.device)  # [B, tgt_len]  (includes <sos> and <eos>)

            # ── teacher forcing split ──────────────────────
            tgt_input = tgt[:, :-1]  # [B, tgt_len-1]  decoder input
            tgt_target = tgt[:, 1:]  # [B, tgt_len-1]  what we want to predict

            # ── forward pass ──────────────────────────────
            logits = self.model(src, tgt_input)
            # logits: [B, tgt_len-1, tgt_vocab_size]

            # ── loss ──────────────────────────────────────
            # CrossEntropyLoss expects [B*T, vocab] and [B*T]
            B, T, V = logits.shape
            loss = self.criterion(
                logits.reshape(B * T, V),
                tgt_target.reshape(B * T),
            )

            # ── backward ──────────────────────────────────
            self.optimizer.zero_grad()
            loss.backward()

            # gradient clipping — prevents exploding gradients
            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.clip_grad_norm,
            )

            # ── update weights + LR ───────────────────────
            self.optimizer.step()
            self.scheduler.step()  # Noam: called per STEP not per epoch

            # accumulate loss (weighted by non-pad token count)
            non_pad = (tgt_target != PAD_IDX).sum().item()
            total_loss += loss.item() * non_pad
            total_tokens += non_pad

        return total_loss / total_tokens if total_tokens > 0 else 0.0

    # ── private: one validation epoch ────────────────────────
    @torch.no_grad()
    def _val_epoch(self) -> float:
        """
        Run one full pass over the validation set.
        No gradients, no weight updates.
        Returns average loss per token.
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        for src, tgt in self.val_loader:
            src = src.to(self.config.device)
            tgt = tgt.to(self.config.device)

            tgt_input = tgt[:, :-1]
            tgt_target = tgt[:, 1:]

            logits = self.model(src, tgt_input)
            B, T, V = logits.shape
            loss = self.criterion(
                logits.reshape(B * T, V),
                tgt_target.reshape(B * T),
            )

            non_pad = (tgt_target != PAD_IDX).sum().item()
            total_loss += loss.item() * non_pad
            total_tokens += non_pad

        return total_loss / total_tokens if total_tokens > 0 else 0.0

    # ── public: load checkpoint ──────────────────────────────
    def load_checkpoint(self, path) -> int:
        """
        Resume training from a saved checkpoint.
        Restores model weights, optimizer state, scheduler step,
        and early stopping best_loss so training continues cleanly.

        Returns the epoch number stored in the checkpoint so the
        training loop can start from the correct epoch number.

        Usage
        ─────
        start_epoch = trainer.load_checkpoint("artifacts/models/transformer_mt.pt")
        history = trainer.run(start_epoch=start_epoch)
        """
        from pathlib import Path as _Path

        path = _Path(path)
        if not path.exists():
            raise CheckpointError(f"Checkpoint not found: {path}")
        try:
            ckpt = torch.load(path, map_location=self.config.device)
            self.model.load_state_dict(ckpt["model_state"])
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
            self.early_stopping.best_loss = ckpt["val_loss"]
            start_epoch = ckpt["epoch"] + 1
            logger.info(f"Resumed from checkpoint: {path}")
            logger.info(f"  Saved at epoch : {ckpt['epoch']}")
            logger.info(f"  Best val loss  : {ckpt['val_loss']:.4f}")
            logger.info(f"  Scheduler step : {self.scheduler._step}")
            logger.info(f"  Resuming from epoch {start_epoch}")
            return start_epoch
        except CheckpointError:
            raise
        except Exception as e:
            raise CheckpointError(f"Failed to load checkpoint: {e}") from e

    # ── private: save checkpoint ─────────────────────────────
    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """Save model + optimizer + scheduler state to disk."""
        path = Path(self.config.save_dir) / self.config.model_name
        try:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "scheduler_state": self.scheduler.state_dict(),
                    "val_loss": val_loss,
                },
                path,
            )
            logger.info(f"  Checkpoint saved → {path}  (val_loss={val_loss:.4f})")
        except Exception as e:
            raise CheckpointError(f"Failed to save checkpoint: {e}") from e

    # ── public: full training run ─────────────────────────────
    def run(self, start_epoch: int = 1) -> dict:
        """
        Execute the full training loop.

        Parameters
        ──────────
        start_epoch : int
            Epoch to start from. Default=1 (fresh training).
            Pass the return value of load_checkpoint() to resume.

        Returns
        ───────
        history : dict with lists of train_loss, val_loss, lr per epoch
        """
        logger.info("══ Training started ════════════════════════════")
        if start_epoch > 1:
            logger.info(f"  Resuming from epoch {start_epoch}")
        history = {"train_loss": [], "val_loss": [], "lr": []}

        try:
            for epoch in range(start_epoch, self.config.epochs + 1):
                with epoch_timer(epoch):
                    train_loss = self._train_epoch()
                    val_loss = self._val_epoch()

                    # perplexity = exp(cross-entropy loss)
                    # useful secondary metric for translation quality
                    train_ppl = math.exp(min(train_loss, 20))
                    val_ppl = math.exp(min(val_loss, 20))

                    current_lr = self.scheduler.current_lr
                    history["train_loss"].append(train_loss)
                    history["val_loss"].append(val_loss)
                    history["lr"].append(current_lr)

                    logger.info(
                        f"Epoch {epoch:>3}/{self.config.epochs} | "
                        f"train_loss={train_loss:.4f} ppl={train_ppl:.2f} | "
                        f"val_loss={val_loss:.4f} ppl={val_ppl:.2f} | "
                        f"lr={current_lr:.2e}"
                    )

                    # early stopping check + conditional checkpoint
                    self.early_stopping(val_loss)
                    if self.early_stopping.improved:
                        self._save_checkpoint(epoch, val_loss)

                    if self.early_stopping.should_stop:
                        logger.info(f"Training stopped early at epoch {epoch}.")
                        break

        except KeyboardInterrupt:
            logger.warning("Training interrupted by user.")
        except Exception as e:
            raise ModelTrainingError(f"Training loop failed: {e}") from e

        logger.info("══ Training complete ═══════════════════════════")
        logger.info(f"  Best val loss : {self.early_stopping.best_loss:.4f}")
        logger.info(
            f"  Best val ppl  : {math.exp(min(self.early_stopping.best_loss, 20)):.2f}"
        )
        return history


# ──────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────
def build_trainer(
    model: Transformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config_path: str = "config/config.yaml",
) -> ModelTrainer:
    cfg = read_yaml(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training device: {device}")

    trainer_config = TrainerConfig(
        epochs=cfg.training.epochs,
        learning_rate=cfg.training.learning_rate,
        clip_grad_norm=cfg.training.clip_grad_norm,
        warmup_steps=cfg.training.warmup_steps,
        save_dir=Path(cfg.training.save_dir),
        model_name=cfg.training.model_name,
        early_stopping_patience=cfg.training.early_stopping_patience,
        d_model=cfg.model.d_model,
        device=device,
    )
    return ModelTrainer(model, train_loader, val_loader, trainer_config)


# ──────────────────────────────────────────────────────────────
# Entry point
# python -m src.components.model_trainer
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.utils.common import load_json
    from src.components.data_preprocessing import Vocabulary

    # ── 1. data ───────────────────────────────────────────────
    logger.info("Loading data …")
    pp = build_data_preprocessing()
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = pp.run()

    # ── 2. model ──────────────────────────────────────────────
    logger.info("Building model …")
    model = build_transformer(len(src_vocab), len(tgt_vocab))
    count_parameters(model)

    # ── 3. trainer ────────────────────────────────────────────
    trainer = build_trainer(model, train_loader, val_loader)

    # ── 4. resume from checkpoint if it exists, else train fresh
    import sys

    checkpoint_path = "artifacts/models/transformer_mt.pt"
    if "--resume" in sys.argv:
        start_epoch = trainer.load_checkpoint(checkpoint_path)
    else:
        start_epoch = 1

    # ── 5. train ──────────────────────────────────────────────
    history = trainer.run(start_epoch=start_epoch)

    # ── 6. print loss curve summary ───────────────────────────
    print("\nLoss curve summary:")
    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>10}  {'LR':>12}")
    print("─" * 46)
    for i, (tl, vl, lr) in enumerate(
        zip(history["train_loss"], history["val_loss"], history["lr"]), 1
    ):
        print(f"{i:>6}  {tl:>10.4f}  {vl:>10.4f}  {lr:>12.2e}")
