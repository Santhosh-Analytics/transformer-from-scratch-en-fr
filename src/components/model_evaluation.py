"""
model_evaluation.py
───────────────────
Responsibility: Evaluate the trained Transformer on the test set.

What this file contains
───────────────────────
1. greedy_decode()      — auto-regressive inference, one token at a time
2. translate()          — translate a single English sentence to French
3. ModelEvaluator       — runs evaluation on the full test set:
                            - corpus BLEU score (sacrebleu)
                            - sample translations printed side-by-side

Key concepts
────────────
Greedy Decoding
  At inference time, teacher forcing is gone. The decoder generates
  tokens one by one, always picking the highest-probability token
  (argmax) at each step and feeding it back as input for the next step.

  Step 1: encoder encodes the full source sentence once → enc_out
  Step 2: decoder starts with [<sos>]
  Step 3: decoder predicts next token → argmax → append to sequence
  Step 4: repeat step 3 until <eos> is predicted or max_len is reached

  This is called "greedy" because it always picks the local best token.
  Beam search (beam_size > 1) explores multiple hypotheses in parallel
  and typically scores 1-2 BLEU points higher, but greedy is simpler
  and sufficient for evaluating that the model has learned to translate.

BLEU Score
  Bilingual Evaluation Understudy — the standard MT evaluation metric.
  Measures n-gram overlap between the model's output and the reference.

  BLEU = BP × exp( Σ wₙ × log pₙ )
    pₙ  = precision of n-grams (n=1,2,3,4)
    BP  = brevity penalty (penalises overly short translations)
    wₙ  = uniform weights (0.25 each for BLEU-4)

  Range: 0 → 100.  Rough guide for Multi30k:
    < 10  : model barely learned anything
    10-20 : poor but some structure visible
    20-30 : reasonable — most from-scratch tutorials land here
    30-40 : good — comparable to early NMT papers
    > 40  : very good for this dataset/model size

  We use sacrebleu — the standard detokenised BLEU implementation.
  It handles tokenisation internally so results are reproducible
  and comparable across papers.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader
import sacrebleu

from src.logger import logger
from src.exception import ModelEvaluationError, CheckpointError
from src.utils.common import read_yaml, load_json, save_json, ensure_dir
from src.components.model import Transformer, build_transformer
from src.components.data_preprocessing import (
    Vocabulary,
    PAD_IDX,
    SOS_IDX,
    EOS_IDX,
    build_data_preprocessing,
)


# ══════════════════════════════════════════════════════════════
# 1. Greedy Decode
# ══════════════════════════════════════════════════════════════
@torch.no_grad()
def greedy_decode(
    model: Transformer,
    src: Tensor,
    max_len: int,
    device: torch.device,
) -> Tensor:
    """
    Auto-regressive greedy decoding for a single source tensor.

    Parameters
    ──────────
    model   : trained Transformer
    src     : [1, src_len]  integer token IDs (batch size = 1)
    max_len : maximum number of tokens to generate
    device  : cpu or cuda

    Returns
    ───────
    output : [tgt_len]  integer token IDs  (no <sos>, may include <eos>)

    How it works step by step
    ─────────────────────────
    1. Encode the source once → enc_out, src_mask
    2. Initialise decoder input as [[<sos>]]
    3. Loop:
         a. Run decoder forward pass → logits [1, cur_len, vocab]
         b. Take logits for the LAST position only → [1, vocab]
         c. argmax → next token ID
         d. If next token == <eos>: stop
         e. Append next token to decoder input → grow by 1
    4. Return generated token sequence (excluding <sos>)
    """
    model.eval()
    src = src.to(device)

    # step 1: encode source once
    enc_out, src_mask = model.encode(src)  # enc_out: [1, src_len, d_model]

    # step 2: initialise decoder input with <sos>
    tgt = torch.tensor([[SOS_IDX]], dtype=torch.long, device=device)
    # tgt: [1, 1]

    generated = []

    # step 3: generate tokens one by one
    for _ in range(max_len):
        logits = model.decode_step(tgt, enc_out, src_mask)
        # logits: [1, cur_len, tgt_vocab_size]

        # take only the last position's logit → next token prediction
        next_token_logits = logits[:, -1, :]  # [1, vocab]
        next_token = next_token_logits.argmax(dim=-1)  # [1]

        token_id = next_token.item()

        if token_id == EOS_IDX:
            break

        generated.append(token_id)
        # append to decoder input for next step
        tgt = torch.cat([tgt, next_token.unsqueeze(0)], dim=1)
        # tgt: [1, cur_len + 1]

    return torch.tensor(generated, dtype=torch.long)


# ══════════════════════════════════════════════════════════════
# 2. Single sentence translation
# ══════════════════════════════════════════════════════════════
def translate(
    sentence: str,
    model: Transformer,
    src_vocab: Vocabulary,
    tgt_vocab: Vocabulary,
    device: torch.device,
    max_len: int = 100,
) -> str:
    """
    Translate a single English sentence to French.

    Parameters
    ──────────
    sentence  : raw English string  e.g. "A dog is running in the park."
    model     : trained Transformer
    src_vocab : English vocabulary
    tgt_vocab : French vocabulary
    device    : cpu or cuda
    max_len   : max tokens to generate

    Returns
    ───────
    translation : French string  e.g. "Un chien court dans le parc."
    """
    import spacy

    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "tagger"])
    tokens = [tok.text.lower() for tok in nlp(sentence.strip())]

    # numericalize — unknown words get UNK_IDX
    token_ids = src_vocab.numericalize(tokens)
    src = torch.tensor([token_ids], dtype=torch.long)  # [1, src_len]

    output_ids = greedy_decode(model, src, max_len, device)

    # convert IDs back to French words
    translation = " ".join(tgt_vocab.itos.get(i.item(), "<unk>") for i in output_ids)
    return translation


# ══════════════════════════════════════════════════════════════
# 3. ModelEvaluator
# ══════════════════════════════════════════════════════════════
class ModelEvaluator:
    """
    Evaluates the best saved checkpoint on the test set.

    Outputs
    ───────
    - Corpus BLEU score (sacrebleu, detokenised)
    - N-gram precision breakdown (BLEU-1 through BLEU-4)
    - Sample translations printed side-by-side
    - Results saved to artifacts/models/eval_results.json
    """

    def __init__(
        self,
        model: Transformer,
        test_loader: DataLoader,
        src_vocab: Vocabulary,
        tgt_vocab: Vocabulary,
        device: torch.device,
        save_dir: Path,
        max_len: int = 100,
    ):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.device = device
        self.save_dir = Path(save_dir)
        self.max_len = max_len
        ensure_dir(save_dir)

    def _load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Load best model weights from checkpoint."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise CheckpointError(f"Checkpoint not found: {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        logger.info(f"Loaded checkpoint: {path}")
        logger.info(f"  Trained epochs : {ckpt['epoch']}")
        logger.info(f"  Best val loss  : {ckpt['val_loss']:.4f}")

    def _ids_to_sentence(self, ids: Tensor) -> str:
        """Convert a tensor of token IDs to a string, skipping special tokens."""
        special = {PAD_IDX, SOS_IDX, EOS_IDX}
        return " ".join(
            self.tgt_vocab.itos[i.item()] for i in ids if i.item() not in special
        )

    def _generate_translations(self) -> tuple[list[str], list[str]]:
        """
        Run greedy decoding on the entire test set.

        Returns
        ───────
        hypotheses  : list of model-generated French sentences
        references  : list of ground-truth French sentences
        """
        hypotheses = []
        references = []

        self.model.eval()
        for batch_idx, (src_batch, tgt_batch) in enumerate(self.test_loader):
            # decode sentence-by-sentence (greedy_decode expects batch size 1)
            for i in range(src_batch.size(0)):
                src_single = src_batch[i].unsqueeze(0)  # [1, src_len]
                tgt_single = tgt_batch[i]  # [tgt_len]

                # generate hypothesis
                output_ids = greedy_decode(
                    self.model, src_single, self.max_len, self.device
                )
                hypothesis = self._ids_to_sentence(output_ids)
                reference = self._ids_to_sentence(tgt_single)

                hypotheses.append(hypothesis)
                references.append(reference)

            if (batch_idx + 1) % 2 == 0:
                logger.info(
                    f"  Decoded {min((batch_idx + 1) * self.test_loader.batch_size, len(hypotheses))}"
                    f" / {len(self.test_loader.dataset)} sentences"
                )

        return hypotheses, references

    def _compute_bleu(self, hypotheses: list[str], references: list[str]) -> dict:
        """
        Compute corpus BLEU using sacrebleu.

        sacrebleu expects:
          hypotheses : list of strings
          references : list of lists of strings  (multiple refs possible)
        """
        result = sacrebleu.corpus_bleu(hypotheses, [references])
        return {
            "bleu": round(result.score, 2),
            "bleu_1": round(result.precisions[0], 2),
            "bleu_2": round(result.precisions[1], 2),
            "bleu_3": round(result.precisions[2], 2),
            "bleu_4": round(result.precisions[3], 2),
            "brevity_penalty": round(result.bp, 4),
            "hypothesis_len": result.sys_len,
            "reference_len": result.ref_len,
        }

    def _print_samples(
        self,
        src_batch: Tensor,
        hypotheses: list[str],
        references: list[str],
        n: int = 10,
    ) -> None:
        """Print n side-by-side translation examples."""
        print("\n" + "═" * 70)
        print("  SAMPLE TRANSLATIONS  (greedy decoding)")
        print("═" * 70)

        src_sentences = []
        for i in range(min(n, src_batch.size(0))):
            src_ids = src_batch[i]
            src_sent = " ".join(
                self.src_vocab.itos[t.item()]
                for t in src_ids
                if t.item() not in {PAD_IDX, SOS_IDX, EOS_IDX}
            )
            src_sentences.append(src_sent)

        for i, (src, hyp, ref) in enumerate(
            zip(src_sentences, hypotheses[:n], references[:n]), 1
        ):
            print(f"\n[{i}]")
            print(f"  SRC  : {src}")
            print(f"  REF  : {ref}")
            print(f"  HYP  : {hyp}")
        print("\n" + "═" * 70)

    def run(self, checkpoint_path: str | Path) -> dict:
        """
        Load best checkpoint → generate translations → compute BLEU.

        Returns
        ───────
        results : dict with BLEU scores and metadata
        """
        logger.info("══ Evaluation started ══════════════════════════")

        # 1. load best model
        self._load_checkpoint(checkpoint_path)

        # 2. generate all translations
        logger.info("Generating translations on test set …")
        hypotheses, references = self._generate_translations()
        logger.info(f"Generated {len(hypotheses):,} translations.")

        # 3. compute BLEU
        logger.info("Computing BLEU score …")
        scores = self._compute_bleu(hypotheses, references)

        # 4. print results
        logger.info("══ Evaluation Results ══════════════════════════")
        logger.info(f"  BLEU-4  : {scores['bleu']}")
        logger.info(f"  BLEU-1  : {scores['bleu_1']}")
        logger.info(f"  BLEU-2  : {scores['bleu_2']}")
        logger.info(f"  BLEU-3  : {scores['bleu_3']}")
        logger.info(f"  BLEU-4  : {scores['bleu_4']}")
        logger.info(f"  BP      : {scores['brevity_penalty']}")
        logger.info("════════════════════════════════════════════════")

        # 5. print sample translations (use first test batch for src display)
        first_src, _ = next(iter(self.test_loader))
        self._print_samples(first_src, hypotheses, references, n=10)

        # 6. save results
        results_path = self.save_dir / "eval_results.json"
        save_json(scores, results_path)

        return scores


# ──────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────
def build_evaluator(
    model: Transformer,
    test_loader: DataLoader,
    src_vocab: Vocabulary,
    tgt_vocab: Vocabulary,
    config_path: str = "config/config.yaml",
) -> ModelEvaluator:
    cfg = read_yaml(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return ModelEvaluator(
        model=model,
        test_loader=test_loader,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        device=device,
        save_dir=Path(cfg.training.save_dir),
        max_len=cfg.model.max_seq_len,
    )


# ──────────────────────────────────────────────────────────────
# Entry point
# python -m src.components.model_evaluation
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── 1. data ───────────────────────────────────────────────
    logger.info("Loading data …")
    pp = build_data_preprocessing()
    _, _, test_loader, src_vocab, tgt_vocab = pp.run()

    # ── 2. model ──────────────────────────────────────────────
    logger.info("Building model …")
    model = build_transformer(len(src_vocab), len(tgt_vocab))

    # ── 3. evaluator ──────────────────────────────────────────
    evaluator = build_evaluator(model, test_loader, src_vocab, tgt_vocab)

    # ── 4. evaluate ───────────────────────────────────────────
    checkpoint = "artifacts/models/transformer_mt.pt"
    scores = evaluator.run(checkpoint)

    print(f"\n{'─' * 40}")
    print(f"  Final BLEU-4 : {scores['bleu']}")
    print(f"{'─' * 40}")

    # ── 5. custom sentence demo ───────────────────────────────
    cfg = read_yaml("config/config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_sentences = [
        "A dog is running in the park.",
        "Two children are playing with a ball.",
        "A woman is reading a book.",
        "The man is riding a bicycle.",
    ]

    print("\n── Custom sentence translations ──────────────────")
    for sent in test_sentences:
        fr = translate(sent, model, src_vocab, tgt_vocab, device)
        print(f"  EN : {sent}")
        print(f"  FR : {fr}")
        print()
