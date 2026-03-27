"""
model.py
────────
Full Transformer architecture from scratch in PyTorch.
Based on: "Attention Is All You Need" — Vaswani et al., 2017
          https://arxiv.org/abs/1706.03762

Build order (bottom-up, each class depends on those above it)
─────────────────────────────────────────────────────────────
1. MultiHeadAttention      — scaled dot-product + multi-head split
2. PositionalEncoding      — sinusoidal position signal
3. PositionwiseFFN         — two-layer feedforward per position
4. EncoderLayer            — self-attn → FFN  (with residual + LayerNorm)
5. DecoderLayer            — self-attn → cross-attn → FFN
6. Encoder                 — stack of N EncoderLayers
7. Decoder                 — stack of N DecoderLayers
8. Transformer             — Encoder + Decoder + output projection

Masking — two masks are used:
  padding mask   : tells attention to ignore <pad> tokens (both encoder & decoder)
  causal mask    : prevents decoder from attending to future tokens (decoder self-attn only)
"""

from __future__ import annotations

import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.logger import logger
from src.exception import ModelBuildError
from src.utils.common import read_yaml, count_parameters


# ══════════════════════════════════════════════════════════════
# 1. MultiHeadAttention
# ══════════════════════════════════════════════════════════════
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention (Vaswani et al. 2017, Section 3.2.2)

    Intuition
    ─────────
    Instead of computing one attention function over d_model-dim vectors,
    we project Q, K, V into h smaller subspaces (d_k = d_model // h),
    compute attention in each subspace in parallel, then concatenate
    and project back to d_model.

    This lets each head attend to different parts of the sequence
    simultaneously — some heads learn syntax, others semantics, etc.

    Shapes throughout
    ─────────────────
    Input  Q, K, V : [B, seq_len, d_model]
    After  split   : [B, h, seq_len, d_k]      (d_k = d_model // h)
    Scores         : [B, h, seq_len_q, seq_len_k]
    After  concat  : [B, seq_len, d_model]
    Output         : [B, seq_len, d_model]
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        if d_model % num_heads != 0:
            raise ModelBuildError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # dimension per head

        # four linear projections — W_Q, W_K, W_V, W_O
        # (using single matrices is equivalent to separate per-head projections)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: Tensor) -> Tensor:
        """
        Reshape [B, seq_len, d_model] → [B, num_heads, seq_len, d_k]
        so each head processes its own slice independently.
        """
        B, seq_len, _ = x.shape
        x = x.view(B, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)  # [B, h, seq_len, d_k]

    def _scaled_dot_product(
        self, Q: Tensor, K: Tensor, V: Tensor, mask: Tensor | None
    ) -> Tensor:
        """
        Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) · V

        Why divide by sqrt(d_k)?
          Dot products grow large in magnitude as d_k increases,
          pushing softmax into regions with tiny gradients.
          Dividing by sqrt(d_k) keeps the variance stable.

        mask:  positions to IGNORE are marked with True
               we fill them with -inf so softmax → 0 after exp
        """
        scale = math.sqrt(self.d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
        # scores: [B, h, seq_len_q, seq_len_k]

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        attn_weights = self.dropout(F.softmax(scores, dim=-1))
        # attn_weights: [B, h, seq_len_q, seq_len_k]

        return torch.matmul(attn_weights, V)  # [B, h, seq_len_q, d_k]

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """
        Parameters
        ──────────
        query : [B, seq_len_q, d_model]
        key   : [B, seq_len_k, d_model]
        value : [B, seq_len_v, d_model]   (seq_len_k == seq_len_v always)
        mask  : broadcastable bool tensor, True = ignore this position

        Returns
        ───────
        [B, seq_len_q, d_model]
        """
        B = query.shape[0]

        # 1. Linear projections
        Q = self.W_q(query)  # [B, seq_len_q, d_model]
        K = self.W_k(key)  # [B, seq_len_k, d_model]
        V = self.W_v(value)  # [B, seq_len_v, d_model]

        # 2. Split into heads
        Q = self._split_heads(Q)  # [B, h, seq_len_q, d_k]
        K = self._split_heads(K)  # [B, h, seq_len_k, d_k]
        V = self._split_heads(V)  # [B, h, seq_len_v, d_k]

        # 3. Scaled dot-product attention per head
        x = self._scaled_dot_product(Q, K, V, mask)
        # x: [B, h, seq_len_q, d_k]

        # 4. Concatenate heads: [B, h, seq_len_q, d_k] → [B, seq_len_q, d_model]
        x = x.transpose(1, 2).contiguous().view(B, -1, self.d_model)

        # 5. Final linear projection W_O
        return self.W_o(x)  # [B, seq_len_q, d_model]


# ══════════════════════════════════════════════════════════════
# 2. PositionalEncoding
# ══════════════════════════════════════════════════════════════
class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding (Vaswani et al. 2017, Section 3.5)

    Why do we need this?
      The Transformer has no recurrence and no convolution — it
      processes all positions simultaneously. Without positional
      information, "dog bites man" and "man bites dog" would be
      identical to the model.

    Formula
    ───────
    PE(pos, 2i)   = sin( pos / 10000^(2i / d_model) )
    PE(pos, 2i+1) = cos( pos / 10000^(2i / d_model) )

    Even dimensions get sine, odd get cosine.
    The 10000 base means low-frequency signals encode coarse position
    and high-frequency signals encode fine position — analogous to
    binary digits where the last bit alternates fastest.

    This is a fixed (non-learned) encoding — the same matrix is
    added to every embedding at every forward pass.
    """

    def __init__(self, d_model: int, max_seq_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # build the PE matrix once, register as a buffer
        # (buffer = part of model state, not a trainable parameter)
        pe = torch.zeros(max_seq_len, d_model)  # [max_len, d_model]

        position = torch.arange(0, max_seq_len).unsqueeze(1).float()  # [max_len, 1]

        # div_term: the 1/10000^(2i/d_model) factor — computed in log space for stability
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd  indices

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]  (batch dim for broadcasting)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        x : [B, seq_len, d_model]  — token embeddings
        Adds the positional signal then applies dropout.
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════
# 3. PositionwiseFFN
# ══════════════════════════════════════════════════════════════
class PositionwiseFFN(nn.Module):
    """
    Position-wise Feed-Forward Network (Section 3.3)

    Applied identically to every position independently:
        FFN(x) = max(0, x W_1 + b_1) W_2 + b_2

    d_ff is typically 4× d_model (e.g. 512 → 2048 → 512).
    This expansion-then-compression acts as a per-position
    "memory" that stores pattern-specific transformations.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ══════════════════════════════════════════════════════════════
# 4. EncoderLayer
# ══════════════════════════════════════════════════════════════
class EncoderLayer(nn.Module):
    """
    Single Encoder Layer — two sub-layers with residual connections:

        x = LayerNorm( x + SelfAttention(x) )
        x = LayerNorm( x + FFN(x) )

    Why residual connections?
      They allow gradients to flow directly through the network
      without passing through the attention/FFN transformations,
      solving the vanishing gradient problem for deep stacks.

    Why LayerNorm?
      Normalises across the feature dimension (not batch), which
      stabilises training when sequence lengths vary widely.
      Applied *after* the residual addition (Post-LN, as in the paper).
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, src_mask: Tensor | None) -> Tensor:
        """
        x        : [B, src_len, d_model]
        src_mask : [B, 1, 1, src_len]  padding mask — True = ignore
        """
        # sub-layer 1: self-attention  (Q = K = V = x)
        _x = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(_x))

        # sub-layer 2: feedforward
        _x = self.ffn(x)
        x = self.norm2(x + self.dropout(_x))

        return x  # [B, src_len, d_model]


# ══════════════════════════════════════════════════════════════
# 5. DecoderLayer
# ══════════════════════════════════════════════════════════════
class DecoderLayer(nn.Module):
    """
    Single Decoder Layer — three sub-layers:

        x = LayerNorm( x + MaskedSelfAttention(x) )     ← causal mask
        x = LayerNorm( x + CrossAttention(x, enc_out) ) ← cross-attention
        x = LayerNorm( x + FFN(x) )

    Key distinction from EncoderLayer
    ──────────────────────────────────
    Sub-layer 1 uses a CAUSAL mask (triangular) so position i
    can only attend to positions 0..i. This enforces auto-regressive
    generation — the decoder cannot cheat by looking at future tokens.

    Sub-layer 2 is CROSS-ATTENTION:
        Q = decoder hidden states   (what the decoder is "asking")
        K = encoder output          (what the encoder "knows")
        V = encoder output          (the information to retrieve)
    This is where the decoder learns to align target tokens with
    the most relevant source tokens — the core of translation.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        enc_out: Tensor,
        src_mask: Tensor | None,
        tgt_mask: Tensor | None,
    ) -> Tensor:
        """
        x        : [B, tgt_len, d_model]   decoder input
        enc_out  : [B, src_len, d_model]   encoder output (fixed during decoding)
        src_mask : [B, 1, 1, src_len]      padding mask for source
        tgt_mask : [B, 1, tgt_len, tgt_len] causal + padding mask for target
        """
        # sub-layer 1: masked self-attention  (Q = K = V = x, causal)
        _x = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(_x))

        # sub-layer 2: cross-attention  (Q=x from decoder, K=V=enc_out)
        _x = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout(_x))

        # sub-layer 3: feedforward
        _x = self.ffn(x)
        x = self.norm3(x + self.dropout(_x))

        return x  # [B, tgt_len, d_model]


# ══════════════════════════════════════════════════════════════
# 6. Encoder
# ══════════════════════════════════════════════════════════════
class Encoder(nn.Module):
    """
    Stack of N identical EncoderLayers.

    Input tokens → embedding → positional encoding → N × EncoderLayer
    Output: contextualised representations for every source position.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float,
        pad_idx: int,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model, max_seq_len, dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src: Tensor, src_mask: Tensor | None) -> Tensor:
        """
        src      : [B, src_len]   integer token IDs
        src_mask : [B, 1, 1, src_len]

        Returns  : [B, src_len, d_model]
        """
        # scale embeddings by sqrt(d_model) as per the paper
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_enc(x)

        for layer in self.layers:
            x = layer(x, src_mask)

        return self.norm(x)


# ══════════════════════════════════════════════════════════════
# 7. Decoder
# ══════════════════════════════════════════════════════════════
class Decoder(nn.Module):
    """
    Stack of N identical DecoderLayers.

    Target tokens → embedding → positional encoding → N × DecoderLayer
    Output: hidden states that will be projected to vocabulary logits.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float,
        pad_idx: int,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model, max_seq_len, dropout)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tgt: Tensor,
        enc_out: Tensor,
        src_mask: Tensor | None,
        tgt_mask: Tensor | None,
    ) -> Tensor:
        """
        tgt      : [B, tgt_len]         integer token IDs
        enc_out  : [B, src_len, d_model]
        src_mask : [B, 1, 1, src_len]
        tgt_mask : [B, 1, tgt_len, tgt_len]

        Returns  : [B, tgt_len, d_model]
        """
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_enc(x)

        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)

        return self.norm(x)


# ══════════════════════════════════════════════════════════════
# 8. Mask utilities
# ══════════════════════════════════════════════════════════════
def make_src_mask(src: Tensor, pad_idx: int) -> Tensor:
    """
    Padding mask for source sequences.
    Marks <pad> positions as True (= ignore in attention).

    Shape: [B, 1, 1, src_len]
    The extra dims allow broadcasting across (num_heads, seq_len_q).
    """
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)
    # [B, src_len] → [B, 1, 1, src_len]


def make_tgt_mask(tgt: Tensor, pad_idx: int) -> Tensor:
    """
    Combined causal + padding mask for target sequences.

    Step 1 — padding mask  : True where token is <pad>
    Step 2 — causal mask   : upper-triangular True matrix
                             position i cannot see j > i

    Final mask is the logical OR of both — True = ignore.
    Shape: [B, 1, tgt_len, tgt_len]
    """
    B, tgt_len = tgt.shape

    # padding mask: [B, 1, 1, tgt_len] → broadcast to [B, 1, tgt_len, tgt_len]
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)

    # causal (no-peek) mask: upper triangle excluding diagonal
    # torch.triu with diagonal=1 keeps everything above the main diagonal
    causal_mask = (
        torch.triu(
            torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=tgt.device),
            diagonal=1,
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )
    # [1, 1, tgt_len, tgt_len]

    return pad_mask | causal_mask
    # [B, 1, tgt_len, tgt_len]


# ══════════════════════════════════════════════════════════════
# 9. Transformer  (top-level model)
# ══════════════════════════════════════════════════════════════
class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for sequence-to-sequence tasks.

    Architecture
    ────────────
    src tokens → Encoder → enc_out
    tgt tokens → Decoder(enc_out) → hidden states
    hidden states → Linear → logits over tgt vocabulary

    The final Linear layer projects d_model → tgt_vocab_size.
    During training, we apply CrossEntropyLoss on these logits.
    During inference, we take argmax (greedy) or beam search.

    Weight tying
    ────────────
    Optionally share weights between the decoder embedding matrix
    and the output projection. Reduces parameters and often improves
    performance on translation tasks (Press & Wolf, 2017).
    We leave this as a future extension.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_enc_layers: int = 3,
        num_dec_layers: int = 3,
        d_ff: int = 512,
        max_seq_len: int = 100,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.pad_idx = pad_idx

        self.encoder = Encoder(
            src_vocab_size,
            d_model,
            num_heads,
            num_enc_layers,
            d_ff,
            max_seq_len,
            dropout,
            pad_idx,
        )
        self.decoder = Decoder(
            tgt_vocab_size,
            d_model,
            num_heads,
            num_dec_layers,
            d_ff,
            max_seq_len,
            dropout,
            pad_idx,
        )
        # final projection: d_model → tgt vocabulary logits
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        # initialise parameters (Xavier uniform — standard for Transformers)
        self._init_weights()
        logger.info("Transformer model initialised.")

    def _init_weights(self) -> None:
        """Xavier uniform initialisation for all linear layers."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
    ) -> Tensor:
        """
        src : [B, src_len]   source token IDs
        tgt : [B, tgt_len]   target token IDs  (teacher-forced during training)

        Returns
        ───────
        logits : [B, tgt_len, tgt_vocab_size]
        """
        src_mask = make_src_mask(src, self.pad_idx)  # [B, 1, 1, src_len]
        tgt_mask = make_tgt_mask(tgt, self.pad_idx)  # [B, 1, tgt_len, tgt_len]

        enc_out = self.encoder(src, src_mask)
        # enc_out: [B, src_len, d_model]

        dec_out = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        # dec_out: [B, tgt_len, d_model]

        logits = self.output_projection(dec_out)
        # logits: [B, tgt_len, tgt_vocab_size]

        return logits

    def encode(self, src: Tensor) -> tuple[Tensor, Tensor]:
        """
        Encode source sequence. Used during inference to run the
        encoder once and reuse its output for all decoding steps.

        Returns: (enc_out, src_mask)
        """
        src_mask = make_src_mask(src, self.pad_idx)
        enc_out = self.encoder(src, src_mask)
        return enc_out, src_mask

    def decode_step(
        self,
        tgt: Tensor,
        enc_out: Tensor,
        src_mask: Tensor,
    ) -> Tensor:
        """
        One decoder forward pass. Used token-by-token during greedy
        inference — we grow `tgt` by one token each step.

        Returns: logits [B, tgt_len, tgt_vocab_size]
        """
        tgt_mask = make_tgt_mask(tgt, self.pad_idx)
        dec_out = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        return self.output_projection(dec_out)


# ──────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────
def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    config_path: str = "config/config.yaml",
) -> Transformer:
    """Build a Transformer from config.yaml + vocab sizes."""
    cfg = read_yaml(config_path)
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=cfg.model.d_model,
        num_heads=cfg.model.num_heads,
        num_enc_layers=cfg.model.num_encoder_layers,
        num_dec_layers=cfg.model.num_decoder_layers,
        d_ff=cfg.model.d_ff,
        max_seq_len=cfg.model.max_seq_len,
        dropout=cfg.model.dropout,
        pad_idx=0,  # always 0 — defined in data_preprocessing.py
    )
    return model


# ──────────────────────────────────────────────────────────────
# Smoke test
# python -m src.components.model
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.utils.common import load_json
    from src.components.data_preprocessing import Vocabulary

    # load real vocab sizes from artifacts
    src_vocab = Vocabulary.from_dict(load_json("artifacts/vocab/src_vocab.json"))
    tgt_vocab = Vocabulary.from_dict(load_json("artifacts/vocab/tgt_vocab.json"))

    print(f"src vocab size : {len(src_vocab):,}")
    print(f"tgt vocab size : {len(tgt_vocab):,}")

    # build model
    model = build_transformer(len(src_vocab), len(tgt_vocab))
    n = count_parameters(model)
    print(f"Trainable parameters : {n:,}")

    # forward pass with dummy data to verify shapes
    B, src_len, tgt_len = 4, 20, 18
    src_dummy = torch.randint(1, len(src_vocab), (B, src_len))  # avoid pad_idx=0
    tgt_dummy = torch.randint(1, len(tgt_vocab), (B, tgt_len))

    logits = model(src_dummy, tgt_dummy)
    print(f"\nForward pass shape check:")
    print(f"  src     : {src_dummy.shape}")
    print(f"  tgt     : {tgt_dummy.shape}")
    print(f"  logits  : {logits.shape}   ← expected [4, 18, tgt_vocab_size]")
    assert logits.shape == (B, tgt_len, len(tgt_vocab)), "Shape mismatch!"
    print("\n✔  model.py smoke test passed.")
