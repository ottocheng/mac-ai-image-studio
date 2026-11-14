"""Query-chunked scaled-dot-product attention for MPS (memory saver).

WHY THIS EXISTS
---------------
On Apple Silicon MPS there is no FlashAttention backend, so PyTorch's
``F.scaled_dot_product_attention`` falls back to the "math" path that
materializes the full ``Lq x Lk`` attention-score matrix in one allocation.
For large-token-grid diffusion transformers (originally observed with
FLUX.2-klein at 1024px ~16k tokens; now equally relevant for Qwen-Image
at higher resolutions) this single allocation is what pushes the
unified-memory working set into disk swap -- the root cause of
multi-hundred-second per-image slowdowns measured on a 16 GB Mac.

WHAT IT DOES
------------
We monkeypatch ``F.scaled_dot_product_attention`` to compute attention in
fixed-size query chunks. Each query still attends to *every* key/value, so
the result is **mathematically identical** to full attention -- only the
peak memory of the score matrix drops (from ``Lq*Lk`` to ``QCHUNK*Lk`` per
call). Quality is unchanged.

A guard limits the patch to large, unmasked, non-causal self-attention --
the exact calls whose full score matrix is the swap trigger. Masked /
causal / small calls fall through to the original kernel untouched, so
smaller pipelines (e.g. Z-Image-Turbo at 512px) are unaffected.

CREDIT
------
Exact-mode subset adapted from
``newideas99/ultra-fast-image-gen``'s ``mps_chunked_attn.py``
(https://github.com/newideas99/ultra-fast-image-gen), which additionally
implements optional image-KV pooling for extra speed. We deliberately keep
only the exact query-chunking path -- no approximation, no quality risk.

TUNING
------
Set the ``QCHUNK`` env var before launch (default 1024, well-tuned for a
16 GB Mac). Smaller = less peak memory, more kernel-launch overhead.
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

# Capture the true original ONCE at import time, before anyone else patches
# SDPA. diffusers / transformers never patch it, so this is safe.
_orig_sdpa = F.scaled_dot_product_attention

QCHUNK = int(os.environ.get("QCHUNK", "1024"))
_patched = False


def _chunked_sdpa(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs
):
    lq = query.shape[-2]
    # Only intercept large, unmasked, non-causal self-attention -- the calls
    # whose full Lq*Lk score matrix is the swap trigger. Masked, causal, and
    # small calls go straight to the original kernel.
    if attn_mask is not None or is_causal or lq <= QCHUNK:
        return _orig_sdpa(
            query, key, value,
            attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal,
            scale=scale, **kwargs,
        )

    outs = []
    for i in range(0, lq, QCHUNK):
        # dropout_p forced to 0 inside chunks: attention dropout is always
        # disabled at inference, and per-chunk RNG would otherwise diverge
        # from a single full call. Matches the upstream reference.
        outs.append(
            _orig_sdpa(
                query[..., i : i + QCHUNK, :],
                key,
                value,
                dropout_p=0.0,
                is_causal=False,
                scale=scale,
                **kwargs,
            )
        )
    return torch.cat(outs, dim=-2)


def patch() -> int:
    """Install the chunked-SDPA monkeypatch on F.scaled_dot_product_attention.

    Idempotent. Returns the QCHUNK size in effect.
    """
    global _patched
    if not _patched:
        F.scaled_dot_product_attention = _chunked_sdpa
        _patched = True
    return QCHUNK


def unpatch() -> None:
    """Restore the original F.scaled_dot_product_attention."""
    global _patched
    F.scaled_dot_product_attention = _orig_sdpa
    _patched = False


def is_patched() -> bool:
    return _patched
