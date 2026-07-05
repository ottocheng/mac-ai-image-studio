"""Pipeline loader and swap manager for the unified Z-Image web UI.

Memory constraint: 16 GB unified memory on the target Mac (M1 Pro).
  - Z-Image-Turbo SDNQ-uint4 peak: ~7.5 GB
  - FLUX.2-klein-4B SDNQ-4bit peak: ~8 GB
Both cannot be resident simultaneously. PipelineManager loads one at a time
and unloads (del + empty_cache) the other before loading.

Models:
  Turbo : Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32       (~3.5 GB download)
  Editor: Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic      (~4 GB download)
          + tokenizer from black-forest-labs/FLUX.2-klein-4B
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

os.environ.setdefault("PYTORCH_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "1")  # MPS fast-math elementwise kernels (bf16 MLP/silu paths)
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")  # let MPS use what it can; OS will swap if starving
# Skip Triton auto-detection in SDNQ — Triton is NVIDIA-CUDA only and the
# check prints a noisy "falling back to PyTorch Eager" banner on every startup
# on Apple Silicon MPS. Eager is the only path that works here anyway.
os.environ.setdefault("SDNQ_USE_TORCH_COMPILE", "0")

import torch

# sdnq registers SDNQ quantization with diffusers/transformers on import
from sdnq import SDNQConfig  # noqa: F401

from diffusers import Flux2KleinPipeline, ZImagePipeline, FlowMatchEulerDiscreteScheduler

# Install the chunked-SDPA memory patch on MPS (see mps_chunked_attn.py). MPS has
# no FlashAttention, so the default SDPA materializes the full Lq*Lk score matrix
# -- the allocation that triggers swap during FLUX edits on 16 GB. An internal
# guard makes it a no-op for small attention grids, so Z-Image-Turbo is untouched.
from mps_chunked_attn import patch as _patch_chunked_sdpa

if torch.backends.mps.is_available():
    _patch_chunked_sdpa()


TURBO_REPO = "Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32"
FLUX_EDITOR_REPO = "Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic"
FLUX_BASE_FOR_TOKENIZER = "black-forest-labs/FLUX.2-klein-4B"


ActivePipeline = Any  # ZImagePipeline | Flux2KleinPipeline


@dataclass
class LoadReport:
    name: str
    seconds: float
    mps_gb: float


class PipelineManager:
    """Singleton-ish holder for the currently-loaded pipeline.

    Not a singleton itself (testability), but the web UI keeps one instance.
    All public methods are guarded by a single mutex so two Gradio clicks
    can't race the load/swap.
    """

    def __init__(self, device: str = "mps") -> None:
        assert torch.backends.mps.is_available(), "MPS not available on this Mac"
        self.device = device
        self._lock = threading.Lock()
        self._active_name: Optional[str] = None  # "turbo" | "flux_editor" | None
        self._pipe: Optional[ActivePipeline] = None

    # ------------------------------------------------------------------ utils
    @property
    def active_name(self) -> Optional[str]:
        return self._active_name

    def mps_gb(self) -> float:
        if not torch.backends.mps.is_available():
            return 0.0
        return torch.mps.current_allocated_memory() / 1024**3

    def _unload(self) -> None:
        if self._pipe is None:
            return
        del self._pipe
        self._pipe = None
        self._active_name = None
        # Python-side MPS cache clear
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            # Give the OS a beat to reclaim
            time.sleep(0.3)

    def _ensure(self, name: str, loader) -> LoadReport:
        """Make the named pipeline the active one, swapping if needed."""
        with self._lock:
            if self._active_name == name and self._pipe is not None:
                return LoadReport(name, 0.0, self.mps_gb())
            # Different model needed: drop the current one first
            self._unload()
            t0 = time.time()
            self._pipe = loader()
            self._active_name = name
            return LoadReport(name, time.time() - t0, self.mps_gb())

    # --------------------------------------------------------------- Turbo
    def load_turbo(self) -> LoadReport:
        def _load() -> ZImagePipeline:
            pipe = ZImagePipeline.from_pretrained(
                TURBO_REPO,
                torch_dtype=torch.float32,  # SDNQ keeps weights quantized; outer dtype is compute dtype
                low_cpu_mem_usage=True,
            )
            # use_beta_sigmas matches the SDNQ model's training; needs scipy
            pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
                pipe.scheduler.config, use_beta_sigmas=True,
            )
            # MPS memory optimizations
            pipe.enable_attention_slicing()
            if hasattr(pipe, "enable_vae_slicing"):
                pipe.enable_vae_slicing()
            if hasattr(pipe.vae, "enable_tiling"):
                pipe.vae.enable_tiling()
            pipe.to(self.device)
            return pipe

        return self._ensure("turbo", _load)

    def get_turbo(self) -> ZImagePipeline:
        """Return the resident Turbo pipeline, loading it if needed."""
        if self._active_name != "turbo" or self._pipe is None:
            self.load_turbo()
        assert self._pipe is not None
        return self._pipe

    # ---------------------------------------------------------- FLUX editor
    def load_flux_editor(self) -> LoadReport:
        def _load() -> Flux2KleinPipeline:
            from transformers import AutoTokenizer

            # SDNQ model is missing vocab files; pull tokenizer from the base repo
            tokenizer = AutoTokenizer.from_pretrained(
                FLUX_BASE_FOR_TOKENIZER,
                subfolder="tokenizer",
                use_fast=False,
            )
            pipe = Flux2KleinPipeline.from_pretrained(
                FLUX_EDITOR_REPO,
                tokenizer=tokenizer,
                # Keep bf16 (per model card): fp16 compute caused a ~20x/step slowdown
                # on this SDNQ-quantized checkpoint (its dequant path is bf16-tuned).
                torch_dtype=torch.bfloat16,
            )
            pipe.enable_attention_slicing()
            if hasattr(pipe, "enable_vae_slicing"):
                pipe.enable_vae_slicing()
            if hasattr(pipe, "enable_vae_tiling"):
                pipe.enable_vae_tiling()
            elif hasattr(getattr(pipe, "vae", None), "enable_tiling"):
                pipe.vae.enable_tiling()
            pipe.to(self.device)
            return pipe

        return self._ensure("flux_editor", _load)

    def get_flux_editor(self) -> Flux2KleinPipeline:
        if self._active_name != "flux_editor" or self._pipe is None:
            self.load_flux_editor()
        assert self._pipe is not None
        return self._pipe


# Module-level singleton for the Gradio app to use
MANAGER = PipelineManager()
