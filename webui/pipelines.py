"""Pipeline loader for Mac (M series) AI Image Studio (Qwen-Image-2.1-Turbo).

Memory constraint: 16 GB unified memory on the target Macs (base M-series).
  - Checkpoint: OzzyGT/Qwen_Image_2_1_Turbo_sdnq_dynamic_4bit
      transformer   ~4.1 GB  (SDNQ dynamic int4)
      text encoder  ~6.7 GB  (SDNQ; Qwen3-VL ~8.8B)
      VAE           ~0.7 GB
      resident      ~11.5 GB total on MPS
  The bf16 original (Qwen/Qwen-Image-2.1-Turbo) is ~32 GB and cannot fit;
  quantization is mandatory on 16 GB.

SDNQ on MPS has no Triton, so weights are dequantized to the compute dtype
(bfloat16) on each forward — a memory technology here, not a speed one. The
text encoder is only touched during prompt encoding; macOS pages it out
during the (long) denoise loop, which keeps the active working set near the
transformer's ~5-7 GB.

The 8-step sampling schedule is baked into the checkpoint (model_index.json
sample_sigmas); `num_inference_steps` does not override it.

License note: Qwen-Image-2.1 is distributed under the Qwen Research License
(non-commercial).
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

os.environ.setdefault("PYTORCH_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "1")  # MPS fast-math elementwise kernels
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")  # let MPS use what it can; OS will swap if starving
# Skip Triton auto-detection in SDNQ — Triton is NVIDIA-CUDA only and the
# check prints a noisy banner on every startup on Apple Silicon MPS.
os.environ.setdefault("SDNQ_USE_TORCH_COMPILE", "0")

import torch

# sdnq registers SDNQ quantization with diffusers/transformers on import
from sdnq import SDNQConfig  # noqa: F401

from diffusers import QwenImage21Pipeline

# Install the chunked-SDPA memory patch on MPS (see mps_chunked_attn.py). MPS
# has no FlashAttention, so the default SDPA materializes the full Lq*Lk score
# matrix -- the allocation that triggers swap on 16 GB at larger grids. The
# guard makes it a no-op for small attention grids.
from .mps_chunked_attn import patch as _patch_chunked_sdpa

if torch.backends.mps.is_available():
    _patch_chunked_sdpa()


QWEN_REPO = "OzzyGT/Qwen_Image_2_1_Turbo_sdnq_dynamic_4bit"


@dataclass
class LoadReport:
    name: str
    seconds: float
    mps_gb: float


class PipelineManager:
    """Holder for the currently-loaded pipeline.

    Not a singleton itself (testability), but the web UI keeps one instance
    (`MANAGER`). All public methods are guarded by a single mutex so two
    Gradio clicks can't race the load/swap.

    The MPS assertion is deliberately *lazy* (checked at load time, not in
    ``__init__``) so the module can be imported and the class unit-tested on
    any machine; pass ``device="cpu"`` with fake loaders in tests.
    """

    def __init__(self, device: str = "mps") -> None:
        self.device = device
        self._lock = threading.Lock()
        self._active_name: Optional[str] = None
        self._pipe: Optional[Any] = None

    # ------------------------------------------------------------------ utils
    @property
    def active_name(self) -> Optional[str]:
        return self._active_name

    def mps_gb(self) -> float:
        if not torch.backends.mps.is_available():
            return 0.0
        return torch.mps.current_allocated_memory() / 1024**3

    def _require_device(self) -> None:
        if self.device == "mps":
            assert torch.backends.mps.is_available(), "MPS not available on this Mac"

    def _unload(self) -> None:
        if self._pipe is None:
            return
        del self._pipe
        self._pipe = None
        self._active_name = None
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            # Give the OS a beat to reclaim
            time.sleep(0.3)

    def _ensure(self, name: str, loader: Callable[[], Any]) -> LoadReport:
        """Make the named pipeline the active one, swapping if needed."""
        with self._lock:
            if self._active_name == name and self._pipe is not None:
                return LoadReport(name, 0.0, self.mps_gb())
            self._require_device()
            # Different model needed: drop the current one first
            self._unload()
            t0 = time.time()
            self._pipe = loader()
            self._active_name = name
            return LoadReport(name, time.time() - t0, self.mps_gb())

    # ------------------------------------------------------------------ Qwen
    def load_qwen(self) -> LoadReport:
        def _load() -> QwenImage21Pipeline:
            pipe = QwenImage21Pipeline.from_pretrained(
                QWEN_REPO,
                # SDNQ keeps weights quantized; the outer dtype is the compute
                # dtype. bf16 matches Qwen training and halves fp32 memory
                # traffic on MPS.
                dtype=torch.bfloat16,
            )
            # MPS memory optimizations (guarded: API surface moves between
            # diffusers versions).
            if hasattr(pipe, "enable_attention_slicing"):
                pipe.enable_attention_slicing()
            if hasattr(pipe.vae, "enable_tiling"):
                pipe.vae.enable_tiling()
            pipe.to(self.device)
            return pipe

        return self._ensure("qwen", _load)

    def get_qwen(self) -> Any:
        """Return the resident Qwen pipeline, loading it if needed.

        Routes through load_qwen -> _ensure (which holds the lock), then
        re-reads under the lock; with a single pipeline name there is no
        read-then-load race window.
        """
        self.load_qwen()
        with self._lock:
            assert self._active_name == "qwen" and self._pipe is not None
            return self._pipe


# Module-level instance for the Gradio app to use. Constructing it does not
# touch MPS (lazy assert) or load any weights.
MANAGER = PipelineManager()
