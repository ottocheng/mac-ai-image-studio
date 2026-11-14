# Mac (M series) AI Image Studio - Based on Qwen Image 2.1

Local AI image generation + editing for **Apple Silicon Macs (M-series)**,
accelerated by MPS (Metal Performance Shaders).

**Powered by [Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo)
— one model, two jobs**: text-to-image **and** instruction-based editing
through a single Gradio web UI, launched by double-clicking a `.command` file.

> ⚠️ **License**: Qwen-Image-2.1 weights are distributed under the
> **Qwen Research License (non-commercial)**. The source code of this repo is
> MIT; the model weights you download are not.

The app runs the SDNQ int4-quantized checkpoint
[`OzzyGT/Qwen_Image_2_1_Turbo_sdnq_dynamic_4bit`](https://huggingface.co/OzzyGT/Qwen_Image_2_1_Turbo_sdnq_dynamic_4bit)
(~11.5 GB resident; the bf16 original is ~32 GB and cannot fit a 16 GB Mac).

## Requirements

> **Apple Silicon Mac only.** Targets **base M-series chips (M1+) with 16 GB
> unified memory** via Apple's **MPS** backend; the web UI checks
> `torch.backends.mps.is_available()` and won't start elsewhere.

| | Minimum | Notes |
|---|---|---|
| **Mac chip** | Apple M1 (2020)+ | First chip with usable MPS |
| **macOS** | 15 (Sequoia)+ | torch 2.15 requirement |
| **Python** | 3.11+ | Provisioned automatically by `uv sync` |
| **Unified memory** | 16 GB | ~11.5 GB of weights resident; macOS pages the text encoder out during denoising |
| **Free disk** | ~15 GB | ~11.5 GB model cache + ~3 GB deps + output headroom |

Faster M-series chips (M2/M3/M4, especially Pro/Max) run proportionally quicker.

---

## Installation (from scratch)

**Prerequisites:** an Apple Silicon Mac (M1 or newer, 16 GB+) running macOS
15+, with ~15 GB free disk for the model cache. Dependencies are managed with
[uv](https://docs.astral.sh/uv/), which also provisions the right Python.

```bash
# 1. Get the code
git clone https://github.com/ottocheng/mac-ai-image-studio.git
cd mac-ai-image-studio

# 2. Install uv (skip if `which uv` already prints a path)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Create the virtualenv and install every dependency (~2 min)
uv sync
```

`uv sync` reads `pyproject.toml` + the committed `uv.lock`, creates `.venv/`,
and installs the full stack: `torch` (2.15+ — Metal attention kernels),
`diffusers` (from git main — needed for `QwenImage21Pipeline` and the
pipeline-configured sampling sigmas), `gradio`, `sdnq` (the SDNQ quantization
engine), `transformers`, `accelerate`, and `safetensors`.

The model checkpoint (~11.5 GB) downloads on **first launch** (or first
generation) into `~/.cache/huggingface/hub/` — outside the repo, so it
survives a re-clone.

---

## Quick start

**Easiest — double-click** `Mac-AI-Image-Studio.command` in Finder. A Terminal
window opens, the server boots, and your browser launches at
http://127.0.0.1:7860. Close the Terminal window to stop.

**Or from the shell** (from the repo root, after [installing](#installation-from-scratch)):

```bash
uv run python -m webui.app
```

---

## Using the web UI

The interface is two-column on each tab and **defaults to dark mode** (your
theme choice is remembered on return). **Left column**: prompt + action
button + output gallery (full image, letterboxed, never cropped). **Right
column**: primary controls.

- **Generate tab** — Qwen-Image-2.1-Turbo text-to-image. Primary controls:
  size preset and number of images (1–4, incrementing seeds). Under
  **Advanced**: seed (-1 for random). First click loads the model (~20 s from
  cache; the first ever run downloads ~11.5 GB).
- **Edit tab** — Qwen-Image-2.1-Turbo instruction editing. Upload an image,
  type an instruction ("turn the apple into an orange", "make it night", "add
  snow"), pick an output size (default 1024×1024). Under **Advanced**: seed.

**No steps/guidance sliders, by design**: Turbo's 8-step sampling schedule is
baked into the checkpoint (loaded via `sample_sigmas`) and `num_inference_steps`
cannot override it — only an explicit `sigmas` argument can. CFG is 1.0
(guidance-distilled). Exposing knobs that do nothing would be dishonest UI.

Size presets stay in the 512–1024 band because the denoise working set scales
with token count on unified memory; the model's native 2K resolutions work
but are slow on 16 GB M-series.

Generated and edited images are saved to `output/` with timestamped filenames.
The browser tab shows a 🎨 favicon.

---

## What's installed

```
mac-ai-image-studio/                    # project root
├── Mac-AI-Image-Studio.command         # ⭐ double-click in Finder to launch
├── webui/                               # the app (a package; run with python -m webui.app)
│   ├── app.py                           #   Gradio: Generate tab + Edit tab (both Qwen)
│   ├── pipelines.py                     #   PipelineManager — lazy-load, mutex-guarded
│   ├── mps_chunked_attn.py              #   query-chunked SDPA memory patch (MPS)
│   └── favicon.png                      #   🎨 browser tab icon (Noto Color Emoji, OFL 1.1)
├── tests/                               # unit tests (PipelineManager + helpers) — pytest
├── output/                              # generated images land here
├── pyproject.toml
└── uv.lock                              # committed for reproducible installs
```

Python deps (in `pyproject.toml`, installed into `.venv/` by `uv sync`):

- `sdnq` (≥0.2.2) — registers SDNQ quantization with diffusers; pure Python, MPS-safe
- `diffusers` (from git main) — `QwenImage21Pipeline` + pipeline-configured sigmas
- `torch` (≥2.15), `gradio` (≥4.44), `transformers` (≥5.17), `accelerate`, `safetensors`

Run the tests: `uv run python -m pytest tests/ -q`

---

## Performance

Indicative timings for the 8-step Turbo schedule (single image, warm disk
cache). Faster M-series chips (M3/M4, especially Pro/Max) run proportionally
quicker; base M1/M2 chips run roughly 2–3× slower than M1 Pro-class due to
memory bandwidth.

| Op | M1 Pro-class 16 GB | Base M1/M2 16 GB |
|---|---|---|
| Model load (cached) | ~20 s | ~40 s |
| Generate 512×512, 8 steps | ~2 min | ~4–6 min |
| Edit 512×512, 8 steps | ~3 min | ~6–10 min |
| Generate 1024×1024, 8 steps | ~7–10 min | avoid on 16 GB |

Editing costs ≈ 1.6× generation (the reference image joins the attention
context), and prompt encoding through the SDNQ-dequantized Qwen3-VL text
encoder adds ~30–50 s per run.

**Memory (16 GB):** the full pipeline is ~11 GB resident on MPS (transformer
~4.1 GB + text encoder ~6.7 GB + VAE ~0.7 GB), and both generation and
editing complete without swapping out. The text encoder is only touched
during prompt encoding, so macOS pages it out during the denoise loop; the
active working set stays near the transformer plus activations.

**If a fresh install fails to load the model** with file-not-found errors,
the HF cache may be corrupt — the hf-xet download path has been observed
leaving dangling symlinks while still reporting success. Clear
`~/.cache/huggingface/hub/models--OzzyGT--Qwen_Image_2_1_Turbo_sdnq_dynamic_4bit`
and re-download with `HF_HUB_DISABLE_XET=1`.

> SDNQ on MPS dequantizes weights to bf16 on every forward — a memory
> technology, not a speed one. If throughput matters more than this app's
> simplicity, [mflux](https://github.com/mflux-community/mflux)/MLX runs the
> same model family with native INT8 matmul and pulls further ahead on M3+
> machines.

---

## Cleanup

Gradio temp files (`/tmp/gradio/*`) can be cleared periodically:

```bash
rm -rf /tmp/gradio/*
```

If you ever want to reclaim the model cache (e.g. before moving the project
off this Mac), the checkpoint lives in:

```bash
~/.cache/huggingface/hub/models--OzzyGT--Qwen_Image_2_1_Turbo_sdnq_dynamic_4bit   # ~11.5 GB
```

The next launch will re-download what it needs.

---

## Credits

- [QwenLM/Qwen-Image-2.1](https://github.com/QwenLM/Qwen-Image-2.1) — the model (Qwen Research License)
- [OzzyGT](https://huggingface.co/OzzyGT) — the SDNQ dynamic int4 quantization of Qwen-Image-2.1-Turbo
- [Disty0/sdnq](https://github.com/Disty0/sdnq) — the SDNQ quantization engine
- [huggingface/diffusers](https://github.com/huggingface/diffusers) — pipeline library (git main)
- [newideas99/ultra-fast-image-gen](https://github.com/newideas99/ultra-fast-image-gen) — source of the chunked-SDPA memory patch in `webui/mps_chunked_attn.py`
- [Noto Color Emoji](https://fonts.google.com/noto/specimen/Noto+Color+Emoji) — 🎨 favicon (OFL 1.1)

---

## License

This project's **source code** is released under the MIT License — see
[LICENSE](./LICENSE).

The **model weights** this project downloads and runs are *not* covered by that
MIT license: Qwen-Image-2.1 (and its quantization) is distributed under the
**Qwen Research License — non-commercial use only**.
