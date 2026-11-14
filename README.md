# AI Image Studio

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

Also includes the [`z-image-mps`](https://github.com/ivanfioravanti/z-image-mps)
CLI (vendored under `src/z_image_mps/`, a legacy side-tool unrelated to the
web UI) for shell-scripting Z-Image-Turbo.

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
git clone https://github.com/ottocheng/ai-image-studio.git
cd ai-image-studio

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

**Easiest — double-click** `AI-Image-Studio.command` in Finder. A Terminal
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

## CLI usage (Z-Image-Turbo, legacy side-tool)

The `z-image-mps` CLI (from ivanfioravanti) is also installed for shell
scripting. It is **independent of the web UI** — the web UI is entirely
Qwen-Image-2.1, while the CLI drives its own full-precision Z-Image-Turbo
model:

```bash
uv run z-image-mps -p "a cozy neon-lit alley, cinematic" --aspect 16:9 --device mps
```

### CLI reference

```
z-image-mps --prompt "..." [options]

-p, --prompt            Text prompt
--negative-prompt       Negative prompt text
-s, --steps             Inference steps (default: 9)
--guidance-scale        CFG scale (Turbo expects 0.0)
--aspect {1:1,16:9,9:16,4:3,3:4}
--height / --width      Exact dimensions (default 1024x1024)
--seed                  Seed for reproducibility
--num-images            Number of images to generate
-o, --output            Output file (otherwise saved to output/)
--outdir                Output directory
--device {auto,mps,cuda,cpu}
--attention-backend     sdpa | flash2 | flash3
--compile               Try torch.compile() on the DiT transformer
--cpu-offload           Enable CPU offload (CUDA only)
--lora                  LoRA name or path (e.g. 'Technically-Color-Z-Image-Turbo')
--lora-scale            LoRA adapter weight scale (default: 1.3)
```

### Examples

```bash
# Square default
z-image-mps -p "Analog film portrait of a skateboarder, shallow depth of field"

# Widescreen
z-image-mps -p "Cyberpunk night market, neon haze" --aspect 16:9

# Multiple images with a fixed seed (increments per image)
z-image-mps -p "Nordic fjord at dawn, misty" --num-images 3 --seed 123
```

LoRA support: drop a `.safetensors` file into `loras/<name>/` and load it with
`--lora <name> --lora-scale 0.8`.

---

## What's installed

```
ai-image-studio/                         # project root
├── AI-Image-Studio.command              # ⭐ double-click in Finder to launch
├── image-backend.py                     # CLI entry-point shim (z-image-mps console script)
├── src/z_image_mps/                     # ivanfioravanti's CLI (z-image-mps package)
│   ├── cli.py
│   ├── gradio_app.py
│   └── lora_utils.py
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

## Performance & test results

All numbers below were measured on the dev machine **through the app's own
code paths** — the actual Gradio handler functions (`qwen_generate`,
`qwen_edit`), not a separate benchmark harness.

**Test environment**

| | |
|---|---|
| Machine | MacBook Pro (2021), Apple M1 Pro — 10-core CPU, 16 GB unified memory (200 GB/s) |
| macOS | 26 (Darwin 27.2.0) |
| Python | 3.12 (uv-managed) |
| Stack | torch 2.15.0 · diffusers 0.44.0.dev0 (git main) · sdnq 0.2.9 · transformers 5.23.0 · gradio 6.32.0 · torchvision 0.29.2 |
| Model | `OzzyGT/Qwen_Image_2_1_Turbo_sdnq_dynamic_4bit` (warm disk cache, 8-step Turbo schedule) |
| Date | Nov 15, 2026 |

### End-to-end runs

| Test | Result | Detail |
|---|---|---|
| Text-to-image 512×512, seed 42 | ✅ **124.6 s** | prompt *"a red apple on a wooden table, studio lighting, photorealistic"*; 8 denoise steps ≈ 85 s (10.7 s/step: 13.5 s peak at step 2, settling to ~10.1 s as the KV prefix warms) + ~40 s prompt encoding (SDNQ-dequantized Qwen3-VL TE), VAE decode, save. RGBA output → `output/` |
| Instruction edit 512×512, seed 7 | ✅ **195.4 s** | reference = the apple generated above; instruction *"make it a green apple on a dark slate table, moody lighting"*; 8 steps ≈ 144 s (18.0 s/step, range 15.2–28.4 s) + ~51 s reference-image preprocessing + prompt encoding + decode. Editing costs ≈ 1.6× generation |
| Model load (cold process, warm cache) | ✅ **17.2–18.7 s** | **10.80 GB resident on MPS** after load |
| App boot (`uv run python -m webui.app`) | ✅ < 20 s | HTTP 200 on http://127.0.0.1:7860 |
| Unit tests (`uv run python -m pytest tests/ -q`) | ✅ **9/9** in ~5 s | PipelineManager load / already-loaded no-op / swap / unload / 8-thread concurrency race, plus UI-helper functions (seed, size clamping, presets, paths) |
| Failure handling | ✅ verified | with `torchvision` deliberately absent, the handler returned a friendly `⚠️ Generation failed — ImportError: …` status instead of a raw traceback (this is exactly how the missing dependency was caught during testing) |

**Memory behavior (16 GB):** all three components live on MPS (transformer
~4.1 GB + text encoder ~6.7 GB + VAE ~0.7 GB ≈ 10.8 GB allocated) and both
generation and editing complete without swap death. The trick: the text
encoder is only touched during prompt encoding, so macOS pages it out during
the long denoise loop, whose active working set is the transformer (~5–7 GB)
plus activations.

**First download:** 16 files, ~11 GB. On this machine the default hf-xet
path produced a broken cache (snapshot symlinks dangling at empty blobs,
while `snapshot_download` still reported success). If a fresh install fails
to load with file-not-found errors, clear
`~/.cache/huggingface/hub/models--OzzyGT--Qwen_Image_2_1_Turbo_sdnq_dynamic_4bit`
and re-download with `HF_HUB_DISABLE_XET=1` — that plain-HTTP path completed
in ~3 min at 60–75 MB/s here.

### Expectations on other machines

| Op | M1 Pro 16 GB (measured) | Base M1/M2 16 GB (estimated) |
|---|---|---|
| Model load (cached) | 17–19 s | ~2× slower |
| Generate 512×512, 8 steps | ~125 s | ~4–6 min |
| Edit 512×512, 8 steps | ~195 s | ~6–10 min |
| Generate 1024×1024, 8 steps | ~7–10 min (est., token count ×4) | avoid on 16 GB |

> Base-chip estimates scale by memory bandwidth (68 vs 200 GB/s) and GPU
> cores. SDNQ on MPS dequantizes weights to bf16 on every forward — a memory
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
- [ivanfioravanti/z-image-mps](https://github.com/ivanfioravanti/z-image-mps) — the vendored CLI
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
