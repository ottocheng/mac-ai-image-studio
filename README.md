# AI Image Studio

Local AI image generation + editing for **Apple Silicon Macs (M-series)**,
accelerated by MPS (Metal Performance Shaders).
Generate with **Z-Image-Turbo** and edit with **FLUX.2-klein-4B** through a
single Gradio web UI, launched by double-clicking a `.command` file.

Built on top of [ivanfioravanti/z-image-mps](https://github.com/ivanfioravanti/z-image-mps)
(the `z-image-mps` CLI) plus a custom web UI that imports the same diffusers
pipelines and keeps them resident in the server process.

## Requirements

> **Apple Silicon Mac only.** Targets **M-series chips (M1 / M2 / M3 / M4 …)**
> via Apple's **MPS** backend; the web UI asserts `torch.backends.mps.is_available()`
> and won't start on Intel Macs, Windows, or Linux.

| | Minimum | Notes |
|---|---|---|
| **Mac chip** | Apple M1 (2020)+ | First chip with usable MPS |
| **macOS** | 13 (Ventura)+ | Stable Metal / MPS support |
| **Unified memory** | 16 GB | Turbo peaks at ~7.5 GB; FLUX editor at ~5–8 GB |
| **Free disk** | ~15 GB | ~11 GB model cache + ~3 GB deps + output headroom |
| **Python** | 3.10 | Provisioned automatically by `uv sync` |

Faster M-series chips (M2/M3/M4, especially Pro/Max) run proportionally quicker.

---

## Installation (from scratch)

**Prerequisites:** an Apple Silicon Mac (M1 or newer) running macOS 13+, with
~11 GB free disk for the model cache. Dependencies are managed with
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

`uv sync` reads `pyproject.toml`, creates `.venv/`, and installs the full stack:
`torch`, `diffusers` (from git main — needed for `Flux2KleinPipeline`), `gradio`,
`sdnq` (the SDNQ quantization engine), `transformers`, `accelerate`, `scipy`, and
`safetensors`. Python ≥3.10 is required; uv provisions it automatically if absent.

The model checkpoints (~11 GB total) download on **first launch** into
`~/.cache/huggingface/hub/` — outside the repo, so they survive a re-clone.

---

## Quick start

**Easiest — double-click** `AI-Image-Studio.command` in Finder. A
Terminal window opens, the server boots (~6 s after first-time setup), and
your browser launches at http://127.0.0.1:7860. Close the Terminal window to
stop.

**Or from the shell** (from the repo root, after [installing](#installation-from-scratch)):

```bash
uv run python webui/app.py
```

First run downloads the model checkpoints (~11 GB total) into
`~/.cache/huggingface/hub/`. Subsequent runs load from cache.

---

## Using the web UI

The interface is two-column on each tab, and **defaults to dark mode** (your
theme choice is remembered on return). The **left column** holds the prompt
+ action button + output gallery (the gallery sits directly under the button,
matches its width, and shows the full image — letterboxed, never cropped). The
**right column** holds the primary controls, with finer tuning tucked under an
**Advanced** accordion.

- **Generate tab** — Z-Image-Turbo text-to-image. Primary controls: size
  preset and number of images. Defaults: 512×512, 1 image. Under **Advanced**:
  steps (8, the Turbo default), guidance (0.0), seed (-1 for random). First
  Generate click loads Turbo (~12 s). Subsequent clicks skip the load.
- **Edit tab** — FLUX.2-klein-4B instruction editing. Upload an image,
  type an instruction ("turn the apple into an orange", "make it night", "add
  snow"). Under **Advanced**: steps (8 default; the model is 4-step-distilled),
  guidance (1.0; the model is guidance-distilled), seed (-1 for random). First
  Edit click downloads ~4 GB and loads (~70 s total).

Generated and edited images are saved to `output/` with timestamped filenames.
The browser tab shows a 🎨 favicon.

---

## CLI usage

The `z-image-mps` CLI (from ivanfioravanti) is also installed for shell
scripting:

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
--lora-scale            LoRA adapter weight scale (default: 1.0)
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
├── webui/                               # custom unified UI
│   ├── app.py                           #   Gradio: Generate tab + Edit tab
│   ├── mps_chunked_attn.py              #   query-chunked SDPA memory patch (MPS)
│   ├── pipelines.py                     #   PipelineManager — load/swap Turbo ↔ FLUX editor
│   └── favicon.png                      #   🎨 browser tab icon (Noto Color Emoji, OFL 1.1)
├── output/                              # generated images land here
└── pyproject.toml
```

Python deps (in `pyproject.toml`, installed into `.venv/` by `uv sync`):

- `sdnq` (≥0.2.1) — registers SDNQ quantization with diffusers; pure Python, MPS-safe
- `scipy` (≥1.11) — required by `FlowMatchEulerDiscreteScheduler(use_beta_sigmas=True)`
- `diffusers` (from git main), `gradio`, `torch`, `transformers`, `accelerate`, `safetensors`

---

## Performance

| Op | Wall time | Memory |
|---|---|---|
| Turbo load (cached) | 11–14 s | — |
| Turbo generate 512×512, 8 steps | 64–72 s | 7.54 GB |
| FLUX editor first load (download + load) | 73 s | 5.08 GB |
| FLUX edit, 20 steps | 201 s | 5.08 GB |
| Pipeline swap (unload + load other) | 12–74 s | — |

> Timings are indicative and vary by chip and memory. The FLUX edit default is
> 8 steps (the model is 4-step-distilled).

Expect roughly 1 minute per Turbo image and a couple of minutes per FLUX edit
at the 8-step default.

---

## Cleanup

Gradio temp files (`/tmp/gradio/*`) can be cleared periodically:

```bash
rm -rf /tmp/gradio/*
```

If you ever want to reclaim the model cache (e.g. before moving the project
off this Mac), the SDNQ checkpoints live in:

```bash
~/.cache/huggingface/hub/models--Disty0--Z-Image-Turbo-SDNQ-uint4-svd-r32     # ~6 GB
~/.cache/huggingface/hub/models--Disty0--FLUX.2-klein-4B-SDNQ-4bit-dynamic    # ~5 GB
~/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B           # ~15 MB (tokenizer)
```

The next launch will re-download what it needs.

---

## Credits

- [Tongyi-MAI/Z-Image](https://github.com/Tongyi-MAI/Z-Image) — original model
- [ivanfioravanti/z-image-mps](https://github.com/ivanfioravanti/z-image-mps) — the CLI base
- [Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32](https://huggingface.co/Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32) — quantized Turbo
- [Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic](https://huggingface.co/Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic) — quantized editor
- [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) — base FLUX editor model (tokenizer source)
- [newideas99/ultra-fast-image-gen](https://github.com/newideas99/ultra-fast-image-gen) — reference for MPS + SDNQ recipe; source of the chunked-SDPA memory patch in `webui/mps_chunked_attn.py`
- [sdnq](https://pypi.org/project/sdnq/) — Disty0's quantization engine
- [huggingface/diffusers](https://github.com/huggingface/diffusers) — pipeline library (git main, needed for `Flux2KleinPipeline`)
- [Noto Color Emoji](https://fonts.google.com/noto/specimen/Noto+Color+Emoji) — 🎨 favicon (OFL 1.1)

---

## License

This project's **source code** is released under the MIT License — see
[LICENSE](./LICENSE).

The **model weights** this project downloads and runs are *not* covered by that
MIT license. Each model is distributed under its own license by its publisher.
