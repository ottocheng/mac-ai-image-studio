"""Unified AI image web UI: Generate (Turbo) + Edit (FLUX.2-klein-4B).

Run:
    cd /Users/ottocheng/Project/image-studio
    uv run python webui/app.py
    # or just double-click AI-Image-Studio.command in Finder

Then open http://127.0.0.1:7860 in your browser.

Architecture note: pipelines are imported from webui/pipelines.py and kept
resident in this process. Spawning the CLI per request would reload the
~3.5 GB Turbo checkpoint every call (~12 s overhead each). Resident pipelines
make the first call pay that cost once and every subsequent call skips it.
The two pipelines cannot both be resident on 16 GB unified memory, so
PipelineManager unloads the other one before loading the requested one
(see pipelines.py).
"""
from __future__ import annotations

import warnings

# Silence one specific Starlette deprecation that leaks through gradio's
# HTTP layer (gradio/routes.py:1379 still references the renamed constant).
# Match by message so real deprecations still surface.
warnings.filterwarnings(
    "ignore",
    message=r"'HTTP_422_UNPROCESSABLE_ENTITY' is deprecated",
)

import gc
import os
import secrets
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import gradio as gr

from pipelines import MANAGER, LoadReport

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


# ---------------------------------------------------------------- helpers
def _seed_from(seed: int | None) -> int:
    """None or -1 → random; otherwise return seed as-is."""
    if seed is None or seed < 0:
        return secrets.randbits(63)
    return int(seed)


def _timestamped_path(prefix: str, ext: str = ".png") -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    short = secrets.token_hex(3)
    return OUTPUT_DIR / f"{prefix}-{ts}-{short}{ext}"


def _fmt_load_report(r: LoadReport, action: str) -> str:
    if r.seconds < 0.1:
        return f"{action} (already loaded) — MPS {r.mps_gb:.2f} GB"
    return f"{action} loaded in {r.seconds:.1f}s — MPS {r.mps_gb:.2f} GB"


# ---------------------------------------------------------------- Turbo
def turbo_generate(
    prompt: str,
    negative_prompt: str,
    size_preset: str,
    custom_height: int,
    custom_width: int,
    steps: int,
    guidance: float,
    seed: int,
    num_images: int,
    progress: gr.Progress = gr.Progress(),
):
    import torch
    from PIL import Image

    if not prompt.strip():
        return None, "Prompt is empty."

    # Resolve size
    if size_preset == "Custom":
        h, w = int(custom_height), int(custom_width)
    else:
        # preset like "512 x 512 (square)" → take the two ints
        parts = [int(s) for s in size_preset.replace("×", "x").split() if s.isdigit()]
        h, w = parts[0], parts[1]
    # Multiple of 16 for VAE
    h = max(256, (h // 16) * 16)
    w = max(256, (w // 16) * 16)

    progress(0.0, desc="Loading Turbo model...")
    load_report = MANAGER.load_turbo()
    pipe = MANAGER.get_turbo()

    n = max(1, min(int(num_images), 4))
    base_seed = _seed_from(seed)
    images: list[Image.Image] = []
    t_start = time.time()

    for i in range(n):
        seed_i = base_seed + i
        progress(
            i / n,
            desc=f"Generating {i+1}/{n}  ({w}x{h}, steps={steps}, seed={seed_i})",
        )

        # Default args pin the loop vars at closure-creation time — without
        # them, every callback would see the final iteration's values (Python
        # late-binding gotcha).
        def on_step_end(pipe, step_index, timestep, callback_kwargs,
                        _i=i, _n=n, _steps=steps, _seed=seed_i):
            progress(
                (_i + (step_index + 1) / _steps) / _n,
                desc=f"Generating {_i + 1}/{_n}  step {step_index + 1}/{_steps}  seed={_seed}",
            )
            return callback_kwargs

        with torch.inference_mode():
            result = pipe(
                prompt=prompt.strip(),
                negative_prompt=negative_prompt.strip() or None,
                height=h,
                width=w,
                num_inference_steps=int(steps),
                guidance_scale=float(guidance),
                generator=torch.Generator("mps").manual_seed(seed_i),
                callback_on_step_end=on_step_end,
            )
        images.append(result.images[0])

    progress(1.0, desc="Done")

    elapsed = time.time() - t_start
    # Save all + return gallery
    saved_paths = []
    for i, img in enumerate(images):
        p = _timestamped_path(f"turbo-{i+1}")
        img.save(p)
        saved_paths.append(str(p))

    status = (
        f"{_fmt_load_report(load_report, 'Turbo')}  |  "
        f"Generated {n} image{'s' if n != 1 else ''} at {w}x{h} in {elapsed:.1f}s "
        f"(~{elapsed/n:.1f}s each, {steps} steps)  |  "
        f"Seeds: {[base_seed + i for i in range(n)]}"
    )
    gallery = [(img, f"img {i+1}") for i, img in enumerate(images)]
    return gallery, status


# ---------------------------------------------------------------- FLUX editor
def flux_edit(
    input_image,
    prompt: str,
    steps: int,
    guidance: float,
    seed: int,
    progress: gr.Progress = gr.Progress(),
):
    import torch
    from PIL import Image

    if input_image is None:
        return None, "Please upload or drop an image to edit."
    if not prompt.strip():
        return None, "Editing instruction is empty."

    if isinstance(input_image, dict):
        input_image = input_image.get("path") or input_image.get("url")
    if isinstance(input_image, str):
        input_image = Image.open(input_image).convert("RGB")

    progress(0.0, desc="Loading FLUX.2-klein editor (first run downloads ~4 GB)...")
    load_report = MANAGER.load_flux_editor()
    pipe = MANAGER.get_flux_editor()

    seed_i = _seed_from(seed)
    progress(0.0, desc=f"Editing  steps={steps}  guidance={guidance}  seed={seed_i}")
    t_start = time.time()

    def on_step_end(pipe, step_index, timestep, callback_kwargs, _steps=steps, _seed=seed_i):
        progress(
            (step_index + 1) / _steps,
            desc=f"Editing  step {step_index + 1}/{_steps}  seed={_seed}",
        )
        return callback_kwargs

    with torch.inference_mode():
        result = pipe(
            image=input_image,
            prompt=prompt.strip(),
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            generator=torch.Generator("cpu").manual_seed(seed_i),
            callback_on_step_end=on_step_end,
        )
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    progress(1.0, desc="Done")
    elapsed = time.time() - t_start
    out = result.images[0]
    p = _timestamped_path("edit")
    out.save(p)

    status = (
        f"{_fmt_load_report(load_report, 'FLUX editor')}  |  "
        f"Edited in {elapsed:.1f}s ({steps} steps, seed={seed_i})  |  "
        f"Saved: {p.name}"
    )
    return [(out, "edited")], status


# ---------------------------------------------------------------- UI
SIZE_PRESETS = [
    "512 x 512 (square, fastest)",
    "768 x 768 (square)",
    "640 x 896 (portrait 9:16)",
    "896 x 640 (landscape 16:9)",
    "832 x 624 (4:3 landscape)",
    "Custom",
]

CUSTOM_CSS = """
#title { text-align: center; }
#title h1 { margin: 0.25em 0; font-weight: 700; }
#title p { margin: 0.25em 0; color: #666; font-size: 0.95em; }
/* Gallery thumbnails must show the whole image (letterbox), never crop. */
.grid-wrap > div,
.thumbnail-item,
.thumbnail-image,
.thumbnail-image img {
    object-fit: contain !important;
}
"""

# Default the UI to dark mode on first visit; once the user picks a theme it is
# remembered in localStorage and this no longer overrides their choice.
DEFAULT_THEME_JS = """
() => {
    if (!localStorage.getItem('theme')) {
        localStorage.setItem('theme', '"dark"');
    }
}
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="AI Image Studio") as demo:
        gr.HTML(
            "<div id='title'>"
            "<h1>🎨 AI Image Studio</h1>"
            "<p>Local Z-Image-Turbo generation + FLUX.2-klein-4B editing on Apple Silicon MPS</p>"
            "</div>"
        )

        with gr.Tab("✨ Generate (Z-Image-Turbo)"):
            with gr.Row():
                with gr.Column(scale=3):
                    prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Describe the image. e.g. 'a red apple on a wooden table, studio lighting'",
                        lines=3,
                    )
                    negative = gr.Textbox(
                        label="Negative prompt (optional)",
                        placeholder="blurry, low quality, distorted",
                        lines=1,
                    )
                    generate_btn = gr.Button("Generate", variant="primary")
                    gallery = gr.Gallery(
                        label="Output",
                        columns=1,
                        show_label=True,
                        preview=True,
                        object_fit="contain",
                    )
                    gen_status = gr.Markdown()
                with gr.Column(scale=2):
                    size = gr.Dropdown(
                        SIZE_PRESETS,
                        value=SIZE_PRESETS[0],
                        label="Size",
                    )
                    with gr.Accordion("Custom size", open=False):
                        custom_h = gr.Slider(256, 1280, value=512, step=16, label="Height")
                        custom_w = gr.Slider(256, 1280, value=512, step=16, label="Width")
                    num_images = gr.Slider(1, 4, value=1, step=1, label="Number of images")
                    with gr.Accordion("Advanced", open=False):
                        steps = gr.Slider(4, 16, value=8, step=1,
                                          label="Steps (8 is Turbo default)")
                        guidance = gr.Slider(0.0, 10.0, value=0.0, step=0.1,
                                             label="Guidance (0.0 for Turbo)")
                        seed = gr.Number(label="Seed (-1 for random)", value=-1)

            generate_btn.click(
                fn=turbo_generate,
                inputs=[prompt, negative, size, custom_h, custom_w, steps, guidance, seed, num_images],
                outputs=[gallery, gen_status],
            )

        with gr.Tab("✂️ Edit (FLUX.2-klein-4B)"):
            with gr.Row():
                with gr.Column(scale=2):
                    input_image = gr.Image(label="Image to edit", type="pil", height=320)
                    edit_gallery = gr.Gallery(
                        label="Edited",
                        columns=1,
                        show_label=True,
                        preview=True,
                        object_fit="contain",
                    )
                    edit_status = gr.Markdown()
                with gr.Column(scale=2):
                    edit_prompt = gr.Textbox(
                        label="Editing instruction",
                        placeholder="e.g. 'make the sky a dramatic sunset', 'turn the apple into an orange', 'add snow'",
                        lines=3,
                    )
                    with gr.Accordion("Advanced", open=False):
                        edit_steps = gr.Slider(4, 50, value=8, step=1,
                                                label="Steps (8 default; model is 4-step-distilled)")
                        edit_guidance = gr.Slider(0.0, 10.0, value=1.0, step=0.1,
                                                   label="Guidance (1.0; model is guidance-distilled)")
                        edit_seed = gr.Number(label="Seed (-1 for random)", value=-1)
                    edit_btn = gr.Button("Edit", variant="primary")

            edit_btn.click(
                fn=flux_edit,
                inputs=[input_image, edit_prompt, edit_steps, edit_guidance, edit_seed],
                outputs=[edit_gallery, edit_status],
            )

        gr.HTML(
            "<hr/><p style='text-align:center;color:#888;font-size:0.85em'>"
            "Turbo: <code>Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32</code> · "
            "Editor: <code>Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic</code>"
            "</p>"
        )

    return demo


if __name__ == "__main__":
    HERE = Path(__file__).resolve().parent
    FAVICON = HERE / "favicon.png"

    app = build_ui()
    app.queue(default_concurrency_limit=1)  # serialize: only one model run at a time on 16 GB
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,  # auto-open browser — right default for the .command double-click flow
        favicon_path=str(FAVICON) if FAVICON.exists() else None,
        theme=gr.themes.Soft(),
        js=DEFAULT_THEME_JS,
        css=CUSTOM_CSS,
    )
