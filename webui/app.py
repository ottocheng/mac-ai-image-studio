"""Unified AI image web UI: Qwen-Image-2.1-Turbo for generation AND editing.

Run (from the repo root):
    uv run python -m webui.app
    # or just double-click Mac-AI-Image-Studio.command in Finder

Then open http://127.0.0.1:7860 in your browser.

Architecture note: the pipeline is imported from webui/pipelines.py and kept
resident in this process. One model serves both tabs — Qwen-Image-2.1-Turbo
is a unified text-to-image + instruction-editing checkpoint. The 8-step
sampling schedule is baked into the checkpoint; the UI deliberately exposes
no steps/guidance controls because they cannot override it (only an explicit
`sigmas` argument can, which we don't pass).

License note: Qwen-Image-2.1 weights are Qwen Research License (non-commercial).
"""
from __future__ import annotations

import warnings

# Silence one specific Starlette deprecation that leaks through gradio's
# HTTP layer. Match by message so real deprecations still surface.
warnings.filterwarnings(
    "ignore",
    message=r"'HTTP_422_UNPROCESSABLE_ENTITY' is deprecated",
)

import gc
import secrets
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import gradio as gr

from .pipelines import MANAGER, LoadReport

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


def _cleanup_mps() -> None:
    gc.collect()
    import torch

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _clamp_size(h: int, w: int) -> tuple[int, int]:
    """Multiple of 16 for the VAE, floored at 256."""
    return max(256, (h // 16) * 16), max(256, (w // 16) * 16)


# ---------------------------------------------------------------- Generate
def qwen_generate(
    prompt: str,
    size_preset: str,
    custom_height: float,
    custom_width: float,
    seed: int,
    num_images: float,
    progress: gr.Progress = gr.Progress(),
):
    import torch

    if not prompt.strip():
        return None, "Prompt is empty."

    if size_preset == "Custom":
        h, w = _clamp_size(int(custom_height), int(custom_width))
    else:
        w, h = SIZE_PRESETS[size_preset]

    load_report = None
    try:
        progress(0.0, desc="Loading Qwen-Image-2.1-Turbo (first run downloads ~11.5 GB)...")
        load_report = MANAGER.load_qwen()
        pipe = MANAGER.get_qwen()

        n = max(1, min(int(num_images), 4))
        base_seed = _seed_from(seed)
        images = []
        t_start = time.time()

        for i in range(n):
            seed_i = base_seed + i
            progress(
                i / n,
                desc=f"Generating {i+1}/{n}  ({w}x{h}, 8-step Turbo schedule, seed={seed_i})",
            )

            # Default args pin the loop vars at closure-creation time — without
            # them, every callback would see the final iteration's values
            # (Python late-binding gotcha).
            def on_step_end(pipe, step_index, timestep, callback_kwargs,
                            _i=i, _n=n, _seed=seed_i):
                progress(
                    (_i + (step_index + 1) / 8) / _n,
                    desc=f"Generating {_i + 1}/{_n}  step {step_index + 1}/8  seed={_seed}",
                )
                return callback_kwargs

            with torch.inference_mode():
                result = pipe(
                    prompt=prompt.strip(),
                    width=w,
                    height=h,
                    true_cfg_scale=1.0,  # Turbo is guidance-distilled
                    use_kv_cache=True,
                    generator=torch.Generator("cpu").manual_seed(seed_i),
                    callback_on_step_end=on_step_end,
                )
            images.append(result.images[0])

        progress(1.0, desc="Done")
        elapsed = time.time() - t_start

        saved_paths = []
        for i, img in enumerate(images):
            p = _timestamped_path(f"qwen-{i+1}")
            img.save(p)
            saved_paths.append(str(p))

        status = (
            f"{_fmt_load_report(load_report, 'Qwen-Image-2.1-Turbo')}  |  "
            f"Generated {n} image{'s' if n != 1 else ''} at {w}x{h} in {elapsed:.1f}s "
            f"(~{elapsed/n:.1f}s each, 8-step schedule)  |  "
            f"Seeds: {[base_seed + i for i in range(n)]}"
        )
        gallery = [(img, f"img {i+1}") for i, img in enumerate(images)]
        return gallery, status

    except Exception as exc:  # friendly failure for a double-click consumer app
        return None, f"⚠️ Generation failed — {type(exc).__name__}: {exc}"
    finally:
        _cleanup_mps()


# ---------------------------------------------------------------- Edit
def qwen_edit(
    input_image,
    prompt: str,
    size_preset: str,
    custom_height: float,
    custom_width: float,
    seed: int,
    progress: gr.Progress = gr.Progress(),
):
    import torch

    if input_image is None:
        return None, "Please upload or drop an image to edit."
    if not prompt.strip():
        return None, "Editing instruction is empty."

    if size_preset == "Custom":
        h, w = _clamp_size(int(custom_height), int(custom_width))
    else:
        w, h = SIZE_PRESETS[size_preset]

    load_report = None
    try:
        progress(0.0, desc="Loading Qwen-Image-2.1-Turbo (first run downloads ~11.5 GB)...")
        load_report = MANAGER.load_qwen()
        pipe = MANAGER.get_qwen()

        image = input_image.convert("RGBA")  # Qwen-Image-2.1 takes RGBA references

        seed_i = _seed_from(seed)
        progress(0.0, desc=f"Editing  ({w}x{h}, 8-step Turbo schedule, seed={seed_i})")
        t_start = time.time()

        def on_step_end(pipe, step_index, timestep, callback_kwargs, _seed=seed_i):
            progress(
                (step_index + 1) / 8,
                desc=f"Editing  step {step_index + 1}/8  seed={_seed}",
            )
            return callback_kwargs

        with torch.inference_mode():
            result = pipe(
                prompt=prompt.strip(),
                image=image,
                width=w,
                height=h,
                true_cfg_scale=1.0,
                use_kv_cache=True,
                generator=torch.Generator("cpu").manual_seed(seed_i),
                callback_on_step_end=on_step_end,
            )

        progress(1.0, desc="Done")
        elapsed = time.time() - t_start
        out = result.images[0]
        p = _timestamped_path("edit")
        out.save(p)

        status = (
            f"{_fmt_load_report(load_report, 'Qwen-Image-2.1-Turbo')}  |  "
            f"Edited in {elapsed:.1f}s (8-step schedule, seed={seed_i})  |  "
            f"Saved: {p.name}"
        )
        return [(out, "edited")], status

    except Exception as exc:
        return None, f"⚠️ Edit failed — {type(exc).__name__}: {exc}"
    finally:
        _cleanup_mps()


# ---------------------------------------------------------------- UI
# Qwen-Image-2.1 is trained at 2K-class resolutions (e.g. 2048x2048 for 1:1).
# On a 16 GB Mac the denoise working set scales with token count, so the
# presets below stay in the 512-1024 band; native 2K works but is slow here.
SIZE_PRESETS: OrderedDict[str, tuple[int, int]] = OrderedDict(
    [
        ("512 x 512 (square, fastest)", (512, 512)),
        ("768 x 768 (square)", (768, 768)),
        ("640 x 896 (portrait 9:16)", (640, 896)),
        ("896 x 640 (landscape 16:9)", (896, 640)),
        ("832 x 624 (4:3 landscape)", (832, 624)),
        ("1024 x 1024 (square, native-class)", (1024, 1024)),
        ("Custom", (0, 0)),  # sentinel; Custom uses the sliders
    ]
)

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

SCHEDULE_NOTE = (
    "Turbo's 8-step sampling schedule is baked into the checkpoint — "
    "there are no steps/guidance knobs to tune."
)


def _size_controls(default_preset: str):
    """Shared size dropdown + custom-size accordion (returned as a tuple)."""
    size = gr.Dropdown(
        list(SIZE_PRESETS.keys()),
        value=default_preset,
        label="Size",
        info="512-1024 px presets fit 16 GB comfortably; the model's native 2K works but is slow on M-series.",
    )
    with gr.Accordion("Custom size", open=False) as custom_box:
        custom_h = gr.Slider(256, 1536, value=512, step=16, label="Height")
        custom_w = gr.Slider(256, 1536, value=512, step=16, label="Width")

    def _toggle(preset: str):
        return gr.update(open=(preset == "Custom"))

    size.change(_toggle, inputs=size, outputs=custom_box)
    return size, custom_h, custom_w


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Mac (M series) AI Image Studio") as demo:
        gr.HTML(
            "<div id='title'>"
            "<h1>🎨 Mac (M series) AI Image Studio</h1>"
            "<p>Based on Qwen Image 2.1 — Turbo generation + instruction editing on Apple Silicon MPS</p>"
            "</div>"
        )

        with gr.Tab("✨ Generate (Qwen-Image-2.1-Turbo)"):
            with gr.Row():
                with gr.Column(scale=3):
                    prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Describe the image. e.g. 'a red apple on a wooden table, studio lighting'",
                        lines=3,
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
                    size, custom_h, custom_w = _size_controls(list(SIZE_PRESETS.keys())[0])
                    num_images = gr.Slider(1, 4, value=1, step=1, label="Number of images")
                    with gr.Accordion("Advanced", open=False):
                        seed = gr.Number(label="Seed (-1 for random)", value=-1)
                        gr.Markdown(f"<small>{SCHEDULE_NOTE}</small>")

            generate_btn.click(
                fn=qwen_generate,
                inputs=[prompt, size, custom_h, custom_w, seed, num_images],
                outputs=[gallery, gen_status],
            )

        with gr.Tab("✂️ Edit (Qwen-Image-2.1-Turbo)"):
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
                    edit_size, edit_custom_h, edit_custom_w = _size_controls(
                        "1024 x 1024 (square, native-class)"
                    )
                    with gr.Accordion("Advanced", open=False):
                        edit_seed = gr.Number(label="Seed (-1 for random)", value=-1)
                        gr.Markdown(f"<small>{SCHEDULE_NOTE}</small>")
                    edit_btn = gr.Button("Edit", variant="primary")

            edit_btn.click(
                fn=qwen_edit,
                inputs=[input_image, edit_prompt, edit_size, edit_custom_h, edit_custom_w, edit_seed],
                outputs=[edit_gallery, edit_status],
            )

        gr.HTML(
            "<hr/><p style='text-align:center;color:#888;font-size:0.85em'>"
            "Model: <code>OzzyGT/Qwen_Image_2_1_Turbo_sdnq_dynamic_4bit</code> (SDNQ int4) · "
            "weights under the Qwen Research License (non-commercial)"
            "</p>"
        )

    return demo


if __name__ == "__main__":
    HERE = Path(__file__).resolve().parent
    FAVICON = HERE / "favicon.png"

    app = build_ui()
    app.queue(default_concurrency_limit=1)  # serialize: one model run at a time on 16 GB
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
