#!/usr/bin/env python3
from __future__ import annotations

import os
import threading

import gradio as gr
import torch
from diffusers import FluxPipeline, StableDiffusionPipeline, ZImagePipeline

BASE_MODEL = os.getenv("BASE_MODEL", "Tongyi-MAI/Z-Image-Turbo")
CPU_FALLBACK_MODEL = os.getenv("CPU_FALLBACK_MODEL", "stabilityai/sd-turbo")
LORA_REPO = os.getenv("LORA_REPO", "")
LORA_WEIGHT_NAME = os.getenv("LORA_WEIGHT_NAME", "pytorch_lora_weights.safetensors")

MODEL_CHOICES = [
    "Tongyi-MAI/Z-Image-Turbo",
    "black-forest-labs/FLUX.1-schnell",
    "stabilityai/sd-turbo",
    "stabilityai/stable-diffusion-2-1-base",
]

DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
GENERATOR_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STABLE_STEPS = 8
STABLE_GUIDANCE = 6.0
STABLE_RESOLUTION = 640
STABLE_FAST_MODE = False
STABLE_QUALITY = "stable"
DEFAULT_STEPS = STABLE_STEPS
DEFAULT_RESOLUTION = STABLE_RESOLUTION
DEFAULT_FAST_MODE = STABLE_FAST_MODE
DEFAULT_QUALITY = STABLE_QUALITY
DEFAULT_NEGATIVE = "blurry, low quality, watermark, text, logo, deformed"

pipe = None
pipeline_kind = ""
requested_model_id = ""
effective_model_id = ""
last_model_note = ""
MODEL_LOCK = threading.Lock()


def _apply_common_pipeline_tuning(current_pipe) -> None:
    if torch.cuda.is_available() and hasattr(current_pipe, "enable_model_cpu_offload"):
        current_pipe.enable_model_cpu_offload()
    if hasattr(current_pipe, "set_progress_bar_config"):
        current_pipe.set_progress_bar_config(disable=True)


def _try_load_lora(current_pipe, kind: str) -> str:
    if not LORA_REPO:
        return ""
    if kind not in {"z-image", "flux"}:
        return f"LoRA ignored for pipeline `{kind}`."
    try:
        current_pipe.load_lora_weights(LORA_REPO, weight_name=LORA_WEIGHT_NAME)
        return f"LoRA loaded from `{LORA_REPO}`."
    except Exception as exc:
        return f"Could not load LoRA `{LORA_REPO}`: {exc}"


def _build_pipeline(target_model: str):
    target_lower = target_model.lower()

    if not torch.cuda.is_available() and ("z-image" in target_lower or "flux" in target_lower):
        cpu_pipe = StableDiffusionPipeline.from_pretrained(
            CPU_FALLBACK_MODEL,
            torch_dtype=torch.float32,
            safety_checker=None,
        )
        note = (
            f"CPU runtime detected: requested `{target_model}`, "
            f"running fallback `{CPU_FALLBACK_MODEL}`."
        )
        return cpu_pipe, "cpu-fallback", CPU_FALLBACK_MODEL, note

    if "z-image" in target_lower:
        return ZImagePipeline.from_pretrained(target_model, torch_dtype=DTYPE), "z-image", target_model, ""

    if "flux" in target_lower:
        return FluxPipeline.from_pretrained(target_model, torch_dtype=DTYPE), "flux", target_model, ""

    sd_pipe = StableDiffusionPipeline.from_pretrained(
        target_model,
        torch_dtype=torch.float32 if not torch.cuda.is_available() else DTYPE,
        safety_checker=None,
    )
    return sd_pipe, "sd", target_model, ""


def _status_text() -> str:
    base = (
        f"Requested model: `{requested_model_id}` | "
        f"Active model: `{effective_model_id}` | "
        f"Pipeline: `{pipeline_kind}`"
    )
    if last_model_note:
        return f"{base}\n\n{last_model_note}"
    return base


def switch_model(target_model: str) -> str:
    global pipe, pipeline_kind, requested_model_id, effective_model_id, last_model_note

    with MODEL_LOCK:
        previous_pipe = pipe
        previous_kind = pipeline_kind
        previous_requested = requested_model_id
        previous_effective = effective_model_id
        previous_note = last_model_note

        try:
            next_pipe, next_kind, next_effective, load_note = _build_pipeline(target_model)
            _apply_common_pipeline_tuning(next_pipe)
            lora_note = _try_load_lora(next_pipe, next_kind)

            pipe = next_pipe
            pipeline_kind = next_kind
            requested_model_id = target_model
            effective_model_id = next_effective

            merged_note = "\n".join([n for n in [load_note, lora_note] if n])
            last_model_note = merged_note

            if previous_pipe is not None:
                del previous_pipe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return _status_text()
        except Exception as exc:
            pipe = previous_pipe
            pipeline_kind = previous_kind
            requested_model_id = previous_requested
            effective_model_id = previous_effective
            last_model_note = previous_note
            return _status_text() + f"\n\nModel switch failed: {exc}"


def build_prompt(subject: str, details: str) -> str:
    subject_map = {
        "horreo": "galician horreo",
        "cruceiro": "galician cruceiro",
        "mixed": "galician ethnographic scene with horreo and cruceiro",
    }
    base = subject_map.get(subject, subject)
    return (
        f"ethnographic photography, {base}, Galicia, stone and wood textures, "
        f"natural light, realistic details, {details}"
    )


def normalize_resolution(resolution: int) -> int:
    clamped = max(512, min(1024, int(resolution)))
    return (clamped // 64) * 64


def apply_quality_profile(steps: int, resolution: int, quality_profile: str, fast_mode: bool) -> tuple[int, int]:
    current_steps = steps
    current_res = resolution

    if quality_profile == "stable":
        current_steps = max(current_steps, STABLE_STEPS)
        current_res = max(current_res, STABLE_RESOLUTION)
    elif quality_profile == "fast":
        current_steps = min(current_steps, 3)
        current_res = min(current_res, 640)
    elif quality_profile == "balanced":
        current_steps = max(current_steps, 6)
        current_res = max(current_res, 768)
    elif quality_profile == "quality":
        current_steps = max(current_steps, 10)
        current_res = max(current_res, 896)

    if fast_mode:
        current_steps = min(current_steps, 3)
        current_res = min(current_res, 640)

    return current_steps, current_res


def apply_stable_preset():
    return (
        STABLE_STEPS,
        STABLE_GUIDANCE,
        STABLE_RESOLUTION,
        STABLE_QUALITY,
        STABLE_FAST_MODE,
        DEFAULT_NEGATIVE,
    )


def generate(
    subject: str,
    details: str,
    negative_details: str,
    seed: int,
    steps: int,
    guidance: float,
    resolution: int,
    fast_mode: bool,
    quality_profile: str,
):
    prompt = build_prompt(subject, details)
    use_resolution = normalize_resolution(resolution)
    use_steps, use_resolution = apply_quality_profile(steps, use_resolution, quality_profile, fast_mode)
    use_seq_len = 128 if fast_mode else 256

    with MODEL_LOCK:
        active_pipe = pipe
        active_kind = pipeline_kind
        status_snapshot = _status_text()

    generator = torch.Generator(device=GENERATOR_DEVICE).manual_seed(seed)

    with torch.inference_mode():
        kwargs = dict(
            prompt=prompt,
            num_inference_steps=use_steps,
            guidance_scale=guidance,
            height=use_resolution,
            width=use_resolution,
            generator=generator,
        )

        negative_text = negative_details.strip()
        if negative_text and active_kind in {"cpu-fallback", "sd"}:
            kwargs["negative_prompt"] = negative_text

        if active_kind in {"z-image", "flux"}:
            kwargs["max_sequence_length"] = use_seq_len

        image = active_pipe(**kwargs).images[0]

    return image, prompt, status_snapshot


# Initialize pipeline once at startup.
switch_model(BASE_MODEL)

with gr.Blocks(title="Galicia Horreos and Cruceiros") as demo:
    gr.Markdown(
        "# Galicia Ethnography Generator\n"
        "Generate images of **horreos** and **cruceiros** and switch models from the UI."
    )

    with gr.Row():
        model_selector = gr.Dropdown(
            choices=MODEL_CHOICES,
            value=BASE_MODEL if BASE_MODEL in MODEL_CHOICES else MODEL_CHOICES[0],
            label="Model",
        )
        apply_model_btn = gr.Button("Apply Model")

    model_status = gr.Markdown(_status_text())
    gr.Markdown(
        "Preset fijo activo: `Calidad estable` (steps 8, guidance 6.0, resolution 640, fast mode OFF)."
    )
    gr.Markdown("Tip: on `cpu-basic`, large models are auto-fallback to lightweight CPU model.")

    with gr.Row():
        subject = gr.Dropdown(
            choices=["horreo", "cruceiro", "mixed"],
            value="horreo",
            label="Subject",
        )
        details = gr.Textbox(
            value="rural landscape, cloudy sky, documentary style",
            label="Prompt details",
        )

    negative_details = gr.Textbox(
        value=DEFAULT_NEGATIVE,
        label="Negative prompt (for SD pipelines)",
    )

    with gr.Row():
        seed = gr.Slider(minimum=0, maximum=2_000_000_000, value=42, step=1, label="Seed")
        steps = gr.Slider(minimum=1, maximum=60, value=DEFAULT_STEPS, step=1, label="Steps")
        guidance = gr.Slider(minimum=0.0, maximum=12.0, value=STABLE_GUIDANCE, step=0.1, label="Guidance")
        resolution = gr.Slider(minimum=512, maximum=1024, value=DEFAULT_RESOLUTION, step=64, label="Resolution")

    with gr.Row():
        quality_profile = gr.Dropdown(
            choices=["stable", "fast", "balanced", "quality"],
            value=DEFAULT_QUALITY,
            label="Quality profile",
        )
        fast_mode = gr.Checkbox(value=DEFAULT_FAST_MODE, label="Fast mode")

    with gr.Row():
        stable_preset_btn = gr.Button("Apply Stable Quality")
        run_btn = gr.Button("Generate")
    output_image = gr.Image(label="Result", type="pil")
    output_prompt = gr.Textbox(label="Final prompt")

    apply_model_btn.click(
        fn=switch_model,
        inputs=[model_selector],
        outputs=[model_status],
    )

    stable_preset_btn.click(
        fn=apply_stable_preset,
        inputs=[],
        outputs=[steps, guidance, resolution, quality_profile, fast_mode, negative_details],
    )

    run_btn.click(
        fn=generate,
        inputs=[subject, details, negative_details, seed, steps, guidance, resolution, fast_mode, quality_profile],
        outputs=[output_image, output_prompt, model_status],
    )


demo.launch()
