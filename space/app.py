#!/usr/bin/env python3
from __future__ import annotations

import os

import gradio as gr
import torch
from diffusers import FluxPipeline, StableDiffusionPipeline, ZImagePipeline

BASE_MODEL = os.getenv("BASE_MODEL", "Tongyi-MAI/Z-Image-Turbo")
CPU_FALLBACK_MODEL = os.getenv("CPU_FALLBACK_MODEL", "stabilityai/sd-turbo")
LORA_REPO = os.getenv("LORA_REPO", "")
LORA_WEIGHT_NAME = os.getenv("LORA_WEIGHT_NAME", "pytorch_lora_weights.safetensors")

DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
GENERATOR_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_STEPS = 6 if torch.cuda.is_available() else 3
DEFAULT_RESOLUTION = 1024 if torch.cuda.is_available() else 640
DEFAULT_FAST_MODE = not torch.cuda.is_available()
if not torch.cuda.is_available():
    DEFAULT_STEPS = 2
    DEFAULT_RESOLUTION = 512

def load_pipeline(base_model: str):
    lower = base_model.lower()
    if not torch.cuda.is_available() and ("z-image" in lower or "flux" in lower):
        print(
            f"CPU runtime detected. Falling back from '{base_model}' to '{CPU_FALLBACK_MODEL}' "
            "to stay within memory limits."
        )
        return (
            StableDiffusionPipeline.from_pretrained(CPU_FALLBACK_MODEL, torch_dtype=torch.float32, safety_checker=None),
            "cpu-fallback",
        )
    if "z-image" in lower:
        return ZImagePipeline.from_pretrained(base_model, torch_dtype=DTYPE), "z-image"
    if "flux" in lower:
        return FluxPipeline.from_pretrained(base_model, torch_dtype=DTYPE), "flux"
    return FluxPipeline.from_pretrained(base_model, torch_dtype=DTYPE), "flux"


pipe, pipeline_kind = load_pipeline(BASE_MODEL)

if LORA_REPO and pipeline_kind in {"z-image", "flux"}:
    try:
        pipe.load_lora_weights(LORA_REPO, weight_name=LORA_WEIGHT_NAME)
        print(f"Loaded LoRA from {LORA_REPO}")
    except Exception as exc:
        print(f"Could not load LoRA ({LORA_REPO}): {exc}")
elif LORA_REPO:
    print(f"LoRA repo '{LORA_REPO}' ignored for pipeline kind '{pipeline_kind}'")

if torch.cuda.is_available():
    pipe.enable_model_cpu_offload()

if hasattr(pipe, "set_progress_bar_config"):
    pipe.set_progress_bar_config(disable=True)


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
    clamped = max(512, min(1024, resolution))
    return (clamped // 64) * 64


def generate(
    subject: str,
    details: str,
    seed: int,
    steps: int,
    guidance: float,
    resolution: int,
    fast_mode: bool,
):
    prompt = build_prompt(subject, details)
    generator = torch.Generator(device=GENERATOR_DEVICE).manual_seed(seed)
    use_steps = min(steps, 3) if fast_mode else steps
    use_resolution = min(normalize_resolution(resolution), 640) if fast_mode else normalize_resolution(resolution)
    use_seq_len = 128 if fast_mode else 256

    with torch.inference_mode():
        kwargs = dict(
            prompt=prompt,
            num_inference_steps=use_steps,
            guidance_scale=guidance,
            height=use_resolution,
            width=use_resolution,
            generator=generator,
        )
        if pipeline_kind in {"z-image", "flux"}:
            kwargs["max_sequence_length"] = use_seq_len
        image = pipe(**kwargs).images[0]

    return image, prompt


with gr.Blocks(title="Galicia Horreos and Cruceiros") as demo:
    gr.Markdown(
        "# Galicia Ethnography Generator\n"
        "Generate images of **horreos** and **cruceiros** using Z-Image-Turbo or FLUX + your LoRA."
    )
    gr.Markdown(f"Loaded model: `{BASE_MODEL}` (`{pipeline_kind}` pipeline)")
    if pipeline_kind == "cpu-fallback":
        gr.Markdown(
            f"CPU fallback active due memory limits. Running lightweight model: `{CPU_FALLBACK_MODEL}`"
        )
    gr.Markdown("Tip: on `cpu-basic`, keep `Fast mode` enabled for lower latency.")

    with gr.Row():
        subject = gr.Dropdown(
            choices=["horreo", "cruceiro", "mixed"],
            value="horreo",
            label="Subject",
        )
        details = gr.Textbox(
            value="rural landscape, cloudy sky, documentary style",
            label="Extra details",
        )

    with gr.Row():
        seed = gr.Slider(minimum=0, maximum=2_000_000_000, value=42, step=1, label="Seed")
        steps = gr.Slider(minimum=1, maximum=50, value=DEFAULT_STEPS, step=1, label="Steps")
        guidance = gr.Slider(minimum=0.0, maximum=8.0, value=0.0, step=0.1, label="Guidance")
        resolution = gr.Slider(minimum=512, maximum=1024, value=DEFAULT_RESOLUTION, step=64, label="Resolution")
        fast_mode = gr.Checkbox(value=DEFAULT_FAST_MODE, label="Fast mode")

    run_btn = gr.Button("Generate")
    output_image = gr.Image(label="Result", type="pil")
    output_prompt = gr.Textbox(label="Final prompt")

    run_btn.click(
        fn=generate,
        inputs=[subject, details, seed, steps, guidance, resolution, fast_mode],
        outputs=[output_image, output_prompt],
    )


demo.launch()
