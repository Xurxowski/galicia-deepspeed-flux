#!/usr/bin/env python3
from __future__ import annotations

import os

import gradio as gr
import torch
from diffusers import FluxPipeline, ZImagePipeline

BASE_MODEL = os.getenv("BASE_MODEL", "Tongyi-MAI/Z-Image-Turbo")
LORA_REPO = os.getenv("LORA_REPO", "")
LORA_WEIGHT_NAME = os.getenv("LORA_WEIGHT_NAME", "pytorch_lora_weights.safetensors")

DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
GENERATOR_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_pipeline(base_model: str):
    lower = base_model.lower()
    if "z-image" in lower:
        return ZImagePipeline.from_pretrained(base_model, torch_dtype=DTYPE), "z-image"
    return FluxPipeline.from_pretrained(base_model, torch_dtype=DTYPE), "flux"


pipe, pipeline_kind = load_pipeline(BASE_MODEL)

if LORA_REPO:
    try:
        pipe.load_lora_weights(LORA_REPO, weight_name=LORA_WEIGHT_NAME)
        print(f"Loaded LoRA from {LORA_REPO}")
    except Exception as exc:
        print(f"Could not load LoRA ({LORA_REPO}): {exc}")

if torch.cuda.is_available():
    pipe.enable_model_cpu_offload()


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


def generate(subject: str, details: str, seed: int, steps: int, guidance: float):
    prompt = build_prompt(subject, details)
    generator = torch.Generator(device=GENERATOR_DEVICE).manual_seed(seed)

    image = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        height=1024,
        width=1024,
        max_sequence_length=256,
        generator=generator,
    ).images[0]

    return image, prompt


with gr.Blocks(title="Galicia Horreos and Cruceiros") as demo:
    gr.Markdown(
        "# Galicia Ethnography Generator\n"
        "Generate images of **horreos** and **cruceiros** using Z-Image-Turbo or FLUX + your LoRA."
    )
    gr.Markdown(f"Loaded model: `{BASE_MODEL}` (`{pipeline_kind}` pipeline)")

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
        steps = gr.Slider(minimum=1, maximum=50, value=9, step=1, label="Steps")
        guidance = gr.Slider(minimum=0.0, maximum=8.0, value=0.0, step=0.1, label="Guidance")

    run_btn = gr.Button("Generate")
    output_image = gr.Image(label="Result", type="pil")
    output_prompt = gr.Textbox(label="Final prompt")

    run_btn.click(
        fn=generate,
        inputs=[subject, details, seed, steps, guidance],
        outputs=[output_image, output_prompt],
    )


demo.launch()
