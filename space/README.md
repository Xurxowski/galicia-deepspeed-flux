---
title: Galicia Horreos Cruceiros Generator
sdk: gradio
app_file: app.py
pinned: false
---

Space focused on Galicia ethnographic elements:
- horreos
- cruceiros

Optional Space variables:
- `BASE_MODEL`: defaults to `Tongyi-MAI/Z-Image-Turbo`
- `CPU_FALLBACK_MODEL`: defaults to `stabilityai/sd-turbo`
- `LORA_REPO`: your LoRA model repo (example: `username/flux-schnell-galicia-lora`)
- `LORA_WEIGHT_NAME`: LoRA file name (default: `pytorch_lora_weights.safetensors`)

The app auto-detects `BASE_MODEL` family:
- `z-image` -> `ZImagePipeline`
- `flux` -> `FluxPipeline`
- other models -> `StableDiffusionPipeline`

If `LORA_REPO` is not set, the app runs with base model only.

UI features:
- model selector + `Apply Model` button
- stable preset button: `Apply Stable Quality`
- quality profiles: `stable`, `fast`, `balanced`, `quality`
- negative prompt textbox (used for SD pipelines)

Performance note:
- current runtime is `cpu-basic`, so first load and high-resolution generations are slow.
- use `Fast mode`, low steps, and `512` or `640` resolution for faster responses.
- if the selected base model exceeds CPU memory limits, the app auto-falls back to `CPU_FALLBACK_MODEL`.
