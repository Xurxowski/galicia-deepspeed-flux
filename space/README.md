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
- otherwise -> `FluxPipeline`

If `LORA_REPO` is not set, the app runs with base model only.

Performance note:
- current runtime is `cpu-basic`, so first load and high-resolution generations are slow.
- use `Fast mode`, low steps, and `640` or `768` resolution for faster responses.
- if the selected base model exceeds CPU memory limits, the app auto-falls back to `CPU_FALLBACK_MODEL`.
