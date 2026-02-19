---
title: Galicia Horreos Cruceiros Generator
sdk: gradio
app_file: app.py
pinned: false
---

Space focused on Galicia ethnographic elements:
- horreos
- cruceiros
- muinos

Optional Space variables:
- `BASE_MODEL`: defaults to `Tongyi-MAI/Z-Image-Turbo`
- `CPU_FALLBACK_MODEL`: defaults to `stabilityai/sd-turbo`
- `LORA_REPO`: your LoRA model repo (example: `username/flux-schnell-galicia-lora`)
- `LORA_REPO_HORREO`: optional LoRA repo for `horreo`
- `LORA_REPO_CRUCEIRO`: optional LoRA repo for `cruceiro`
- `LORA_REPO_MUINO`: optional LoRA repo for `muino`
- `LORA_WEIGHT_NAME`: LoRA file name (default: `pytorch_lora_weights.safetensors`)
- `TOKEN_HORREO`: defaults to `<gal_horreo>`
- `TOKEN_CRUCEIRO`: defaults to `<gal_cruceiro>`
- `TOKEN_MUINO`: defaults to `<gal_muino>`

The app auto-detects `BASE_MODEL` family:
- `z-image` -> `ZImagePipeline`
- `flux` -> `FluxPipeline`
- other models -> `StableDiffusionPipeline`

If `LORA_REPO` is not set, the app runs with base model only.
If `LORA_REPO_HORREO` / `LORA_REPO_CRUCEIRO` are set, the app auto-switches LoRA by selected subject.

UI features:
- model selector + `Apply Model` button
- stable preset button: `Apply Stable Quality`
- quality profiles: `stable`, `fast`, `balanced`, `quality`
- negative prompt textbox (used for SD pipelines)

Performance note:
- current runtime is `cpu-basic`, so first load and high-resolution generations are slow.
- use `Fast mode`, low steps, and `512` or `640` resolution for faster responses.
- if the selected base model exceeds CPU memory limits, the app auto-falls back to `CPU_FALLBACK_MODEL`.
