#!/usr/bin/env python3
from __future__ import annotations

import os
import threading

import gradio as gr
import torch
from diffusers import FluxPipeline, StableDiffusionPipeline, ZImagePipeline
from huggingface_hub import InferenceClient, get_token

BASE_MODEL = os.getenv("BASE_MODEL", "Tongyi-MAI/Z-Image-Turbo")
CPU_FALLBACK_MODEL = os.getenv("CPU_FALLBACK_MODEL", "stabilityai/sd-turbo")
USE_INFERENCE_API = os.getenv("USE_INFERENCE_API", "1").strip().lower() not in {"0", "false", "no", "off"}
INFERENCE_TIMEOUT_SEC = float(os.getenv("INFERENCE_TIMEOUT_SEC", "120"))
LORA_REPO = os.getenv("LORA_REPO", "")
LORA_REPO_HORREO = os.getenv("LORA_REPO_HORREO", "")
LORA_REPO_CRUCEIRO = os.getenv("LORA_REPO_CRUCEIRO", "")
LORA_REPO_MUINO = os.getenv("LORA_REPO_MUINO", "")
LORA_WEIGHT_NAME = os.getenv("LORA_WEIGHT_NAME", "pytorch_lora_weights.safetensors")
LORA_TARGET = os.getenv("LORA_TARGET", "flux").strip().lower()  # flux | z-image | both/auto
TOKEN_HORREO = os.getenv("TOKEN_HORREO", "<gal_horreo>")
TOKEN_CRUCEIRO = os.getenv("TOKEN_CRUCEIRO", "<gal_cruceiro>")
TOKEN_MUINO = os.getenv("TOKEN_MUINO", "<gal_muino>")

MODEL_CHOICES = [
    "Tongyi-MAI/Z-Image-Turbo",
    "black-forest-labs/FLUX.1-schnell",
    "stabilityai/sd-turbo",
    "stabilityai/stable-diffusion-2-1-base",
]

def _pick_torch_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    # T4 does not support bf16; prefer fp16 unless bf16 is supported.
    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


DTYPE = _pick_torch_dtype()
GENERATOR_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_QUALITY = "stable"
DEFAULT_NEGATIVE = (
    "blurry, low quality, watermark, text, logo, deformed, "
    "modern house, cottage, cabin, villa, apartment building"
)

pipe = None
pipeline_kind = ""
requested_model_id = ""
effective_model_id = ""
last_model_note = ""
lora_active = False
active_lora_repo = ""
MODEL_LOCK = threading.Lock()


def _apply_common_pipeline_tuning(current_pipe) -> None:
    if torch.cuda.is_available() and hasattr(current_pipe, "enable_model_cpu_offload"):
        current_pipe.enable_model_cpu_offload()
    if hasattr(current_pipe, "set_progress_bar_config"):
        current_pipe.set_progress_bar_config(disable=True)


def _pick_lora_repo_for_subject(subject: str) -> str:
    per_subject = {
        "horreo": LORA_REPO_HORREO,
        "cruceiro": LORA_REPO_CRUCEIRO,
        "muino": LORA_REPO_MUINO,
    }
    return per_subject.get(subject, "") or LORA_REPO


def _try_load_lora(current_pipe, kind: str, lora_repo: str) -> tuple[str, bool]:
    if not lora_repo:
        return "", False
    if kind.startswith("remote-"):
        return "LoRA disabled: remote Inference API mode does not support applying LoRAs.", False
    if kind not in {"z-image", "flux"}:
        return f"LoRA ignored for pipeline `{kind}`.", False
    if LORA_TARGET in {"flux", "flux-only"} and kind != "flux":
        return "LoRA configured for `flux`; ignored for `z-image`.", False
    if LORA_TARGET in {"z-image", "zimage", "z-image-only"} and kind != "z-image":
        return "LoRA configured for `z-image`; ignored for `flux`.", False
    try:
        current_pipe.load_lora_weights(lora_repo, weight_name=LORA_WEIGHT_NAME)
        return f"LoRA loaded from `{lora_repo}`.", True
    except Exception as exc:
        return f"Could not load LoRA `{lora_repo}`: {exc}", False


def _build_pipeline(target_model: str):
    target_lower = target_model.lower()

    if not torch.cuda.is_available() and ("z-image" in target_lower or "flux" in target_lower):
        if USE_INFERENCE_API:
            token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or get_token()
            client = InferenceClient(model=target_model, token=token, timeout=INFERENCE_TIMEOUT_SEC)
            note = (
                f"CPU runtime detected: using Hugging Face Inference API for `{target_model}`.\n"
                "If generation fails, add `HF_TOKEN` as a Space secret and ensure you accepted the model license."
            )
            return client, f"remote-{target_model.split('/')[-1].lower()}", target_model, note

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
    if lora_active and active_lora_repo:
        base += f" | LoRA: `{active_lora_repo}`"
    if last_model_note:
        return f"{base}\n\n{last_model_note}"
    if pipeline_kind.startswith("remote-"):
        return (
            f"{base}\n\n"
            "Running in **remote Inference API** mode (no GPU required). "
            "If it errors, add `HF_TOKEN` as a Space secret."
        )
    if pipeline_kind == "cpu-fallback":
        return (
            f"{base}\n\n"
            "Tip: this Space is running on CPU. To use FLUX / Z-Image, switch the Space Hardware to a GPU "
            "(T4 or A10) in the Space Settings."
        )
    return base


def switch_model(target_model: str) -> str:
    global pipe, pipeline_kind, requested_model_id, effective_model_id, last_model_note, lora_active, active_lora_repo

    with MODEL_LOCK:
        previous_pipe = pipe
        previous_kind = pipeline_kind
        previous_requested = requested_model_id
        previous_effective = effective_model_id
        previous_note = last_model_note
        previous_lora_active = lora_active
        previous_lora_repo = active_lora_repo

        try:
            next_pipe, next_kind, next_effective, load_note = _build_pipeline(target_model)
            _apply_common_pipeline_tuning(next_pipe)
            startup_lora_repo = _pick_lora_repo_for_subject("horreo")
            lora_note, lora_loaded = _try_load_lora(next_pipe, next_kind, startup_lora_repo)

            pipe = next_pipe
            pipeline_kind = next_kind
            requested_model_id = target_model
            effective_model_id = next_effective
            lora_active = lora_loaded
            active_lora_repo = startup_lora_repo if lora_loaded else ""

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
            lora_active = previous_lora_active
            active_lora_repo = previous_lora_repo
            return _status_text() + f"\n\nModel switch failed: {exc}"


def _ensure_subject_lora(subject: str) -> None:
    global last_model_note, lora_active, active_lora_repo

    target_repo = _pick_lora_repo_for_subject(subject)
    if not target_repo:
        lora_active = False
        active_lora_repo = ""
        return

    if pipeline_kind.startswith("remote-") or pipeline_kind not in {"z-image", "flux"}:
        lora_active = False
        active_lora_repo = ""
        return

    if lora_active and active_lora_repo == target_repo:
        return

    try:
        if hasattr(pipe, "unload_lora_weights"):
            pipe.unload_lora_weights()
        lora_note, lora_loaded = _try_load_lora(pipe, pipeline_kind, target_repo)
        lora_active = lora_loaded
        active_lora_repo = target_repo if lora_loaded else ""
        if lora_note:
            last_model_note = lora_note
    except Exception as exc:
        lora_active = False
        active_lora_repo = ""
        last_model_note = f"Could not switch LoRA `{target_repo}`: {exc}"


def build_prompt(subject: str, details: str) -> str:
    token_horreo = TOKEN_HORREO if lora_active else ""
    token_cruceiro = TOKEN_CRUCEIRO if lora_active else ""
    token_muino = TOKEN_MUINO if lora_active else ""
    subject_map = {
        "horreo": (
            f"{token_horreo} traditional Galician horreo (raised granary), "
            "long narrow granary with slatted chamber, on stone pillars (pegollos) "
            "with capstones, exterior view, full structure visible, rural Galicia, no modern house, no interior"
        ),
        "cruceiro": (
            f"{token_cruceiro} Galician cruceiro, carved granite cross on stone pedestal, "
            "historic village context, rural Galicia"
        ),
        "muino": (
            f"{token_muino} traditional Galician muino (water mill), "
            "stone millhouse near a stream, moss and granite textures, rural Galicia"
        ),
        "mixed": (
            "traditional Galician horreo and a Galician cruceiro in the same ethnographic scene, rural Galicia"
        ),
    }
    base = subject_map.get(subject, subject)
    return (
        f"ethnographic documentary photography, {base}, "
        f"stone and wood textures, natural light, realistic details, {details}"
    )


def normalize_resolution(resolution: int) -> int:
    clamped = max(512, min(1024, int(resolution)))
    return (clamped // 64) * 64


def apply_quality_profile(steps: int, resolution: int, quality_profile: str, fast_mode: bool) -> tuple[int, int]:
    current_steps = steps
    current_res = resolution

    # stable: keep user/model preset as-is.
    if quality_profile == "fast":
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


def recommended_preset():
    # Pipeline-aware presets so you can just press Generate.
    with MODEL_LOCK:
        kind = pipeline_kind
        model_id = effective_model_id

    if kind == "flux" or kind.startswith("remote-"):
        return (4, 3.5, 1024, "stable", False, DEFAULT_NEGATIVE)
    if kind == "z-image":
        return (4, 3.5, 1024, "stable", False, DEFAULT_NEGATIVE)
    if model_id == "stabilityai/sd-turbo" or kind == "cpu-fallback":
        return (1, 0.0, 512, "stable", False, DEFAULT_NEGATIVE)
    # Generic SD pipeline (higher-quality but slower).
    return (25, 7.0, 768, "stable", False, DEFAULT_NEGATIVE)


def apply_stable_preset():
    return (
        *recommended_preset(),
    )


def switch_model_and_apply_preset(target_model: str):
    status = switch_model(target_model)
    steps, guidance, resolution, quality_profile, fast_mode, negative = recommended_preset()
    return status, steps, guidance, resolution, quality_profile, fast_mode, negative


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
    use_resolution = normalize_resolution(resolution)
    use_steps, use_resolution = apply_quality_profile(steps, use_resolution, quality_profile, fast_mode)
    use_seq_len = 128 if fast_mode else 256

    with MODEL_LOCK:
        _ensure_subject_lora(subject)
        prompt = build_prompt(subject, details)
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

        if active_kind.startswith("remote-"):
            try:
                image = active_pipe.text_to_image(
                    prompt,
                    height=use_resolution,
                    width=use_resolution,
                    num_inference_steps=use_steps,
                    guidance_scale=guidance,
                    seed=seed,
                )
            except Exception as exc:
                raise gr.Error(
                    "Remote Inference API call failed.\n\n"
                    "Common fixes:\n"
                    "- Add `HF_TOKEN` as a Space secret\n"
                    "- Accept the model license on its model page\n"
                    "- If using Z-Image-Turbo: it may require paid inference credits\n\n"
                    f"Error: {exc}"
                ) from exc
        else:
            image = active_pipe(**kwargs).images[0]

    return image, prompt, status_snapshot


# Initialize pipeline once at startup.
switch_model(BASE_MODEL)
INIT_STEPS, INIT_GUIDANCE, INIT_RESOLUTION, INIT_QUALITY, INIT_FAST_MODE, INIT_NEGATIVE = recommended_preset()

with gr.Blocks(title="Galicia Horreos and Cruceiros") as demo:
    gr.Markdown(
        "# Galicia Ethnography Generator\n"
        "Generate images of **horreos**, **cruceiros** and **muinos** and switch models from the UI."
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
        "Preset recomendado activo. Si cambias el modelo, pulsa `Apply Model` para reajustar los valores."
    )
    gr.Markdown("Tip: on `cpu-basic`, large models are auto-fallback to lightweight CPU model.")

    with gr.Row():
        subject = gr.Dropdown(
            choices=["horreo", "cruceiro", "muino", "mixed"],
            value="horreo",
            label="Subject",
        )
        details = gr.Textbox(
            value="rural landscape, cloudy sky, documentary style",
            label="Prompt details",
        )

    negative_details = gr.Textbox(
        value=INIT_NEGATIVE,
        label="Negative prompt (for SD pipelines)",
    )

    with gr.Row():
        seed = gr.Slider(minimum=0, maximum=2_000_000_000, value=42, step=1, label="Seed")
        steps = gr.Slider(minimum=1, maximum=60, value=INIT_STEPS, step=1, label="Steps")
        guidance = gr.Slider(minimum=0.0, maximum=12.0, value=INIT_GUIDANCE, step=0.1, label="Guidance")
        resolution = gr.Slider(minimum=512, maximum=1024, value=INIT_RESOLUTION, step=64, label="Resolution")

    with gr.Row():
        quality_profile = gr.Dropdown(
            choices=["stable", "fast", "balanced", "quality"],
            value=INIT_QUALITY,
            label="Quality profile",
        )
        fast_mode = gr.Checkbox(value=INIT_FAST_MODE, label="Fast mode")

    with gr.Row():
        stable_preset_btn = gr.Button("Apply Stable Quality")
        run_btn = gr.Button("Generate")
    output_image = gr.Image(label="Result", type="pil")
    output_prompt = gr.Textbox(label="Final prompt")

    apply_model_btn.click(
        fn=switch_model_and_apply_preset,
        inputs=[model_selector],
        outputs=[model_status, steps, guidance, resolution, quality_profile, fast_mode, negative_details],
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
