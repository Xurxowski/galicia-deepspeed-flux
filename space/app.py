#!/usr/bin/env python3
from __future__ import annotations

import os
import threading

import gradio as gr
import torch
from diffusers import (
    FluxImg2ImgPipeline,
    FluxPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
)
from huggingface_hub import InferenceClient, get_token

BASE_MODEL = os.getenv("BASE_MODEL", "black-forest-labs/FLUX.1-schnell")
CPU_FALLBACK_MODEL = os.getenv("CPU_FALLBACK_MODEL", "LanguageMachines/stable-diffusion-2-1-base")
USE_INFERENCE_API = os.getenv("USE_INFERENCE_API", "1").strip().lower() not in {"0", "false", "no", "off"}
INFERENCE_TIMEOUT_SEC = float(os.getenv("INFERENCE_TIMEOUT_SEC", "120"))
LORA_REPO = os.getenv("LORA_REPO", "")
LORA_REPO_HORREO = os.getenv("LORA_REPO_HORREO", "")
LORA_REPO_CRUCEIRO = os.getenv("LORA_REPO_CRUCEIRO", "")
LORA_REPO_MUINO = os.getenv("LORA_REPO_MUINO", "")
LORA_WEIGHT_NAME = os.getenv("LORA_WEIGHT_NAME", "pytorch_lora_weights.safetensors")
LORA_WEIGHT_NAME_HORREO = os.getenv("LORA_WEIGHT_NAME_HORREO", "")
LORA_WEIGHT_NAME_CRUCEIRO = os.getenv("LORA_WEIGHT_NAME_CRUCEIRO", "")
LORA_WEIGHT_NAME_MUINO = os.getenv("LORA_WEIGHT_NAME_MUINO", "")
LORA_TARGET = os.getenv("LORA_TARGET", "flux").strip().lower()  # flux|sd|both|auto
TOKEN_HORREO = os.getenv("TOKEN_HORREO", "<gal_horreo>")
TOKEN_CRUCEIRO = os.getenv("TOKEN_CRUCEIRO", "<gal_cruceiro>")
TOKEN_MUINO = os.getenv("TOKEN_MUINO", "<gal_muino>")
APP_VERSION = os.getenv("APP_VERSION", "2026-02-28-1")

MODEL_CHOICES = [
    "black-forest-labs/FLUX.1-schnell",
    "LanguageMachines/stable-diffusion-2-1-base",
    "stabilityai/sdxl-turbo",
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
    "blurry, low quality, watermark, text, logo, deformed geometry, modern elements, "
    "modern cross, cemetery, gravestone, wooden cross, metal cross, church interior, "
    "asturian horreo, square stone pillars, round pegollos, wooden pillars, stairs, ground-level granary, "
    "thatched roof, modern barn, metal roof, concrete base, generic christian cross, calvary, "
    "close-up crop, interior view, apartment building, cabin, chalet"
)
SUBJECT_DETAIL_PRESETS = {
    "horreo": (
        "traditional Galician granary called horreo, elevated structure on cylindrical stone pegollos, "
        "granite feet (pies) with ant guards (tornaformigas), tornarratos on pillars, "
        "rectangular wooden chest with vertical duelas/tablillas and ventilation gaps, horizontal fajas, "
        "interior lintel (dintel interior), wooden door with penal lock, slate roof with sobrepens and cornisa, "
        "pinche and cross adornos on ridge, no stairs, not Asturian, full exterior visible, rural Galicia"
    ),
    "cruceiro": (
        "traditional Galician cruceiro in granite, square stepped base with three tiers, "
        "octagonal shaft with carved geometric motifs, decorated capital with vegetal/volute motifs, "
        "latin cross with flared arms, full monument visible, weathered granite with lichen and moss, "
        "rural Galicia (crossroads or churchyard), overcast atlantic daylight, "
        "plataforma_escalonada, pousadoiro, fuste_octogonal"
    ),
    "muino": (
        "stone millhouse beside flowing water, weathered granite and moss, "
        "traditional rural Galicia, documentary realism"
    ),
    "mixed": (
        "both elements fully visible, clear separation between structures, "
        "rural Galicia, natural daylight, documentary realism"
    ),
}

pipe = None
img2img_pipe = None
pipeline_kind = ""
requested_model_id = ""
effective_model_id = ""
last_model_note = ""
lora_active = False
active_lora_repo = ""
active_lora_weight_name = ""
MODEL_LOCK = threading.Lock()


def _apply_common_pipeline_tuning(current_pipe) -> None:
    if torch.cuda.is_available() and hasattr(current_pipe, "enable_model_cpu_offload"):
        current_pipe.enable_model_cpu_offload()
    if hasattr(current_pipe, "set_progress_bar_config"):
        current_pipe.set_progress_bar_config(disable=True)


def _pick_lora_spec_for_subject(subject: str) -> tuple[str, str]:
    per_subject_repo = {
        "horreo": LORA_REPO_HORREO,
        "cruceiro": LORA_REPO_CRUCEIRO,
        "muino": LORA_REPO_MUINO,
    }
    per_subject_weight = {
        "horreo": LORA_WEIGHT_NAME_HORREO,
        "cruceiro": LORA_WEIGHT_NAME_CRUCEIRO,
        "muino": LORA_WEIGHT_NAME_MUINO,
    }
    repo = per_subject_repo.get(subject, "") or LORA_REPO
    weight_name = per_subject_weight.get(subject, "") or LORA_WEIGHT_NAME
    return repo, weight_name


def _try_load_lora(current_pipe, kind: str, lora_repo: str, lora_weight_name: str) -> tuple[str, bool]:
    if not lora_repo:
        return "", False
    if kind.startswith("remote-"):
        return "LoRA disabled: remote Inference API mode does not support applying LoRAs.", False
    if kind not in {"flux", "sd"}:
        return f"LoRA ignored for pipeline `{kind}`.", False
    allowed_targets = {
        "flux": {"flux", "flux-only", "auto", "both", "all"},
        "sd": {"sd", "sd-only", "auto", "both", "all"},
    }
    if LORA_TARGET not in allowed_targets[kind]:
        return f"LoRA disabled for pipeline `{kind}` by LORA_TARGET `{LORA_TARGET}`.", False
    try:
        current_pipe.load_lora_weights(lora_repo, weight_name=lora_weight_name)
        return f"LoRA loaded from `{lora_repo}` (`{lora_weight_name}`).", True
    except Exception as exc:
        return f"Could not load LoRA `{lora_repo}` (`{lora_weight_name}`): {exc}", False


def _build_pipeline(target_model: str):
    target_lower = target_model.lower()

    is_sdxl = "sdxl" in target_lower

    if not torch.cuda.is_available() and ("flux" in target_lower or is_sdxl):
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

    if "flux" in target_lower:
        return FluxPipeline.from_pretrained(target_model, torch_dtype=DTYPE), "flux", target_model, ""

    if is_sdxl:
        # SDXL is much heavier than SD2; avoid fp16 on CPU.
        sdxl_dtype = DTYPE if torch.cuda.is_available() else torch.float32
        sdxl_pipe = StableDiffusionXLPipeline.from_pretrained(
            target_model,
            torch_dtype=sdxl_dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        return sdxl_pipe, "sdxl", target_model, ""

    sd_pipe = StableDiffusionPipeline.from_pretrained(
        target_model,
        torch_dtype=torch.float32 if not torch.cuda.is_available() else DTYPE,
        safety_checker=None,
    )
    return sd_pipe, "sd", target_model, ""


def _build_img2img_pipeline(base_pipe, kind: str):
    if kind.startswith("remote-"):
        return None
    if kind == "flux":
        return FluxImg2ImgPipeline(**base_pipe.components)
    if kind in {"sd", "cpu-fallback"}:
        return StableDiffusionImg2ImgPipeline(**base_pipe.components)
    if kind == "sdxl":
        return StableDiffusionXLImg2ImgPipeline(**base_pipe.components)
    return None


def _status_text() -> str:
    base = (
        f"Requested model: `{requested_model_id}` | "
        f"Active model: `{effective_model_id}` | "
        f"Pipeline: `{pipeline_kind}`"
    )
    if lora_active and active_lora_repo:
        if active_lora_weight_name:
            base += f" | LoRA: `{active_lora_repo}` (`{active_lora_weight_name}`)"
        else:
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
            "Tip: this Space is running on CPU. To use FLUX with LoRA, switch the Space Hardware to a GPU "
            "(T4 or A10) in the Space Settings."
        )
    return base


def switch_model(target_model: str) -> str:
    global pipe, img2img_pipe, pipeline_kind, requested_model_id, effective_model_id, last_model_note, lora_active
    global active_lora_repo, active_lora_weight_name

    with MODEL_LOCK:
        previous_pipe = pipe
        previous_img2img_pipe = img2img_pipe
        previous_kind = pipeline_kind
        previous_requested = requested_model_id
        previous_effective = effective_model_id
        previous_note = last_model_note
        previous_lora_active = lora_active
        previous_lora_repo = active_lora_repo
        previous_lora_weight = active_lora_weight_name

        try:
            next_pipe, next_kind, next_effective, load_note = _build_pipeline(target_model)
            next_img2img_pipe = _build_img2img_pipeline(next_pipe, next_kind)
            _apply_common_pipeline_tuning(next_pipe)
            if next_img2img_pipe is not None:
                _apply_common_pipeline_tuning(next_img2img_pipe)
            startup_lora_repo, startup_lora_weight = _pick_lora_spec_for_subject("horreo")
            lora_note, lora_loaded = _try_load_lora(
                next_pipe, next_kind, startup_lora_repo, startup_lora_weight
            )

            pipe = next_pipe
            img2img_pipe = next_img2img_pipe
            pipeline_kind = next_kind
            requested_model_id = target_model
            effective_model_id = next_effective
            lora_active = lora_loaded
            active_lora_repo = startup_lora_repo if lora_loaded else ""
            active_lora_weight_name = startup_lora_weight if lora_loaded else ""

            merged_note = "\n".join([n for n in [load_note, lora_note] if n])
            last_model_note = merged_note

            if previous_pipe is not None:
                del previous_pipe
            if previous_img2img_pipe is not None:
                del previous_img2img_pipe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return _status_text()
        except Exception as exc:
            pipe = previous_pipe
            img2img_pipe = previous_img2img_pipe
            pipeline_kind = previous_kind
            requested_model_id = previous_requested
            effective_model_id = previous_effective
            last_model_note = previous_note
            lora_active = previous_lora_active
            active_lora_repo = previous_lora_repo
            active_lora_weight_name = previous_lora_weight
            return _status_text() + f"\n\nModel switch failed: {exc}"


def _ensure_subject_lora(subject: str) -> None:
    global last_model_note, lora_active, active_lora_repo, active_lora_weight_name

    target_repo, target_weight_name = _pick_lora_spec_for_subject(subject)
    if not target_repo:
        lora_active = False
        active_lora_repo = ""
        active_lora_weight_name = ""
        return

    if pipeline_kind.startswith("remote-") or pipeline_kind not in {"flux", "sd"}:
        lora_active = False
        active_lora_repo = ""
        active_lora_weight_name = ""
        return

    if lora_active and active_lora_repo == target_repo and active_lora_weight_name == target_weight_name:
        return

    try:
        if hasattr(pipe, "unload_lora_weights"):
            pipe.unload_lora_weights()
        lora_note, lora_loaded = _try_load_lora(pipe, pipeline_kind, target_repo, target_weight_name)
        lora_active = lora_loaded
        active_lora_repo = target_repo if lora_loaded else ""
        active_lora_weight_name = target_weight_name if lora_loaded else ""
        if lora_note:
            last_model_note = lora_note
    except Exception as exc:
        lora_active = False
        active_lora_repo = ""
        active_lora_weight_name = ""
        last_model_note = f"Could not switch LoRA `{target_repo}` (`{target_weight_name}`): {exc}"


def build_prompt(subject: str, details: str) -> str:
    token_horreo = TOKEN_HORREO if lora_active else ""
    token_cruceiro = TOKEN_CRUCEIRO if lora_active else ""
    token_muino = TOKEN_MUINO if lora_active else ""
    subject_map = {
        "horreo": (
            f"{token_horreo} traditional Galician granary called horreo, elevated high above ground on four "
            "cylindrical stone pegollos with tornarratos, granite feet (pies) with tornaformigas, "
            "rectangular wooden slatted chamber with ventilation gaps and horizontal fajas, "
            "interior lintel (dintel interior), wooden door with penal lock, dark gray slate roof with sobrepens "
            "and cornisa, pinche and cross adornos, weathered gray wood and granite, full exterior visible, "
            "rural Galicia, no stairs, no interior, not Asturian style, "
            "pies_granito, tornaformigas, pegollos_cilindricos, tornarratos, duelas_tablillas, fajas_horizontales, "
            "dintel_interior, penal_cerradura, sobrepens, cornisa_dintel, pinche, adornos_cruz"
        ),
        "cruceiro": (
            f"{token_cruceiro} traditional Galician cruceiro de granito, square stepped base (gradas), "
            "octagonal shaft (varal) with carved motifs, decorated capital, latin cross with flared arms, "
            "full monument visible, weathered granite with moss and lichen, churchyard or crossroads, rural Galicia, "
            "plataforma_escalonada, pousadoiro, fuste_octogonal, ofrenda_votiva, capitel_volutas"
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
    sd_boost_map = {
        "horreo": (
            "(horreo gallego:1.4), (Galician granary:1.3), (elevated on stone pegollos:1.3), "
            "(four cylindrical stone pillars:1.2), (wooden slatted walls:1.2), (slate roof tairona:1.2), "
            "(pies stone base:1.2), (tornaformigas ant guards:1.1), (tornarratos rat guards:1.2), "
            "(duelas tablillas with gaps:1.2), (fajas horizontales:1.1), (dintel interior:1.1), "
            "(penal lock:1.1), (sobrepens eaves:1.2), (cornisa dintel:1.2), (pinche pinnacle:1.1), "
            "(adornos cross:1.1), (no stairs:1.4), (not Asturian horreo:1.3)"
        ),
        "cruceiro": (
            "(cruceiro gallego:1.4), (granite wayside cross:1.3), (stepped base:1.3), "
            "(octagonal shaft:1.3), (decorated capital:1.2), (latin stone cross:1.2), "
            "(Galician rural heritage:1.2)"
        ),
        "muino": "",
        "mixed": "",
    }
    sd_boost = sd_boost_map.get(subject, "") if pipeline_kind in {"sd", "cpu-fallback"} else ""
    chunks = [
        "ethnographic documentary photography",
        base,
        "stone and wood textures",
        "natural light",
        "realistic details",
        details,
    ]
    if sd_boost:
        chunks.append(sd_boost)
    return ", ".join(chunks)


def suggest_details(subject: str) -> str:
    return SUBJECT_DETAIL_PRESETS.get(subject, SUBJECT_DETAIL_PRESETS["horreo"])


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

    if kind.startswith("remote-"):
        # Remote Inference API is usually queued / shared; keep defaults small for responsiveness.
        return (3, 3.0, 640, "fast", True, DEFAULT_NEGATIVE)
    if kind == "flux":
        return (4, 3.5, 1024, "stable", False, DEFAULT_NEGATIVE)
    if kind == "sdxl":
        # SDXL Turbo is designed for very low steps.
        return (4, 3.0, 512, "stable", False, DEFAULT_NEGATIVE)
    if kind == "cpu-fallback":
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
    generation_mode: str,
    init_image,
    img2img_strength: float,
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
        active_img2img_pipe = img2img_pipe
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

        if generation_mode == "image-to-image":
            if init_image is None:
                raise gr.Error("Sube una imagen de referencia para usar image-to-image.")
            if active_kind.startswith("remote-"):
                raise gr.Error(
                    "Image-to-image no está disponible en modo remoto (Inference API) en este Space (CPU). "
                    "Cambia a `texto-a-imagen` o usa un modelo local (SD2 en CPU) / hardware GPU para img2img local."
                )
            if active_img2img_pipe is None:
                raise gr.Error("Image-to-image no está disponible para el modelo activo.")

            reference_image = init_image.convert("RGB").resize((use_resolution, use_resolution))
            kwargs["image"] = reference_image
            kwargs["strength"] = max(0.05, min(1.0, float(img2img_strength)))
            if active_kind == "flux":
                kwargs["max_sequence_length"] = use_seq_len
            image = active_img2img_pipe(**kwargs).images[0]
        elif active_kind.startswith("remote-"):
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
                    "- Accept the model license on its model page\n\n"
                    f"Model: `{effective_model_id}`\n"
                    f"Error: {exc!r}"
                ) from exc
        else:
            if active_kind == "flux":
                kwargs["max_sequence_length"] = use_seq_len
            image = active_pipe(**kwargs).images[0]

    return image, prompt, status_snapshot


# Initialize pipeline once at startup.
switch_model(BASE_MODEL)
INIT_STEPS, INIT_GUIDANCE, INIT_RESOLUTION, INIT_QUALITY, INIT_FAST_MODE, INIT_NEGATIVE = recommended_preset()

with gr.Blocks(title="Hórreos y Cruceiros de Galicia") as demo:
    gr.Markdown(
        "# Generador de Etnografía Gallega\n"
        f"Genera imágenes de **hórreos**, **cruceiros** y **muiños** y cambia de modelo desde la interfaz.\n\n"
        f"`app_version: {APP_VERSION}`"
    )

    with gr.Row():
        model_selector = gr.Dropdown(
            choices=MODEL_CHOICES,
            value=BASE_MODEL if BASE_MODEL in MODEL_CHOICES else MODEL_CHOICES[0],
            label="Modelo",
        )
        apply_model_btn = gr.Button("Aplicar modelo")

    model_status = gr.Markdown(_status_text())
    gr.Markdown(
        "Preset recomendado activo. Si cambias el modelo, pulsa `Aplicar modelo` para reajustar los valores."
    )
    gr.Markdown("Consejo: en `cpu-basic`, FLUX se ejecuta vía API remota y LoRA está desactivado en modo remoto.")

    with gr.Row():
        subject = gr.Dropdown(
            choices=["horreo", "cruceiro", "muino", "mixed"],
            value="horreo",
            label="Tema",
        )
        details = gr.Textbox(
            value=SUBJECT_DETAIL_PRESETS["horreo"],
            label="Detalles del prompt",
        )

    negative_details = gr.Textbox(
        value=INIT_NEGATIVE,
        label="Prompt negativo (para pipelines SD)",
    )

    with gr.Row():
        generation_mode = gr.Radio(
            choices=[("texto-a-imagen", "text-to-image"), ("imagen-a-imagen", "image-to-image")],
            value="text-to-image",
            label="Modo de generación",
        )
        img2img_strength = gr.Slider(
            minimum=0.15,
            maximum=0.95,
            value=0.55,
            step=0.05,
            label="Fuerza de la imagen (img2img)",
        )

    init_image = gr.Image(
        type="pil",
        label="Imagen de referencia (para imagen-a-imagen)",
    )

    with gr.Row():
        seed = gr.Slider(minimum=0, maximum=2_000_000_000, value=42, step=1, label="Semilla")
        steps = gr.Slider(minimum=1, maximum=60, value=INIT_STEPS, step=1, label="Pasos")
        guidance = gr.Slider(minimum=0.0, maximum=12.0, value=INIT_GUIDANCE, step=0.1, label="Orientación")
        resolution = gr.Slider(minimum=512, maximum=1024, value=INIT_RESOLUTION, step=64, label="Resolución")

    with gr.Row():
        quality_profile = gr.Dropdown(
            choices=[("estable", "stable"), ("rápido", "fast"), ("equilibrado", "balanced"), ("calidad", "quality")],
            value=INIT_QUALITY,
            label="Perfil de calidad",
        )
        fast_mode = gr.Checkbox(value=INIT_FAST_MODE, label="Modo rápido")

    with gr.Row():
        stable_preset_btn = gr.Button("Aplicar calidad estable")
        run_btn = gr.Button("Generar")
    output_image = gr.Image(label="Resultado", type="pil")
    output_prompt = gr.Textbox(label="Prompt final")

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

    subject.change(
        fn=suggest_details,
        inputs=[subject],
        outputs=[details],
    )

    run_btn.click(
        fn=generate,
        inputs=[
            subject,
            details,
            negative_details,
            generation_mode,
            init_image,
            img2img_strength,
            seed,
            steps,
            guidance,
            resolution,
            fast_mode,
            quality_profile,
        ],
        outputs=[output_image, output_prompt, model_status],
    )


demo.launch()
