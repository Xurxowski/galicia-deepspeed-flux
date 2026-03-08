#!/usr/bin/env python3
from __future__ import annotations

import os
import random
import threading

import gradio as gr
import torch
from diffusers import (
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
REMOTE_TXT2IMG_MODEL = os.getenv(
    "REMOTE_TXT2IMG_MODEL", "LanguageMachines/stable-diffusion-2-1-base"
).strip()
REMOTE_TXT2IMG_FALLBACKS = [
    m.strip()
    for m in os.getenv(
        "REMOTE_TXT2IMG_FALLBACKS",
        "LanguageMachines/stable-diffusion-2-1-base,stabilityai/stable-diffusion-xl-base-1.0",
    ).split(",")
    if m.strip()
]
REMOTE_IMG2IMG_MODEL = os.getenv(
    "REMOTE_IMG2IMG_MODEL", "radames/stable-diffusion-v1-5-img2img"
).strip()
SD2_REMOTE_ON_CPU = os.getenv("SD2_REMOTE_ON_CPU", "0").strip().lower() not in {"0", "false", "no", "off"}
SDXL_REMOTE_ON_CPU = os.getenv("SDXL_REMOTE_ON_CPU", "1").strip().lower() not in {"0", "false", "no", "off"}
IMG2IMG_SPACE = os.getenv("IMG2IMG_SPACE", "fffiloni/stable-diffusion-img2img").strip()
IMG2IMG_API_NAME = os.getenv("IMG2IMG_API_NAME", "/predict").strip() or "/predict"
IMG2IMG_UPLOAD_ENDPOINT = os.getenv("IMG2IMG_UPLOAD_ENDPOINT", "/gradio_api/upload").strip() or "/gradio_api/upload"
IMG2IMG_RUN_ENDPOINT = os.getenv("IMG2IMG_RUN_ENDPOINT", "/gradio_api/run/predict").strip() or "/gradio_api/run/predict"
IMG2IMG_FN_INDEX = int(os.getenv("IMG2IMG_FN_INDEX", "0").strip() or "0")
UPSCALE_MODEL = os.getenv("UPSCALE_MODEL", "")
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
APP_VERSION = os.getenv("APP_VERSION", "2026-03-02-4")

MODEL_CHOICES = [
    "black-forest-labs/FLUX.1-schnell",
    "LanguageMachines/stable-diffusion-2-1-base",
    "stabilityai/stable-diffusion-xl-base-1.0",
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
    "modern cross, minimalist cross, abstract cross, cemetery, graveyard, gravestone, tombstone, headstone, "
    "grave marker, burial ground, wooden cross, metal cross, church interior, "
    "celtic cross with circle, irish cross, scottish cross, "
    "asturian horreo, square stone pillars, round pegollos, wooden pillars, stairs, ground-level granary, "
    "thatched roof, modern barn, metal roof, concrete base, generic christian cross, calvary, "
    "close-up, cropped, cut off, partial monument, close-up crop, interior view, apartment building, cabin, chalet"
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
        "traditional Galician cruceiro, weathered gray granite, rural Galicia, overcast sky, "
        "full monument visible, no cropping, centered composition, wide long shot, vertical framing, "
        "stepped stone platform base (plataforma escalonada) with stone bench (pousadoiro), "
        "pedestal, very tall slender octagonal shaft (fuste/varal) with carved geometric patterns, "
        "votive offering figure or saint on the shaft, decorated capital with scroll volutes, "
        "small latin cross on top, shaft 6x to 8x taller than the cross, cross occupies only the top 10 to 15 percent of total height, "
        "photorealistic architectural documentary photo, realistic proportions, sharp stone texture, moss, lichen"
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

MODEL_SUBJECT_DETAIL_PRESETS = {
    "flux": {
        "horreo": (
            "A traditional Galician granary called horreo, elevated structure on stone pillars, "
            "granite base feet (pies) with ant guards (tornaformigas) at ground level, "
            "tall cylindrical stone pillars (pegollos) raising the structure high, "
            "rat guards (tornarratos) on the pillars, rectangular wooden chest construction, "
            "vertical wooden slats (duelas or tablillas) with gaps for ventilation, "
            "horizontal wooden bands (fajas) reinforcing the walls, "
            "interior lintel (dintel interior) framing the doorway, "
            "wooden door with traditional lock mechanism (penal), "
            "sloped slate roof with overhanging eaves (sobrepens), decorative cornice (cornisa), "
            "small pinnacle (pinche) and cross adornment (adornos) at roof peak, "
            "weathered gray wood texture, dark gray slate tiles, no stairs, elevated on pillars only, "
            "rural Galician farm setting, green fields, photorealistic, detailed wood grain"
        ),
        "cruceiro": (
            "Traditional Galician cruceiro, weathered gray granite, rural Galicia, overcast sky. "
            "Full monument visible (no cropping), centered wide long shot, vertical framing. "
            "Stepped stone platform base (plataforma escalonada) with stone bench (pousadoiro). "
            "Pedestal + very tall slender octagonal shaft (fuste/varal) with carved geometric patterns, "
            "votive offering figure/saint on the shaft, decorated capital with scroll volutes. "
            "Small Latin cross on top: shaft 6x to 8x taller than the cross, cross occupies only the top 10 to 15 percent of total height. "
            "Photorealistic architectural documentary photo, realistic proportions, sharp stone texture, moss and lichen. "
            "No cemetery, no graveyard."
        ),
        "muino": (
            "Traditional Galician muino (water mill), small stone millhouse beside a stream, mossy granite, "
            "rural Galicia, overcast atlantic light, documentary realism, full building visible"
        ),
        "mixed": (
            "Traditional Galician horreo and a Galician cruceiro in the same rural ethnographic scene, "
            "both fully visible, wide shot, documentary realism, overcast atlantic light"
        ),
    },
    "sdxl_fast": {
        "horreo": (
            "Galician horreo (traditional granary), elevated high on cylindrical stone pegollos, "
            "tornaformigas, tornarratos, wooden slatted walls, slate roof, rural Galicia, photorealistic"
        ),
        "cruceiro": (
            "Galician cruceiro in granite, full monument visible, wide shot, stepped base, octagonal shaft, "
            "decorated capital, small latin cross on top, rural Galicia, photorealistic"
        ),
        "muino": (
            "Galician stone water millhouse beside a stream, mossy granite, rural Galicia, photorealistic"
        ),
        "mixed": (
            "Galician horreo and Galician cruceiro together, both fully visible, rural Galicia, photorealistic"
        ),
    },
    "sdxl_quality": {
        "horreo": (
            "Traditional Galician horreo, elevated on cylindrical stone pegollos with tornaformigas, tornarratos, "
            "wooden slatted chamber with fajas, slate roof with sobrepens, rural Galicia, architectural documentary photo, wide shot"
        ),
        "cruceiro": (
            "Traditional Galician cruceiro de granito, full monument visible, wide long shot, stepped base, pousadoiro, "
            "octagonal shaft with carved motifs, decorated capital with scroll volutes, small latin cross on top, "
            "realistic proportions, sharp granite texture with moss and lichen, rural Galicia, overcast sky"
        ),
        "muino": (
            "Traditional Galician muino (water mill), stone millhouse beside a stream, mossy granite, rural Galicia, "
            "architectural documentary photo, wide shot"
        ),
        "mixed": (
            "Traditional Galician horreo and cruceiro together in rural Galicia, both fully visible, wide shot, "
            "architectural documentary photo, overcast atlantic daylight"
        ),
    },
}

MODEL_NEGATIVE_PRESETS = {
    "flux": (
        "cemetery, graveyard, tombstone, grave marker, headstone, burial ground, "
        "modern metal cross, wooden cross, minimalist cross, abstract cross, "
        "celtic cross with circle, Irish cross, Scottish cross, church interior, "
        "close-up, cropped, cut off, partial monument"
    ),
    "sdxl": DEFAULT_NEGATIVE,
    "sd": DEFAULT_NEGATIVE,
}

HORREO_VARIANT_CHOICES = [
    ("madera (tablillas)", "wood"),
    ("piedra (lajas/perpiaño)", "stone"),
    ("mixto (madera + piedra)", "mixed"),
]

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
    is_flux = "flux" in target_lower
    is_sdxl = _is_sdxl_model_id(target_model)
    is_sd2 = _is_sd2_model_id(target_model)

    # CPU architecture:
    # - FLUX stays remote for responsiveness.
    # - SD2 defaults local so LoRA can be applied.
    # - SDXL can stay remote on CPU by default (toggle via SDXL_REMOTE_ON_CPU).
    use_remote = (
        not torch.cuda.is_available()
        and USE_INFERENCE_API
        and (
            is_flux
            or (is_sdxl and SDXL_REMOTE_ON_CPU)
            or (is_sd2 and SD2_REMOTE_ON_CPU)
        )
    )
    if use_remote:
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or get_token()
        client_kwargs = dict(model=target_model, token=token, timeout=INFERENCE_TIMEOUT_SEC)
        if token:
            client_kwargs["provider"] = "hf-inference"
        client = InferenceClient(**client_kwargs)
        note = (
            f"CPU runtime detected: using Hugging Face Inference API for `{target_model}`.\n"
            "If generation fails, add `HF_TOKEN` as a Space secret and ensure you accepted the model license."
        )
        return client, f"remote-{target_model.split('/')[-1].lower()}", target_model, note

    if not torch.cuda.is_available() and not USE_INFERENCE_API and (is_flux or is_sdxl):
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

    if is_flux:
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
    base += f"\nApp version: `{APP_VERSION}`"
    base += f"\nRemote img2img backend: `{IMG2IMG_SPACE or 'disabled'}`"
    base += (
        f"\nRemote img2img upload: `{IMG2IMG_UPLOAD_ENDPOINT}` | run: `{IMG2IMG_RUN_ENDPOINT}` "
        f"(fn_index={IMG2IMG_FN_INDEX})"
    )
    base += f"\nRemote img2img model: `{REMOTE_IMG2IMG_MODEL}`"
    base += f"\nRemote txt2img fallback: `{REMOTE_TXT2IMG_MODEL}`"
    if REMOTE_TXT2IMG_FALLBACKS:
        base += f"\nRemote txt2img fallbacks: `{', '.join(REMOTE_TXT2IMG_FALLBACKS)}`"
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


def _model_family_from_id(model_id: str) -> str:
    lower = (model_id or "").lower()
    if "flux" in lower:
        return "flux"
    if "stable-diffusion-xl" in lower or lower.endswith("sdxl-base-1.0"):
        return "sdxl_quality"
    return "sd"


def _is_sdxl_model_id(model_id: str) -> bool:
    lower = (model_id or "").lower()
    return (
        "sdxl" in lower
        or (
            "xl" in lower
            and "stable-diffusion-2" not in lower
            and "stable-diffusion-1" not in lower
            and "sd-2" not in lower
            and "sd-1" not in lower
        )
    )


def _is_sd2_model_id(model_id: str) -> bool:
    lower = (model_id or "").lower()
    return "stable-diffusion-2" in lower or "sd-2" in lower or "sd2" in lower


def suggest_details_for_model(subject: str, model_id: str) -> tuple[str, str]:
    family = _model_family_from_id(model_id)
    details = MODEL_SUBJECT_DETAIL_PRESETS.get(family, SUBJECT_DETAIL_PRESETS).get(
        subject, SUBJECT_DETAIL_PRESETS.get(subject, SUBJECT_DETAIL_PRESETS["horreo"])
    )
    if family == "flux":
        negative = MODEL_NEGATIVE_PRESETS["flux"]
    elif family.startswith("sdxl"):
        negative = MODEL_NEGATIVE_PRESETS["sdxl"]
    else:
        negative = MODEL_NEGATIVE_PRESETS["sd"]
    return details, negative


def horreo_variant_update(subject: str):
    return gr.update(visible=subject == "horreo")


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
        # Remote Inference API is usually queued/shared; keep defaults small for responsiveness.
        if _is_sdxl_model_id(model_id):
            return (8, 4.0, 640, "stable", False, DEFAULT_NEGATIVE)
        if _is_sd2_model_id(model_id):
            return (10, 5.5, 640, "stable", False, DEFAULT_NEGATIVE)
        return (3, 3.0, 640, "fast", True, DEFAULT_NEGATIVE)
    if kind == "flux":
        return (4, 3.5, 1024, "stable", False, DEFAULT_NEGATIVE)
    if kind == "sdxl":
        return (10, 5.0, 640, "stable", False, DEFAULT_NEGATIVE)
    if kind == "cpu-fallback":
        return (1, 0.0, 512, "stable", False, DEFAULT_NEGATIVE)
    # SD2 local on CPU: default to moderate quality/cost.
    return (16, 6.5, 640, "stable", False, DEFAULT_NEGATIVE)


def apply_stable_preset():
    return (
        *recommended_preset(),
    )

def generation_mode_update():
    with MODEL_LOCK:
        kind = pipeline_kind
        model_id = effective_model_id
        has_local_i2i = img2img_pipe is not None

    eligible_model = _is_sd2_model_id(model_id) or _is_sdxl_model_id(model_id)
    if kind.startswith("remote-"):
        img2img_supported = eligible_model
    else:
        img2img_supported = eligible_model and has_local_i2i

    if img2img_supported:
        return gr.update(
            choices=[("texto-a-imagen", "text-to-image"), ("imagen-a-imagen", "image-to-image")],
            value="text-to-image",
        )
    return gr.update(choices=[("texto-a-imagen", "text-to-image")], value="text-to-image")


def img2img_controls_update():
    with MODEL_LOCK:
        kind = pipeline_kind
        model_id = effective_model_id
        has_local_i2i = img2img_pipe is not None

    eligible_model = _is_sd2_model_id(model_id) or _is_sdxl_model_id(model_id)
    if kind.startswith("remote-"):
        visible = eligible_model
    else:
        visible = eligible_model and has_local_i2i

    return (
        gr.update(visible=visible),
        gr.update(visible=visible),
    )


def switch_model_and_apply_preset(target_model: str, subject: str):
    status = switch_model(target_model)
    steps, guidance, resolution, quality_profile, fast_mode, negative = recommended_preset()
    mode_update = generation_mode_update()
    init_image_update, strength_update = img2img_controls_update()
    details_text, negative_text = suggest_details_for_model(subject, target_model)
    return (
        status,
        steps,
        guidance,
        resolution,
        quality_profile,
        fast_mode,
        negative_text,
        mode_update,
        init_image_update,
        strength_update,
        details_text,
    )


def generate(
    subject: str,
    details: str,
    negative_details: str,
    generation_mode: str,
    init_image,
    horreo_variant: str,
    img2img_strength: float,
    randomize_seed: bool,
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
        prompt_details = details
        if subject == "horreo":
            if horreo_variant == "stone":
                prompt_details = f"{prompt_details}, stone wall panels, granite slabs, perpiaño stonework"
            elif horreo_variant == "mixed":
                prompt_details = f"{prompt_details}, mixed materials, some stone wall panels and some wooden slats"
            else:
                prompt_details = f"{prompt_details}, wooden slats on wall panels"

        prompt = build_prompt(subject, prompt_details)
        active_pipe = pipe
        active_img2img_pipe = img2img_pipe
        active_kind = pipeline_kind
        active_effective_model_id = effective_model_id
        status_snapshot = _status_text()

    is_sdxl_model = _is_sdxl_model_id(active_effective_model_id)
    is_sd2_model = _is_sd2_model_id(active_effective_model_id)
    remote_img2img_supported = active_kind.startswith("remote-") and (is_sdxl_model or is_sd2_model)

    seed_value = random.randint(0, 2_000_000_000) if randomize_seed else int(seed)
    generator = torch.Generator(device=GENERATOR_DEVICE).manual_seed(seed_value)
    status_snapshot = status_snapshot + f"\n\nSeed usado: `{seed_value}`"

    with torch.inference_mode():
        image = None
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
            if not (is_sdxl_model or is_sd2_model):
                raise gr.Error("Image-to-image solo está disponible para SD 2.1 base y SDXL base 1.0.")
            if active_kind.startswith("remote-"):
                if not remote_img2img_supported:
                    raise gr.Error(
                        "Image-to-image en modo remoto (Inference API) solo está habilitado para SD2/SDXL en este Space. "
                        "Para FLUX remoto usa `texto-a-imagen`."
                    )

                reference_image = init_image.convert("RGB").resize((use_resolution, use_resolution))
                try:
                    import requests
                    from tempfile import NamedTemporaryFile

                    if not IMG2IMG_SPACE:
                        raise gr.Error("Remote img2img backend disabled: set IMG2IMG_SPACE.")

                    # Convert `user/space-name` -> `https://user-space-name.hf.space`
                    space_host = IMG2IMG_SPACE.replace("/", "-")
                    base_url = f"https://{space_host}.hf.space"

                    strength_value = max(0.05, min(1.0, float(img2img_strength)))
                    guide_value = float(guidance)
                    steps_value = int(use_steps)
                    seed_value = int(seed_value)

                    with NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        tmp_path = tmp.name
                    try:
                        reference_image.save(tmp_path, format="PNG")

                        upload_endpoints = []
                        if IMG2IMG_UPLOAD_ENDPOINT:
                            upload_endpoints.append(IMG2IMG_UPLOAD_ENDPOINT)
                        upload_endpoints.extend(["/gradio_api/upload", "/upload", "/api/upload"])

                        upload_resp = None
                        upload_errors = []

                        for ep in upload_endpoints:
                            url = f"{base_url}{ep}"
                            # Gradio upload endpoints have historically accepted slightly different multipart shapes.
                            # Try the two common forms:
                            # - files={"file": (name, fp, mime)}
                            # - files=[("file", (name, fp, mime))]
                            for mode in ("dict", "list"):
                                try:
                                    with open(tmp_path, "rb") as f:
                                        if mode == "dict":
                                            files = {"file": (os.path.basename(tmp_path), f, "image/png")}
                                        else:
                                            files = [("file", (os.path.basename(tmp_path), f, "image/png"))]
                                        resp = requests.post(
                                            url,
                                            files=files,
                                            timeout=INFERENCE_TIMEOUT_SEC,
                                        )
                                    if resp.status_code == 404:
                                        continue
                                    resp.raise_for_status()
                                    upload_resp = resp
                                    break
                                except Exception as exc:
                                    status = getattr(getattr(exc, "response", None), "status_code", None)
                                    body = getattr(getattr(exc, "response", None), "text", None)
                                    upload_errors.append((url, mode, status, body, repr(exc)))
                            if upload_resp is not None:
                                break

                        if upload_resp is None:
                            raise gr.Error(
                                "Backend upload failed for all endpoints/modes.\n"
                                + "\n".join([f"- {u} ({m}) -> {s}: {e}" for (u, m, s, _b, e) in upload_errors])
                            )
                        upload_resp.raise_for_status()
                        upload_json = upload_resp.json()

                        uploaded_path = None
                        if isinstance(upload_json, str):
                            uploaded_path = upload_json
                        elif isinstance(upload_json, dict):
                            files = upload_json.get("files")
                            if isinstance(files, list) and files:
                                item = files[0]
                                if isinstance(item, dict):
                                    uploaded_path = item.get("path") or item.get("name")
                                elif isinstance(item, str):
                                    uploaded_path = item
                            uploaded_path = uploaded_path or upload_json.get("path")
                        elif isinstance(upload_json, list) and upload_json:
                            item = upload_json[0]
                            if isinstance(item, dict):
                                uploaded_path = item.get("path") or item.get("name")
                            elif isinstance(item, str):
                                uploaded_path = item

                        if not uploaded_path:
                            raise gr.Error(f"Could not parse backend upload response: {upload_json!r}")

                        payload = {
                            "fn_index": IMG2IMG_FN_INDEX,
                            "data": [
                                uploaded_path,
                                prompt,
                                guide_value,
                                steps_value,
                                seed_value,
                                strength_value,
                            ],
                        }
                        run_endpoints = []
                        if IMG2IMG_RUN_ENDPOINT:
                            run_endpoints.append(IMG2IMG_RUN_ENDPOINT)
                        run_endpoints.extend(["/gradio_api/run/predict", "/run/predict", "/api/predict"])

                        predict_resp = None
                        last_run_exc = None
                        for ep in run_endpoints:
                            try:
                                predict_resp = requests.post(
                                    f"{base_url}{ep}",
                                    json=payload,
                                    timeout=INFERENCE_TIMEOUT_SEC,
                                )
                                if predict_resp.status_code == 404:
                                    continue
                                predict_resp.raise_for_status()
                                break
                            except Exception as exc:
                                last_run_exc = exc
                                predict_resp = None

                        if predict_resp is None:
                            raise gr.Error(
                                f"Backend run failed for endpoints {run_endpoints!r}: {last_run_exc!r}"
                            )
                        predict_resp.raise_for_status()
                        result = predict_resp.json()
                    finally:
                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass
                except Exception as exc:
                    raise gr.Error(
                        "Remote img2img call failed (Space backend).\n\n"
                        "Common fixes:\n"
                        "- The backend Space may be cold-starting or queued; retry in 30-60s\n"
                        "- Ensure IMG2IMG_SPACE points to a running Space\n"
                        "- Ensure IMG2IMG_RUN_ENDPOINT and IMG2IMG_FN_INDEX match the backend API\n\n"
                        f"Remote img2img backend: `{IMG2IMG_SPACE}`\n"
                        f"Remote img2img upload: `{IMG2IMG_UPLOAD_ENDPOINT}`\n"
                        f"Remote img2img run: `{IMG2IMG_RUN_ENDPOINT}` (fn_index={IMG2IMG_FN_INDEX})\n"
                        f"Remote img2img model: `{REMOTE_IMG2IMG_MODEL}`\n"
                        f"Model: `{effective_model_id}`\n"
                        f"Pipeline: `{active_kind}`\n"
                        f"Error: {exc!r}"
                    ) from exc

                from PIL import Image

                # Gradio HTTP responses usually contain a dict with a `data` field.
                image = None
                out_path = None
                if isinstance(result, dict) and "data" in result and isinstance(result["data"], list) and result["data"]:
                    out0 = result["data"][0]
                    if isinstance(out0, dict):
                        out_path = out0.get("path") or out0.get("name")
                    elif isinstance(out0, str):
                        out_path = out0

                if not out_path:
                    raise gr.Error(f"Remote img2img backend returned unexpected payload: {result!r}")

                # Many Spaces return a temporary path that must be fetched via /file=...
                try:
                    if out_path.startswith("http://") or out_path.startswith("https://"):
                        file_url = out_path
                    else:
                        file_url = f"{base_url}/file={out_path.lstrip('/')}"
                    file_resp = requests.get(file_url, timeout=INFERENCE_TIMEOUT_SEC)
                    file_resp.raise_for_status()
                    from io import BytesIO

                    image = Image.open(BytesIO(file_resp.content)).convert("RGB")
                except Exception as exc:
                    raise gr.Error(f"Could not fetch backend image from `{out_path}`: {exc!r}") from exc

                return image, prompt, status_snapshot, gr.update(visible=True)
            if active_img2img_pipe is None:
                raise gr.Error("Image-to-image no está disponible para el modelo activo.")

            reference_image = init_image.convert("RGB").resize((use_resolution, use_resolution))
            kwargs["image"] = reference_image
            kwargs["strength"] = max(0.05, min(1.0, float(img2img_strength)))
            image = active_img2img_pipe(**kwargs).images[0]
        elif active_kind.startswith("remote-"):
            try:
                remote_kwargs = dict(
                    prompt=prompt,
                    height=use_resolution,
                    width=use_resolution,
                    num_inference_steps=use_steps,
                    guidance_scale=guidance,
                    seed=seed_value,
                )
                if negative_text:
                    remote_kwargs["negative_prompt"] = negative_text

                def _is_remote_404(error: Exception) -> bool:
                    try:
                        response = getattr(error, "response", None)
                        status_code = getattr(response, "status_code", None)
                        if status_code == 404:
                            return True
                    except Exception:
                        pass
                    text = repr(error)
                    return "404" in text and "Not Found" in text

                def _remote_txt2img_try_models(primary_model_id: str, call_kwargs: dict):
                    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or get_token()
                    tried = []

                    def _client_for(model_id: str):
                        client_kwargs = dict(model=model_id, token=token, timeout=INFERENCE_TIMEOUT_SEC)
                        if token:
                            client_kwargs["provider"] = "hf-inference"
                        return InferenceClient(**client_kwargs)

                    candidates = []
                    if primary_model_id:
                        candidates.append(primary_model_id)
                    if REMOTE_TXT2IMG_MODEL and REMOTE_TXT2IMG_MODEL not in candidates:
                        candidates.append(REMOTE_TXT2IMG_MODEL)
                    for mid in REMOTE_TXT2IMG_FALLBACKS:
                        if mid not in candidates:
                            candidates.append(mid)

                    last_exc = None
                    for model_id in candidates:
                        tried.append(model_id)
                        try:
                            per_model_kwargs = dict(call_kwargs)
                            result = _client_for(model_id).text_to_image(**per_model_kwargs)
                            return result, model_id, tried
                        except Exception as exc:
                            last_exc = exc
                            if _is_remote_404(exc):
                                continue
                            raise
                    raise last_exc

                result, used_model_id, tried_models = _remote_txt2img_try_models(
                    active_effective_model_id, remote_kwargs
                )
                if hasattr(result, "convert"):
                    image = result
                else:
                    from io import BytesIO

                    from PIL import Image

                    image = Image.open(BytesIO(result)).convert("RGB")

                if used_model_id != active_effective_model_id:
                    status_snapshot = (
                        status_snapshot
                        + "\n\n"
                        + f"Remote txt2img router fallback used: `{used_model_id}` (requested `{active_effective_model_id}`)."
                        + f"\nTried: {', '.join(tried_models)}"
                    )
            except Exception as exc:
                if image is not None:
                    return image, prompt, status_snapshot, gr.update(visible=True)
                raise gr.Error(
                    "Remote Inference API call failed.\n\n"
                    "Common fixes:\n"
                    "- Add `HF_TOKEN` as a Space secret\n"
                    "- Accept the model license on its model page\n\n"
                    f"Model: `{active_effective_model_id}`\n"
                    f"Pipeline: `{active_kind}`\n"
                    f"Remote txt2img fallback: `{REMOTE_TXT2IMG_MODEL}`\n"
                    f"Error: {exc!r}"
                ) from exc
        else:
            if active_kind == "flux":
                kwargs["max_sequence_length"] = use_seq_len
            image = active_pipe(**kwargs).images[0]

    return image, prompt, status_snapshot, gr.update(visible=True)


def upscale_x2(image):
    if image is None:
        raise gr.Error("No hay imagen para escalar. Genera una imagen primero.")

    base_image = image.convert("RGB")
    base_w, base_h = base_image.size
    status = _status_text()
    real_esrgan_error = None
    remote_upscale_error = None

    # Preferred path: call a Real-ESRGAN Space (better quality than classical resize).
    # This is optional; if the Space/API is unavailable we fall back to local upscaling.
    realesrgan_space = os.getenv("REAL_ESRGAN_SPACE", "Nick088/Real-ESRGAN_Pytorch").strip()
    if realesrgan_space:
        try:
            from gradio_client import Client

            client = Client(realesrgan_space)
            api_name = os.getenv("REAL_ESRGAN_API_NAME", "/predict").strip() or "/predict"
            size_modifier = os.getenv("REAL_ESRGAN_SIZE", "2").strip() or "2"

            from tempfile import NamedTemporaryFile

            from PIL import Image

            with NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                base_image.save(tmp_path, format="PNG")
                result_path = client.predict(tmp_path, size_modifier, api_name=api_name)
                upscaled = Image.open(result_path).convert("RGB")
                return upscaled, status + f"\n\nUpscaler: Real-ESRGAN (`{realesrgan_space}`)."
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
        except Exception as exc:
            real_esrgan_error = repr(exc)

    if USE_INFERENCE_API and UPSCALE_MODEL.strip():
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or get_token()
        if token:
            client = InferenceClient(model=UPSCALE_MODEL, token=token, timeout=INFERENCE_TIMEOUT_SEC)
            try:
                result = client.image_to_image(prompt="high quality", image=base_image)
                if hasattr(result, "convert"):
                    upscaled = result
                else:
                    from io import BytesIO

                    from PIL import Image

                    upscaled = Image.open(BytesIO(result)).convert("RGB")

                target_size = (base_w * 2, base_h * 2)
                if upscaled.size != target_size:
                    upscaled = upscaled.resize(target_size, resample=3)
                return upscaled, status + f"\n\nUpscaler: Inference API (`{UPSCALE_MODEL}`)."
            except Exception as exc:
                remote_upscale_error = repr(exc)

    from PIL import Image, ImageEnhance, ImageFilter

    target_size = (base_w * 2, base_h * 2)
    upscaled = base_image.resize(target_size, resample=Image.Resampling.LANCZOS)
    upscaled = upscaled.filter(ImageFilter.DETAIL)
    upscaled = upscaled.filter(ImageFilter.UnsharpMask(radius=1.2, percent=160, threshold=2))
    upscaled = ImageEnhance.Contrast(upscaled).enhance(1.06)
    fallback_note = "Upscaler: fallback local (Lanczos + UnsharpMask)."
    if real_esrgan_error:
        fallback_note += f"\nReal-ESRGAN no disponible: `{real_esrgan_error}`"
    if remote_upscale_error:
        fallback_note += f"\nInference API upscale no disponible: `{remote_upscale_error}`"
    return upscaled, status + f"\n\n{fallback_note}"


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
        horreo_variant = gr.Dropdown(
            choices=HORREO_VARIANT_CHOICES,
            value="wood",
            label="Variante de hórreo",
            visible=True,
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
            choices=[("texto-a-imagen", "text-to-image")],
            value="text-to-image",
            label="Modo de generación",
        )
        img2img_strength = gr.Slider(
            minimum=0.15,
            maximum=0.95,
            value=0.55,
            step=0.05,
            label="Fuerza de la imagen (img2img)",
            visible=False,
        )

    init_image = gr.Image(
        type="pil",
        label="Imagen de referencia (para imagen-a-imagen)",
        visible=False,
    )

    with gr.Row():
        seed = gr.Slider(minimum=0, maximum=2_000_000_000, value=42, step=1, label="Semilla fija")
        randomize_seed = gr.Checkbox(value=True, label="Semilla aleatoria en cada clic")
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
    output_image = gr.Image(label="Resultado", type="pil", interactive=False)
    upscale_btn = gr.Button("Upscale x2 (opcional)", visible=False)
    output_prompt = gr.Textbox(label="Prompt final")

    apply_model_btn.click(
        fn=switch_model_and_apply_preset,
        inputs=[model_selector, subject],
        outputs=[
            model_status,
            steps,
            guidance,
            resolution,
            quality_profile,
            fast_mode,
            negative_details,
            generation_mode,
            init_image,
            img2img_strength,
            details,
        ],
    )

    stable_preset_btn.click(
        fn=apply_stable_preset,
        inputs=[],
        outputs=[steps, guidance, resolution, quality_profile, fast_mode, negative_details],
    )

    subject.change(
        fn=lambda s, m: (*suggest_details_for_model(s, m),),
        inputs=[subject, model_selector],
        outputs=[details, negative_details],
    )

    subject.change(
        fn=horreo_variant_update,
        inputs=[subject],
        outputs=[horreo_variant],
    )

    run_btn.click(
        fn=generate,
        inputs=[
            subject,
            details,
            negative_details,
            generation_mode,
            init_image,
            horreo_variant,
            img2img_strength,
            randomize_seed,
            seed,
            steps,
            guidance,
            resolution,
            fast_mode,
            quality_profile,
        ],
        outputs=[output_image, output_prompt, model_status, upscale_btn],
    )

    upscale_btn.click(
        fn=upscale_x2,
        inputs=[output_image],
        outputs=[output_image, model_status],
    )


demo.queue(default_concurrency_limit=1, max_size=24).launch()
