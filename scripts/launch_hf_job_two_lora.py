#!/usr/bin/env python3
"""Launch/monitor a Hugging Face Job to train two FLUX LoRAs on cloud GPU."""

from __future__ import annotations

import argparse
import shlex
import sys
import textwrap
import time
from typing import Iterable

from huggingface_hub import HfApi, get_token


TERMINAL_STATUSES = {"COMPLETED", "FAILED", "CANCELED", "TIMEOUT"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train horreo+cruceiro LoRAs on HF Jobs (T4/A10), upload, and update Space."
    )
    parser.add_argument("--job-id", default=None, help="Only inspect/follow an existing job id.")
    parser.add_argument("--follow", action="store_true", help="Stream status/logs until the job finishes.")
    parser.add_argument("--poll-sec", type=int, default=20, help="Polling interval while following.")
    parser.add_argument("--tail-lines", type=int, default=80, help="How many log lines to show in status mode.")

    parser.add_argument("--hf-username", default="Xurxowsky")
    parser.add_argument("--namespace", default=None, help="HF namespace for Jobs. Defaults to --hf-username.")
    parser.add_argument(
        "--repo-url",
        default="https://github.com/Xurxowski/galicia-deepspeed-flux.git",
        help="Git repo cloned inside the job runner.",
    )
    parser.add_argument("--repo-ref", default="codex/galicia-deepspeed-flux", help="Git branch/tag/ref to checkout.")
    parser.add_argument(
        "--image",
        default="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
        help="Docker image used by HF Jobs.",
    )
    parser.add_argument("--hardware", default="t4-medium", help="HF Jobs hardware flavor (e.g. t4-small, a10g-small).")
    parser.add_argument("--timeout", default="8h", help="HF Jobs timeout (e.g. 4h, 8h).")

    parser.add_argument(
        "--dataset-repo-horreo",
        default=None,
        help="Dataset repo id for horreo images (imagefolder with image/text columns).",
    )
    parser.add_argument(
        "--dataset-repo-cruceiro",
        default=None,
        help="Dataset repo id for cruceiro images (imagefolder with image/text columns).",
    )
    parser.add_argument("--model-repo-horreo", default=None, help="Output model repo for horreo LoRA.")
    parser.add_argument("--model-repo-cruceiro", default=None, help="Output model repo for cruceiro LoRA.")
    parser.add_argument("--space-id", default=None, help="Space to update at the end.")
    parser.add_argument("--base-model", default="black-forest-labs/FLUX.1-schnell")

    parser.add_argument("--max-train-steps-horreo", type=int, default=500)
    parser.add_argument("--max-train-steps-cruceiro", type=int, default=700)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--grad-acc", type=int, default=1)
    parser.add_argument("--learning-rate", default="1e-4")
    parser.add_argument("--mixed-precision", default="fp16")
    parser.add_argument("--use-deepspeed", default="auto", choices=["auto", "0", "1"])
    parser.add_argument("--skip-space-update", action="store_true")

    return parser.parse_args()


def _q(value: str) -> str:
    return shlex.quote(value)


def defaulted(value: str | None, fallback: str) -> str:
    return value if value else fallback


def build_remote_script(args: argparse.Namespace) -> str:
    hf_username = args.hf_username
    dataset_repo_horreo = defaulted(args.dataset_repo_horreo, f"{hf_username}/galicia-horreo-dataset-private")
    dataset_repo_cruceiro = defaulted(args.dataset_repo_cruceiro, f"{hf_username}/galicia-cruceiro-dataset-private")
    model_repo_horreo = defaulted(args.model_repo_horreo, f"{hf_username}/flux-schnell-galicia-lora-horreo")
    model_repo_cruceiro = defaulted(args.model_repo_cruceiro, f"{hf_username}/flux-schnell-galicia-lora-cruceiro")
    space_id = defaulted(args.space_id, f"{hf_username}/galicia-horreos-cruceiros")

    commands: list[str] = [
        "set -euo pipefail",
        "export DEBIAN_FRONTEND=noninteractive",
        "apt-get update && apt-get install -y git git-lfs",
        "git lfs install",
        "mkdir -p /workspace",
        "rm -rf /workspace/galicia-deepspeed-flux",
        f"git clone --depth 1 --branch {_q(args.repo_ref)} {_q(args.repo_url)} /workspace/galicia-deepspeed-flux",
        "cd /workspace/galicia-deepspeed-flux",
        "python -m pip install -U pip",
        "python -m pip install -r requirements.txt",
        "export HF_HUB_DISABLE_XET=1",
        "export HF_HUB_ENABLE_HF_TRANSFER=1",
        f"export BASE_MODEL={_q(args.base_model)}",
        f"export MIXED_PRECISION={_q(args.mixed_precision)}",
        f"export USE_DEEPSPEED={_q(args.use_deepspeed)}",
        f"export RESOLUTION={_q(str(args.resolution))}",
        f"export RANK={_q(str(args.rank))}",
        f"export GRAD_ACC={_q(str(args.grad_acc))}",
        f"export LEARNING_RATE={_q(args.learning_rate)}",
        "export CHECKPOINTING_STEPS=100",
        "echo '==> Train horreo from dataset repo'",
        f"export DATASET_NAME={_q(dataset_repo_horreo)}",
        "unset DATA_DIR",
        "export IMAGE_COLUMN=image",
        "export CAPTION_COLUMN=text",
        "export OUTPUT_DIR=/workspace/outputs/horreo",
        "export INSTANCE_PROMPT='ethnographic documentary photo of a traditional galician horreo, raised granary on stone pillars, elongated slatted chamber, rural Galicia'",
        f"export MAX_TRAIN_STEPS={_q(str(args.max_train_steps_horreo))}",
        "bash scripts/train_flux_lora_deepspeed.sh",
        f"python scripts/upload_to_hub.py --repo_id {_q(model_repo_horreo)} --repo_type model --folder_path /workspace/outputs/horreo",
        "echo '==> Train cruceiro from dataset repo'",
        f"export DATASET_NAME={_q(dataset_repo_cruceiro)}",
        "unset DATA_DIR",
        "export IMAGE_COLUMN=image",
        "export CAPTION_COLUMN=text",
        "export OUTPUT_DIR=/workspace/outputs/cruceiro",
        "export INSTANCE_PROMPT='ethnographic documentary photo of a galician cruceiro, carved granite cross on stone pedestal, rural Galicia, historic village context'",
        f"export MAX_TRAIN_STEPS={_q(str(args.max_train_steps_cruceiro))}",
        "bash scripts/train_flux_lora_deepspeed.sh",
        f"python scripts/upload_to_hub.py --repo_id {_q(model_repo_cruceiro)} --repo_type model --folder_path /workspace/outputs/cruceiro",
    ]

    if not args.skip_space_update:
        commands.extend(
            [
                "echo '==> Update Space variables/app'",
                "python scripts/create_or_update_space.py "
                f"--space_id {_q(space_id)} "
                "--space_dir ./space "
                f"--base_model {_q(args.base_model)} "
                "--lora_target flux "
                f"--lora_repo_horreo {_q(model_repo_horreo)} "
                f"--lora_repo_cruceiro {_q(model_repo_cruceiro)}",
            ]
        )

    return "\n".join(commands)


def iter_log_lines(api: HfApi, job_id: str, namespace: str) -> list[str]:
    try:
        return list(api.fetch_job_logs(job_id=job_id, namespace=namespace))
    except Exception as exc:  # pragma: no cover - best effort polling
        return [f"[log-fetch-error] {exc}"]


def follow_job(api: HfApi, job_id: str, namespace: str, poll_sec: int) -> str:
    shown = 0
    last_status = ""
    while True:
        job = api.inspect_job(job_id=job_id, namespace=namespace)
        status = str(job.status)
        if status != last_status:
            print(f"[status] {status}", flush=True)
            last_status = status

        lines = iter_log_lines(api, job_id, namespace)
        if shown < len(lines):
            for line in lines[shown:]:
                sys.stdout.write(line)
                if not line.endswith("\n"):
                    sys.stdout.write("\n")
            sys.stdout.flush()
            shown = len(lines)

        if status in TERMINAL_STATUSES:
            return status
        time.sleep(max(poll_sec, 3))


def show_job_status(api: HfApi, job_id: str, namespace: str, tail_lines: int) -> None:
    job = api.inspect_job(job_id=job_id, namespace=namespace)
    print(f"job_id: {job.id}")
    print(f"status: {job.status}")
    print(f"url: {job.url}")
    logs = iter_log_lines(api, job_id, namespace)
    if logs:
        print("---- logs (tail) ----")
        for line in logs[-tail_lines:]:
            sys.stdout.write(line)
            if not line.endswith("\n"):
                sys.stdout.write("\n")


def main() -> None:
    args = parse_args()
    token = get_token()
    if not token:
        raise SystemExit("Missing HF token. Run: hf auth login")

    namespace = args.namespace or args.hf_username
    api = HfApi(token=token)

    if args.job_id:
        show_job_status(api, args.job_id, namespace, args.tail_lines)
        if args.follow:
            final_status = follow_job(api, args.job_id, namespace, args.poll_sec)
            raise SystemExit(0 if final_status == "COMPLETED" else 1)
        return

    remote_script = build_remote_script(args)
    print("Submitting job with:")
    print(f"- namespace: {namespace}")
    print(f"- hardware: {args.hardware}")
    print(f"- timeout: {args.timeout}")
    print(f"- repo: {args.repo_url}@{args.repo_ref}")

    job = api.run_job(
        image=args.image,
        command=["bash", "-lc", remote_script],
        flavor=args.hardware,
        timeout=args.timeout,
        namespace=namespace,
        secrets={"HF_TOKEN": token},
        labels={"project": "galicia-ethnography", "task": "train-two-lora"},
    )
    print(f"job_id: {job.id}")
    print(f"url: {job.url}")
    print(f"status: {job.status}")

    if args.follow:
        final_status = follow_job(api, job.id, namespace, args.poll_sec)
        raise SystemExit(0 if final_status == "COMPLETED" else 1)


if __name__ == "__main__":
    main()
