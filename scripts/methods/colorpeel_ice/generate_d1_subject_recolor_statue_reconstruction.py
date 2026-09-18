"""Generate fixed training-prompt reconstructions for three statue-subject checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


MODEL_ID = "CompVis/stable-diffusion-v1-4"
WEIGHTS = "pytorch_custom_diffusion_weights.bin"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_manifest(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    sampling_variants = protocol.get("sampling_variants")
    if sampling_variants is None:
        sampling_variants = [{"id": None, **protocol["sampling"]}]
    expected_count = protocol.get("expected_image_count")
    if expected_count is None:
        expected_count = protocol["sampling"]["expected_image_count"]
    for checkpoint in protocol["source_checkpoints"]:
        for sampling in sampling_variants:
            for item in protocol["prompts"]:
                for seed in sampling["seeds"]:
                    step = checkpoint["steps"]
                    checkpoint_id = checkpoint.get("id", f"step-{step}")
                    image_group = checkpoint.get("id", f"step_{step}")
                    sampling_id = sampling["id"]
                    sampling_group = "" if sampling_id is None else f"/{sampling_id}"
                    sampling_label = "" if sampling_id is None else f"-{sampling_id}"
                    rows.append({
                        "id": f"{checkpoint_id}{sampling_label}-{item['color']}-seed-{seed}", "checkpoint_steps": step,
                        "checkpoint_id": checkpoint_id, "sampling_id": sampling_id,
                        "model_dir": checkpoint["model_dir"], "color": item["color"], "prompt": item["prompt"],
                        "seed": seed, "num_inference_steps": sampling["num_inference_steps"],
                        "guidance_scale": sampling["guidance_scale"],
                        "image_path": f"images/{image_group}{sampling_group}/{item['color']}-seed-{seed}.png",
                    })
    if len(rows) != expected_count:
        raise ValueError("protocol expected_image_count does not match its grid")
    return rows


def validate_model_dir(path: Path, protocol: dict[str, Any]) -> dict[str, str]:
    resolved = path.resolve()
    checkpoint = next((item for item in protocol["source_checkpoints"] if Path(item["model_dir"]).resolve() == resolved), None)
    if checkpoint is None:
        raise ValueError("model directory is not bound by the protocol")
    token_artifacts = tuple(protocol.get("required_token_artifacts", ("<S*>.bin",)))
    required = (*token_artifacts, WEIGHTS, "embedding_update_audit.json", "training_metrics.jsonl")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError("missing checkpoint artifacts: " + ", ".join(missing))
    forbidden = [name for name in protocol["forbidden_token_artifacts"] if (path / name).exists()]
    if forbidden:
        raise ValueError("forbidden token artifacts: " + ", ".join(forbidden))
    hashes = {name: sha256(path / name) for name in required}
    if checkpoint.get("model_sha256") and hashes[WEIGHTS] != checkpoint["model_sha256"]:
        raise ValueError("checkpoint weights do not match the protocol hash")
    for name, expected in checkpoint.get("token_artifact_sha256", {}).items():
        if name not in token_artifacts or hashes.get(name) != expected:
            raise ValueError(f"checkpoint token artifact does not match the protocol hash: {name}")
    if checkpoint.get("run_manifest_sha256"):
        run_dir = Path(checkpoint["run_dir"])
        if path.parent.resolve() != run_dir.resolve():
            raise ValueError("model directory does not belong to the protocol run directory")
        manifest = run_dir / "manifest.json"
        if not manifest.is_file() or sha256(manifest) != checkpoint["run_manifest_sha256"]:
            raise ValueError("run manifest does not match the protocol hash")
    return hashes


def load_pipeline(model_dir: Path, protocol: dict[str, Any], args: argparse.Namespace) -> Any:
    import torch
    from diffusers import DiffusionPipeline

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    pipe = DiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, low_cpu_mem_usage=False, torch_dtype=dtype, local_files_only=True).to(args.device)
    pipe.unet.load_attn_procs(str(model_dir), weight_name=WEIGHTS)
    for token_artifact in protocol.get("required_token_artifacts", ("<S*>.bin",)):
        pipe.load_textual_inversion(str(model_dir), weight_name=token_artifact)
    return pipe


def all_black(image: Image.Image) -> bool:
    return image.convert("RGB").getextrema() == ((0, 0), (0, 0), (0, 0))


def generate(rows: list[dict[str, Any]], protocol: dict[str, Any], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    import torch

    statuses, artifacts = [], {}
    for model_dir_text in dict.fromkeys(row["model_dir"] for row in rows):
        model_dir = Path(model_dir_text)
        artifacts[str(model_dir)] = validate_model_dir(model_dir, protocol)
        pipe = load_pipeline(model_dir, protocol, args)
        for row in (item for item in rows if item["model_dir"] == model_dir_text):
            path = args.output_dir / row["image_path"]
            status = {"id": row["id"], "image_path": str(path), "status": None, "failure_reason": None, "image_sha256": None, "nsfw_content_detected": None}
            try:
                result = pipe(row["prompt"], num_inference_steps=row["num_inference_steps"], guidance_scale=row["guidance_scale"], generator=torch.Generator(device=args.device).manual_seed(row["seed"]))
                image = result.images[0]
                detected = getattr(result, "nsfw_content_detected", None)
                status["nsfw_content_detected"] = bool(detected[0]) if isinstance(detected, (list, tuple)) and detected else False
                if not isinstance(image, Image.Image) or image.mode != "RGB" or image.size != (512, 512):
                    raise ValueError("pipeline must return a 512x512 RGB PIL image")
                path.parent.mkdir(parents=True, exist_ok=True)
                image.save(path)
                status["image_sha256"] = sha256(path)
                if status["nsfw_content_detected"]:
                    status.update(status="failure", failure_reason="safety_checker_filtered")
                elif all_black(image):
                    status.update(status="failure", failure_reason="all_black_output")
                else:
                    status["status"] = "ok"
            except Exception as error:
                status.update(status="failure", failure_reason=f"generation_error:{type(error).__name__}:{error}")
            statuses.append(status)
        del pipe
        torch.cuda.empty_cache()
    return statuses, artifacts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--pretrained-model-name-or-path", default=MODEL_ID)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    protocol = read_json(args.protocol)
    rows = build_manifest(protocol)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(rows, args.output_dir / "generation_manifest.jsonl")
    provenance = {"protocol_path": str(args.protocol.resolve()), "protocol_sha256": sha256(args.protocol), "model_artifact_sha256": None}
    if args.dry_run:
        write_json(args.output_dir / "provenance.json", provenance)
        return 0
    statuses, artifact_hashes = generate(rows, protocol, args)
    provenance["model_artifact_sha256"] = artifact_hashes
    write_json(args.output_dir / "provenance.json", provenance)
    write_jsonl(statuses, args.output_dir / "generation_status.jsonl")
    return 0 if all(row["status"] == "ok" for row in statuses) else 1


if __name__ == "__main__":
    raise SystemExit(main())
