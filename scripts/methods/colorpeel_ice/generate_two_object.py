"""Generate the locked seen/unseen two-object evaluation set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.methods.colorpeel_ice.two_object_evaluation_protocol import (  # noqa: E402
    PROTOCOL_ID,
    build_manifest,
    file_sha256,
    read_protocol,
)

TOKENS = ("<s1*>", "<s2*>", "<s3*>", "<c1*>", "<c2*>", "<c3*>", "<m1*>", "<m2*>")
CUSTOM_DIFFUSION_WEIGHTS = "pytorch_custom_diffusion_weights.bin"
MODEL_ARTIFACTS = (CUSTOM_DIFFUSION_WEIGHTS, *(f"{token}.bin" for token in TOKENS))


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(row: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def valid_image(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with Image.open(path) as image:
            image.load()
            return image.mode == "RGB" and image.size == (512, 512)
    except (OSError, ValueError):
        return False


def validate_model_dir(model_dir: Path) -> None:
    missing = [name for name in MODEL_ARTIFACTS if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing trained weights:\n" + "\n".join(missing))


def model_provenance(model_dir: Path, training_run: Path) -> dict:
    model_dir, training_run = model_dir.resolve(), training_run.resolve()
    if model_dir != training_run / "checkpoints":
        raise ValueError("model-dir must be PARENT_TRAINING_RUN/checkpoints")
    manifest_path = training_run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "succeeded" or manifest.get("returncode") != 0:
        raise ValueError("parent training run is not successful")
    if manifest.get("run", {}).get("variant") != "controlled_two_object_seed42":
        raise ValueError("parent training variant is not the locked two-object seed-42 run")
    validate_model_dir(model_dir)
    hashes = {name: file_sha256(model_dir / name) for name in MODEL_ARTIFACTS}
    aggregate = hashlib.sha256("".join(
        f"{name}:{hashes[name]}\n" for name in sorted(hashes)
    ).encode()).hexdigest()
    return {
        "evaluation_protocol_id": PROTOCOL_ID,
        "parent_training_run": str(training_run),
        "parent_training_variant": "controlled_two_object_seed42",
        "parent_training_git": manifest["git"],
        "training_manifest_sha256": file_sha256(manifest_path),
        "training_config_sha256": file_sha256(training_run / "config.yaml"),
        "model_dir": str(model_dir),
        "model_artifact_sha256": hashes,
        "model_fingerprint_sha256": aggregate,
    }


def pending_rows(rows: list[dict], args: argparse.Namespace) -> list[dict]:
    if not args.skip_existing:
        existing = [args.output_dir / row["image_path"] for row in rows
                    if (args.output_dir / row["image_path"]).exists()]
        if existing:
            raise RuntimeError("output images already exist; use explicit --skip-existing")
        return rows
    if not args.status_path.is_file():
        if any((args.output_dir / row["image_path"]).exists() for row in rows):
            raise RuntimeError("existing images have no generation status ledger")
        return rows
    statuses = {row.get("id"): row for row in read_jsonl(args.status_path)}
    pending = []
    for row in rows:
        image = args.output_dir / row["image_path"]
        if not image.exists():
            pending.append(row)
            continue
        status = statuses.get(row["id"])
        if status is None or status.get("status") != "ok":
            raise RuntimeError(f"existing image lacks successful status: {row['id']}")
        for field in ("model_fingerprint_sha256", "protocol_fingerprint_sha256"):
            if status.get(field) != row[field]:
                raise RuntimeError(f"existing image {field} mismatch: {row['id']}")
        if not valid_image(image) or status.get("image_sha256") != file_sha256(image):
            raise RuntimeError(f"existing image is invalid or changed: {row['id']}")
    return pending


def generate(rows: list[dict], args: argparse.Namespace) -> None:
    import torch
    from diffusers import DiffusionPipeline

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    pipe = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path, low_cpu_mem_usage=False, torch_dtype=dtype
    ).to(args.device)
    pipe.safety_checker = None
    if hasattr(pipe, "requires_safety_checker"):
        pipe.requires_safety_checker = False
    pipe.unet.load_attn_procs(str(args.model_dir), weight_name=CUSTOM_DIFFUSION_WEIGHTS)
    for token in TOKENS:
        pipe.load_textual_inversion(str(args.model_dir), weight_name=f"{token}.bin")
    for row in rows:
        path = args.output_dir / row["image_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        generator = torch.Generator(device=args.device).manual_seed(row["generation_seed"])
        image = pipe(
            row["prompt"],
            num_inference_steps=row["num_inference_steps"],
            guidance_scale=row["guidance_scale"],
            generator=generator,
        ).images[0]
        image.save(path)
        append_jsonl(status_row(row, args, None), args.status_path)


def status_row(row: dict, args: argparse.Namespace, error: str | None) -> dict:
    path = args.output_dir / row["image_path"]
    ok = valid_image(path)
    return {
        "id": row["id"],
        "image_path": str(path),
        "status": "ok" if ok else "failure",
        "failure_reason": None if ok else (error or "missing_or_invalid_image"),
        "image_sha256": file_sha256(path) if ok else None,
        "model_fingerprint_sha256": row.get("model_fingerprint_sha256"),
        "protocol_fingerprint_sha256": row.get("protocol_fingerprint_sha256"),
    }


def output_status(rows: list[dict], args: argparse.Namespace, error: str | None) -> list[dict]:
    return [status_row(row, args, error) for row in rows]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--parent-training-run", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluation-protocol", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--status-path", type=Path)
    parser.add_argument("--provenance-path", type=Path)
    parser.add_argument("--pretrained-model-name-or-path", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--disable-safety-checker", action="store_true")
    parser.add_argument("--acknowledge-safety-risk", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not (args.disable_safety_checker and args.acknowledge_safety_risk):
        parser.error("two-object evaluation requires the paired safety-checker disable flags")
    if not args.dry_run and (args.model_dir is None or args.parent_training_run is None):
        parser.error("model-dir and parent-training-run are required unless dry-run")
    args.manifest_path = args.manifest_path or args.output_dir / "generation_manifest.jsonl"
    args.status_path = args.status_path or args.output_dir / "generation_status.jsonl"
    args.provenance_path = args.provenance_path or args.output_dir / "generation_provenance.json"
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    protocol = read_protocol(args.evaluation_protocol)
    rows = build_manifest(protocol)
    if args.smoke:
        rows = [
            row for row in rows
            if row["pair_index"] == 0 and row["generation_seed"] == 42
        ]
    rows = [{
        **row,
        "dtype": args.dtype,
        "safety_checker_disabled": True,
        "safety_risk_acknowledged": True,
        "protocol_fingerprint_sha256": protocol["_source_sha256"],
    } for row in rows]
    if args.model_dir is not None:
        provenance = model_provenance(args.model_dir, args.parent_training_run)
        args.provenance_path.parent.mkdir(parents=True, exist_ok=True)
        args.provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        rows = [{
            **row,
            "parent_training_run": provenance["parent_training_run"],
            "parent_training_git_commit": provenance["parent_training_git"]["commit"],
            "training_manifest_sha256": provenance["training_manifest_sha256"],
            "training_config_sha256": provenance["training_config_sha256"],
            "model_fingerprint_sha256": provenance["model_fingerprint_sha256"],
        } for row in rows]
    write_jsonl(rows, args.manifest_path)
    if args.dry_run:
        return
    error = None
    try:
        generate(pending_rows(rows, args), args)
    except Exception as exc:
        error = f"{type(exc).__name__}:{exc}"
        raise
    finally:
        write_jsonl(output_status(rows, args, error), args.status_path)


if __name__ == "__main__":
    main()
