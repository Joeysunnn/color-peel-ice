"""Generate the locked complete-bundle CLEVR multiview held-out protocol."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import generate as baseline_generate
from src.methods.colorpeel_ice.multiview_evaluation_protocol import (
    EVALUATION_PROTOCOL_ID,
    build_manifest,
    read_evaluation_protocol,
)


MODEL_ARTIFACTS = (
    baseline_generate.CUSTOM_DIFFUSION_WEIGHTS,
    *(f"{token}.bin" for _, token, _ in (*baseline_generate.SUBJECTS, *baseline_generate.COLORS)),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_provenance(model_dir: Path, parent_training_run: Path) -> dict:
    model_dir = model_dir.resolve()
    parent_training_run = parent_training_run.resolve()
    if model_dir != parent_training_run / "checkpoints":
        raise ValueError("model-dir must be PARENT_TRAINING_RUN/checkpoints")
    manifest_path = parent_training_run / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing parent training manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "succeeded" or manifest.get("returncode") != 0:
        raise ValueError("parent training run is not succeeded with return code 0")
    baseline_generate.validate_model_dir(model_dir)
    hashes = {name: sha256_file(model_dir / name) for name in MODEL_ARTIFACTS}
    aggregate = hashlib.sha256(
        "".join(f"{name}:{hashes[name]}\n" for name in sorted(hashes)).encode("utf-8")
    ).hexdigest()
    return {
        "evaluation_protocol_id": EVALUATION_PROTOCOL_ID,
        "parent_training_run": str(parent_training_run),
        "parent_training_variant": manifest["run"]["variant"],
        "parent_training_seed": manifest["run"]["seed"],
        "parent_training_git": manifest["git"],
        "training_manifest_sha256": sha256_file(manifest_path),
        "training_config_sha256": sha256_file(parent_training_run / "config.yaml"),
        "model_dir": str(model_dir),
        "model_artifact_sha256": hashes,
        "model_fingerprint_sha256": aggregate,
    }


def output_status(rows: list[dict], output_dir: Path, generation_error: str | None) -> list[dict]:
    statuses = []
    for row in rows:
        image_path = output_dir / row["image_path"]
        valid = baseline_generate.is_decodable_image(image_path)
        width = height = mode = None
        if valid:
            with baseline_generate.Image.open(image_path) as image:
                width, height = image.size
                mode = image.mode
        statuses.append(
            {
                "id": row["id"],
                "image_path": str(image_path),
                "status": "ok" if valid else "failure",
                "failure_reason": None if valid else (generation_error or "missing_or_invalid_image"),
                "image_sha256": sha256_file(image_path) if valid else None,
                "width": width,
                "height": height,
                "mode": mode,
                "model_fingerprint_sha256": row.get("model_fingerprint_sha256"),
                "protocol_fingerprint_sha256": row.get("protocol_fingerprint_sha256"),
            }
        )
    return statuses


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def resume_pending_rows(rows: list[dict], args: argparse.Namespace) -> list[dict]:
    if not args.skip_existing:
        return rows
    existing_images = [
        args.output_dir / row["image_path"]
        for row in rows
        if (args.output_dir / row["image_path"]).exists()
    ]
    if not args.status_path.is_file():
        if existing_images:
            raise RuntimeError("existing images have no generation status ledger")
        return rows
    status_rows = baseline_generate.read_jsonl(args.status_path) if hasattr(
        baseline_generate, "read_jsonl"
    ) else [
        json.loads(line)
        for line in args.status_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    statuses = {row.get("id"): row for row in status_rows}
    if len(statuses) != len(status_rows):
        raise RuntimeError("generation status ledger contains duplicate ids")
    pending = []
    for row in rows:
        image_path = args.output_dir / row["image_path"]
        status = statuses.get(row["id"])
        if not image_path.exists():
            pending.append(row)
            continue
        if status is None or status.get("status") != "ok":
            raise RuntimeError(f"existing image lacks successful status: {row['id']}")
        if status.get("model_fingerprint_sha256") != row["model_fingerprint_sha256"]:
            raise RuntimeError(f"existing image model fingerprint mismatch: {row['id']}")
        if status.get("protocol_fingerprint_sha256") != row["protocol_fingerprint_sha256"]:
            raise RuntimeError(f"existing image protocol fingerprint mismatch: {row['id']}")
        if not baseline_generate.is_decodable_image(image_path):
            raise RuntimeError(f"existing image is invalid: {row['id']}")
        if status.get("image_sha256") != sha256_file(image_path):
            raise RuntimeError(f"existing image hash mismatch: {row['id']}")
    return pending


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--parent-training-run", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--status-path", type=Path)
    parser.add_argument("--provenance-path", type=Path)
    parser.add_argument("--evaluation-protocol", type=Path, required=True)
    parser.add_argument("--fold-id", choices=("A", "B", "C"), required=True)
    parser.add_argument("--training-seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument(
        "--pretrained-model-name-or-path",
        default="CompVis/stable-diffusion-v1-4",
    )
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--disable-safety-checker", action="store_true")
    parser.add_argument("--acknowledge-safety-risk", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.disable_safety_checker != args.acknowledge_safety_risk:
        parser.error(
            "--disable-safety-checker and --acknowledge-safety-risk must be used together"
        )
    if not args.dry_run and args.model_dir is None:
        parser.error("--model-dir is required unless --dry-run is used")
    if (args.model_dir is None) != (args.parent_training_run is None):
        parser.error("--model-dir and --parent-training-run must be used together")
    if args.manifest_path is None:
        args.manifest_path = args.output_dir / "generation_manifest.jsonl"
    if args.status_path is None:
        args.status_path = args.output_dir / "generation_status.jsonl"
    if args.provenance_path is None:
        args.provenance_path = args.output_dir / "generation_provenance.json"
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    protocol = read_evaluation_protocol(args.evaluation_protocol)
    protocol_fingerprint = sha256_file(args.evaluation_protocol)
    rows = [
        {
            **row,
            "safety_checker_disabled": bool(args.disable_safety_checker),
            "safety_risk_acknowledged": bool(args.acknowledge_safety_risk),
            "dtype": args.dtype,
            "protocol_fingerprint_sha256": protocol_fingerprint,
        }
        for row in build_manifest(
            protocol,
            fold_id=args.fold_id,
            training_seed=args.training_seed,
        )
    ]
    provenance = None
    if args.model_dir is not None:
        provenance = model_provenance(args.model_dir, args.parent_training_run)
        if provenance["parent_training_seed"] != args.training_seed:
            raise ValueError("training-seed does not match parent training manifest")
        expected_variant = f"multiview_v2_fold_{args.fold_id.lower()}_seed{args.training_seed}"
        if provenance["parent_training_variant"] != expected_variant:
            raise ValueError("fold-id/training-seed do not match parent training variant")
        rows = [
            {
                **row,
                "parent_training_run": provenance["parent_training_run"],
                "parent_training_variant": provenance["parent_training_variant"],
                "parent_training_git_commit": provenance["parent_training_git"]["commit"],
                "training_manifest_sha256": provenance["training_manifest_sha256"],
                "training_config_sha256": provenance["training_config_sha256"],
                "model_fingerprint_sha256": provenance["model_fingerprint_sha256"],
            }
            for row in rows
        ]
    baseline_generate.write_manifest(rows, args.manifest_path)
    if provenance is not None:
        args.provenance_path.parent.mkdir(parents=True, exist_ok=True)
        args.provenance_path.write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if not args.dry_run:
        generation_error = None
        try:
            pending = resume_pending_rows(rows, args)
            generation_args = copy.copy(args)
            generation_args.skip_existing = False
            baseline_generate.generate(pending, generation_args)
        except Exception as error:
            generation_error = f"{type(error).__name__}:{error}"
            raise
        finally:
            statuses = output_status(rows, args.output_dir, generation_error)
            write_jsonl(statuses, args.status_path)
        if any(status["status"] != "ok" for status in statuses):
            raise RuntimeError("generation completed with missing or invalid images")


if __name__ == "__main__":
    main()
