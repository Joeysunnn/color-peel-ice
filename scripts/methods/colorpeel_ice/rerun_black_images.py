"""Run an isolated three-stage diagnostic for effectively black generations.

Source outputs are read-only. Reruns always go to a distinct output directory.
Disabling the Stable Diffusion safety checker requires both explicit CLI flags.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


MODEL_ID = "CompVis/stable-diffusion-v1-4"
TOKENS = ("<s1*>", "<s2*>", "<s3*>", "<c1*>", "<c2*>", "<c3*>")
CUSTOM_DIFFUSION_WEIGHTS = "pytorch_custom_diffusion_weights.bin"
BLACK_MAX_U8 = 1
EXPECTED_SIZE = (512, 512)
STAGE_PROTOCOL = {
    "safety_flag": {"dtype": "float16", "disable_safety_checker": False},
    "disable_safety": {"dtype": "float16", "disable_safety_checker": True},
    "fp32_finite": {"dtype": "float32", "disable_safety_checker": True},
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def image_audit(path: Path) -> dict[str, Any]:
    with Image.open(path) as handle:
        image = np.asarray(handle.convert("RGB"), dtype=np.uint8)
    return {
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "min_u8": int(image.min()),
        "max_u8": int(image.max()),
        "mean_u8": float(image.mean()),
        "std_u8": float(image.std()),
        "is_black": bool(image.max() <= BLACK_MAX_U8),
    }


def select_black_items(
    manifest_rows: Iterable[dict[str, Any]],
    image_dir: Path,
    requested_ids: set[str] | None = None,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    manifest_rows = list(manifest_rows)
    manifest = {row.get("id"): row for row in manifest_rows}
    if None in manifest or len(manifest) != len(manifest_rows):
        raise ValueError("manifest ids must be present and unique")
    if requested_ids is not None:
        missing = sorted(requested_ids - set(manifest))
        if missing:
            raise ValueError("requested ids absent from manifest: " + ", ".join(missing))
    selected = []
    for item_id, item in manifest.items():
        if requested_ids is not None and item_id not in requested_ids:
            continue
        image_path = image_dir / Path(str(item.get("image_path", "")))
        if not image_path.is_file():
            raise FileNotFoundError(f"source image missing: {image_path}")
        audit = image_audit(image_path)
        if requested_ids is not None and not audit["is_black"]:
            raise ValueError(f"requested image is not black: {item_id}")
        if audit["is_black"]:
            selected.append((item, audit))
    return selected


def continuing_ids(prior_status_path: Path) -> set[str]:
    rows = read_jsonl(prior_status_path)
    if not rows:
        raise ValueError(f"prior status is empty: {prior_status_path}")
    ids = set()
    for row in rows:
        item_id = row.get("id")
        output_audit = row.get("output_audit")
        if (
            isinstance(item_id, str)
            and isinstance(output_audit, dict)
            and output_audit.get("is_black") is True
        ):
            ids.add(item_id)
    if not ids:
        raise ValueError("prior stage has no still-black ids to continue")
    return ids


def _tensor_values(value: Any) -> Iterable[Any]:
    if isinstance(value, dict):
        for nested in value.values():
            yield from _tensor_values(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _tensor_values(nested)
    elif hasattr(value, "is_floating_point"):
        yield value


def audit_checkpoint_finite(model_dir: Path, torch: Any) -> dict[str, Any]:
    paths = [model_dir / CUSTOM_DIFFUSION_WEIGHTS]
    paths.extend(model_dir / f"{token}.bin" for token in TOKENS)
    result = {"files": [], "tensor_count": 0, "finite": True}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"trained weight missing: {path}")
        payload = torch.load(path, map_location="cpu")
        tensors = list(_tensor_values(payload))
        file_finite = all(
            not tensor.is_floating_point() or bool(torch.isfinite(tensor).all())
            for tensor in tensors
        )
        result["files"].append(
            {"path": str(path), "tensor_count": len(tensors), "finite": file_finite}
        )
        result["tensor_count"] += len(tensors)
        result["finite"] = result["finite"] and file_finite
    return result


def load_pipeline(
    args: argparse.Namespace, finite_audit: dict[str, Any] | None = None
) -> tuple[Any, dict[str, Any]]:
    import torch
    from diffusers import DiffusionPipeline

    model_dir = args.model_dir.resolve()
    if args.diagnostic_stage == "fp32_finite":
        finite_audit = finite_audit or audit_checkpoint_finite(model_dir, torch)
        finite_audit["status"] = "completed"
        if not finite_audit["finite"]:
            raise FloatingPointError("checkpoint contains non-finite learned tensors")
    else:
        finite_audit = {
            "status": "not_run",
            "reason": "checkpoint finite audit is reserved for fp32_finite stage",
        }
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    pipe = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        low_cpu_mem_usage=False,
        torch_dtype=dtype,
        local_files_only=True,
    ).to(args.device)
    if args.diagnostic_stage == "safety_flag" and getattr(pipe, "safety_checker", None) is None:
        raise RuntimeError("safety_flag stage requires the original safety checker")
    pipe.unet.load_attn_procs(
        str(model_dir), weight_name=CUSTOM_DIFFUSION_WEIGHTS
    )
    for token in TOKENS:
        pipe.load_textual_inversion(str(model_dir), weight_name=f"{token}.bin")
    if args.disable_safety_checker:
        pipe.safety_checker = None
        if hasattr(pipe, "requires_safety_checker"):
            pipe.requires_safety_checker = False
    return pipe, finite_audit


def rerun(
    selected: Iterable[tuple[dict[str, Any], dict[str, Any]]],
    pipe: Any,
    args: argparse.Namespace,
    finite_audit: dict[str, Any],
) -> list[dict[str, Any]]:
    import torch

    statuses = []
    for item, source_audit in selected:
        output_path = args.output_dir / Path(item["image_path"])
        status = {
            "id": item["id"],
            "prompt": item["prompt"],
            "seed": item["seed"],
            "source_image_path": str(args.image_dir / Path(item["image_path"])),
            "output_image_path": str(output_path),
            "source_audit": source_audit,
            "checkpoint_finite_audit": finite_audit,
            "diagnostic_stage": args.diagnostic_stage,
            "torch_dtype": args.dtype,
            "safety_checker_disabled": bool(args.disable_safety_checker),
            "safety_risk_acknowledged": bool(args.acknowledge_safety_risk),
            "status": None,
            "failure_reason": None,
            "nsfw_content_detected": None,
            "output_audit": None,
        }
        try:
            generator = torch.Generator(device=args.device).manual_seed(item["seed"])
            result = pipe(
                item["prompt"],
                num_inference_steps=item["num_inference_steps"],
                guidance_scale=item["guidance_scale"],
                generator=generator,
                output_type="np",
            )
            pixels = np.asarray(result.images[0], dtype=np.float32)
            if pixels.shape != (EXPECTED_SIZE[1], EXPECTED_SIZE[0], 3):
                raise ValueError(f"unexpected output shape: {pixels.shape}")
            if not np.isfinite(pixels).all():
                raise FloatingPointError("generated FP32 pixels are not finite")
            clipped = np.clip(np.rint(pixels * 255.0), 0, 255).astype(np.uint8)
            output_audit = {
                "finite": True,
                "min_float": float(pixels.min()),
                "max_float": float(pixels.max()),
                "mean_float": float(pixels.mean()),
                "std_float": float(pixels.std()),
                "is_black": bool(clipped.max() <= BLACK_MAX_U8),
            }
            status["output_audit"] = output_audit
            detected = getattr(result, "nsfw_content_detected", None)
            status["nsfw_content_detected"] = (
                bool(detected[0]) if isinstance(detected, (list, tuple)) and detected else None
            )
            if output_audit["is_black"]:
                status.update(status="failure", failure_reason="rerun_still_black")
            else:
                status["status"] = "ok"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(clipped, mode="RGB").save(output_path)
        except Exception as error:
            status.update(
                status="failure",
                failure_reason=f"rerun_error:{type(error).__name__}:{error}",
            )
        statuses.append(status)
    return statuses


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--status-path", type=Path)
    parser.add_argument("--ids", nargs="+")
    parser.add_argument(
        "--diagnostic-stage", choices=tuple(STAGE_PROTOCOL), required=True
    )
    parser.add_argument("--prior-status", type=Path)
    parser.add_argument("--dtype", choices=("float16", "float32"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pretrained-model-name-or-path", default=MODEL_ID)
    parser.add_argument("--disable-safety-checker", action="store_true")
    parser.add_argument("--acknowledge-safety-risk", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    expected = STAGE_PROTOCOL[args.diagnostic_stage]
    if args.dtype != expected["dtype"]:
        parser.error(
            f"{args.diagnostic_stage} requires --dtype {expected['dtype']}"
        )
    if args.disable_safety_checker != expected["disable_safety_checker"]:
        parser.error(
            f"{args.diagnostic_stage} requires disable_safety_checker="
            f"{expected['disable_safety_checker']}"
        )
    if args.disable_safety_checker and not args.acknowledge_safety_risk:
        parser.error("disabling the safety checker requires --acknowledge-safety-risk")
    if not args.disable_safety_checker and args.acknowledge_safety_risk:
        parser.error("--acknowledge-safety-risk is only valid when disabling the checker")
    if args.diagnostic_stage == "safety_flag" and args.prior_status is not None:
        parser.error("safety_flag is the first stage and does not accept --prior-status")
    if args.diagnostic_stage != "safety_flag" and args.prior_status is None:
        parser.error(f"{args.diagnostic_stage} requires --prior-status")
    if not args.dry_run and args.model_dir is None:
        parser.error("--model-dir is required unless --dry-run is used")
    if args.status_path is None:
        args.status_path = args.output_dir / "rerun_status.jsonl"
    if args.image_dir.resolve() == args.output_dir.resolve():
        parser.error("--output-dir must differ from the source --image-dir")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    requested_ids = set(args.ids) if args.ids else None
    if args.prior_status is not None:
        prior_ids = continuing_ids(args.prior_status)
        requested_ids = prior_ids if requested_ids is None else requested_ids & prior_ids
        if not requested_ids:
            raise ValueError("no requested ids remain black in the prior stage")
    selected = select_black_items(
        read_jsonl(args.manifest),
        args.image_dir,
        requested_ids,
    )
    if args.dry_run or not selected:
        write_jsonl(
            (
                {
                    "id": item["id"],
                    "status": "planned",
                    "source_audit": audit,
                    "safety_checker_disabled": bool(args.disable_safety_checker),
                    "diagnostic_stage": args.diagnostic_stage,
                    "torch_dtype": args.dtype,
                }
                for item, audit in selected
            ),
            args.status_path,
        )
        return 0
    finite_audit = None
    try:
        if args.diagnostic_stage == "fp32_finite":
            import torch

            finite_audit = audit_checkpoint_finite(args.model_dir.resolve(), torch)
            finite_audit["status"] = "completed"
            if not finite_audit["finite"]:
                raise FloatingPointError("checkpoint contains non-finite learned tensors")
        pipe, finite_audit = load_pipeline(args, finite_audit)
    except Exception as error:
        setup_failure = [
            {
                "id": item["id"],
                "diagnostic_stage": args.diagnostic_stage,
                "torch_dtype": args.dtype,
                "safety_checker_disabled": bool(args.disable_safety_checker),
                "status": "failure",
                "failure_reason": f"pipeline_setup_error:{type(error).__name__}:{error}",
                "source_audit": source_audit,
                "checkpoint_finite_audit": finite_audit,
            }
            for item, source_audit in selected
        ]
        write_jsonl(setup_failure, args.status_path)
        return 1
    statuses = rerun(selected, pipe, args, finite_audit)
    write_jsonl(statuses, args.status_path)
    return 0 if all(row["status"] == "ok" for row in statuses) else 1


if __name__ == "__main__":
    raise SystemExit(main())
