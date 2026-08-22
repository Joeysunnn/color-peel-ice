#!/usr/bin/env python3
"""Validate and stage the fixed ColorPeel-on-CLEVR 3x3 training set.

GT masks are read only for dataset validation and audit evidence.  The staged
concept directories intentionally contain only ``img.jpg`` so ColorPeel's
directory-based training dataset cannot consume masks or scene metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Iterable

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "experiments"
    / "clevr_subject_color_3x3"
    / "manifests"
    / "clevr_3x3_manifest.json"
)
IMAGE_SIZE = (512, 512)
DATASET_SHAPES = ("cube", "sphere", "cylinder")
DATASET_COLORS = ("gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow")
DATASET_MATERIALS = ("rubber", "metal")
SAMPLE_DIR_PATTERN = re.compile(r"^\d{3}_(cube|sphere|cylinder)_[a-z]+_(rubber|metal)$")


class ValidationError(RuntimeError):
    """Raised when the source dataset does not match the locked protocol."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"Cannot read valid JSON from {path}: {exc}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValidationError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
                if not isinstance(value, dict):
                    raise ValidationError(f"Expected an object at {path}:{line_number}")
                records.append(value)
    except OSError as exc:
        raise ValidationError(f"Cannot read {path}: {exc}") from exc
    return records


def load_experiment_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    manifest = _read_json(path)
    if not isinstance(manifest, dict):
        raise ValidationError(f"Experiment manifest must be a JSON object: {path}")
    return manifest


def expected_dataset_samples() -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    index = 0
    for shape in DATASET_SHAPES:
        for color in DATASET_COLORS:
            for material in DATASET_MATERIALS:
                samples.append(
                    {
                        "index": index,
                        "id": f"{index:03d}_{shape}_{color}_{material}",
                        "shape": shape,
                        "color": color,
                        "material": material,
                    }
                )
                index += 1
    return samples


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _validate_experiment_manifest(manifest: dict[str, Any]) -> None:
    shapes = manifest.get("shapes", [])
    colors = manifest.get("colors", [])
    samples = manifest.get("samples", [])
    _require([entry.get("name") for entry in shapes] == list(DATASET_SHAPES), "Shape axis must be cube/sphere/cylinder")
    _require([entry.get("name") for entry in colors] == ["red", "cyan", "gray"], "Color axis must be red/cyan/gray")
    _require(manifest.get("material") == "metal", "Training material must be metal")
    _require(len(samples) == 9, "Experiment manifest must contain exactly nine samples")

    shape_map = {entry["name"]: entry for entry in shapes}
    color_map = {entry["name"]: entry for entry in colors}
    expected_pairs = {(shape, color) for shape in shape_map for color in color_map}
    actual_pairs = {(entry.get("shape"), entry.get("color")) for entry in samples}
    _require(actual_pairs == expected_pairs, "Experiment samples must be the complete 3x3 Cartesian product")

    for sample in samples:
        shape = sample["shape"]
        color = sample["color"]
        subject_token = shape_map[shape]["token"]
        color_token = color_map[color]["token"]
        expected_prompt = manifest["prompt_template"].format(
            subject_token=subject_token,
            color_token=color_token,
        )
        _require(sample.get("subject_token") == subject_token, f"Wrong subject token for {sample.get('id')}")
        _require(sample.get("color_token") == color_token, f"Wrong color token for {sample.get('id')}")
        _require(sample.get("rgb") == color_map[color]["rgb"], f"Wrong RGB for {sample.get('id')}")
        _require(sample.get("material") == "metal", f"Wrong material for {sample.get('id')}")
        _require(sample.get("instance_prompt") == [expected_prompt], f"Wrong prompt for {sample.get('id')}")


def _validate_image(path: Path, expected_mode: str) -> None:
    _require(path.is_file(), f"Missing required image: {path}")
    try:
        with Image.open(path) as image:
            image.load()
            _require(image.size == IMAGE_SIZE, f"Expected 512x512 image at {path}, got {image.size}")
            _require(image.mode == expected_mode, f"Expected {expected_mode} image at {path}, got {image.mode}")
    except (OSError, ValueError) as exc:
        raise ValidationError(f"Cannot decode image {path}: {exc}") from exc


def _validate_mask(path: Path, expected_foreground: int) -> tuple[list[int], int]:
    _validate_image(path, "L")
    with Image.open(path) as mask:
        histogram = mask.histogram()
        values = [value for value, count in enumerate(histogram) if count]
        foreground = histogram[255]
    _require(values == [0, 255], f"Mask must contain exactly values [0, 255]: {path} has {values}")
    _require(
        foreground == expected_foreground,
        f"Wrong foreground pixel count for {path}: expected {expected_foreground}, got {foreground}",
    )
    return values, foreground


def _manifest_records_by_id(dataset_root: Path) -> dict[str, dict[str, Any]]:
    records = _read_jsonl(dataset_root / "manifest.jsonl")
    _require(len(records) == 48, f"Expected 48 manifest records, found {len(records)}")
    records_by_id = {record.get("id"): record for record in records}
    _require(len(records_by_id) == 48 and None not in records_by_id, "Dataset manifest IDs must be present and unique")
    return records_by_id


def validate_dataset(dataset_root: Path, experiment_manifest: dict[str, Any]) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    _require(dataset_root.is_dir(), f"Dataset root is not a directory: {dataset_root}")
    _validate_experiment_manifest(experiment_manifest)

    metadata = _read_json(dataset_root / "metadata.json")
    _require(metadata.get("sample_count") == 48, "metadata.json sample_count must be 48")
    _require(metadata.get("resolution") == [512, 512], "metadata.json resolution must be [512, 512]")

    expected = expected_dataset_samples()
    expected_by_id = {sample["id"]: sample for sample in expected}
    actual_sample_dirs = {
        path.name: path
        for path in dataset_root.iterdir()
        if path.is_dir() and SAMPLE_DIR_PATTERN.fullmatch(path.name)
    }
    _require(
        set(actual_sample_dirs) == set(expected_by_id),
        "Dataset sample directories do not match the fixed 3 shapes x 8 colors x 2 materials inventory",
    )

    records_by_id = _manifest_records_by_id(dataset_root)
    _require(set(records_by_id) == set(expected_by_id), "manifest.jsonl IDs do not match the 48 sample directories")
    selected_by_id = {sample["id"]: sample for sample in experiment_manifest["samples"]}
    foreground_by_shape = {
        entry["name"]: entry["foreground_pixels"] for entry in experiment_manifest["shapes"]
    }

    sample_audits: list[dict[str, Any]] = []
    for sample in expected:
        sample_id = sample["id"]
        sample_dir = actual_sample_dirs[sample_id]
        image_path = sample_dir / "img.jpg"
        scene_path = sample_dir / "scene.json"
        expected_mask_path = sample_dir / f"mask_{sample['shape']}_0.png"
        masks = sorted(sample_dir.glob("mask_*.png"))
        _require(masks == [expected_mask_path], f"Expected exactly one GT mask for {sample_id}: {expected_mask_path.name}")

        _validate_image(image_path, "RGB")
        scene = _read_json(scene_path)
        objects = scene.get("objects", [])
        _require(scene.get("image_index") == sample["index"], f"Wrong scene image_index for {sample_id}")
        _require(scene.get("image_filename") == "img.jpg", f"Wrong scene image_filename for {sample_id}")
        _require(len(objects) == 1, f"Expected exactly one scene object for {sample_id}")
        for field in ("shape", "color", "material"):
            _require(objects[0].get(field) == sample[field], f"Wrong scene {field} for {sample_id}")

        dataset_record = records_by_id[sample_id]
        for field in ("index", "id", "shape", "color", "material"):
            _require(dataset_record.get(field) == sample[field], f"Wrong manifest {field} for {sample_id}")
        _require(dataset_record.get("image_path") == f"{sample_id}/img.jpg", f"Wrong image_path for {sample_id}")
        _require(dataset_record.get("scene_path") == f"{sample_id}/scene.json", f"Wrong scene_path for {sample_id}")

        selected = selected_by_id.get(sample_id)
        if selected is not None:
            _require(dataset_record.get("rgb") == selected["rgb"], f"Wrong manifest RGB for {sample_id}")

        values, foreground = _validate_mask(expected_mask_path, foreground_by_shape[sample["shape"]])
        sample_audits.append(
            {
                "id": sample_id,
                "selected_for_training": selected is not None,
                "image_sha256": _sha256(image_path),
                "scene_sha256": _sha256(scene_path),
                "mask_file": expected_mask_path.name,
                "mask_sha256": _sha256(expected_mask_path),
                "mask_values": values,
                "mask_foreground_pixels": foreground,
            }
        )

    return {
        "dataset_root": str(dataset_root),
        "sample_count": len(sample_audits),
        "image_count": len(sample_audits),
        "gt_mask_count": len(sample_audits),
        "selected_sample_count": len(selected_by_id),
        "training_uses_gt_masks": False,
        "samples": sample_audits,
    }


def _stage_image(source: Path, destination: Path) -> str:
    if destination.exists() or destination.is_symlink():
        _require(destination.is_file(), f"Staging destination is not a file: {destination}")
        _require(_sha256(destination) == _sha256(source), f"Existing staged image differs from source: {destination}")
        return "existing_symlink" if destination.is_symlink() else "existing_file"

    try:
        os.symlink(source.resolve(), destination)
        return "symlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy_fallback"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stage_training_data(
    dataset_root: Path,
    output_dir: Path,
    experiment_manifest: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    concepts: list[dict[str, Any]] = []
    staging_methods: dict[str, str] = {}
    for sample in experiment_manifest["samples"]:
        sample_id = sample["id"]
        sample_dir = output_dir / sample_id
        sample_dir.mkdir(exist_ok=True)
        unexpected = [path.name for path in sample_dir.iterdir() if path.name != "img.jpg"]
        _require(not unexpected, f"Staging directory must contain only img.jpg: {sample_dir} has {unexpected}")
        staging_methods[sample_id] = _stage_image(dataset_root / sample_id / "img.jpg", sample_dir / "img.jpg")
        concepts.append(
            {
                "instance_prompt": sample["instance_prompt"],
                "instance_data_dir": str(sample_dir),
            }
        )

    concepts_path = output_dir / "concepts.json"
    audit_path = output_dir / "dataset_audit.json"
    audit = dict(audit)
    audit["staging_root"] = str(output_dir)
    audit["staging_methods"] = staging_methods
    audit["training_files"] = [f"{sample['id']}/img.jpg" for sample in experiment_manifest["samples"]]
    _write_json(concepts_path, concepts)
    _write_json(audit_path, audit)
    return {
        "output_dir": str(output_dir),
        "concepts_json": str(concepts_path),
        "audit_json": str(audit_path),
        "staging_methods": staging_methods,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True, help="Root of clevr_basic_neutral_stage1_gt")
    parser.add_argument("--output-dir", type=Path, required=True, help="Isolated ColorPeel training staging directory")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Locked experiment manifest (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the audit summary without writing staging files")
    return parser


def main(argv: Iterable[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    experiment_manifest = load_experiment_manifest(args.manifest)
    audit = validate_dataset(args.dataset_root, experiment_manifest)
    if args.dry_run:
        result = {
            "status": "validated",
            "dry_run": True,
            "output_dir": str(args.output_dir.resolve()),
            "sample_count": audit["sample_count"],
            "selected_sample_count": audit["selected_sample_count"],
            "training_uses_gt_masks": False,
        }
    else:
        result = {"status": "staged", "dry_run": False, **stage_training_data(args.dataset_root, args.output_dir, experiment_manifest, audit)}
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
