#!/usr/bin/env python3
"""Plan and validate the locked CLEVR 3x3 multiview held-out protocol.

The ``plan`` command emits render requests only.  Camera, lighting, background,
scene and file fields remain null until a real renderer supplies a realization
manifest.  The ``realize`` command validates all 180 rendered views and builds
separate, image-only ColorPeel assets for folds A/B/C.  It never modifies the
single-view baseline staging and never places GT masks in training directories.
"""

from __future__ import annotations

import argparse
import filecmp
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable

from PIL import Image, ImageStat

try:
    import yaml
except ImportError:  # pragma: no cover - only needed while realizing train configs
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_DIR = REPO_ROOT / "experiments" / "clevr_subject_color_3x3" / "manifests"
DEFAULT_BASE_MANIFEST = MANIFEST_DIR / "clevr_3x3_manifest.json"
DEFAULT_PROTOCOL = MANIFEST_DIR / "clevr_multiview_protocol.json"
DEFAULT_BASE_CONFIG = (
    REPO_ROOT
    / "experiments"
    / "clevr_subject_color_3x3"
    / "configs"
    / "multiview_base_turquoise.yaml"
)
RENDERER_FIELDS = ("camera", "light", "background", "scene_json", "image", "mask")
EXPECTED_MODIFIER_TOKEN = "<s1*>+<s2*>+<s3*>+<c1*>+<c2*>+<c3*>"
EXPECTED_INITIALIZER_TOKEN = "cube+sphere+cylinder+red+turquoise+gray"
EXPECTED_FOLDS = {
    "A": {("cube", "red"), ("sphere", "cyan"), ("cylinder", "gray")},
    "B": {("cube", "cyan"), ("sphere", "gray"), ("cylinder", "red")},
    "C": {("cube", "gray"), ("sphere", "red"), ("cylinder", "cyan")},
}
TRAINING_SEEDS = (42, 43, 44)


class ProtocolError(RuntimeError):
    """Raised when the locked protocol or a renderer realization is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolError(message)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read valid JSON from {path}: {exc}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                _require(isinstance(value, dict), f"Expected an object at {path}:{line_number}")
                records.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read valid JSONL from {path}: {exc}") from exc
    return records


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")


def load_inputs(base_manifest_path: Path, protocol_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    base_manifest = _read_json(base_manifest_path)
    protocol = _read_json(protocol_path)
    _require(isinstance(base_manifest, dict), "Base manifest must be an object")
    _require(isinstance(protocol, dict), "Protocol manifest must be an object")
    validate_protocol(base_manifest, protocol)
    return base_manifest, protocol


def validate_protocol(base_manifest: dict[str, Any], protocol: dict[str, Any]) -> None:
    _require(protocol.get("$schema") == "clevr_multiview_protocol.schema.json", "Protocol schema changed")
    _require(protocol.get("version") == 1, "Protocol version must be 1")
    _require(
        protocol.get("protocol_id") == "clevr_subject_color_3x3_multiview_v1",
        "Protocol ID changed",
    )
    _require(protocol.get("base_manifest") == "clevr_3x3_manifest.json", "Protocol base manifest changed")
    samples = base_manifest.get("samples", [])
    _require(len(samples) == 9, "Base manifest must contain exactly nine cells")
    _require(len({sample.get("id") for sample in samples}) == 9, "Cell IDs must be unique")
    expected_grid = {
        (shape, color)
        for shape in ("cube", "sphere", "cylinder")
        for color in ("red", "cyan", "gray")
    }
    actual_grid = {(sample.get("shape"), sample.get("color")) for sample in samples}
    _require(actual_grid == expected_grid, "Base manifest must be the locked 3x3 Cartesian grid")
    _require(all(sample.get("material") == "metal" for sample in samples), "All cells must use metal")

    _require(protocol.get("views_per_cell") == 20, "views_per_cell must be 20")
    _require(protocol.get("view_splits") == {
        "train": {"start": 0, "stop": 16},
        "audit": {"start": 16, "stop": 20},
    }, "View split must be train 0:16 and audit 16:20")
    _require(protocol.get("render_seed") == {
        "base": 420000,
        "cell_stride": 100,
        "formula": "base + cell_index * cell_stride + view_index",
    }, "Render seed rule differs from the locked protocol")

    folds = protocol.get("folds", [])
    actual_folds = {
        fold.get("id"): {tuple(pair) for pair in fold.get("held_out", [])}
        for fold in folds
    }
    _require(actual_folds == EXPECTED_FOLDS, "Held-out folds A/B/C differ from the locked matchings")
    _require(len(folds) == 3, "Protocol must define exactly three folds")
    for fold_id, held_out in actual_folds.items():
        train_cells = expected_grid - held_out
        for shape in ("cube", "sphere", "cylinder"):
            partners = {color for candidate_shape, color in train_cells if candidate_shape == shape}
            _require(len(partners) == 2, f"Fold {fold_id} leaks subject axis: {shape} has {len(partners)} partners")
        for color in ("red", "cyan", "gray"):
            partners = {shape for shape, candidate_color in train_cells if candidate_color == color}
            _require(len(partners) == 2, f"Fold {fold_id} leaks color axis: {color} has {len(partners)} partners")

    contract = protocol.get("realization_contract", {})
    _require(contract.get("renderer_owned_fields") == list(RENDERER_FIELDS), "Renderer-owned fields changed")
    _require(contract.get("resolution") == [512, 512], "Realized views must be 512x512")
    _require(contract.get("image_mode") == "RGB", "Realized image mode must be RGB")
    _require(contract.get("mask_mode") == "L", "Realized mask mode must be L")
    _require(contract.get("mask_values") == [0, 255], "Realized mask values must be [0, 255]")
    _require(contract.get("empirical_rgb") == {
        "space": "srgb_u8",
        "statistic": "masked_mean",
        "source": "realized_view_gt_mask",
    }, "Empirical RGB contract changed")


def build_render_requests(base_manifest: dict[str, Any], protocol: dict[str, Any]) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for cell_index, cell in enumerate(base_manifest["samples"]):
        for view_index in range(20):
            split = "train" if view_index < 16 else "audit"
            request = {
                "cell_id": cell["id"],
                "cell_index": cell_index,
                "shape": cell["shape"],
                "color": cell["color"],
                "material": "metal",
                "subject_token": cell["subject_token"],
                "color_token": cell["color_token"],
                "nominal_rgb": cell["rgb"],
                "view_index": view_index,
                "split": split,
                "render_seed": 420000 + cell_index * 100 + view_index,
                "camera": None,
                "light": None,
                "background": None,
                "scene_json": None,
                "image": None,
                "mask": None,
                "empirical_rgb": None,
            }
            requests.append(request)
    validate_render_requests(requests, base_manifest, protocol)
    return requests


def validate_render_requests(
    requests: list[dict[str, Any]],
    base_manifest: dict[str, Any],
    protocol: dict[str, Any],
) -> None:
    _require(len(requests) == 180, f"Expected 180 render requests, got {len(requests)}")
    keys = {(record.get("cell_id"), record.get("view_index")) for record in requests}
    _require(len(keys) == 180, "Each cell/view pair must be unique")
    cells = {sample["id"]: (index, sample) for index, sample in enumerate(base_manifest["samples"])}
    for record in requests:
        cell_id = record.get("cell_id")
        _require(cell_id in cells, f"Unknown cell_id: {cell_id}")
        cell_index, cell = cells[cell_id]
        view_index = record.get("view_index")
        _require(isinstance(view_index, int) and 0 <= view_index < 20, f"Invalid view_index for {cell_id}")
        expected_split = "train" if view_index < 16 else "audit"
        _require(record.get("split") == expected_split, f"Wrong split for {cell_id} view {view_index}")
        _require(record.get("cell_index") == cell_index, f"Wrong cell_index for {cell_id}")
        _require(record.get("render_seed") == 420000 + cell_index * 100 + view_index,
                 f"Wrong render_seed for {cell_id} view {view_index}")
        for field in ("shape", "color", "subject_token", "color_token"):
            _require(record.get(field) == cell[field], f"Wrong {field} for {cell_id}")
        _require(record.get("nominal_rgb") == cell["rgb"], f"Wrong nominal_rgb for {cell_id}")
        _require(record.get("material") == "metal", f"Wrong material for {cell_id}")
        for field in RENDERER_FIELDS:
            _require(record.get(field) is None, f"Protocol generator must not fabricate {field}")
        _require(record.get("empirical_rgb") is None, "Protocol generator must not fabricate empirical_rgb")

    for cell_id in cells:
        records = [record for record in requests if record["cell_id"] == cell_id]
        _require(sum(record["split"] == "train" for record in records) == 16, f"{cell_id} needs 16 train views")
        _require(sum(record["split"] == "audit" for record in records) == 4, f"{cell_id} needs 4 audit views")


def plan_protocol(
    base_manifest: dict[str, Any],
    protocol: dict[str, Any],
    output_dir: Path,
    renderer: Path | None,
) -> dict[str, Any]:
    requests = build_render_requests(base_manifest, protocol)
    output_dir = output_dir.resolve()
    _require_empty_output_dir(output_dir)
    requests_path = output_dir / "render_requests.jsonl"
    _write_jsonl(requests_path, requests)
    renderer_path = renderer.resolve() if renderer is not None else None
    renderer_available = renderer_path is not None and renderer_path.is_file()
    status = {
        "status": "planned" if renderer_available else "blocked",
        "blocked_reason": None if renderer_available else "multiview_renderer_not_provided_or_missing",
        "renderer": str(renderer_path) if renderer_path is not None else None,
        "renderer_available": renderer_available,
        "render_request_manifest": str(requests_path),
        "request_count": 180,
        "images_created": 0,
        "renderer_owned_fields_populated": False,
    }
    _write_json(output_dir / "protocol_status.json", status)
    return status


def _resolved_under(root: Path, relative: Any, field: str) -> Path:
    _require(isinstance(relative, str) and relative, f"Realized {field} must be a nonempty relative path")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ProtocolError(f"Realized {field} escapes render root: {relative}") from exc
    _require(candidate.is_file(), f"Missing realized {field}: {candidate}")
    return candidate


def _require_empty_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        _require(output_dir.is_dir(), f"Output path is not a directory: {output_dir}")
        _require(not any(output_dir.iterdir()), f"Output directory must be empty: {output_dir}")
    else:
        output_dir.mkdir(parents=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_key(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _validate_realized_image(image_path: Path, mask_path: Path) -> tuple[list[float], int]:
    try:
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            image.load()
            mask.load()
            _require(image.size == (512, 512) and image.mode == "RGB", f"Invalid RGB image: {image_path}")
            _require(mask.size == (512, 512) and mask.mode == "L", f"Invalid L mask: {mask_path}")
            histogram = mask.histogram()
            values = [value for value, count in enumerate(histogram) if count]
            _require(values == [0, 255], f"Mask must contain exactly [0, 255]: {mask_path}")
            foreground = histogram[255]
            _require(foreground > 0, f"Mask is empty: {mask_path}")
            mean = [round(value, 6) for value in ImageStat.Stat(image, mask=mask).mean]
            return mean, foreground
    except OSError as exc:
        raise ProtocolError(f"Cannot decode realized image or mask: {exc}") from exc


def validate_realization(
    render_root: Path,
    realized_records: list[dict[str, Any]],
    base_manifest: dict[str, Any],
    protocol: dict[str, Any],
) -> list[dict[str, Any]]:
    render_root = render_root.resolve()
    _require(render_root.is_dir(), f"Render root is not a directory: {render_root}")
    expected = build_render_requests(base_manifest, protocol)
    expected_by_key = {(record["cell_id"], record["view_index"]): record for record in expected}
    _require(len(realized_records) == 180, f"Expected 180 realized views, got {len(realized_records)}")
    actual_by_key = {(record.get("cell_id"), record.get("view_index")): record for record in realized_records}
    _require(len(actual_by_key) == 180 and set(actual_by_key) == set(expected_by_key),
             "Realization must contain every cell/view exactly once")

    realized: list[dict[str, Any]] = []
    used_paths: dict[str, set[Path]] = {field: set() for field in ("scene_json", "image", "mask")}
    image_hashes: dict[str, set[str]] = {sample["id"]: set() for sample in base_manifest["samples"]}
    metadata_values: dict[str, dict[str, set[str]]] = {
        sample["id"]: {field: set() for field in ("camera", "light", "background")}
        for sample in base_manifest["samples"]
    }
    for expected_record in expected:
        key = (expected_record["cell_id"], expected_record["view_index"])
        supplied = actual_by_key[key]
        for field in (
            "cell_index", "shape", "color", "material", "subject_token", "color_token",
            "nominal_rgb", "split", "render_seed",
        ):
            _require(supplied.get(field) == expected_record[field], f"Realization changed {field} for {key}")
        for field in ("camera", "light", "background"):
            _require(isinstance(supplied.get(field), dict) and supplied[field],
                     f"Renderer must populate nonempty {field} metadata for {key}")
            metadata_values[expected_record["cell_id"]][field].add(_metadata_key(supplied[field]))

        paths = {field: _resolved_under(render_root, supplied.get(field), field)
                 for field in ("scene_json", "image", "mask")}
        for field, path in paths.items():
            _require(path not in used_paths[field], f"Realized {field} is reused: {path}")
            used_paths[field].add(path)
        scene = _read_json(paths["scene_json"])
        _require(isinstance(scene, dict), f"Scene JSON must be an object: {paths['scene_json']}")
        for field in ("render_seed", "camera", "light", "background"):
            _require(scene.get(field) == supplied[field], f"Scene {field} disagrees with realization for {key}")
        objects = scene.get("objects", [])
        _require(len(objects) == 1, f"Scene must contain one object for {key}")
        for field in ("shape", "color", "material"):
            _require(objects[0].get(field) == expected_record[field], f"Scene {field} disagrees for {key}")

        mean, foreground = _validate_realized_image(paths["image"], paths["mask"])
        image_hashes[expected_record["cell_id"]].add(_file_sha256(paths["image"]))
        record = {**expected_record, **supplied}
        record["empirical_rgb"] = {
            "value": mean,
            "space": "srgb_u8",
            "statistic": "masked_mean",
            "source": "realized_view_gt_mask",
            "source_image": supplied["image"],
            "source_mask": supplied["mask"],
            "foreground_pixels": foreground,
        }
        realized.append(record)
    for cell_id, hashes in image_hashes.items():
        _require(len(hashes) == 20, f"Cell {cell_id} must contain 20 distinct rendered images")
        for field, values in metadata_values[cell_id].items():
            _require(len(values) > 1, f"Cell {cell_id} has no realized {field} variation")
    return realized


def _stage_image(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        same_target = destination.is_symlink() and destination.resolve() == source.resolve()
        same_content = destination.is_file() and filecmp.cmp(destination, source, shallow=False)
        _require(same_target or same_content, f"Existing staged file differs: {destination}")
        return
    try:
        os.symlink(source.resolve(), destination)
    except OSError:
        shutil.copy2(source, destination)


def _load_training_config(path: Path) -> dict[str, Any]:
    _require(yaml is not None, "PyYAML is required to derive executable fold training configs")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ProtocolError(f"Cannot read baseline training config: {path}") from exc
    _require(isinstance(value, dict) and value.get("stage") == "train", "Baseline config must be a train mapping")
    args = value.get("args", {})
    _require(args.get("modifier_token") == EXPECTED_MODIFIER_TOKEN, "Multiview modifier token mapping changed")
    _require(
        args.get("initializer_token") == EXPECTED_INITIALIZER_TOKEN,
        "Multiview base config must use the selected single-token turquoise initializer",
    )
    return value


def build_fold_outputs(
    render_root: Path,
    realized: list[dict[str, Any]],
    base_manifest: dict[str, Any],
    protocol: dict[str, Any],
    output_dir: Path,
    base_config_path: Path,
) -> list[dict[str, Any]]:
    render_root = render_root.resolve()
    base_config = _load_training_config(base_config_path)
    cells_by_pair = {(cell["shape"], cell["color"]): cell for cell in base_manifest["samples"]}
    fold_summaries: list[dict[str, Any]] = []
    realized_manifest_path = output_dir.resolve() / "realized_views.jsonl"
    for fold in protocol["folds"]:
        fold_id = fold["id"]
        held_out = {tuple(pair) for pair in fold["held_out"]}
        train_cells = set(cells_by_pair) - held_out
        train_records = [
            record for record in realized
            if (record["shape"], record["color"]) in train_cells and record["split"] == "train"
        ]
        _require(len(train_records) == 96, f"Fold {fold_id} must contain 96 training views")
        _require(not any((record["shape"], record["color"]) in held_out for record in train_records),
                 f"Fold {fold_id} contains a held-out cell")
        _require(not any(record["split"] == "audit" for record in train_records),
                 f"Fold {fold_id} contains audit views")

        fold_dir = output_dir.resolve() / "folds" / f"fold_{fold_id.lower()}"
        assets_dir = fold_dir / "train_assets"
        concepts: list[dict[str, Any]] = []
        for pair in sorted(train_cells, key=lambda pair: next(
            index for index, cell in enumerate(base_manifest["samples"])
            if (cell["shape"], cell["color"]) == pair
        )):
            cell = cells_by_pair[pair]
            cell_dir = assets_dir / cell["id"]
            _require(not cell_dir.exists(), f"Training asset directory already exists: {cell_dir}")
            cell_dir.mkdir(parents=True, exist_ok=True)
            records = sorted(
                (record for record in train_records if record["cell_id"] == cell["id"]),
                key=lambda record: record["view_index"],
            )
            _require(len(records) == 16, f"Fold {fold_id} cell {cell['id']} must have 16 views")
            for record in records:
                source = _resolved_under(render_root, record["image"], "image")
                _stage_image(source, cell_dir / f"view_{record['view_index']:02d}.jpg")
            expected_names = {f"view_{index:02d}.jpg" for index in range(16)}
            actual_names = {path.name for path in cell_dir.iterdir()}
            _require(actual_names == expected_names, f"Fold {fold_id} cell {cell['id']} staging is contaminated")
            concepts.append({"instance_prompt": cell["instance_prompt"], "instance_data_dir": str(cell_dir)})

        concepts_path = fold_dir / "concepts.json"
        _write_json(concepts_path, concepts)
        seen_audit = [
            record for record in realized
            if (record["shape"], record["color"]) in train_cells and record["split"] == "audit"
        ]
        held_out_records = [
            record for record in realized if (record["shape"], record["color"]) in held_out
        ]
        held_out_train_views = [record for record in held_out_records if record["split"] == "train"]
        held_out_audit_views = [record for record in held_out_records if record["split"] == "audit"]
        fold_protocol = {
            "fold_id": fold_id,
            "training_seeds": list(TRAINING_SEEDS),
            "held_out_cells": [list(pair) for pair in fold["held_out"]],
            "train_cells": [list(pair) for pair in sorted(train_cells)],
            "train_view_count": len(train_records),
            "seen_audit_view_count": len(seen_audit),
            "held_out_view_count": len(held_out_records),
            "held_out_train_view_count": len(held_out_train_views),
            "held_out_audit_view_count": len(held_out_audit_views),
            "training_uses_gt_masks": False,
            "train_record_ids": [f"{record['cell_id']}:v{record['view_index']:02d}" for record in train_records],
            "seen_audit_record_ids": [f"{record['cell_id']}:v{record['view_index']:02d}" for record in seen_audit],
            "held_out_record_ids": [f"{record['cell_id']}:v{record['view_index']:02d}" for record in held_out_records],
            "held_out_train_record_ids": [
                f"{record['cell_id']}:v{record['view_index']:02d}" for record in held_out_train_views
            ],
            "held_out_audit_record_ids": [
                f"{record['cell_id']}:v{record['view_index']:02d}" for record in held_out_audit_views
            ],
        }
        _write_json(fold_dir / "fold_protocol.json", fold_protocol)

        for seed in TRAINING_SEEDS:
            train_config = json.loads(json.dumps(base_config))
            train_config["run"]["variant"] = f"multiview_fold_{fold_id.lower()}_seed{seed}"
            train_config["run"]["seed"] = seed
            train_config["data_manifest"] = str(realized_manifest_path)
            train_config["args"]["concepts_list"] = str(concepts_path)
            train_config["args"]["seed"] = seed
            train_config.setdefault("protocol", {})["multiview_protocol"] = protocol["protocol_id"]
            train_config["protocol"]["fold_id"] = fold_id
            train_config["protocol"]["held_out_cells"] = fold_protocol["held_out_cells"]
            train_config["protocol"]["views_per_training_cell"] = 16
            train_config["protocol"]["training_seed"] = seed
            _write_json(fold_dir / f"train_config_seed{seed}.json", train_config)
        fold_summaries.append(fold_protocol)
    return fold_summaries


def realize_protocol(
    base_manifest: dict[str, Any],
    protocol: dict[str, Any],
    render_root: Path,
    render_manifest: Path,
    output_dir: Path,
    base_config: Path,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    _require_empty_output_dir(output_dir)
    records = _read_jsonl(render_manifest)
    realized = validate_realization(render_root, records, base_manifest, protocol)
    realized_path = output_dir / "realized_views.jsonl"
    _write_jsonl(realized_path, realized)
    folds = build_fold_outputs(render_root, realized, base_manifest, protocol, output_dir, base_config)
    result = {
        "status": "validated",
        "realized_view_count": len(realized),
        "train_views_per_cell": 16,
        "audit_views_per_cell": 4,
        "fold_train_view_count": 96,
        "training_seeds": list(TRAINING_SEEDS),
        "training_uses_gt_masks": False,
        "realized_manifest": str(realized_path),
        "folds": folds,
    }
    _write_json(output_dir / "protocol_status.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, default=DEFAULT_BASE_MANIFEST)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="Write deterministic render requests without fabricating views")
    plan.add_argument("--output-dir", type=Path, required=True)
    plan.add_argument("--renderer", type=Path, help="Real multiview renderer entrypoint; absence is recorded as blocked")
    realize = subparsers.add_parser("realize", help="Validate renderer outputs and build image-only fold assets")
    realize.add_argument("--render-root", type=Path, required=True)
    realize.add_argument("--render-manifest", type=Path, required=True)
    realize.add_argument("--output-dir", type=Path, required=True)
    realize.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    return parser


def main(argv: Iterable[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    base_manifest, protocol = load_inputs(args.base_manifest, args.protocol)
    if args.command == "plan":
        result = plan_protocol(base_manifest, protocol, args.output_dir, args.renderer)
    else:
        result = realize_protocol(
            base_manifest, protocol, args.render_root, args.render_manifest, args.output_dir, args.base_config
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
