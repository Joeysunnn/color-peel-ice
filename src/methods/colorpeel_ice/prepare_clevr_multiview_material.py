#!/usr/bin/env python3
"""Plan and realize the locked CLEVR 3x3x2 material multiview protocol."""

from __future__ import annotations

import argparse
import csv
import filecmp
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable

from PIL import Image, ImageChops, ImageDraw, ImageStat

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.methods.colorpeel_ice.multiview_render_contract import (  # noqa: E402
    EXPECTED_PROFILE_V3,
    canonical_sha256,
)
from src.methods.colorpeel_ice.prepare_clevr_multiview import (  # noqa: E402
    ProtocolError,
    _file_sha256,
    _require,
    _require_empty_output_dir,
    _resolved_under,
    _validate_realized_image,
    _validate_view_metadata,
    _vector_matches,
    _write_json,
    _write_jsonl,
)

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


EXPERIMENT_DIR = REPO_ROOT / "experiments" / "clevr_subject_color_material_3x3x2"
DEFAULT_BASE_MANIFEST = EXPERIMENT_DIR / "manifests" / "clevr_material_manifest.json"
DEFAULT_PROTOCOL = EXPERIMENT_DIR / "manifests" / "clevr_multiview_material_protocol.json"
DEFAULT_BASE_CONFIG = EXPERIMENT_DIR / "configs" / "material_base.yaml"
SHAPES = ("cube", "sphere", "cylinder")
COLORS = ("red", "cyan", "gray")
MATERIALS = ("metal", "rubber")
TRAINING_SEEDS = (42, 43, 44)
MODIFIER_TOKEN = "<s1*>+<s2*>+<s3*>+<c1*>+<c2*>+<c3*>+<m1*>+<m2*>"
INITIALIZER_TOKEN = "cube+sphere+cylinder+red+turquoise+gray+metal+rubber"
RENDERER_FIELDS = ("camera", "light", "background", "scene_json", "image", "mask", "background_mask")
V2_METAL_EQUIVALENCE_V1 = {
    "id": "decoded_pixel_equivalence_v1",
    "rgb": {
        "comparison": "decoded_rgb_u8",
        "max_abs_difference": 1,
        "mean_abs_difference": 0.001,
    },
    "mask": {"comparison": "decoded_pixel_exact"},
    "background_mask": {"comparison": "decoded_pixel_exact"},
    "raw_sha256": "record_only",
}
V2_METAL_EQUIVALENCE = {
    "id": "decoded_pixel_equivalence_v2",
    "rgb": {
        "comparison": "decoded_rgb_u8",
        "mean_abs_difference": 0.001,
        "changed_channel_fraction": 0.001,
        "max_abs_difference": "record_only",
    },
    "mask": {"comparison": "decoded_pixel_exact"},
    "background_mask": {"comparison": "decoded_pixel_exact"},
    "raw_sha256": "record_only",
}
EXPECTED_FOLDS = {
    "A": {
        ("cube", "red", "metal"), ("sphere", "cyan", "metal"),
        ("cylinder", "gray", "metal"), ("cube", "cyan", "rubber"),
        ("sphere", "gray", "rubber"), ("cylinder", "red", "rubber"),
    },
    "B": {
        ("cube", "cyan", "metal"), ("sphere", "gray", "metal"),
        ("cylinder", "red", "metal"), ("cube", "gray", "rubber"),
        ("sphere", "red", "rubber"), ("cylinder", "cyan", "rubber"),
    },
    "C": {
        ("cube", "gray", "metal"), ("sphere", "red", "metal"),
        ("cylinder", "cyan", "metal"), ("cube", "red", "rubber"),
        ("sphere", "cyan", "rubber"), ("cylinder", "gray", "rubber"),
    },
}


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read valid JSON from {path}: {exc}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    value = json.loads(line)
                    _require(isinstance(value, dict), f"Expected object at {path}:{line_number}")
                    records.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read valid JSONL from {path}: {exc}") from exc
    return records


def _dimensions(manifest: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], ...]:
    dimensions = []
    for field, expected in (("shapes", SHAPES), ("colors", COLORS), ("materials", MATERIALS)):
        values = manifest.get(field, [])
        _require([item.get("name") for item in values] == list(expected), f"Locked {field} order changed")
        _require(all(isinstance(item.get("token"), str) for item in values), f"Missing {field} token")
        dimensions.append({item["name"]: item for item in values})
    return tuple(dimensions)


def build_cells(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    shape_map, color_map, material_map = _dimensions(manifest)
    _require(manifest.get("version") == 1, "Material manifest version changed")
    _require(manifest.get("study") == "clevr_subject_color_material_3x3x2", "Study changed")
    template = "a photo of {subject_token} shape in {color_token} color with {material_token} material"
    _require(manifest.get("prompt_template") == template, "Prompt template changed")
    _require([shape_map[name]["initializer"] for name in SHAPES] == list(SHAPES), "Shape initializers changed")
    _require([color_map[name]["initializer"] for name in COLORS] == ["red", "turquoise", "gray"],
             "Color initializers changed")
    _require([material_map[name]["initializer"] for name in MATERIALS] == list(MATERIALS),
             "Material initializers changed")
    cells = []
    for shape_color_index, (shape, color) in enumerate((s, c) for s in SHAPES for c in COLORS):
        for material_index, material in enumerate(MATERIALS):
            values = {
                "subject_token": shape_map[shape]["token"],
                "color_token": color_map[color]["token"],
                "material_token": material_map[material]["token"],
            }
            cells.append({
                "id": f"{shape}_{color}_{material}",
                "cell_index": shape_color_index * 2 + material_index,
                "shape_color_index": shape_color_index,
                "shape": shape,
                "color": color,
                "material": material,
                **values,
                "rgb": color_map[color]["rgb"],
                "instance_prompt": [template.format(**values)],
            })
    return cells


def validate_protocol(manifest: dict[str, Any], protocol: dict[str, Any]) -> list[dict[str, Any]]:
    cells = build_cells(manifest)
    _require(protocol.get("version") == 3, "Protocol version changed")
    _require(protocol.get("protocol_id") == "clevr_subject_color_material_3x3x2_multiview_v3",
             "Protocol ID changed")
    _require(protocol.get("base_manifest") == "clevr_material_manifest.json", "Base manifest changed")
    _require(protocol.get("views_per_cell") == 20, "views_per_cell must be 20")
    _require(protocol.get("view_splits") == {
        "train": {"start": 0, "stop": 16}, "audit": {"start": 16, "stop": 20},
    }, "View split changed")
    _require(protocol.get("render_seed") == {
        "base": 420000,
        "shape_color_stride": 100,
        "formula": "base + shape_color_index * shape_color_stride + view_index",
        "paired_axes": ["shape", "color", "view_index"],
        "paired_materials": ["metal", "rubber"],
    }, "Paired render seed rule changed")
    expected_renderer = {
        "id": "multiview_render_v3_material",
        "config": "../configs/multiview_render_v3_material.json",
        "background": EXPECTED_PROFILE_V3["background"],
    }
    _require(protocol.get("renderer_profile") == expected_renderer, "Renderer profile changed")
    realization_contract = protocol.get("realization_contract", {})
    _require(realization_contract.get("v2_metal_reference_equivalence") == V2_METAL_EQUIVALENCE,
             "v2 metal reference equivalence gate changed")
    folds = {fold.get("id"): {tuple(cell) for cell in fold.get("held_out", [])}
             for fold in protocol.get("folds", [])}
    _require(folds == EXPECTED_FOLDS, "Material held-out folds changed")
    grid = {(s, c, m) for s in SHAPES for c in COLORS for m in MATERIALS}
    _require(set().union(*folds.values()) == grid and sum(len(value) for value in folds.values()) == 18,
             "Held-out folds must partition all 18 cells")
    for fold_id, held_out in folds.items():
        train = grid - held_out
        for shape in SHAPES:
            _require(sum(cell[0] == shape for cell in train) == 4, f"Fold {fold_id} shape imbalance")
        for color in COLORS:
            _require(sum(cell[1] == color for cell in train) == 4, f"Fold {fold_id} color imbalance")
        for material in MATERIALS:
            _require(sum(cell[2] == material for cell in train) == 6, f"Fold {fold_id} material imbalance")
    return cells


def load_inputs(manifest_path: Path, protocol_path: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    manifest, protocol = _read_json(manifest_path), _read_json(protocol_path)
    _require(isinstance(manifest, dict) and isinstance(protocol, dict), "Inputs must be JSON objects")
    return manifest, protocol, validate_protocol(manifest, protocol)


def build_render_requests(manifest: dict[str, Any], protocol: dict[str, Any]) -> list[dict[str, Any]]:
    cells = validate_protocol(manifest, protocol)
    requests = []
    # Material is innermost so --limit 2 is a paired metal/rubber runtime smoke.
    for shape_color_index in range(9):
        paired_cells = [cell for cell in cells if cell["shape_color_index"] == shape_color_index]
        for view_index in range(20):
            for cell in paired_cells:
                requests.append({
                    **{key: cell[key] for key in (
                        "id", "cell_index", "shape_color_index", "shape", "color", "material",
                        "subject_token", "color_token", "material_token", "rgb",
                    )},
                    "cell_id": cell["id"],
                    "nominal_rgb": cell["rgb"],
                    "view_index": view_index,
                    "split": "train" if view_index < 16 else "audit",
                    "render_seed": 420000 + shape_color_index * 100 + view_index,
                    "renderer_profile_id": EXPECTED_PROFILE_V3["profile_id"],
                    "renderer_profile_sha256": canonical_sha256(EXPECTED_PROFILE_V3),
                    **{field: None for field in RENDERER_FIELDS},
                    "empirical_rgb": None,
                })
                requests[-1].pop("id")
                requests[-1].pop("rgb")
    validate_render_requests(requests, cells)
    return requests


def validate_render_requests(requests: list[dict[str, Any]], cells: list[dict[str, Any]]) -> None:
    _require(len(requests) == 360, f"Expected 360 render requests, got {len(requests)}")
    by_id = {cell["id"]: cell for cell in cells}
    _require(len({(row.get("cell_id"), row.get("view_index")) for row in requests}) == 360,
             "Each material cell/view pair must be unique")
    for row in requests:
        cell = by_id.get(row.get("cell_id"))
        _require(cell is not None, f"Unknown cell: {row.get('cell_id')}")
        for field in ("cell_index", "shape_color_index", "shape", "color", "material",
                      "subject_token", "color_token", "material_token"):
            _require(row.get(field) == cell[field], f"Wrong {field} for {cell['id']}")
        view = row.get("view_index")
        _require(isinstance(view, int) and 0 <= view < 20, f"Invalid view for {cell['id']}")
        _require(row.get("render_seed") == 420000 + cell["shape_color_index"] * 100 + view,
                 f"Wrong paired seed for {cell['id']} view {view}")
        _require(row.get("split") == ("train" if view < 16 else "audit"), "Wrong split")
        _require(row.get("nominal_rgb") == cell["rgb"], "Wrong nominal RGB")
        _require(row.get("renderer_profile_id") == EXPECTED_PROFILE_V3["profile_id"], "Wrong profile")
        _require(row.get("renderer_profile_sha256") == canonical_sha256(EXPECTED_PROFILE_V3),
                 "Wrong profile hash")
        _require(all(row.get(field) is None for field in RENDERER_FIELDS), "Fabricated renderer field")


def plan_protocol(manifest: dict[str, Any], protocol: dict[str, Any], output_dir: Path,
                  renderer: Path | None) -> dict[str, Any]:
    requests = build_render_requests(manifest, protocol)
    _require_empty_output_dir(output_dir)
    path = output_dir / "render_requests.jsonl"
    _write_jsonl(path, requests)
    available = renderer is not None and renderer.resolve().is_file()
    status = {
        "status": "planned" if available else "blocked",
        "blocked_reason": None if available else "multiview_renderer_not_provided_or_missing",
        "renderer": str(renderer.resolve()) if renderer else None,
        "renderer_available": available,
        "renderer_profile_id": EXPECTED_PROFILE_V3["profile_id"],
        "renderer_profile_sha256": canonical_sha256(EXPECTED_PROFILE_V3),
        "render_request_manifest": str(path.resolve()),
        "request_count": 360,
        "paired_smoke_prefix_count": 2,
    }
    _write_json(output_dir / "protocol_status.json", status)
    return status


def _stage(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        same = destination.is_symlink() and destination.resolve() == source.resolve()
        same = same or (destination.is_file() and filecmp.cmp(destination, source, shallow=False))
        _require(same, f"Existing staged file differs: {destination}")
        return
    try:
        os.symlink(source.resolve(), destination)
    except OSError:
        shutil.copy2(source, destination)


def _load_base_config(path: Path) -> dict[str, Any]:
    _require(yaml is not None, "PyYAML is required")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(config["args"]["modifier_token"] == MODIFIER_TOKEN, "Eight-token modifier order changed")
    _require(config["args"]["initializer_token"] == INITIALIZER_TOKEN, "Initializer order changed")
    return config


def _write_cell_assets(render_root: Path, records: list[dict[str, Any]], cells: list[dict[str, Any]],
                       root: Path) -> list[dict[str, Any]]:
    concepts = []
    for cell in cells:
        cell_dir = root / cell["id"]
        _require(not cell_dir.exists(), f"Training directory already exists: {cell_dir}")
        cell_dir.mkdir(parents=True)
        selected = sorted((row for row in records if row["cell_id"] == cell["id"] and row["split"] == "train"),
                          key=lambda row: row["view_index"])
        _require(len(selected) == 16, f"Cell {cell['id']} must have 16 training views")
        for row in selected:
            _stage(_resolved_under(render_root, row["image"], "image"), cell_dir / f"view_{row['view_index']:02d}.jpg")
        _require({path.name for path in cell_dir.iterdir()} == {f"view_{i:02d}.jpg" for i in range(16)},
                 f"Training staging is contaminated: {cell_dir}")
        concepts.append({"instance_prompt": cell["instance_prompt"], "instance_data_dir": str(cell_dir)})
    return concepts


def build_training_outputs(render_root: Path, realized: list[dict[str, Any]], cells: list[dict[str, Any]],
                           protocol: dict[str, Any], output_dir: Path, base_config_path: Path) -> list[dict[str, Any]]:
    base_config = _load_base_config(base_config_path)
    full_dir = output_dir / "full_grid"
    full_concepts = _write_cell_assets(render_root, realized, cells, full_dir / "train_assets")
    _write_json(full_dir / "concepts.json", full_concepts)
    full_config = json.loads(json.dumps(base_config))
    full_config["args"]["concepts_list"] = str((full_dir / "concepts.json").resolve())
    full_config["data_manifest"] = str((output_dir / "realized_views.jsonl").resolve())
    _write_json(full_dir / "train_config_seed42.json", full_config)

    for smoke_name, smoke_cells, config_name in (
        ("smoke_2step", cells[:2], "smoke_2step.yaml"),
        ("smoke_18step", cells, "smoke_18step.yaml"),
    ):
        smoke_dir = output_dir / "smokes" / smoke_name
        concepts = []
        for cell in smoke_cells:
            cell_dir = smoke_dir / "train_assets" / cell["id"]
            cell_dir.mkdir(parents=True)
            row = next(row for row in realized if row["cell_id"] == cell["id"] and row["view_index"] == 0)
            _stage(_resolved_under(render_root, row["image"], "image"), cell_dir / "view_00.jpg")
            concepts.append({"instance_prompt": cell["instance_prompt"], "instance_data_dir": str(cell_dir)})
        concepts_path = smoke_dir / "concepts.json"
        _write_json(concepts_path, concepts)
        smoke_config = yaml.safe_load((EXPERIMENT_DIR / "configs" / config_name).read_text(encoding="utf-8"))
        smoke_config["args"]["concepts_list"] = str(concepts_path.resolve())
        smoke_config["data_manifest"] = str((output_dir / "realized_views.jsonl").resolve())
        _write_json(smoke_dir / "train_config.json", smoke_config)

    by_triple = {(cell["shape"], cell["color"], cell["material"]): cell for cell in cells}
    summaries = []
    for fold in protocol["folds"]:
        held_out = {tuple(value) for value in fold["held_out"]}
        train_triples = set(by_triple) - held_out
        train_cells = [cell for cell in cells if (cell["shape"], cell["color"], cell["material"]) in train_triples]
        _require(len(train_cells) == 12, f"Fold {fold['id']} must train 12 cells")
        fold_dir = output_dir / "folds" / f"fold_{fold['id'].lower()}"
        concepts = _write_cell_assets(render_root, realized, train_cells, fold_dir / "train_assets")
        _write_json(fold_dir / "concepts.json", concepts)
        summary = {
            "fold_id": fold["id"], "held_out_cells": fold["held_out"],
            "train_cells": [[c["shape"], c["color"], c["material"]] for c in train_cells],
            "train_view_count": 192, "seen_generation_count_per_checkpoint": 240,
            "held_out_generation_count_per_checkpoint": 120, "training_seeds": list(TRAINING_SEEDS),
            "training_uses_gt_masks": False,
        }
        _write_json(fold_dir / "fold_protocol.json", summary)
        for seed in TRAINING_SEEDS:
            config = json.loads(json.dumps(base_config))
            config["status"] = "pending_full_grid_human_gate"
            config["run"].update({"variant": f"material_fold_{fold['id'].lower()}_seed{seed}", "seed": seed})
            config["args"].update({"concepts_list": str((fold_dir / "concepts.json").resolve()), "seed": seed})
            config["data_manifest"] = str((output_dir / "realized_views.jsonl").resolve())
            config["protocol"].update({"fold_id": fold["id"], "held_out_cells": fold["held_out"]})
            _write_json(fold_dir / f"train_config_seed{seed}.json", config)
        summaries.append(summary)
    eval_protocol = str((EXPERIMENT_DIR / "manifests" / "clevr_material_heldout_eval.json").resolve())
    evaluation_dir = output_dir / "evaluation_configs"
    baseline_generation = {
        "schema_version": 1, "status": "pending_full_grid_training", "stage": "generate_material_multiview",
        "run": {"study": "clevr_subject_color_material_3x3x2",
                "variant": "material_full_grid_generate_seed42", "seed": 42},
        "environment": {"CUDA_VISIBLE_DEVICES": "3"}, "data_manifest": str((output_dir / "realized_views.jsonl").resolve()),
        "args": {"model-dir": "${COLORPEEL_MATERIAL_FULL_GRID_RUN}/checkpoints",
                 "parent-training-run": "${COLORPEEL_MATERIAL_FULL_GRID_RUN}", "evaluation-protocol": eval_protocol,
                 "mode": "full-grid", "training-seed": 42, "device": "cuda:0", "dtype": "float16",
                 "disable-safety-checker": True, "acknowledge-safety-risk": True, "skip-existing": True},
    }
    _write_json(evaluation_dir / "generate_full_grid_seed42.json", baseline_generation)
    for fold_id in "ABC":
        for seed in TRAINING_SEEDS:
            env_name = f"COLORPEEL_MATERIAL_FOLD_{fold_id}_SEED{seed}_RUN"
            config = {"schema_version": 1, "status": "pending_full_grid_human_gate",
                      "stage": "generate_material_multiview",
                      "run": {"study": "clevr_subject_color_material_3x3x2",
                              "variant": f"material_heldout_generate_fold_{fold_id.lower()}_train{seed}", "seed": seed},
                      "environment": {"CUDA_VISIBLE_DEVICES": "3"},
                      "data_manifest": str((output_dir / "realized_views.jsonl").resolve()),
                      "args": {"model-dir": f"${{{env_name}}}/checkpoints", "parent-training-run": f"${{{env_name}}}",
                               "evaluation-protocol": eval_protocol, "mode": "heldout", "fold-id": fold_id,
                               "training-seed": seed, "device": "cuda:0", "dtype": "float16",
                               "disable-safety-checker": True, "acknowledge-safety-risk": True,
                               "skip-existing": True}}
            _write_json(evaluation_dir / f"generate_fold_{fold_id.lower()}_seed{seed}.json", config)
    return summaries


def _decoded_rgb_difference(candidate: Path, reference: Path) -> dict[str, Any]:
    with Image.open(candidate) as candidate_image, Image.open(reference) as reference_image:
        _require(candidate_image.mode == reference_image.mode == "RGB", "RGB comparison requires RGB inputs")
        _require(candidate_image.size == reference_image.size, "RGB comparison dimensions differ")
        difference = ImageChops.difference(candidate_image, reference_image)
        max_abs = max(maximum for _, maximum in difference.getextrema())
        mean_abs = sum(ImageStat.Stat(difference).mean) / 3.0
        changed = sum(value != 0 for value in difference.tobytes())
        total = candidate_image.width * candidate_image.height * 3
        changed_fraction = changed / total
    return {
        "max_abs_difference": max_abs,
        "mean_abs_difference": mean_abs,
        "changed_channel_values": changed,
        "total_channel_values": total,
        "changed_channel_fraction": changed_fraction,
        "pixel_equivalent": (
            mean_abs <= V2_METAL_EQUIVALENCE["rgb"]["mean_abs_difference"]
            and changed_fraction <= V2_METAL_EQUIVALENCE["rgb"]["changed_channel_fraction"]
        ),
    }


def _decoded_pixels_equal(candidate: Path, reference: Path) -> bool:
    with Image.open(candidate) as candidate_image, Image.open(reference) as reference_image:
        if candidate_image.mode != reference_image.mode or candidate_image.size != reference_image.size:
            return False
        return ImageChops.difference(candidate_image, reference_image).getbbox() is None


def _reference_artifacts(v2_root: Path, v2_manifest: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    refs = {}
    for row in _read_jsonl(v2_manifest):
        key = (row.get("shape"), row.get("color"), row.get("view_index"))
        _require(key not in refs, f"Duplicate v2 reference: {key}")
        refs[key] = {}
        for field in ("image", "mask", "background_mask"):
            path = _resolved_under(v2_root, row[field], field)
            refs[key][field] = {"path": path, "sha256": _file_sha256(path)}
    _require(len(refs) == 180, "Accepted v2 reference must contain 180 views")
    return refs


def validate_realization(render_root: Path, records: list[dict[str, Any]], manifest: dict[str, Any],
                         protocol: dict[str, Any], v2_root: Path, v2_manifest: Path) -> list[dict[str, Any]]:
    cells = validate_protocol(manifest, protocol)
    expected = build_render_requests(manifest, protocol)
    _require(len(records) == 360, f"Expected 360 realized views, got {len(records)}")
    expected_by_key = {(row["cell_id"], row["view_index"]): row for row in expected}
    actual_by_key = {(row.get("cell_id"), row.get("view_index")): row for row in records}
    _require(set(actual_by_key) == set(expected_by_key) and len(actual_by_key) == 360,
             "Realization must contain every material cell/view exactly once")
    contract = _read_json(render_root / "render_contract.json")
    _require(contract.get("profile_id") == EXPECTED_PROFILE_V3["profile_id"], "Render contract profile changed")
    _require(contract.get("profile_sha256") == canonical_sha256(EXPECTED_PROFILE_V3), "Profile hash changed")
    _require(contract.get("requests_sha256") == canonical_sha256(expected), "Request hash changed")
    _require(contract.get("request_count") == 360, "Contract request count changed")
    _require(set(contract.get("asset_sha256", {})) >= {"material_metal", "material_rubber"},
             "Both material asset hashes are required")
    contract_hash = canonical_sha256(contract)
    v2_artifacts = _reference_artifacts(v2_root.resolve(), v2_manifest.resolve())
    realized = []
    image_hashes = {cell["id"]: set() for cell in cells}
    pair_metadata: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = {}
    for key, expected_row in expected_by_key.items():
        row = actual_by_key[key]
        for field in ("cell_index", "shape_color_index", "shape", "color", "material", "subject_token",
                      "color_token", "material_token", "nominal_rgb", "view_index", "split", "render_seed",
                      "renderer_profile_id", "renderer_profile_sha256"):
            _require(row.get(field) == expected_row[field], f"Realization changed {field} for {key}")
        _require(row.get("background") == EXPECTED_PROFILE_V3["background"], f"Background changed for {key}")
        _validate_view_metadata(row, key, EXPECTED_PROFILE_V3)
        paths = {field: _resolved_under(render_root, row.get(field), field)
                 for field in ("scene_json", "image", "mask", "background_mask")}
        hashes = row.get("artifact_sha256", {})
        _require(all(hashes.get(field) == _file_sha256(path) for field, path in paths.items()),
                 f"Artifact hash changed for {key}")
        scene = _read_json(paths["scene_json"])
        _require(scene.get("renderer_profile_id") == EXPECTED_PROFILE_V3["profile_id"],
                 f"Scene renderer profile changed for {key}")
        _require(scene.get("render_seed") == expected_row["render_seed"], f"Scene seed changed for {key}")
        _require(scene.get("cycles_seed") == expected_row["render_seed"], f"Cycles seed changed for {key}")
        _require(scene.get("camera") == row["camera"] and scene.get("light") == row["light"],
                 f"Scene view metadata changed for {key}")
        _require(scene.get("background") == EXPECTED_PROFILE_V3["background"], f"Scene background changed for {key}")
        _require(scene.get("asset_sha256") == contract["asset_sha256"], f"Scene asset hashes changed for {key}")
        renderer = scene.get("renderer", {})
        _require(renderer.get("blender_version") == "4.2.11" and renderer.get("engine") == "CYCLES",
                 f"Renderer runtime changed for {key}")
        _require(renderer.get("cycles_device") == "CUDA" and renderer.get("cycles_samples") == 512,
                 f"Cycles runtime changed for {key}")
        _require(len(renderer.get("cuda_devices", [])) == 1 and
                 "V100" in renderer["cuda_devices"][0].get("name", ""), f"GPU changed for {key}")
        objects = scene.get("objects")
        _require(isinstance(objects, list) and len(objects) == 1, f"Scene must contain one object for {key}")
        obj = objects[0]
        _require(all(obj.get(field) == expected_row[field] for field in ("shape", "color", "material")),
                 f"Object labels changed for {key}")
        expected_asset = {"metal": "MyMetal", "rubber": "Rubber"}[expected_row["material"]]
        _require(obj.get("material_asset_name") == expected_asset, f"Wrong material asset for {key}")
        _require(obj.get("material_asset_sha256") == contract["asset_sha256"][f"material_{expected_row['material']}"],
                 f"Wrong material asset hash for {key}")
        _require(_vector_matches(row["camera"].get("look_at_target"), obj.get("3d_coords")),
                 f"Camera target changed for {key}")
        mean, foreground = _validate_realized_image(paths["image"], paths["mask"], paths["background_mask"])
        _require(row.get("foreground_pixels") == foreground, f"Foreground count changed for {key}")
        _require(row.get("render_contract_sha256") == contract_hash, f"Contract hash changed for {key}")
        image_hash = _file_sha256(paths["image"])
        image_hashes[expected_row["cell_id"]].add(image_hash)
        pair_key = (row["shape"], row["color"], row["view_index"])
        pair_metadata.setdefault(pair_key, {})[row["material"]] = {
            "seed": row["render_seed"], "camera": row["camera"], "light": row["light"],
            "mask": paths["mask"], "background_mask": paths["background_mask"],
        }
        equivalence = None
        if row["material"] == "metal":
            reference = v2_artifacts[pair_key]
            rgb = _decoded_rgb_difference(paths["image"], reference["image"]["path"])
            mask_equal = _decoded_pixels_equal(paths["mask"], reference["mask"]["path"])
            background_mask_equal = _decoded_pixels_equal(
                paths["background_mask"], reference["background_mask"]["path"]
            )
            _require(rgb["pixel_equivalent"], f"v3 metal RGB exceeds v2 pixel gate for {pair_key}: {rgb}")
            _require(mask_equal, f"v3 metal mask pixels differ from accepted v2 for {pair_key}")
            _require(background_mask_equal,
                     f"v3 metal background mask pixels differ from accepted v2 for {pair_key}")
            equivalence = {
                "gate": V2_METAL_EQUIVALENCE,
                "accepted_v2_raw_sha256": {
                    field: reference[field]["sha256"] for field in ("image", "mask", "background_mask")
                },
                "v3_raw_sha256": {
                    "image": image_hash,
                    "mask": _file_sha256(paths["mask"]),
                    "background_mask": _file_sha256(paths["background_mask"]),
                },
                "rgb": rgb,
                "mask_pixel_equal": mask_equal,
                "background_mask_pixel_equal": background_mask_equal,
                "passed": True,
            }
        merged = {**expected_row, **row}
        merged["empirical_rgb"] = {"value": mean, "space": "srgb_u8", "statistic": "masked_mean",
                                   "source": "realized_view_gt_mask", "foreground_pixels": foreground}
        if equivalence is not None:
            merged["v2_metal_reference_equivalence"] = equivalence
        realized.append(merged)
    for cell_id, hashes in image_hashes.items():
        _require(len(hashes) == 20, f"Cell {cell_id} must have 20 unique RGB images")
    for pair_key, pair in pair_metadata.items():
        _require(set(pair) == set(MATERIALS), f"Missing paired material for {pair_key}")
        for field in ("seed", "camera", "light"):
            _require(pair["metal"][field] == pair["rubber"][field],
                     f"Paired material {field} differs for {pair_key}")
        for field in ("mask", "background_mask"):
            _require(_decoded_pixels_equal(pair["metal"][field], pair["rubber"][field]),
                     f"Paired material {field} pixels differ for {pair_key}")
    return realized


def write_review_outputs(render_root: Path, realized: list[dict[str, Any]], cells: list[dict[str, Any]],
                         output_dir: Path) -> tuple[Path, Path, list[Path]]:
    review_path = output_dir / "material_render_human_review.csv"
    fields = ["generation_id", "cell_id", "shape", "color", "material", "render_seed", "view_index",
              "split", "image", "mask", "background_mask", "observed_shape", "observed_color",
              "observed_material", "object_complete", "object_clipped", "mask_aligned", "lighting_ok",
              "background_neutral", "material_ok", "black_cap", "artifact_or_invalid", "confidence",
              "reviewer_id", "comment"]
    with review_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in realized:
            writer.writerow({key: row.get(key, "") for key in fields} | {
                "generation_id": f"{row['cell_id']}:v{row['view_index']:02d}"})
    selected = (0, 4, 8, 12, 16)
    by_key = {(row["cell_id"], row["view_index"]): row for row in realized}
    sheet = Image.new("RGB", (1000, 18 * 224), "white")
    draw = ImageDraw.Draw(sheet)
    for row_index, cell in enumerate(cells):
        for column, view in enumerate(selected):
            record = by_key[(cell["id"], view)]
            with Image.open(_resolved_under(render_root, record["image"], "image")) as source:
                tile = source.convert("RGB"); tile.thumbnail((192, 192), Image.Resampling.LANCZOS)
                sheet.paste(tile, (column * 200 + (200 - tile.width) // 2, row_index * 224 + 24))
            draw.text((column * 200 + 4, row_index * 224 + 4), f"{cell['id']} v{view:02d}", fill="black")
    sheet_path = output_dir / "material_18cell_contact_sheet.png"
    sheet.save(sheet_path)
    pair_paths = []
    for shape in SHAPES:
        for color in COLORS:
            pair_sheet = Image.new("RGB", (400, 20 * 224), "white")
            pair_draw = ImageDraw.Draw(pair_sheet)
            for view in range(20):
                for column, material in enumerate(MATERIALS):
                    cell_id = f"{shape}_{color}_{material}"
                    record = by_key[(cell_id, view)]
                    with Image.open(_resolved_under(render_root, record["image"], "image")) as source:
                        tile = source.convert("RGB"); tile.thumbnail((192, 192), Image.Resampling.LANCZOS)
                        pair_sheet.paste(tile, (column * 200 + (200 - tile.width) // 2, view * 224 + 24))
                    pair_draw.text((column * 200 + 4, view * 224 + 4), f"{material} v{view:02d}", fill="black")
            path = output_dir / "material_pairs" / f"{shape}_{color}.png"
            path.parent.mkdir(parents=True, exist_ok=True); pair_sheet.save(path); pair_paths.append(path)
    return review_path, sheet_path, pair_paths


def realize_protocol(manifest: dict[str, Any], protocol: dict[str, Any], render_root: Path,
                     render_manifest: Path, v2_render_root: Path, v2_render_manifest: Path,
                     output_dir: Path, base_config: Path) -> dict[str, Any]:
    _require_empty_output_dir(output_dir)
    cells = validate_protocol(manifest, protocol)
    realized = validate_realization(render_root.resolve(), _read_jsonl(render_manifest), manifest, protocol,
                                    v2_render_root, v2_render_manifest)
    realized_path = output_dir / "realized_views.jsonl"
    _write_jsonl(realized_path, realized)
    equivalence_records = [
        {"cell_id": row["cell_id"], "shape": row["shape"], "color": row["color"],
         "view_index": row["view_index"], "render_seed": row["render_seed"],
         **row["v2_metal_reference_equivalence"]}
        for row in realized if row["material"] == "metal"
    ]
    _require(len(equivalence_records) == 180, "Expected 180 v2 metal equivalence records")
    equivalence_path = output_dir / "v2_metal_pixel_equivalence.jsonl"
    _write_jsonl(equivalence_path, equivalence_records)
    folds = build_training_outputs(render_root.resolve(), realized, cells, protocol, output_dir, base_config)
    review, sheet, pair_sheets = write_review_outputs(render_root.resolve(), realized, cells, output_dir)
    gate = {"status": "pending_human_review", "training_authorized": False,
            "required_checks": ["metal_rubber_distinguishable", "mask_alignment", "object_not_clipped",
                                "fixed_neutral_background", "paired_view_consistency", "lighting_span"]}
    _write_json(output_dir / "human_gate_decision.json", gate)
    result = {"status": "validated_pending_human_review", "realized_view_count": 360,
              "full_grid_train_view_count": 288, "fold_train_view_count": 192,
              "training_uses_gt_masks": False, "realized_manifest": str(realized_path),
              "v2_metal_equivalence": {
                  "status": "passed_180_of_180",
                  "gate": V2_METAL_EQUIVALENCE,
                  "audit_manifest": str(equivalence_path),
                  "maximum_observed_abs_difference": max(
                      row["rgb"]["max_abs_difference"] for row in equivalence_records
                  ),
                  "maximum_observed_mean_abs_difference": max(
                      row["rgb"]["mean_abs_difference"] for row in equivalence_records
                  ),
                  "maximum_observed_changed_channel_fraction": max(
                      row["rgb"]["changed_channel_fraction"] for row in equivalence_records
                  ),
                  "raw_rgb_sha256_match_count": sum(
                      row["accepted_v2_raw_sha256"]["image"] == row["v3_raw_sha256"]["image"]
                      for row in equivalence_records
                  ),
              },
              "human_review_csv": str(review), "contact_sheet": str(sheet),
              "material_pair_sheets": [str(path) for path in pair_sheets], "folds": folds}
    _write_json(output_dir / "protocol_status.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "realize"))
    parser.add_argument("--base-manifest", type=Path, default=DEFAULT_BASE_MANIFEST)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--renderer", type=Path)
    parser.add_argument("--render-root", type=Path)
    parser.add_argument("--render-manifest", type=Path)
    parser.add_argument("--v2-render-root", type=Path)
    parser.add_argument("--v2-render-manifest", type=Path)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    return parser


def main(argv: Iterable[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    manifest, protocol, _ = load_inputs(args.base_manifest, args.protocol)
    if args.command == "plan":
        result = plan_protocol(manifest, protocol, args.output_dir.resolve(), args.renderer)
    else:
        for name in ("render_root", "render_manifest", "v2_render_root", "v2_render_manifest"):
            _require(getattr(args, name) is not None, f"--{name.replace('_', '-')} is required for realize")
        result = realize_protocol(manifest, protocol, args.render_root, args.render_manifest,
                                  args.v2_render_root, args.v2_render_manifest, args.output_dir.resolve(),
                                  args.base_config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
