#!/usr/bin/env python3
"""Render the locked v4 two-object CLEVR request manifest with Blender 4.2."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.methods.colorpeel_ice.multiview_render_contract import (  # noqa: E402
    EXPECTED_PROFILE_V4,
    canonical_sha256,
    validate_profile as validate_locked_profile,
    validate_two_object_render_requests,
)

_BASE_PATH = Path(__file__).with_name("render_clevr_multiview.py")
_SPEC = importlib.util.spec_from_file_location("colorpeel_single_object_renderer", _BASE_PATH)
BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(BASE)
bpy = BASE.bpy

LEFT_MASK_PREFIX = "__left_mask_"
RIGHT_MASK_PREFIX = "__right_mask_"
BACKGROUND_MASK_PREFIX = "__background_mask_"


class RendererError(RuntimeError):
    """Raised when a v4 renderer input, runtime, or resume artifact is invalid."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RendererError(message)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--properties-json", type=Path, required=True)
    parser.add_argument("--base-scene-blendfile", type=Path, required=True)
    parser.add_argument("--shape-dir", type=Path, required=True)
    parser.add_argument("--material-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="Smoke-only prefix length")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def validate_profile(value: Any) -> dict[str, Any]:
    try:
        profile = validate_locked_profile(value)
    except ValueError as exc:
        raise RendererError(str(exc)) from exc
    require(profile == EXPECTED_PROFILE_V4, "Two-object renderer requires the locked v4 profile")
    return profile


def validate_requests(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    try:
        validate_two_object_render_requests(records)
    except Exception as exc:
        raise RendererError(str(exc)) from exc
    return records


def _expected_names() -> set[str]:
    return {"img.jpg", "mask_left.png", "mask_right.png", "background.png", "scene.json", ".record.json"}


def verify_completed_record(output_root: Path, record: dict[str, Any], expected: dict[str, Any],
                            contract: dict[str, Any]) -> None:
    for field, value in expected.items():
        if field in {"camera", "light", "background", "scene_json", "image", "masks", "background_mask"}:
            continue
        require(record.get(field) == value, f"Resume record changed {field}")
    require(record.get("render_contract_sha256") == canonical_sha256(contract), "Resume contract hash changed")
    relative_paths = {
        "image": record.get("image"), "scene_json": record.get("scene_json"),
        "background_mask": record.get("background_mask"),
        "mask_left": (record.get("masks") or {}).get("left"),
        "mask_right": (record.get("masks") or {}).get("right"),
    }
    hashes = record.get("artifact_sha256", {})
    resolved = {}
    for field, relative in relative_paths.items():
        require(isinstance(relative, str) and relative, f"Missing resume artifact {field}")
        path = (output_root / relative).resolve()
        try:
            path.relative_to(output_root.resolve())
        except ValueError as exc:
            raise RendererError(f"Resume artifact escapes output root: {relative}") from exc
        require(path.is_file(), f"Missing resume artifact: {path}")
        require(hashes.get(field) == BASE.file_sha256(path), f"Resume artifact hash changed: {field}")
        resolved[field] = path
    view_dir = resolved["scene_json"].parent
    require({path.name for path in view_dir.iterdir()} == _expected_names(),
            f"Resume view directory is contaminated: {view_dir}")
    require(BASE.load_json(view_dir / ".record.json") == record, f"Resume .record.json disagrees: {view_dir}")


def load_completed_records(output_root: Path, expected_by_key: dict[tuple[str, int], dict[str, Any]],
                           contract: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    manifest_path = output_root / "renderer_realization.jsonl"
    records = BASE.load_jsonl(manifest_path) if manifest_path.is_file() else []
    completed = {}
    for record in records:
        key = (record.get("scene_id"), record.get("view_index"))
        require(key in expected_by_key, f"Unknown resume record: {key}")
        require(key not in completed, f"Duplicate resume record: {key}")
        verify_completed_record(output_root, record, expected_by_key[key], contract)
        completed[key] = record
    expected_scenes = {row["scene_id"] for row in expected_by_key.values()}
    for path in output_root.iterdir():
        if not path.is_dir():
            continue
        require(path.name in expected_scenes, f"Resume output contains unknown directory: {path}")
        for view_dir in path.iterdir():
            require(view_dir.is_dir(), f"Resume scene contains non-directory: {view_dir}")
            try:
                view_index = int(view_dir.name.removeprefix("view_"))
            except ValueError as exc:
                raise RendererError(f"Unknown resume view directory: {view_dir}") from exc
            require((path.name, view_index) in completed, f"Resume found orphan final directory: {view_dir}")
    return completed


def add_object(spec: dict[str, Any], profile: dict[str, Any], properties: dict[str, Any],
               shape_dir: Path, material_dir: Path):
    source_name = properties["shapes"][spec["shape"]]
    obj = BASE.append_shape(shape_dir, source_name)
    scale = profile["objects"]["scale"]
    if source_name == "SmoothCube_v2":
        scale /= math.sqrt(2)
    position = profile["objects"]["positions_xy"][spec["side"]]
    obj.location = (position[0], position[1], scale)
    obj.rotation_euler = (0.0, 0.0, math.radians(profile["objects"]["rotation_z_degrees"]))
    obj.scale = (scale, scale, scale)
    material = BASE.create_asset_material(
        f"ColorPeel_{spec['side']}_{spec['state_id']}", spec["nominal_rgb"], spec["material"],
        material_dir, properties,
    )
    obj.data.materials.clear()
    obj.data.materials.append(material)
    return obj, properties["materials"][spec["material"]]


def configure_mask_outputs(sample_dir: Path, left_obj, right_obj) -> None:
    scene = bpy.context.scene
    scene.use_nodes = True
    scene.view_layers[0].use_pass_object_index = True
    left_obj.pass_index = 1
    right_obj.pass_index = 2
    nodes, links = scene.node_tree.nodes, scene.node_tree.links
    nodes.clear()
    render_layers = nodes.new("CompositorNodeRLayers")
    composite = nodes.new("CompositorNodeComposite")
    links.new(render_layers.outputs["Image"], composite.inputs["Image"])
    object_masks = []
    for index in (1, 2):
        mask = nodes.new("CompositorNodeIDMask")
        mask.index = index
        mask.use_antialiasing = False
        links.new(render_layers.outputs["IndexOB"], mask.inputs["ID value"])
        object_masks.append(mask)
    union = nodes.new("CompositorNodeMath")
    union.operation = "ADD"
    links.new(object_masks[0].outputs["Alpha"], union.inputs[0])
    links.new(object_masks[1].outputs["Alpha"], union.inputs[1])
    background = nodes.new("CompositorNodeMath")
    background.operation = "SUBTRACT"
    background.inputs[0].default_value = 1.0
    links.new(union.outputs["Value"], background.inputs[1])
    outputs = ((object_masks[0].outputs["Alpha"], LEFT_MASK_PREFIX),
               (object_masks[1].outputs["Alpha"], RIGHT_MASK_PREFIX),
               (background.outputs["Value"], BACKGROUND_MASK_PREFIX))
    for value, prefix in outputs:
        output = nodes.new("CompositorNodeOutputFile")
        output.base_path = str(sample_dir)
        output.file_slots[0].path = prefix
        output.format.file_format = "PNG"
        output.format.color_mode = "BW"
        output.format.color_depth = "8"
        links.new(value, output.inputs[0])


def _validate_masks(left_path: Path, right_path: Path, background_path: Path) -> dict[str, int]:
    images = [bpy.data.images.load(str(path), check_existing=False)
              for path in (left_path, right_path, background_path)]
    try:
        require(all(tuple(image.size) == (512, 512) for image in images), "Masks must be 512x512")
        pixels = [image.pixels[:] for image in images]
        counts = [0, 0]
        for index in range(0, len(pixels[0]), 4):
            values = [channel[index] for channel in pixels]
            require(all(abs(value) < 1e-6 or abs(value - 1.0) < 1e-6 for value in values),
                    "Masks must be binary")
            require(not (values[0] > 0.5 and values[1] > 0.5), "Object masks overlap")
            require(abs(values[0] + values[1] + values[2] - 1.0) < 1e-6,
                    "Object/background masks do not partition the image")
            counts[0] += values[0] > 0.5
            counts[1] += values[1] > 0.5
        for count in counts:
            require(0.005 <= count / (512 * 512) <= 0.45, "Object mask ratio outside 0.005-0.45")
        return {"left": counts[0], "right": counts[1]}
    finally:
        for image in images:
            bpy.data.images.remove(image)


def render_one(request: dict[str, Any], profile: dict[str, Any], contract: dict[str, Any],
               properties: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    bpy.ops.wm.open_mainfile(filepath=str(args.base_scene_blendfile.resolve()))
    cuda_devices = BASE.configure_render(profile, request["render_seed"])
    BASE.neutralize_scene(profile)
    BASE.clear_base_scene_geometry()
    rendered = []
    blender_objects = {}
    for spec in request["objects"]:
        obj, material_asset_name = add_object(
            spec, profile, properties, args.shape_dir, args.material_dir
        )
        blender_objects[spec["side"]] = obj
        rendered.append({
            "side": spec["side"], "state_id": spec["state_id"], "shape": spec["shape"],
            "color": spec["color"], "material": spec["material"],
            "material_backend": "clevr_asset_node_group",
            "material_asset_name": material_asset_name,
            "material_asset_sha256": contract["asset_sha256"][f"material_{spec['material']}"],
            "nominal_scale": profile["objects"]["scale"], "applied_scale": BASE._tuple(obj.scale),
            "3d_coords": BASE._tuple(obj.location), "rotation": profile["objects"]["rotation_z_degrees"],
        })
    bpy.context.view_layer.update()
    midpoint = bpy.data.objects.new("ColorPeel_scene_midpoint", None)
    midpoint.location = tuple(
        (float(blender_objects["left"].location[axis]) + float(blender_objects["right"].location[axis])) / 2.0
        for axis in range(3)
    )
    bpy.context.collection.objects.link(midpoint)
    bpy.context.view_layer.update()
    camera_metadata, light_metadata = BASE.apply_orbit_view(profile, request["render_seed"], midpoint)
    camera = bpy.data.objects[profile["camera"]["name"]]
    for metadata in rendered:
        metadata["pixel_coords"] = BASE.camera_pixel_coords(camera, blender_objects[metadata["side"]])

    final_dir = args.output_root / request["scene_id"] / f"view_{request['view_index']:02d}"
    require(not final_dir.exists(), f"Final output already exists without valid resume: {final_dir}")
    partial_dir = args.output_root / ".partial" / f"{request['scene_id']}__view_{request['view_index']:02d}"
    require(not partial_dir.exists(), f"Incomplete partial output already exists: {partial_dir}")
    partial_dir.mkdir(parents=True)
    configure_mask_outputs(partial_dir, blender_objects["left"], blender_objects["right"])
    bpy.context.scene.render.filepath = str(partial_dir / "img.jpg")
    bpy.ops.render.render(write_still=True)
    left_mask = BASE.finalize_mask(partial_dir, LEFT_MASK_PREFIX, "mask_left.png")
    right_mask = BASE.finalize_mask(partial_dir, RIGHT_MASK_PREFIX, "mask_right.png")
    background_mask = BASE.finalize_mask(partial_dir, BACKGROUND_MASK_PREFIX, "background.png")
    foreground = _validate_masks(left_mask, right_mask, background_mask)
    scene = {
        "renderer_profile_id": profile["profile_id"], "render_seed": request["render_seed"],
        "cycles_seed": request["render_seed"], "camera": camera_metadata, "light": light_metadata,
        "background": profile["background"],
        "renderer": {"blender_version": ".".join(str(part) for part in bpy.app.version),
                     "engine": "CYCLES", "cycles_samples": 512, "cycles_device": "CUDA",
                     "cuda_devices": cuda_devices},
        "asset_sha256": contract["asset_sha256"], "objects": rendered,
    }
    BASE.write_json(partial_dir / "scene.json", scene)
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(partial_dir, final_dir)
    relative = final_dir.relative_to(args.output_root).as_posix()
    record = {
        **request, "camera": camera_metadata, "light": light_metadata,
        "background": profile["background"], "scene_json": f"{relative}/scene.json",
        "image": f"{relative}/img.jpg",
        "masks": {"left": f"{relative}/mask_left.png", "right": f"{relative}/mask_right.png"},
        "background_mask": f"{relative}/background.png", "rendered_objects": rendered,
        "render_contract_sha256": canonical_sha256(contract), "foreground_pixels": foreground,
    }
    artifact_paths = {
        "image": record["image"], "scene_json": record["scene_json"],
        "background_mask": record["background_mask"], "mask_left": record["masks"]["left"],
        "mask_right": record["masks"]["right"],
    }
    record["artifact_sha256"] = {
        field: BASE.file_sha256(args.output_root / path) for field, path in artifact_paths.items()
    }
    BASE.write_json(final_dir / ".record.json", record)
    return record


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(BASE.extract_blender_args() if argv is None else argv)
    args.output_root = args.output_root.resolve()
    profile = validate_profile(BASE.load_json(args.profile.resolve()))
    requests = validate_requests(BASE.load_jsonl(args.requests.resolve()))
    require(args.limit is None or 1 <= args.limit <= len(requests),
            f"--limit must be between 1 and {len(requests)}")
    properties, asset_hashes = BASE.collect_asset_hashes(
        args.properties_json.resolve(), args.base_scene_blendfile.resolve(),
        args.shape_dir.resolve(), args.material_dir.resolve(), profile,
    )
    for request in requests:
        for obj in request["objects"]:
            require(properties["colors"].get(obj["color"]) == obj["nominal_rgb"],
                    f"Request RGB differs from properties.json: {obj['state_id']}")
    contract = BASE.stable_contract(requests, profile, asset_hashes)
    if args.validate_only:
        result = {"status": "validated", **contract}
        print(json.dumps(result, indent=2, sort_keys=True))
        return result
    require(bpy is not None, "Rendering must run through Blender")
    require(".".join(str(part) for part in bpy.app.version) == profile["blender"]["version"],
            f"Blender version must be {profile['blender']['version']}")
    try:
        BASE.prepare_output_root(args.output_root, contract, args.resume)
    except Exception as exc:
        raise RendererError(str(exc)) from exc
    expected_by_key = {(row["scene_id"], row["view_index"]): row for row in requests}
    completed = load_completed_records(args.output_root, expected_by_key, contract) if args.resume else {}
    selected = requests[:args.limit] if args.limit is not None else requests
    manifest_path = args.output_root / "renderer_realization.jsonl"
    status_path = args.output_root / "renderer_status.jsonl"
    for index, request in enumerate(selected, start=1):
        key = (request["scene_id"], request["view_index"])
        if key in completed:
            print(f"[{index:03d}/{len(selected):03d}] resume skip {key}")
            continue
        print(f"[{index:03d}/{len(selected):03d}] render {key}")
        try:
            record = render_one(request, profile, contract, properties, args)
            BASE.append_jsonl(manifest_path, record)
            BASE.append_jsonl(status_path, {"scene_id": key[0], "view_index": key[1], "status": "ok"})
            completed[key] = record
        except Exception as exc:
            BASE.append_jsonl(status_path, {"scene_id": key[0], "view_index": key[1], "status": "failed",
                                            "error_type": type(exc).__name__, "error": str(exc)})
            raise
    partial_root = args.output_root / ".partial"
    if partial_root.is_dir() and not any(partial_root.iterdir()):
        partial_root.rmdir()
    state = "succeeded" if args.limit is None and len(completed) == len(requests) else "partial_smoke"
    result = {"status": state, "completed_count": len(completed), "selected_count": len(selected),
              "request_count": len(requests), "resume": args.resume, "limit": args.limit,
              "profile_id": profile["profile_id"], "render_contract_sha256": canonical_sha256(contract),
              "git_commit": BASE.git_commit(REPO_ROOT)}
    BASE.write_json(args.output_root / "renderer_status.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
