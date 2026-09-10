#!/usr/bin/env python3
"""Plan, render, and analyze only RC-1's frozen 18-request Rubber matrix.

Plan/analyze use ordinary Python; render runs inside Blender 4.2.11.
Every render starts from the hashed base scene. Outputs cannot be resumed.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import math
import os
from pathlib import Path
import struct
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from scripts.methods.colorpeel_ice import render_d1_color_calibration_preflight as shared

require = shared.require
ReducedDirectError = shared.PreflightError
canonical_sha256 = shared.canonical_sha256
file_sha256 = shared.file_sha256
atomic_json = shared.atomic_json
load_json = shared.load_json
PLAN_NAME = "reduced_direct_plan.json"
CONTRACT_NAME = "reduced_direct_contract.json"
MANIFEST_NAME = "reduced_direct_render_manifest.json"
SUMMARY_NAME = "reduced_direct_analysis.json"
PREFIX = "d1_rubber_reduced_direct"
REQUEST_HASH = "0cbbd38a7411fe02faaa4573130a40c3e6da53adc86fdeda6174af6f70254585"
ASSETS = {"base_scene": "data/base_scene.blend", "shape_asset": "data/shapes/Sphere.blend",
          "material_asset": "data/materials/Rubber.blend"}


def _script_hash() -> str:
    return file_sha256(Path(__file__).resolve())


def _requests() -> list[dict[str, Any]]:
    # Keep RC-1/Pillow imports out of Blender's Python environment.
    from src.methods.colorpeel_ice import renderer_color_calibration as calibration
    rows = calibration.build_reduced_direct_requests()
    require(len(rows) == 18 and canonical_sha256(rows) == REQUEST_HASH, "Frozen requests differ")
    require(canonical_sha256(calibration.load_protocol()) == shared.FROZEN_PROTOCOL_SHA256, "Protocol differs")
    return rows


def make_plan(output_root: Path, asset_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    shared._new_or_empty(output_root)
    requests = _requests()
    assets = {}
    for name, relative in ASSETS.items():
        path = shared._inside(asset_root, relative, name)
        require(path.is_file(), f"Missing asset {name}")
        assets[name] = {"relative_path": relative, "sha256": file_sha256(path)}
    contract = {"schema": f"{PREFIX}_contract/v1", "git_commit": shared._git_commit(),
                "adapter_script_sha256": _script_hash(), "assets": assets, "material": "Rubber",
                "protocol_canonical_sha256": shared.FROZEN_PROTOCOL_SHA256,
                "requests_sha256": REQUEST_HASH, "request_count": 18}
    plan = {"schema": f"{PREFIX}_plan/v1", "requests": requests, "requests_sha256": REQUEST_HASH,
            "protocol_canonical_sha256": shared.FROZEN_PROTOCOL_SHA256,
            "contract_sha256": canonical_sha256(contract)}
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(output_root / CONTRACT_NAME, contract)
    atomic_json(output_root / PLAN_NAME, plan)
    return plan, contract


def _load_plan_contract(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    plan, contract = load_json(root / PLAN_NAME), load_json(root / CONTRACT_NAME)
    require(set(plan) == {"schema", "requests", "requests_sha256", "protocol_canonical_sha256", "contract_sha256"},
            "Plan fields differ")
    require(plan["schema"] == f"{PREFIX}_plan/v1", "Plan schema differs")
    require(isinstance(plan["requests"], list) and len(plan["requests"]) == 18
            and canonical_sha256(plan["requests"]) == REQUEST_HASH, "Plan request payload differs")
    require(plan["requests_sha256"] == REQUEST_HASH
            and plan["protocol_canonical_sha256"] == shared.FROZEN_PROTOCOL_SHA256, "Plan hashes differ")
    require(set(contract) == {"schema", "git_commit", "adapter_script_sha256", "assets", "material",
                              "protocol_canonical_sha256", "requests_sha256", "request_count"}, "Contract fields differ")
    require(contract["schema"] == f"{PREFIX}_contract/v1" and contract["material"] == "Rubber", "Contract identity differs")
    require(contract["protocol_canonical_sha256"] == shared.FROZEN_PROTOCOL_SHA256
            and contract["requests_sha256"] == REQUEST_HASH and contract["request_count"] == 18, "Contract hashes differ")
    require(contract["git_commit"] == shared._git_commit(), "Current repository HEAD differs")
    require(contract["adapter_script_sha256"] == _script_hash(), "Adapter script hash differs")
    require(plan["contract_sha256"] == canonical_sha256(contract), "Plan contract hash differs")
    require(isinstance(contract["assets"], dict) and set(contract["assets"]) == set(ASSETS), "Contract assets differ")
    for name, relative in ASSETS.items():
        row = contract["assets"][name]
        require(isinstance(row, dict) and set(row) == {"relative_path", "sha256"}
                and row["relative_path"] == relative, f"Contract {name} path differs")
        require(isinstance(row["sha256"], str) and len(row["sha256"]) == 64
                and all(c in "0123456789abcdef" for c in row["sha256"]), f"Contract {name} hash differs")
    return plan, contract


def _verify_runtime_assets(asset_root: Path, contract: Mapping[str, Any]) -> list[Path]:
    paths = []
    for name in ASSETS:
        row = contract["assets"][name]
        path = shared._inside(asset_root, row["relative_path"], name)
        require(path.is_file() and file_sha256(path) == row["sha256"], f"Runtime {name} hash differs")
        paths.append(path)
    return paths


def rubber_socket_rgba(request: Mapping[str, Any]) -> tuple[float, ...]:
    require(request.get("material") == "Rubber", "Request material must be Rubber")
    return shared.emission_socket_rgba(request)


def _float32(values: Sequence[float]) -> list[float]:
    return [struct.unpack("f", struct.pack("f", value))[0] for value in values]


def _configure_rubber(obj, request: Mapping[str, Any], asset: Path) -> dict[str, Any]:
    bpy = shared.bpy
    require(bpy.data.node_groups.get("Rubber") is None, "Base scene already contains Rubber node group")
    before = set(bpy.data.node_groups.keys())
    bpy.ops.wm.append(directory=str(asset / "NodeTree") + os.sep, filename="Rubber", link=False)
    added = [group for group in bpy.data.node_groups if group.name not in before]
    require(len(added) == 1 and added[0].name == "Rubber", "Append must produce unique Rubber node group")
    material = bpy.data.materials.new("D1_Rubber")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    group = nodes.new("ShaderNodeGroup")
    group.node_tree = added[0]
    group.inputs["Color"].default_value = rubber_socket_rgba(request)
    material.node_tree.links.new(group.outputs["Shader"], output.inputs["Surface"])
    obj.data.materials.clear()
    obj.data.materials.append(material)
    observed = [float(value) for value in group.inputs["Color"].default_value]
    require(observed == _float32(rubber_socket_rgba(request)), "Rubber Color socket readback differs")
    return {"node_group": group.node_tree.name, "input": "Color", "output": "Shader",
            "socket_rgba": observed, "surface_linked": True}


def view_location(base_camera: Mapping[str, Any], view_index: int) -> list[float]:
    require(type(view_index) is int and view_index in (0, 8, 16), "Invalid reduced view")
    azimuth = math.radians(base_camera["base_scene_camera_azimuth_degrees"] + 18 * view_index)
    elevation = math.radians(base_camera["base_scene_camera_elevation_degrees"])
    radius = base_camera["base_scene_camera_radius"]
    target = base_camera["look_at"]
    return [target[0] + radius * math.cos(elevation) * math.cos(azimuth),
            target[1] + radius * math.cos(elevation) * math.sin(azimuth),
            target[2] + radius * math.sin(elevation)]


def _camera_and_lights(obj, view_index: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    base, lights = shared._camera_and_lights(obj)
    camera = shared.bpy.data.objects["Camera"]
    camera.location = view_location(base, view_index)
    camera.rotation_quaternion = (shared.Vector(base["look_at"]) - camera.location).to_track_quat("-Z", "Y")
    shared.bpy.context.scene.camera = camera
    shared.bpy.context.view_layer.update()
    actual = dict(base, view_index=view_index, location=[float(v) for v in camera.location],
                  forward=[float(v) for v in camera.rotation_quaternion @ shared.Vector((0, 0, -1))])
    return actual, lights, {"camera": base, "lights": deepcopy(lights)}


def _render_one(root: Path, request: dict[str, Any], contract: dict[str, Any],
                base_scene: Path, shape_asset: Path, material_asset: Path) -> dict[str, Any]:
    bpy = shared.bpy
    partial = root / ".partial" / request["request_id"]
    final = root / "renders" / request["request_id"]
    require(not partial.exists() and not final.exists(), "Render output already exists")
    partial.mkdir(parents=True)
    bpy.ops.wm.open_mainfile(filepath=str(base_scene))
    version = shared.canonical_blender_version(bpy.app.version)
    scene = bpy.context.scene
    devices = shared._configure_cycles(scene, request["render_seed"])
    color_management = shared.configure_color_management(scene)
    for obj in list(scene.objects):
        if obj.type == "MESH" and obj.name != "Ground":
            bpy.data.objects.remove(obj, do_unlink=True)
    world, ground = shared._configure_background(scene)
    obj = shared._append_sphere(shape_asset)
    material = _configure_rubber(obj, request, material_asset)
    camera, lights, base = _camera_and_lights(obj, request["view_index"])
    shared._configure_mask_output(partial, obj)
    scene.render.filepath = str(partial / "image.png")
    bpy.ops.render.render(write_still=True)
    mask = shared._finalize_mask(partial)
    foreground = shared._validate_blender_mask(mask)
    relative = f"renders/{request['request_id']}"
    metadata = {"request": request, "render_contract_sha256": canonical_sha256(contract),
                "blender_version": version, "blender_build_identifier": str(bpy.app.build_hash),
                "renderer": {"engine": scene.render.engine, "cuda_devices": devices, "samples": scene.cycles.samples,
                             "resolution": [scene.render.resolution_x, scene.render.resolution_y],
                             "image": {"format": scene.render.image_settings.file_format,
                                       "mode": scene.render.image_settings.color_mode,
                                       "bits_per_channel": int(scene.render.image_settings.color_depth)}},
                "render_seed": scene.cycles.seed, "material": material, "base_scene_state": base,
                "object": {"shape": "sphere", "location": list(obj.location), "scale": list(obj.scale),
                           "rotation_euler": list(obj.rotation_euler)},
                "color_management": color_management, "camera": camera, "lights": lights,
                "world": world, "ground": ground, "foreground_pixels": foreground,
                "image_relative_path": f"{relative}/image.png", "image_sha256": file_sha256(partial / "image.png"),
                "mask_relative_path": f"{relative}/mask.png", "mask_sha256": file_sha256(mask)}
    for name, asset in contract["assets"].items():
        metadata[f"{name}_relative_path"] = asset["relative_path"]
        metadata[f"{name}_sha256"] = asset["sha256"]
    _verify_runtime_metadata(metadata, request, contract, foreground)
    atomic_json(partial / "metadata.json", metadata)
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(partial, final)
    return {"request_id": request["request_id"],
            **{key: metadata[key] for key in ("image_relative_path", "image_sha256", "mask_relative_path", "mask_sha256")},
            "metadata_relative_path": f"{relative}/metadata.json", "metadata_sha256": file_sha256(final / "metadata.json")}


def render(root: Path, asset_root: Path) -> dict[str, Any]:
    require(shared.bpy is not None, "Render requires Blender")
    root = root.resolve()
    plan, contract = _load_plan_contract(root)
    paths = _verify_runtime_assets(asset_root, contract)
    require({p.name for p in root.iterdir()} == {PLAN_NAME, CONTRACT_NAME}, "Render root contains existing output")
    records = [_render_one(root, request, contract, *paths) for request in plan["requests"]]
    manifest = {"schema": f"{PREFIX}_manifest/v1", "contract_sha256": canonical_sha256(contract),
                "request_count": 18, "records": records}
    atomic_json(root / MANIFEST_NAME, manifest)
    return manifest


def _verify_runtime_metadata(metadata: dict[str, Any], request: dict[str, Any],
                             contract: dict[str, Any], object_pixels: int) -> None:
    extra = {"material_asset_relative_path", "material_asset_sha256", "material", "base_scene_state", "object", "render_seed"}
    require(extra <= set(metadata), "Metadata Rubber fields missing")
    base = metadata["base_scene_state"]
    require(isinstance(base, dict) and set(base) == {"camera", "lights"}, "Metadata base scene state differs")
    camera = metadata["camera"]
    require(isinstance(camera, dict) and set(camera) == set(base["camera"]) | {"forward"}, "Metadata camera fields differ")
    require(camera["view_index"] == request["view_index"], "Metadata camera view differs")
    require({k: v for k, v in camera.items() if k not in {"location", "view_index", "forward"}}
            == {k: v for k, v in base["camera"].items() if k not in {"location", "view_index"}}, "Metadata camera base fields differ")
    location = shared._finite_vector(camera["location"], 3, "camera location")
    forward = shared._finite_vector(camera["forward"], 3, "camera forward")
    expected = view_location(base["camera"], request["view_index"])
    tolerance = shared.CAMERA_DERIVED_FLOAT32_ABS_TOLERANCE
    require(all(math.isclose(a, b, rel_tol=0, abs_tol=tolerance) for a, b in zip(location, expected)), "Metadata camera mapping differs")
    direction = [target - value for target, value in zip(camera["look_at"], location)]
    length = math.sqrt(sum(value * value for value in direction))
    require(length > 0 and all(math.isclose(a, b / length, rel_tol=0, abs_tol=tolerance)
                              for a, b in zip(forward, direction)), "Metadata camera look-at differs")
    require(metadata["lights"] == base["lights"], "Metadata lights changed from base scene")
    require(metadata["render_seed"] == request["render_seed"], "Metadata render seed differs")
    require(metadata["object"] == {"shape": "sphere", "location": _float32([0, 0, 1.3]),
                                   "scale": _float32([1.3] * 3), "rotation_euler": [0, 0, 0]}, "Metadata object differs")
    require(camera["look_at"] == metadata["object"]["location"], "Metadata target is not object center")
    require(metadata["material"] == {"node_group": "Rubber", "input": "Color", "output": "Shader",
                                     "socket_rgba": _float32(rubber_socket_rgba(request)), "surface_linked": True},
            "Metadata Rubber socket or graph differs")
    asset = contract["assets"]["material_asset"]
    require(metadata["material_asset_relative_path"] == asset["relative_path"]
            and metadata["material_asset_sha256"] == asset["sha256"], "Metadata material asset differs")
    # Reuse the accepted validator solely for common renderer/base-scene evidence.
    # The projection is in memory; persisted metadata always describes Rubber.
    common = {key: value for key, value in metadata.items() if key not in extra}
    common["camera"] = base["camera"]
    common["emission_socket_rgba"] = list(rubber_socket_rgba(request))
    shared._verify_runtime_metadata(common, request, contract, object_pixels)


def measure_pixels(pixels, mask_values, request: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    mask = np.asarray(mask_values)
    require(mask.size == 512 * 512 and set(np.unique(mask).tolist()) <= {0, 255}, "Mask must be binary 512x512")
    interior, geometry = shared._mask_interior((mask.reshape(-1) == 255).tolist(), 512, 512)
    rgb = np.asarray(pixels).reshape(-1, 3)
    require(rgb.shape == (512 * 512, 3), "RGB dimensions differ")
    # Evaluate unique RGB bytes once, preserving per-pixel Lab statistics.
    unique, inverse = np.unique(rgb[interior], axis=0, return_inverse=True)
    lab = np.asarray([shared._linear_to_lab([shared._srgb_byte_to_linear(int(c)) for c in value])
                      for value in unique])[inverse.reshape(-1)]
    return _measure_lab(lab, geometry, request)


def _measure_lab(lab, geometry: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    from src.methods.colorpeel_ice import renderer_color_calibration as calibration
    eligible = lab[(lab[:, 0] > 5) & (lab[:, 0] < 95)]
    geometry = calibration.derive_interior_contract(
        geometry["object_area"], geometry["bbox_w"], geometry["bbox_h"], image_area=512 * 512,
        background_area=geometry["background_area"], interior_area=len(lab), eligible_count=len(eligible))
    require(geometry["eligible_ok"], "Eligible pixel count fails frozen threshold")
    median = np.median(eligible, axis=0)
    iqr = np.percentile(eligible, 75, axis=0) - np.percentile(eligible, 25, axis=0)
    row = {**request, **{f"median_{key}": float(median[i]) for i, key in enumerate("Lab")},
           **{f"iqr_{key}": float(iqr[i]) for i, key in enumerate("Lab")},
           "object_ratio_ok": geometry["object_ratio_ok"], "interior_ok": geometry["interior_ok"],
           "eligible_ok": geometry["eligible_ok"], "measurement_stats": geometry}
    return calibration.summarize_render_measurement(row)


def _decode_record(root: Path, request: dict[str, Any], contract: dict[str, Any], record: dict[str, Any]):
    Image = shared._require_pillow()
    require(set(record) == {"request_id", "image_relative_path", "image_sha256", "mask_relative_path", "mask_sha256",
                           "metadata_relative_path", "metadata_sha256"}, "Manifest record fields differ")
    require(record["request_id"] == request["request_id"], "Manifest request differs")
    paths = {}
    for kind, suffix in (("image", "png"), ("mask", "png"), ("metadata", "json")):
        relative = f"renders/{request['request_id']}/{kind}.{suffix}"
        require(record[f"{kind}_relative_path"] == relative, f"Manifest {kind} path differs")
        paths[kind] = shared._artifact_path(root, relative, kind)
        require(file_sha256(paths[kind]) == record[f"{kind}_sha256"], f"Manifest {kind} hash differs")
    require({p.name for p in paths["image"].parent.iterdir()} == {"image.png", "mask.png", "metadata.json"},
            "Render artifacts differ")
    metadata = load_json(paths["metadata"])
    for key in ("image_relative_path", "image_sha256", "mask_relative_path", "mask_sha256"):
        require(metadata.get(key) == record[key], f"Metadata {key} differs")
    with Image.open(paths["image"]) as image, Image.open(paths["mask"]) as mask:
        require(image.format == "PNG" and image.mode == "RGB" and image.size == (512, 512), "Image must be RGB 512x512 PNG")
        require(mask.format == "PNG" and mask.mode == "L" and mask.size == (512, 512), "Mask must be L 512x512 PNG")
        import numpy as np
        pixels, mask_values = np.asarray(image), np.asarray(mask)
    _verify_runtime_metadata(metadata, request, contract, int((mask_values == 255).sum()))
    row = measure_pixels(pixels, mask_values, request)
    row["provenance"] = {"render_contract_sha256": canonical_sha256(contract), **record}
    row["decision"] = "measurement_valid"
    return row, metadata["base_scene_state"]


def analyze(root: Path) -> dict[str, Any]:
    root = root.resolve()
    require(not (root / SUMMARY_NAME).exists(), "Analysis output already exists")
    plan, contract = _load_plan_contract(root)
    require(plan["requests"] == _requests(), "Frozen request payload differs")
    manifest = load_json(root / MANIFEST_NAME)
    require(set(manifest) == {"schema", "contract_sha256", "request_count", "records"}, "Manifest fields differ")
    require(manifest["schema"] == f"{PREFIX}_manifest/v1" and manifest["request_count"] == 18
            and manifest["contract_sha256"] == canonical_sha256(contract), "Manifest contract differs")
    records = manifest["records"]
    require(isinstance(records, list) and len(records) == 18, "Manifest needs 18 records")
    by_id = {r["request_id"]: r for r in plan["requests"]}
    seen, rows, base = set(), [], None
    for record in records:
        require(isinstance(record, dict) and isinstance(record.get("request_id"), str), "Invalid manifest record")
        identity = record["request_id"]
        require(identity in by_id and identity not in seen, "Manifest duplicate or extra request")
        seen.add(identity)
        row, state = _decode_record(root, by_id[identity], contract, record)
        require(base is None or base == state, "Base scene evidence differs across requests")
        base = state
        rows.append(row)
    require({p.name for p in (root / "renders").iterdir()} == seen, "Render coverage differs")
    from src.methods.colorpeel_ice import renderer_color_calibration as calibration
    gate = calibration.summarize_chroma_measurements(rows, "reduced_direct")
    result = {"schema": f"{PREFIX}_analysis/v1", "status": "passed" if gate["overall_pass"] else "failed",
              "contract_sha256": canonical_sha256(contract), "manifest_sha256": file_sha256(root / MANIFEST_NAME),
              "measurements": rows, "gate": gate}
    atomic_json(root / SUMMARY_NAME, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "render", "analyze"):
        command = commands.add_parser(name)
        command.add_argument("--output-root", required=True, type=Path)
        if name != "analyze":
            command.add_argument("--asset-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(shared._extract_args(argv))
    try:
        if args.command == "plan":
            make_plan(args.output_root, args.asset_root)
        elif args.command == "render":
            render(args.output_root, args.asset_root)
        else:
            result = analyze(args.output_root)
            print(result["status"])
            return 0 if result["gate"]["overall_pass"] else 1
        print(f"{args.command} complete")
        return 0
    except Exception as exc:
        # Preserve failed runs without overwriting earlier analysis/failure evidence.
        failure = args.output_root / f"reduced_direct_{args.command}_failure.json"
        if args.command != "plan" and args.output_root.is_dir() and not failure.exists():
            atomic_json(failure, {"schema": f"{PREFIX}_failure/v1", "status": "failed",
                                  "command": args.command, "error_type": type(exc).__name__})
        print(f"reduced direct aborted: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
