#!/usr/bin/env python3
"""Run only the frozen D1 four-probe emission-encoding preflight.

Use ordinary Python for planning and analysis, and Blender only for rendering:

  python render_d1_color_calibration_preflight.py plan --output-root OUT --asset-root CLEVR \
    --base-scene-relative-path data/base_scene.blend --shape-relative-path data/shapes/Sphere.blend
  blender --background --python-exit-code 1 --python render_d1_color_calibration_preflight.py -- \
    render --output-root OUT --asset-root CLEVR
  python render_d1_color_calibration_preflight.py analyze --output-root OUT

This is deliberately not a Rubber/photometric renderer.  It realizes exactly
the RC-1 emission probes, with their linear RGBA values connected directly to
an Emission shader.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

try:  # Blender is intentionally optional for ordinary Python plan/analyze.
    import bpy
    from mathutils import Vector
except ImportError:
    bpy = None
    Vector = None


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PLAN_SCHEMA = "d1_emission_encoding_preflight_plan/v1"
CONTRACT_SCHEMA = "d1_emission_encoding_preflight_contract/v1"
FROZEN_PROTOCOL_SHA256 = "476d01c5a383ead7d1ceec5c276bb26d809cf2fabf51cd81e0c49695cc4aa48e"
FROZEN_PREFLIGHT_REQUESTS_SHA256 = "c2b5a4078f9407c2e92d555924a89ad5ba5640eeee1ef5c38e41402253ea08ff"
PLAN_NAME = "preflight_plan.json"
CONTRACT_NAME = "preflight_contract.json"
MANIFEST_NAME = "preflight_render_manifest.json"
SUMMARY_NAME = "preflight_analysis.json"
RENDER_ROOT = "renders"
# Blender persists object/camera coordinates as float32.  Re-deriving angles
# from their JSON-decoded coordinates can move elevation by about 1.2e-6 deg.
CAMERA_DERIVED_FLOAT32_ABS_TOLERANCE = 2e-6


class PreflightError(RuntimeError):
    """Raised when the frozen emission preflight is invalid or incomplete."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PreflightError(message)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _adapter_script_sha256() -> str:
    return file_sha256(Path(__file__).resolve())


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError(f"Cannot read JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON must be an object: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    require(not partial.exists(), f"Partial metadata already exists: {partial}")
    partial.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, path)


def _inside(root: Path, relative: str, label: str) -> Path:
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute(),
            f"{label} must be a nonempty relative path")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise PreflightError(f"{label} escapes its root: {relative}") from exc
    return path


def _new_or_empty(root: Path) -> None:
    require(not root.exists() or root.is_dir(), f"Output root is not a directory: {root}")
    require(not root.exists() or not any(root.iterdir()), f"Output root must be new or empty: {root}")


def _git_commit() -> str:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL
        ).decode("ascii").strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PreflightError("Cannot record repository HEAD") from exc
    require(len(commit) == 40 and all(char in "0123456789abcdef" for char in commit), "Invalid repository HEAD")
    return commit


def _preflight_requests() -> list[dict[str, Any]]:
    # This import remains inside the normal-Python planning path.  Blender must
    # not import RC-1 because that dependency chain may require Pillow.
    from src.methods.colorpeel_ice import renderer_color_calibration as calibration

    requests = calibration.build_preflight_requests()
    require(len(requests) == 4, "RC-1 preflight request count drifted")
    require(canonical_sha256(requests) == FROZEN_PREFLIGHT_REQUESTS_SHA256,
            "RC-1 preflight request hash drifted")
    return requests


def make_plan(output_root: Path, asset_root: Path, base_scene_relative_path: str,
              shape_relative_path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Freeze exactly the RC-1 requests and runtime asset identities."""
    output_root = output_root.resolve()
    asset_root = asset_root.resolve()
    _new_or_empty(output_root)
    require(asset_root.is_dir(), f"Asset root is not a directory: {asset_root}")
    base_scene = _inside(asset_root, base_scene_relative_path, "base scene path")
    shape_asset = _inside(asset_root, shape_relative_path, "shape asset path")
    require(base_scene.is_file(), f"Missing base scene: {base_scene}")
    require(shape_asset.is_file(), f"Missing shape asset: {shape_asset}")
    requests = _preflight_requests()
    from src.methods.colorpeel_ice import renderer_color_calibration as calibration

    protocol = calibration.load_protocol()
    require(canonical_sha256(protocol) == FROZEN_PROTOCOL_SHA256, "RC-1 protocol hash drifted")
    assets = {
        "base_scene": {"relative_path": base_scene_relative_path.replace("\\", "/"), "sha256": file_sha256(base_scene)},
        "shape_asset": {"relative_path": shape_relative_path.replace("\\", "/"), "sha256": file_sha256(shape_asset)},
    }
    contract = {
        "schema": CONTRACT_SCHEMA,
        "git_commit": _git_commit(),
        "protocol_canonical_sha256": FROZEN_PROTOCOL_SHA256,
        "requests_sha256": canonical_sha256(requests),
        "request_count": 4,
        "assets": assets,
        "emission_material": "built_in_blender_emission",
        "adapter_script_sha256": _adapter_script_sha256(),
    }
    plan = {
        "schema": PLAN_SCHEMA,
        "protocol_canonical_sha256": FROZEN_PROTOCOL_SHA256,
        "requests": requests,
        "requests_sha256": canonical_sha256(requests),
        "contract_sha256": canonical_sha256(contract),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(output_root / CONTRACT_NAME, contract)
    atomic_json(output_root / PLAN_NAME, plan)
    return plan, contract


def _load_plan_contract(output_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    output_root = output_root.resolve()
    plan = load_json(output_root / PLAN_NAME)
    contract = load_json(output_root / CONTRACT_NAME)
    require(set(plan) == {"schema", "protocol_canonical_sha256", "requests", "requests_sha256", "contract_sha256"},
            "Plan fields differ")
    require(plan["schema"] == PLAN_SCHEMA, "Plan schema differs")
    require(plan["protocol_canonical_sha256"] == FROZEN_PROTOCOL_SHA256, "Plan protocol hash differs")
    require(isinstance(plan["requests"], list) and len(plan["requests"]) == 4, "Plan must have exactly four requests")
    require(canonical_sha256(plan["requests"]) == FROZEN_PREFLIGHT_REQUESTS_SHA256,
            "Plan request payload hash differs")
    require(plan["requests_sha256"] == FROZEN_PREFLIGHT_REQUESTS_SHA256, "Plan request hash field differs")
    require(set(contract) == {"schema", "git_commit", "protocol_canonical_sha256", "requests_sha256", "request_count", "assets", "emission_material", "adapter_script_sha256"},
            "Contract fields differ")
    require(contract["schema"] == CONTRACT_SCHEMA, "Contract schema differs")
    require(contract["protocol_canonical_sha256"] == FROZEN_PROTOCOL_SHA256, "Contract protocol hash differs")
    require(contract["requests_sha256"] == FROZEN_PREFLIGHT_REQUESTS_SHA256 and contract["request_count"] == 4,
            "Contract request identity differs")
    require(isinstance(contract["git_commit"], str) and len(contract["git_commit"]) == 40
            and all(char in "0123456789abcdef" for char in contract["git_commit"]), "Contract git commit differs")
    require(contract["git_commit"] == _git_commit(), "Current repository HEAD differs from contract")
    require(contract["adapter_script_sha256"] == _adapter_script_sha256(), "Adapter script hash differs from contract")
    require(plan["contract_sha256"] == canonical_sha256(contract), "Plan contract hash differs")
    require(contract["emission_material"] == "built_in_blender_emission", "Contract material differs")
    assets = contract["assets"]
    require(isinstance(assets, dict) and set(assets) == {"base_scene", "shape_asset"}, "Contract assets differ")
    for name in ("base_scene", "shape_asset"):
        row = assets[name]
        require(isinstance(row, dict) and set(row) == {"relative_path", "sha256"}, f"Contract {name} differs")
        require(isinstance(row["relative_path"], str) and not Path(row["relative_path"]).is_absolute(),
                f"Contract {name} path is not relative")
        require(isinstance(row["sha256"], str) and len(row["sha256"]) == 64, f"Contract {name} hash differs")
    return plan, contract


def _verify_runtime_assets(asset_root: Path, contract: Mapping[str, Any]) -> tuple[Path, Path]:
    asset_root = asset_root.resolve()
    require(asset_root.is_dir(), f"Asset root is not a directory: {asset_root}")
    found = []
    for name in ("base_scene", "shape_asset"):
        row = contract["assets"][name]
        path = _inside(asset_root, row["relative_path"], f"{name} path")
        require(path.is_file(), f"Missing runtime {name}: {path}")
        require(file_sha256(path) == row["sha256"], f"Runtime {name} hash differs")
        found.append(path)
    return found[0], found[1]


def emission_socket_rgba(request: Mapping[str, Any]) -> tuple[float, float, float, float]:
    """Return the frozen linear socket value; never convert, scale, or clip it."""
    rgba = request.get("socket_rgba")
    linear = request.get("linear_rgb")
    require(isinstance(rgba, list) and len(rgba) == 4 and isinstance(linear, list) and len(linear) == 3,
            "Emission request lacks linear RGBA")
    require(rgba[:3] == linear and rgba[3] == 1.0, "Emission socket value differs from linear RGB")
    require(all(type(channel) in {int, float} and math.isfinite(float(channel)) and 0.0 <= float(channel) <= 1.0
                for channel in rgba), "Emission socket is not in-gamut linear RGBA")
    return tuple(float(channel) for channel in rgba)


def canonical_blender_version(version: Sequence[Any]) -> str:
    require(tuple(version) == (4, 2, 11), f"Blender must be 4.2.11, got {tuple(version)}")
    return "4.2.11"


def configure_color_management(scene) -> dict[str, Any]:
    """Apply the RC-1 color-management contract or report supported choices."""
    try:
        scene.display_settings.display_device = "sRGB"
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"
        scene.view_settings.exposure = 0.0
        scene.view_settings.gamma = 1.0
        require(scene.display_settings.display_device == "sRGB", "display device did not remain sRGB")
        require(scene.view_settings.view_transform == "Standard", "view transform did not remain Standard")
        require(scene.view_settings.look == "None", "look did not remain None")
        require(float(scene.view_settings.exposure) == 0.0 and float(scene.view_settings.gamma) == 1.0,
                "exposure or gamma differs")
    except (AttributeError, TypeError, ValueError, PreflightError) as exc:
        available = {
            "display_device": getattr(scene.display_settings, "display_device", None),
            "view_transform": getattr(scene.view_settings, "view_transform", None),
            "look": getattr(scene.view_settings, "look", None),
        }
        raise PreflightError(f"Unsupported color-management setting; available/current={available}") from exc
    return {"display_device": "sRGB", "view_transform": "Standard", "look": None, "exposure": 0.0, "gamma": 1.0}


def _configure_cycles(scene, seed: int) -> list[dict[str, str]]:
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.cycles.samples = 512
    scene.cycles.seed = int(seed)
    scene.cycles.device = "GPU"
    try:
        preferences = bpy.context.preferences.addons["cycles"].preferences
        preferences.compute_device_type = "CUDA"
        preferences.get_devices()
        devices = [device for device in preferences.devices if device.type == "CUDA"]
        require(devices, "No CUDA device is available to Cycles")
        for device in preferences.devices:
            device.use = device.type == "CUDA"
    except (KeyError, RuntimeError, TypeError, AttributeError, PreflightError) as exc:
        raise PreflightError(f"CUDA Cycles initialization failed: {exc}") from exc
    return [{"name": str(device.name), "type": str(device.type), "id": str(device.id)} for device in devices]


def _append_sphere(shape_asset: Path):
    before = set(bpy.data.objects.keys())
    bpy.ops.wm.append(directory=str(shape_asset / "Object") + os.sep, filename="Sphere", link=False)
    added = [obj for obj in bpy.data.objects if obj.name not in before]
    require(len(added) == 1 and added[0].type == "MESH", "Sphere append did not produce exactly one mesh")
    obj = added[0]
    obj.location = (0.0, 0.0, 1.3)
    obj.rotation_euler = (0.0, 0.0, 0.0)
    obj.scale = (1.3, 1.3, 1.3)
    return obj


def _configure_emission(obj, request: Mapping[str, Any]) -> None:
    rgba = emission_socket_rgba(request)
    material = bpy.data.materials.new("D1_Preflight_Emission")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = rgba
    emission.inputs["Strength"].default_value = 1.0
    material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    obj.data.materials.clear()
    obj.data.materials.append(material)


def _configure_background(scene) -> tuple[dict[str, Any], dict[str, Any]]:
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True
    background = next((node for node in world.node_tree.nodes if node.type == "BACKGROUND"), None)
    if background is None:
        background = world.node_tree.nodes.new("ShaderNodeBackground")
    world_rgba = [0.05, 0.05, 0.05, 1.0]
    background.inputs["Color"].default_value = tuple(world_rgba)
    ground = bpy.data.objects.get("Ground")
    require(ground is not None and ground.type == "MESH", "Base scene is missing mesh Ground")
    material = bpy.data.materials.new("D1_Preflight_Neutral_Ground")
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    require(principled is not None, "Neutral ground Principled BSDF is missing")
    ground_rgba = [0.5, 0.5, 0.5, 1.0]
    principled.inputs["Base Color"].default_value = tuple(ground_rgba)
    principled.inputs["Metallic"].default_value = 0.0
    principled.inputs["Roughness"].default_value = 1.0
    ground.data.materials.clear()
    ground.data.materials.append(material)
    return ({"rgba": world_rgba}, {"rgba": ground_rgba, "metallic": 0.0, "roughness": 1.0})


def _camera_and_lights(obj) -> tuple[dict[str, Any], dict[str, Any]]:
    camera = bpy.data.objects.get("Camera")
    require(camera is not None and camera.type == "CAMERA" and camera.parent is None, "Base scene lacks an unparented Camera")
    require(abs(float(camera.data.shift_x)) <= 1e-9 and abs(float(camera.data.shift_y)) <= 1e-9,
            "Base scene camera lens shift must be zero")
    for constraint in camera.constraints:
        constraint.mute = True
    bpy.context.view_layer.update()
    camera.data.dof.use_dof = False
    target = Vector(obj.matrix_world.translation)
    location = Vector(camera.matrix_world.translation)
    direction = target - location
    require(direction.length > 0.0, "Camera cannot occupy object center")
    delta = location - target
    horizontal = math.hypot(float(delta.x), float(delta.y))
    base_radius = float(delta.length)
    base_azimuth = math.degrees(math.atan2(float(delta.y), float(delta.x)))
    base_elevation = math.degrees(math.atan2(float(delta.z), horizontal))
    camera.rotation_mode = "QUATERNION"
    camera.rotation_quaternion = direction.to_track_quat("-Z", "Y")
    lights = []
    for name in ("Lamp_Key", "Lamp_Back", "Lamp_Fill", "Area"):
        light = bpy.data.objects.get(name)
        require(light is not None and light.type == "LIGHT", f"Base scene is missing light {name}")
        light.data.color = (1.0, 1.0, 1.0)
        lights.append({"name": name, "position": [float(v) for v in light.location], "energy": float(light.data.energy),
                       "rgb": [1.0, 1.0, 1.0]})
    return ({"name": "Camera", "view_index": 0, "jitter": "none", "location": [float(v) for v in camera.location],
             "look_at": [float(v) for v in target], "lens_mm": float(camera.data.lens),
             "sensor_width_mm": float(camera.data.sensor_width), "sensor_height_mm": float(camera.data.sensor_height),
             "shift_x": float(camera.data.shift_x), "shift_y": float(camera.data.shift_y),
             "base_scene_camera_azimuth_degrees": base_azimuth,
             "base_scene_camera_radius": base_radius,
             "base_scene_camera_elevation_degrees": base_elevation},
            {"jitter": "none", "records": lights})


def _configure_mask_output(partial: Path, obj) -> None:
    scene = bpy.context.scene
    scene.use_nodes = True
    scene.view_layers[0].use_pass_object_index = True
    obj.pass_index = 1
    nodes, links = scene.node_tree.nodes, scene.node_tree.links
    nodes.clear()
    layers = nodes.new("CompositorNodeRLayers")
    composite = nodes.new("CompositorNodeComposite")
    links.new(layers.outputs["Image"], composite.inputs["Image"])
    object_mask = nodes.new("CompositorNodeIDMask")
    object_mask.index = 1
    object_mask.use_antialiasing = False
    links.new(layers.outputs["IndexOB"], object_mask.inputs["ID value"])
    output = nodes.new("CompositorNodeOutputFile")
    output.base_path = str(partial)
    output.file_slots[0].path = "__mask_"
    output.format.file_format = "PNG"
    output.format.color_mode = "BW"
    output.format.color_depth = "8"
    links.new(object_mask.outputs["Alpha"], output.inputs[0])


def _finalize_mask(partial: Path) -> Path:
    masks = sorted(partial.glob("__mask_*.png"))
    require(len(masks) == 1, f"Expected exactly one mask output, got {len(masks)}")
    target = partial / "mask.png"
    os.replace(masks[0], target)
    return target


def _validate_blender_mask(mask_path: Path) -> int:
    image = bpy.data.images.load(str(mask_path), check_existing=False)
    try:
        require(tuple(image.size) == (512, 512), "Mask must be 512x512")
        pixels = image.pixels[:]
        count = 0
        for index in range(0, len(pixels), 4):
            value = pixels[index]
            require(abs(value) < 1e-6 or abs(value - 1.0) < 1e-6, "Mask is not binary")
            count += value > 0.5
        ratio = count / (512 * 512)
        require(0.005 <= ratio < 0.90, "Object mask ratio is outside [0.005, 0.90)")
        return count
    finally:
        bpy.data.images.remove(image)


def _render_one(output_root: Path, request: dict[str, Any], contract: dict[str, Any],
                base_scene: Path, shape_asset: Path) -> dict[str, Any]:
    bpy.ops.wm.open_mainfile(filepath=str(base_scene))
    blender_version = canonical_blender_version(bpy.app.version)
    scene = bpy.context.scene
    devices = _configure_cycles(scene, request["render_seed"])
    color_management = configure_color_management(scene)
    for scene_obj in list(scene.objects):
        if scene_obj.type == "MESH" and scene_obj.name != "Ground":
            bpy.data.objects.remove(scene_obj, do_unlink=True)
    world, ground = _configure_background(scene)
    obj = _append_sphere(shape_asset)
    _configure_emission(obj, request)
    camera, lights = _camera_and_lights(obj)
    final = output_root / RENDER_ROOT / request["request_id"]
    partial = output_root / ".partial" / request["request_id"]
    require(not final.exists() and not partial.exists(), f"Output already exists for {request['request_id']}")
    partial.mkdir(parents=True)
    try:
        _configure_mask_output(partial, obj)
        image = partial / "image.png"
        scene.render.filepath = str(image)
        bpy.ops.render.render(write_still=True)
        mask = _finalize_mask(partial)
        foreground = _validate_blender_mask(mask)
        relative = f"{RENDER_ROOT}/{request['request_id']}"
        metadata = {
            "request": request,
            "render_contract_sha256": canonical_sha256(contract),
            "blender_version": blender_version,
            "blender_build_identifier": str(getattr(bpy.app, "build_hash", "")),
            "renderer": {"engine": "CYCLES", "cuda_devices": devices, "samples": 512, "resolution": [512, 512],
                         "image": {"format": "PNG", "mode": "RGB", "bits_per_channel": 8}},
            "base_scene_relative_path": contract["assets"]["base_scene"]["relative_path"],
            "base_scene_sha256": contract["assets"]["base_scene"]["sha256"],
            "shape_asset_relative_path": contract["assets"]["shape_asset"]["relative_path"],
            "shape_asset_sha256": contract["assets"]["shape_asset"]["sha256"],
            "emission_socket_rgba": list(emission_socket_rgba(request)),
            "color_management": color_management,
            "camera": camera,
            "lights": lights,
            "world": world,
            "ground": ground,
            "foreground_pixels": foreground,
            "image_relative_path": f"{relative}/image.png",
            "image_sha256": file_sha256(image),
            "mask_relative_path": f"{relative}/mask.png",
            "mask_sha256": file_sha256(mask),
        }
        atomic_json(partial / "metadata.json", metadata)
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(partial, final)
        return {"request_id": request["request_id"], "image_relative_path": metadata["image_relative_path"],
                "image_sha256": metadata["image_sha256"], "mask_relative_path": metadata["mask_relative_path"],
                "mask_sha256": metadata["mask_sha256"], "metadata_relative_path": f"{relative}/metadata.json",
                "metadata_sha256": file_sha256(final / "metadata.json")}
    except Exception:
        # Preserve the partial directory as evidence and deliberately do not resume it.
        raise


def render(output_root: Path, asset_root: Path) -> dict[str, Any]:
    require(bpy is not None, "Rendering must run through Blender with bpy available")
    output_root = output_root.resolve()
    plan, contract = _load_plan_contract(output_root)
    base_scene, shape_asset = _verify_runtime_assets(asset_root, contract)
    allowed = {PLAN_NAME, CONTRACT_NAME}
    require({path.name for path in output_root.iterdir()} == allowed, "Render output root has unexpected existing content")
    records = [_render_one(output_root, request, contract, base_scene, shape_asset) for request in plan["requests"]]
    require(len(records) == 4 and len({row["request_id"] for row in records}) == 4, "Render request coverage differs")
    manifest = {"schema": "d1_emission_encoding_preflight_manifest/v1", "contract_sha256": canonical_sha256(contract),
                "request_count": 4, "records": records}
    atomic_json(output_root / MANIFEST_NAME, manifest)
    return manifest


def _require_pillow():
    try:
        from PIL import Image
    except ImportError as exc:
        raise PreflightError("analyze requires Pillow in ordinary Python") from exc
    return Image


def _mask_interior(mask: list[int], width: int, height: int) -> tuple[list[int], dict[str, Any]]:
    """Apply RC-1's tight-bbox radius and its production exact Euclidean EDT."""
    try:
        import numpy as np
        from src.methods.colorpeel_ice import renderer_color_calibration as calibration
        from src.methods.colorpeel_ice.natural_image_masks import exact_edt
    except ImportError as exc:
        raise PreflightError("analyze requires RC-1 NumPy EDT support") from exc
    raw = np.asarray(mask, dtype=bool).reshape((height, width))
    ys, xs = np.nonzero(raw)
    object_area = int(raw.sum())
    require(object_area > 0, "Object mask is empty")
    bbox_w = int(xs.max() - xs.min() + 1)
    bbox_h = int(ys.max() - ys.min() + 1)
    interior_contract = calibration.derive_interior_contract(
        object_area, bbox_w, bbox_h, image_area=width * height, background_area=width * height - object_area,
    )
    require(interior_contract["object_ratio_ok"], "Object mask ratio is outside [0.005, 0.90)")
    interior = np.flatnonzero(raw & (exact_edt(raw) >= interior_contract["radius"])).tolist()
    interior_contract = calibration.derive_interior_contract(
        object_area, bbox_w, bbox_h, image_area=width * height, background_area=width * height - object_area,
        interior_area=len(interior),
    )
    require(interior_contract["interior_ok"], "Mask exact EDT interior is too small")
    return interior, interior_contract


def _median_uint8(values: list[int]) -> int:
    require(values, "Cannot decode an empty interior")
    ordered = sorted(values)
    count = len(ordered)
    value = ordered[count // 2] if count % 2 else (ordered[count // 2 - 1] + ordered[count // 2]) / 2
    return int(math.floor(value + 0.5))


def _linear_to_lab(rgb: Sequence[float]) -> list[float]:
    r, g, b = [float(value) for value in rgb]
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    epsilon, kappa = 216 / 24389, 24389 / 27
    def f(value: float) -> float:
        return value ** (1 / 3) if value > epsilon else (kappa * value + 16) / 116
    fx, fy, fz = f(x / 0.95047), f(y), f(z / 1.08883)
    return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)]


def _srgb_byte_to_linear(value: int) -> float:
    channel = value / 255.0
    return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4


def _artifact_path(root: Path, relative: Any, label: str) -> Path:
    require(isinstance(relative, str), f"Missing {label} relative path")
    path = _inside(root, relative, label)
    require(path.is_file(), f"Missing {label}: {path}")
    return path


def _finite_vector(value: Any, count: int, label: str) -> list[float]:
    require(isinstance(value, list) and len(value) == count
            and all(type(item) in {int, float} and math.isfinite(float(item)) for item in value),
            f"{label} must be {count} finite numbers")
    return [float(item) for item in value]


def derived_camera_scalars_match(camera: Mapping[str, Any], location: Sequence[float], target: Sequence[float]) -> bool:
    """Compare camera scalars after Blender float32 coordinate round-tripping."""
    delta = [float(location[index]) - float(target[index]) for index in range(3)]
    radius = math.sqrt(sum(value * value for value in delta))
    if radius <= 0.0:
        return False
    expected = {
        "base_scene_camera_radius": radius,
        "base_scene_camera_azimuth_degrees": math.degrees(math.atan2(delta[1], delta[0])),
        "base_scene_camera_elevation_degrees": math.degrees(math.atan2(delta[2], math.hypot(delta[0], delta[1]))),
    }
    return all(math.isclose(float(camera[field]), value, rel_tol=0.0,
                            abs_tol=CAMERA_DERIVED_FLOAT32_ABS_TOLERANCE)
               for field, value in expected.items())


def _verify_runtime_metadata(metadata: dict[str, Any], request: dict[str, Any], contract: dict[str, Any],
                             object_pixels: int) -> None:
    expected_fields = {
        "request", "render_contract_sha256", "blender_version", "blender_build_identifier", "renderer",
        "base_scene_relative_path", "base_scene_sha256", "shape_asset_relative_path", "shape_asset_sha256",
        "emission_socket_rgba", "color_management", "camera", "lights", "world", "ground", "foreground_pixels",
        "image_relative_path", "image_sha256", "mask_relative_path", "mask_sha256",
    }
    require(set(metadata) == expected_fields, "Metadata fields differ")
    require(metadata["request"] == request, "Metadata request payload differs")
    require(metadata["render_contract_sha256"] == canonical_sha256(contract), "Metadata contract hash differs")
    require(metadata["blender_version"] == "4.2.11", "Metadata Blender version differs")
    require(isinstance(metadata["blender_build_identifier"], str) and metadata["blender_build_identifier"],
            "Metadata Blender build identifier differs")
    require(isinstance(metadata["renderer"], dict), "Metadata renderer evidence differs")
    require(metadata["renderer"] == {
        "engine": "CYCLES", "cuda_devices": metadata["renderer"].get("cuda_devices"), "samples": 512,
        "resolution": [512, 512], "image": {"format": "PNG", "mode": "RGB", "bits_per_channel": 8},
    }, "Metadata renderer evidence differs")
    devices = metadata["renderer"]["cuda_devices"]
    require(isinstance(devices, list) and devices, "Metadata CUDA device list is empty")
    for device in devices:
        require(isinstance(device, dict) and set(device) == {"name", "type", "id"}
                and device["type"] == "CUDA" and all(isinstance(device[key], str) and device[key] for key in device),
                "Metadata CUDA device differs")
    for asset_name, path_key, hash_key in (
        ("base_scene", "base_scene_relative_path", "base_scene_sha256"),
        ("shape_asset", "shape_asset_relative_path", "shape_asset_sha256"),
    ):
        require(metadata[path_key] == contract["assets"][asset_name]["relative_path"]
                and metadata[hash_key] == contract["assets"][asset_name]["sha256"], f"Metadata {asset_name} differs")
    require(metadata["emission_socket_rgba"] == list(emission_socket_rgba(request)), "Metadata emission socket differs")
    require(metadata["color_management"] == {"display_device": "sRGB", "view_transform": "Standard", "look": None,
                                              "exposure": 0.0, "gamma": 1.0}, "Metadata color management differs")
    camera = metadata["camera"]
    camera_fields = {"name", "view_index", "jitter", "location", "look_at", "lens_mm", "sensor_width_mm",
                     "sensor_height_mm", "shift_x", "shift_y", "base_scene_camera_azimuth_degrees",
                     "base_scene_camera_radius", "base_scene_camera_elevation_degrees"}
    require(isinstance(camera, dict) and set(camera) == camera_fields and camera["name"] == "Camera"
            and camera["view_index"] == 0 and camera["jitter"] == "none", "Metadata camera identity differs")
    location = _finite_vector(camera["location"], 3, "Metadata camera location")
    target = _finite_vector(camera["look_at"], 3, "Metadata camera target")
    scalar_fields = ["lens_mm", "sensor_width_mm", "sensor_height_mm", "shift_x", "shift_y",
                     "base_scene_camera_azimuth_degrees", "base_scene_camera_radius", "base_scene_camera_elevation_degrees"]
    require(all(type(camera[field]) in {int, float} and math.isfinite(float(camera[field])) for field in scalar_fields),
            "Metadata camera scalar differs")
    require(abs(float(camera["shift_x"])) <= 1e-9 and abs(float(camera["shift_y"])) <= 1e-9,
            "Metadata camera shift differs")
    require(derived_camera_scalars_match(camera, location, target),
            "Metadata base-scene camera mapping differs")
    lights = metadata["lights"]
    require(isinstance(lights, dict) and set(lights) == {"jitter", "records"} and lights["jitter"] == "none"
            and isinstance(lights["records"], list) and [row.get("name") for row in lights["records"]] == ["Lamp_Key", "Lamp_Back", "Lamp_Fill", "Area"],
            "Metadata lights differ")
    for light in lights["records"]:
        require(set(light) == {"name", "position", "energy", "rgb"}, "Metadata light fields differ")
        _finite_vector(light["position"], 3, "Metadata light position")
        require(type(light["energy"]) in {int, float} and math.isfinite(float(light["energy"])) and light["rgb"] == [1.0, 1.0, 1.0],
                "Metadata light evidence differs")
    require(metadata["world"] == {"rgba": [0.05, 0.05, 0.05, 1.0]}
            and metadata["ground"] == {"rgba": [0.5, 0.5, 0.5, 1.0], "metallic": 0.0, "roughness": 1.0},
            "Metadata world or ground differs")
    require(type(metadata["foreground_pixels"]) is int and metadata["foreground_pixels"] == object_pixels,
            "Metadata foreground pixel count differs")


def _decode_record(output_root: Path, request: dict[str, Any], contract: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    Image = _require_pillow()
    expected_dir = f"{RENDER_ROOT}/{request['request_id']}"
    require(record.get("request_id") == request["request_id"], "Manifest request ID differs")
    for field, name in (("image_relative_path", "image.png"), ("mask_relative_path", "mask.png"),
                        ("metadata_relative_path", "metadata.json")):
        require(record.get(field) == f"{expected_dir}/{name}", f"Manifest {field} differs")
    image_path = _artifact_path(output_root, record["image_relative_path"], "image")
    mask_path = _artifact_path(output_root, record["mask_relative_path"], "mask")
    metadata_path = _artifact_path(output_root, record["metadata_relative_path"], "metadata")
    for field, path in (("image", image_path), ("mask", mask_path), ("metadata", metadata_path)):
        require(record.get(f"{field}_sha256") == file_sha256(path), f"Manifest {field} hash differs")
    metadata = load_json(metadata_path)
    with Image.open(image_path) as image, Image.open(mask_path) as mask_image:
        require(image.format == "PNG" and image.size == (512, 512) and image.mode == "RGB", "Image must be RGB 512x512 PNG")
        require(mask_image.format == "PNG" and mask_image.size == (512, 512) and mask_image.mode == "L", "Mask must be L 512x512 PNG")
        pixels = list(image.get_flattened_data() if hasattr(image, "get_flattened_data") else image.getdata())
        mask_values = list(mask_image.get_flattened_data() if hasattr(mask_image, "get_flattened_data") else mask_image.getdata())
    require({path.name for path in image_path.parent.iterdir()} == {"image.png", "mask.png", "metadata.json"},
            "Render directory has missing or extra artifacts")
    require(set(mask_values) <= {0, 255}, "Mask must contain only 0 and 255")
    require(metadata.get("image_relative_path") == record["image_relative_path"] and metadata.get("image_sha256") == record["image_sha256"],
            "Metadata image provenance differs")
    require(metadata.get("mask_relative_path") == record["mask_relative_path"] and metadata.get("mask_sha256") == record["mask_sha256"],
            "Metadata mask provenance differs")
    object_pixels = sum(value == 255 for value in mask_values)
    _verify_runtime_metadata(metadata, request, contract, object_pixels)
    interior, interior_contract = _mask_interior([value == 255 for value in mask_values], 512, 512)
    decoded = [_median_uint8([pixels[index][channel] for index in interior]) for channel in range(3)]
    observed_lab = _linear_to_lab([_srgb_byte_to_linear(value) for value in decoded])
    target_lab = _linear_to_lab(request["linear_rgb"])
    delta = math.sqrt(sum((actual - expected) ** 2 for actual, expected in zip(observed_lab, target_lab)))
    return {**request, "decoded_png_rgb_uint8": decoded, "decoded_lab_delta": delta,
            "interior_pixel_count": len(interior), "object_pixel_count": object_pixels,
            "interior_contract": interior_contract}


def analyze(output_root: Path) -> dict[str, Any]:
    """Verify final artifacts and pass the decoded values through the RC-1 gate."""
    output_root = output_root.resolve()
    plan, contract = _load_plan_contract(output_root)
    manifest = load_json(output_root / MANIFEST_NAME)
    require(set(manifest) == {"schema", "contract_sha256", "request_count", "records"}, "Manifest fields differ")
    require(manifest["schema"] == "d1_emission_encoding_preflight_manifest/v1", "Manifest schema differs")
    require(manifest["contract_sha256"] == canonical_sha256(contract) and manifest["request_count"] == 4,
            "Manifest contract differs")
    records = manifest["records"]
    require(isinstance(records, list) and len(records) == 4, "Manifest must contain exactly four records")
    by_id = {request["request_id"]: request for request in plan["requests"]}
    require(len(by_id) == 4, "Plan requests are duplicate")
    seen = set()
    decoded_rows = []
    for record in records:
        require(isinstance(record, dict) and isinstance(record.get("request_id"), str), "Manifest record lacks request ID")
        request_id = record["request_id"]
        require(request_id in by_id and request_id not in seen, "Manifest has missing, extra, or duplicate request")
        seen.add(request_id)
        decoded_rows.append(_decode_record(output_root, by_id[request_id], contract, record))
    require(seen == set(by_id), "Manifest request coverage differs")
    render_root = output_root / RENDER_ROOT
    require(render_root.is_dir() and {path.name for path in render_root.iterdir()} == seen,
            "Render directory has missing or extra artifacts")
    from src.methods.colorpeel_ice import renderer_color_calibration as calibration
    summary = calibration.summarize_preflight_measurements(decoded_rows)
    result = {"schema": "d1_emission_encoding_preflight_analysis/v1", "status": "passed" if summary["overall_pass"] else "failed",
              "contract_sha256": canonical_sha256(contract), "measurements": decoded_rows, "gate": summary}
    atomic_json(output_root / SUMMARY_NAME, result)
    return result


def _extract_args(argv: Sequence[str] | None) -> list[str]:
    values = list(sys.argv if argv is None else argv)
    if values and values[0] in {"plan", "render", "analyze"}:
        return values
    return values[values.index("--") + 1:] if "--" in values else values[1:]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--output-root", type=Path, required=True)
    plan.add_argument("--asset-root", type=Path, required=True)
    plan.add_argument("--base-scene-relative-path", required=True)
    plan.add_argument("--shape-relative-path", required=True)
    render_parser = commands.add_parser("render")
    render_parser.add_argument("--output-root", type=Path, required=True)
    render_parser.add_argument("--asset-root", type=Path, required=True)
    analyze_parser = commands.add_parser("analyze")
    analyze_parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(_extract_args(argv))
    try:
        if args.command == "plan":
            plan, contract = make_plan(args.output_root, args.asset_root, args.base_scene_relative_path, args.shape_relative_path)
            print(json.dumps({"status": "planned", "requests_sha256": plan["requests_sha256"],
                              "contract_sha256": canonical_sha256(contract)}, sort_keys=True))
            return 0
        if args.command == "render":
            manifest = render(args.output_root, args.asset_root)
            print(json.dumps({"status": "rendered", "request_count": manifest["request_count"]}, sort_keys=True))
            return 0
        result = analyze(args.output_root)
        print(json.dumps({"status": result["status"], "overall_pass": result["gate"]["overall_pass"]}, sort_keys=True))
        return 0 if result["gate"]["overall_pass"] else 1
    except PreflightError as exc:
        if getattr(args, "command", None) == "analyze" and args.output_root.is_dir():
            atomic_json(args.output_root / SUMMARY_NAME, {"schema": "d1_emission_encoding_preflight_analysis/v1",
                                                          "status": "failed", "error": str(exc)})
        if getattr(args, "command", None) == "render" and args.output_root.is_dir():
            atomic_json(args.output_root / "preflight_render_failure.json", {
                "schema": "d1_emission_encoding_preflight_render_failure/v1", "status": "failed", "error": str(exc),
            })
        print(f"d1 emission preflight aborted: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
