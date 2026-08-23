#!/usr/bin/env python3
"""Render the locked CLEVR 3x3 multiview request manifest with Blender 4.2.

Run through Blender, for example:

  blender --background --python-exit-code 1 --cycles-device CUDA \
    --cycles-print-stats --python render_clevr_multiview.py -- \
    --requests render_requests.jsonl --profile multiview_render.json \
    --output-root rendered --properties-json data/properties.json \
    --base-scene-blendfile data/base_scene.blend \
    --shape-dir data/shapes --material-dir data/materials

The scientific renderer parameters live in the tracked profile and cannot be
overridden on the command line. ``--limit`` is smoke-only. ``--resume`` only
accepts artifacts whose request, contract fingerprint, and hashes still match.
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
from typing import Any, Iterable

try:
    import bpy
    from bpy_extras.object_utils import world_to_camera_view
except ImportError:  # ordinary Python is supported for validation/unit tests
    bpy = None
    world_to_camera_view = None


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.methods.colorpeel_ice.multiview_render_contract import (  # noqa: E402
    EXPECTED_PROFILE,
    canonical_sha256,
    official_jitter_metadata,
    validate_profile as validate_locked_profile,
)

REQUEST_FIELDS = (
    "cell_id", "cell_index", "shape", "color", "material", "subject_token",
    "color_token", "nominal_rgb", "view_index", "split", "render_seed",
    "renderer_profile_id", "renderer_profile_sha256",
)
RENDERER_OWNED_FIELDS = (
    "camera", "light", "background", "scene_json", "image", "mask", "background_mask",
)
OBJECT_MASK_PREFIX = "__object_mask_"
BACKGROUND_MASK_PREFIX = "__background_mask_"


class RendererError(RuntimeError):
    """Raised when a renderer input, runtime, or resume artifact is invalid."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RendererError(message)


def extract_blender_args(argv: list[str] | None = None) -> list[str]:
    argv = sys.argv if argv is None else argv
    return argv[argv.index("--") + 1:] if "--" in argv else argv[1:]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--properties-json", type=Path, required=True)
    parser.add_argument("--base-scene-blendfile", type=Path, required=True)
    parser.add_argument("--shape-dir", type=Path, required=True)
    parser.add_argument("--material-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="Smoke-only prefix length; full realization requires no limit")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RendererError(f"Cannot read valid JSON from {path}: {exc}") from exc


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                require(isinstance(value, dict), f"Expected object at {path}:{line_number}")
                records.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise RendererError(f"Cannot read valid JSONL from {path}: {exc}") from exc
    return records


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, allow_nan=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def validate_profile(profile: Any) -> dict[str, Any]:
    try:
        return validate_locked_profile(profile)
    except ValueError as exc:
        raise RendererError(str(exc)) from exc


def validate_requests(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    require(len(records) == 180, f"Expected 180 render requests, got {len(records)}")
    seen: set[tuple[str, int]] = set()
    for record in records:
        require(all(field in record for field in REQUEST_FIELDS), "Render request is missing locked fields")
        key = (record["cell_id"], record["view_index"])
        require(key not in seen, f"Duplicate render request: {key}")
        seen.add(key)
        require(record["cell_index"] in range(9), f"Invalid cell_index for {key}")
        require(record["view_index"] in range(20), f"Invalid view_index for {key}")
        require(record["material"] == "metal", f"Non-metal request: {key}")
        require(record["split"] == ("train" if record["view_index"] < 16 else "audit"),
                f"Invalid split for {key}")
        expected_seed = 420000 + record["cell_index"] * 100 + record["view_index"]
        require(record["render_seed"] == expected_seed, f"Invalid render_seed for {key}")
        require(record["renderer_profile_id"] == EXPECTED_PROFILE["profile_id"],
                f"Invalid renderer profile for {key}")
        require(record["renderer_profile_sha256"] == canonical_sha256(EXPECTED_PROFILE),
                f"Invalid renderer profile hash for {key}")
        require(record["shape"] in {"cube", "sphere", "cylinder"}, f"Invalid shape for {key}")
        require(record["color"] in {"red", "cyan", "gray"}, f"Invalid color for {key}")
        for field in RENDERER_OWNED_FIELDS:
            require(record.get(field) is None, f"Request fabricates renderer field {field}: {key}")
    return records


def _asset_path(root: Path, stem: str) -> Path:
    path = root / f"{stem}.blend"
    require(path.is_file(), f"Missing Blender asset: {path}")
    return path


def collect_asset_hashes(
    properties_path: Path,
    base_scene_path: Path,
    shape_dir: Path,
    material_dir: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    properties = load_json(properties_path)
    require(isinstance(properties, dict), "properties.json must be an object")
    for key in ("shapes", "colors", "materials", "sizes"):
        require(key in properties, f"properties.json is missing {key}")
    require(base_scene_path.is_file(), f"Missing base scene: {base_scene_path}")
    assets = {
        "properties_json": properties_path,
        "base_scene_blendfile": base_scene_path,
    }
    for shape in ("cube", "sphere", "cylinder"):
        require(shape in properties["shapes"], f"properties.json is missing shape {shape}")
        assets[f"shape_{shape}"] = _asset_path(shape_dir, properties["shapes"][shape])
    require("metal" in properties["materials"], "properties.json is missing metal material")
    assets["material_metal"] = _asset_path(material_dir, properties["materials"]["metal"])
    return properties, {name: file_sha256(path) for name, path in assets.items()}


def stable_contract(
    requests: list[dict[str, Any]],
    profile: dict[str, Any],
    asset_hashes: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "profile_id": profile["profile_id"],
        "profile_sha256": canonical_sha256(profile),
        "requests_sha256": canonical_sha256(requests),
        "request_count": len(requests),
        "asset_sha256": asset_hashes,
    }


def prepare_output_root(output_root: Path, contract: dict[str, Any], resume: bool) -> None:
    contract_path = output_root / "render_contract.json"
    if resume:
        require(output_root.is_dir(), f"Resume output root does not exist: {output_root}")
        require(contract_path.is_file(), f"Resume contract is missing: {contract_path}")
        require(load_json(contract_path) == contract, "Resume contract differs from current inputs")
        partial_root = output_root / ".partial"
        require(not partial_root.exists(), f"Resume found an incomplete partial directory: {partial_root}")
        return
    require(not output_root.exists() or output_root.is_dir(),
            f"Output root exists but is not a directory: {output_root}")
    require(not output_root.exists() or not any(output_root.iterdir()),
            f"Output root must be new or empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(contract_path, contract)


def _resolve_artifact(output_root: Path, relative: Any, field: str) -> Path:
    require(isinstance(relative, str) and relative, f"Missing artifact path: {field}")
    path = (output_root / relative).resolve()
    try:
        path.relative_to(output_root)
    except ValueError as exc:
        raise RendererError(f"Artifact escapes output root: {relative}") from exc
    require(path.is_file(), f"Missing resume artifact: {path}")
    return path


def verify_completed_record(
    output_root: Path,
    record: dict[str, Any],
    expected: dict[str, Any],
    contract: dict[str, Any],
) -> None:
    for field in REQUEST_FIELDS:
        require(record.get(field) == expected[field], f"Resume record changed {field}")
    require(record.get("renderer_profile_id") == contract["profile_id"], "Resume profile ID changed")
    require(record.get("render_contract_sha256") == canonical_sha256(contract), "Resume contract hash changed")
    hashes = record.get("artifact_sha256", {})
    for field in ("image", "mask", "background_mask", "scene_json"):
        path = _resolve_artifact(output_root, record.get(field), field)
        require(hashes.get(field) == file_sha256(path), f"Resume artifact hash changed: {field}")
    view_dir = _resolve_artifact(output_root, record["scene_json"], "scene_json").parent
    expected_names = {"img.jpg", f"mask_{expected['shape']}_0.png", "background.png", "scene.json", ".record.json"}
    require({path.name for path in view_dir.iterdir()} == expected_names, f"Resume view directory is contaminated: {view_dir}")
    require(load_json(view_dir / ".record.json") == record, f"Resume .record.json disagrees: {view_dir}")


def load_completed_records(
    output_root: Path,
    expected_by_key: dict[tuple[str, int], dict[str, Any]],
    contract: dict[str, Any],
) -> dict[tuple[str, int], dict[str, Any]]:
    manifest = output_root / "renderer_realization.jsonl"
    records = load_jsonl(manifest) if manifest.is_file() else []
    completed: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        key = (record.get("cell_id"), record.get("view_index"))
        require(key in expected_by_key, f"Unknown resume record: {key}")
        require(key not in completed, f"Duplicate resume record: {key}")
        verify_completed_record(output_root, record, expected_by_key[key], contract)
        completed[key] = record

    expected_cells = {record["cell_id"] for record in expected_by_key.values()}
    for path in output_root.iterdir():
        if not path.is_dir():
            continue
        require(path.name in expected_cells, f"Resume output contains an unknown directory: {path}")
        for view_dir in path.iterdir():
            require(view_dir.is_dir(), f"Resume cell directory contains a non-directory: {view_dir}")
            try:
                view_index = int(view_dir.name.removeprefix("view_"))
            except ValueError as exc:
                raise RendererError(f"Resume output contains an unknown view directory: {view_dir}") from exc
            key = (path.name, view_index)
            require(key in completed, f"Resume output contains an orphan final directory: {view_dir}")
    return completed


def _tuple(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values]


def configure_render(profile: dict[str, Any], render_seed: int) -> list[dict[str, str]]:
    require(bpy is not None, "Blender Python API is unavailable")
    scene = bpy.context.scene
    config = profile["blender"]
    scene.render.engine = config["render_engine"]
    scene.render.resolution_x, scene.render.resolution_y = config["resolution"]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "JPEG"
    scene.render.image_settings.quality = config["jpeg_quality"]
    scene.cycles.samples = config["cycles_samples"]
    scene.cycles.seed = render_seed
    scene.cycles.device = "GPU"
    scene.use_nodes = False

    try:
        preferences = bpy.context.preferences.addons["cycles"].preferences
        preferences.compute_device_type = "CUDA"
        preferences.get_devices()
    except (KeyError, RuntimeError, TypeError) as exc:
        raise RendererError(f"CUDA Cycles initialization failed: {exc}") from exc
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "3", "CUDA_VISIBLE_DEVICES must be exactly 3")
    cuda_devices = [device for device in preferences.devices if device.type == "CUDA"]
    require(len(cuda_devices) == 1, f"Expected exactly one visible CUDA device, found {len(cuda_devices)}")
    require("V100" in cuda_devices[0].name, f"Expected a V100 CUDA device, found {cuda_devices[0].name}")
    for device in preferences.devices:
        device.use = device.type == "CUDA"
    return [{"name": device.name, "type": device.type, "id": device.id} for device in cuda_devices]


def neutralize_scene(profile: dict[str, Any]) -> None:
    background_profile = profile["background"]
    for obj in bpy.context.scene.objects:
        if obj.type == "LIGHT":
            obj.data.color = tuple(profile["lights"]["rgb"])
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    background = next((node for node in world.node_tree.nodes if node.type == "BACKGROUND"), None)
    if background is None:
        background = world.node_tree.nodes.new("ShaderNodeBackground")
    background.inputs["Color"].default_value = tuple(background_profile["world_rgba"])

    ground = bpy.data.objects.get("Ground")
    require(ground is not None and ground.type == "MESH", "Base scene is missing mesh Ground")
    material = bpy.data.materials.new("ColorPeel_Neutral_Ground")
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    require(principled is not None, "Neutral ground Principled BSDF is missing")
    principled.inputs["Base Color"].default_value = tuple(background_profile["ground_rgba"])
    principled.inputs["Metallic"].default_value = 0.0
    principled.inputs["Roughness"].default_value = 1.0
    ground.data.materials.clear()
    ground.data.materials.append(material)


def clear_base_scene_geometry() -> None:
    for obj in list(bpy.context.scene.objects):
        if obj.type == "MESH" and obj.name != "Ground":
            bpy.data.objects.remove(obj, do_unlink=True)


def append_shape(shape_dir: Path, object_name: str):
    blend_path = _asset_path(shape_dir, object_name)
    existing_names = set(bpy.data.objects.keys())
    bpy.ops.wm.append(directory=str(blend_path / "Object") + os.sep, filename=object_name, link=False)
    appended = [obj for obj in bpy.data.objects if obj.name not in existing_names]
    require(len(appended) == 1, f"Expected one object from {blend_path}, found {len(appended)}")
    obj = appended[0]
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    return obj


def create_asset_material(name: str, rgb: list[int], material_dir: Path, properties: dict[str, Any]):
    source_name = properties["materials"]["metal"]
    blend_path = _asset_path(material_dir, source_name)
    bpy.ops.wm.append(directory=str(blend_path / "NodeTree") + os.sep, filename=source_name, link=False)
    node_group = bpy.data.node_groups.get(source_name)
    require(node_group is not None, f"Could not append material node group {source_name}")
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    output = material.node_tree.nodes.get("Material Output")
    require(output is not None, "Material Output node is missing")
    group_node = material.node_tree.nodes.new("ShaderNodeGroup")
    group_node.node_tree = node_group
    group_node.inputs["Color"].default_value = tuple(value / 255.0 for value in rgb) + (1.0,)
    material.node_tree.links.new(group_node.outputs["Shader"], output.inputs["Surface"])
    return material


def add_object(request: dict[str, Any], profile: dict[str, Any], properties: dict[str, Any],
               shape_dir: Path, material_dir: Path):
    source_name = properties["shapes"][request["shape"]]
    obj = append_shape(shape_dir, source_name)
    scale = profile["object"]["scale"]
    if source_name == "SmoothCube_v2":
        scale /= math.sqrt(2)
    obj.location = (profile["object"]["position_xy"][0], profile["object"]["position_xy"][1], scale)
    obj.rotation_euler = (0.0, 0.0, math.radians(profile["object"]["rotation_z_degrees"]))
    obj.scale = (scale, scale, scale)
    name = f"ColorPeel_{request['shape']}_{request['color']}_metal"
    material = create_asset_material(name, request["nominal_rgb"], material_dir, properties)
    obj.data.materials.clear()
    obj.data.materials.append(material)
    return obj, "clevr_asset_node_group"


def apply_view_jitter(profile: dict[str, Any], render_seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    offsets = official_jitter_metadata(render_seed, profile)
    camera = bpy.data.objects.get(profile["camera"]["name"])
    require(camera is not None, "Base scene is missing Camera")
    camera.data.dof.use_dof = False
    camera_base = _tuple(camera.location)
    camera_rotation = _tuple(camera.rotation_euler)
    for axis, offset in enumerate(offsets["camera_offset"]):
        camera.location[axis] += offset
    camera_metadata = {
        "name": profile["camera"]["name"],
        "base_location": camera_base,
        "jitter_offset": offsets["camera_offset"],
        "final_location": _tuple(camera.location),
        "rotation_euler": camera_rotation,
        "rotation_policy": profile["camera"]["rotation_policy"],
        "lens": float(camera.data.lens),
        "sensor_width": float(camera.data.sensor_width),
    }

    light_metadata: dict[str, Any] = {
        "order": profile["lights"]["order"],
        "lights": {},
        "fixed_lights": {},
    }
    for name in profile["lights"]["order"]:
        light = bpy.data.objects.get(name)
        require(light is not None and light.type == "LIGHT", f"Base scene is missing light {name}")
        base = _tuple(light.location)
        offset = offsets["light_offsets"][name]
        for axis, value in enumerate(offset):
            light.location[axis] += value
        light.data.color = tuple(profile["lights"]["rgb"])
        light_metadata["lights"][name] = {
            "base_location": base,
            "jitter_offset": offset,
            "final_location": _tuple(light.location),
            "rgb": profile["lights"]["rgb"],
            "type": light.data.type,
            "energy": float(light.data.energy),
        }
    for name in profile["lights"]["fixed_order"]:
        light = bpy.data.objects.get(name)
        require(light is not None and light.type == "LIGHT", f"Base scene is missing fixed light {name}")
        location = _tuple(light.location)
        light.data.color = tuple(profile["lights"]["rgb"])
        light_metadata["fixed_lights"][name] = {
            "base_location": location,
            "final_location": location,
            "rgb": profile["lights"]["rgb"],
            "type": light.data.type,
            "energy": float(light.data.energy),
        }
    return camera_metadata, light_metadata


def configure_mask_outputs(sample_dir: Path, obj) -> None:
    scene = bpy.context.scene
    scene.use_nodes = True
    scene.view_layers[0].use_pass_object_index = True
    obj.pass_index = 1
    nodes = scene.node_tree.nodes
    links = scene.node_tree.links
    nodes.clear()
    render_layers = nodes.new("CompositorNodeRLayers")
    composite = nodes.new("CompositorNodeComposite")
    links.new(render_layers.outputs["Image"], composite.inputs["Image"])
    object_mask = nodes.new("CompositorNodeIDMask")
    object_mask.index = 1
    object_mask.use_antialiasing = False
    links.new(render_layers.outputs["IndexOB"], object_mask.inputs["ID value"])
    background_mask = nodes.new("CompositorNodeMath")
    background_mask.operation = "SUBTRACT"
    background_mask.inputs[0].default_value = 1.0
    links.new(object_mask.outputs["Alpha"], background_mask.inputs[1])
    for value_node, prefix in ((object_mask, OBJECT_MASK_PREFIX), (background_mask, BACKGROUND_MASK_PREFIX)):
        output = nodes.new("CompositorNodeOutputFile")
        output.base_path = str(sample_dir)
        output.file_slots[0].path = prefix
        output.format.file_format = "PNG"
        output.format.color_mode = "BW"
        output.format.color_depth = "8"
        links.new(value_node.outputs["Alpha"] if value_node is object_mask else value_node.outputs["Value"],
                  output.inputs[0])


def finalize_mask(sample_dir: Path, prefix: str, target_name: str) -> Path:
    matches = sorted(sample_dir.glob(f"{prefix}*.png"))
    require(len(matches) == 1, f"Expected one mask output for {prefix}, found {len(matches)}")
    target = sample_dir / target_name
    matches[0].replace(target)
    return target


def validate_masks(object_mask: Path, background_mask: Path) -> int:
    object_image = bpy.data.images.load(str(object_mask), check_existing=False)
    background_image = bpy.data.images.load(str(background_mask), check_existing=False)
    try:
        require(tuple(object_image.size) == (512, 512), "Object mask must be 512x512")
        require(tuple(background_image.size) == (512, 512), "Background mask must be 512x512")
        object_pixels = object_image.pixels[:]
        background_pixels = background_image.pixels[:]
        foreground = 0
        for index in range(0, len(object_pixels), 4):
            object_value = object_pixels[index]
            background_value = background_pixels[index]
            require(abs(object_value) < 1e-6 or abs(object_value - 1.0) < 1e-6, "Object mask is not binary")
            require(abs(background_value) < 1e-6 or abs(background_value - 1.0) < 1e-6,
                    "Background mask is not binary")
            require(abs(object_value + background_value - 1.0) < 1e-6, "Masks are not complements")
            foreground += object_value > 0.5
        require(0.005 <= foreground / (512 * 512) <= 0.90, "Object mask ratio is outside 0.005-0.90")
        return foreground
    finally:
        bpy.data.images.remove(object_image)
        bpy.data.images.remove(background_image)


def camera_pixel_coords(camera, obj) -> list[float]:
    point = world_to_camera_view(bpy.context.scene, camera, obj.location)
    return [round(point.x * 512), round(512 - point.y * 512), float(point.z)]


def render_one(
    request: dict[str, Any],
    profile: dict[str, Any],
    contract: dict[str, Any],
    properties: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    bpy.ops.wm.open_mainfile(filepath=str(args.base_scene_blendfile.resolve()))
    cuda_devices = configure_render(profile, request["render_seed"])
    neutralize_scene(profile)
    clear_base_scene_geometry()
    camera_metadata, light_metadata = apply_view_jitter(profile, request["render_seed"])
    obj, material_backend = add_object(request, profile, properties, args.shape_dir, args.material_dir)

    final_dir = args.output_root / request["cell_id"] / f"view_{request['view_index']:02d}"
    require(not final_dir.exists(), f"Final output already exists without a valid resume record: {final_dir}")
    partial_root = args.output_root / ".partial"
    partial_dir = partial_root / f"{request['cell_id']}__view_{request['view_index']:02d}"
    require(not partial_dir.exists(), f"Incomplete partial output already exists: {partial_dir}")
    partial_dir.mkdir(parents=True)
    configure_mask_outputs(partial_dir, obj)
    image_path = partial_dir / "img.jpg"
    bpy.context.scene.render.filepath = str(image_path)
    bpy.ops.render.render(write_still=True)
    object_mask = finalize_mask(partial_dir, OBJECT_MASK_PREFIX, f"mask_{request['shape']}_0.png")
    background_mask = finalize_mask(partial_dir, BACKGROUND_MASK_PREFIX, "background.png")
    foreground = validate_masks(object_mask, background_mask)

    background_metadata = profile["background"]
    scene = {
        "renderer_profile_id": profile["profile_id"],
        "render_seed": request["render_seed"],
        "cycles_seed": request["render_seed"],
        "camera": camera_metadata,
        "light": light_metadata,
        "background": background_metadata,
        "renderer": {
            "blender_version": ".".join(str(part) for part in bpy.app.version),
            "engine": "CYCLES",
            "cycles_samples": 512,
            "cycles_device": "CUDA",
            "cuda_devices": cuda_devices,
        },
        "asset_sha256": contract["asset_sha256"],
        "objects": [{
            "shape": request["shape"],
            "color": request["color"],
            "material": request["material"],
            "material_backend": material_backend,
            "nominal_scale": profile["object"]["scale"],
            "applied_scale": _tuple(obj.scale),
            "3d_coords": _tuple(obj.location),
            "rotation": profile["object"]["rotation_z_degrees"],
            "pixel_coords": camera_pixel_coords(bpy.data.objects[profile["camera"]["name"]], obj),
        }],
    }
    write_json(partial_dir / "scene.json", scene)

    final_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(partial_dir, final_dir)
    relative_base = final_dir.relative_to(args.output_root).as_posix()
    record = {
        **request,
        "camera": camera_metadata,
        "light": light_metadata,
        "background": background_metadata,
        "scene_json": f"{relative_base}/scene.json",
        "image": f"{relative_base}/img.jpg",
        "mask": f"{relative_base}/mask_{request['shape']}_0.png",
        "background_mask": f"{relative_base}/background.png",
        "renderer_profile_id": profile["profile_id"],
        "render_contract_sha256": canonical_sha256(contract),
        "foreground_pixels": foreground,
    }
    record["artifact_sha256"] = {
        field: file_sha256(args.output_root / record[field])
        for field in ("image", "mask", "background_mask", "scene_json")
    }
    write_json(final_dir / ".record.json", record)
    return record


def git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root), stderr=subprocess.DEVNULL
        ).decode("ascii").strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(extract_blender_args() if argv is None else argv)
    args.output_root = args.output_root.resolve()
    profile = validate_profile(load_json(args.profile.resolve()))
    requests = validate_requests(load_jsonl(args.requests.resolve()))
    require(args.limit is None or 1 <= args.limit <= len(requests), "--limit must be between 1 and 180")
    properties, asset_hashes = collect_asset_hashes(
        args.properties_json.resolve(), args.base_scene_blendfile.resolve(),
        args.shape_dir.resolve(), args.material_dir.resolve(),
    )
    for request in requests:
        require(properties["colors"].get(request["color"]) == request["nominal_rgb"],
                f"Request RGB differs from properties.json: {request['cell_id']}")
    contract = stable_contract(requests, profile, asset_hashes)
    if args.validate_only:
        result = {"status": "validated", **contract}
        print(json.dumps(result, indent=2, sort_keys=True))
        return result
    require(bpy is not None, "Rendering must run through Blender")
    require(".".join(str(part) for part in bpy.app.version) == profile["blender"]["version"],
            f"Blender version must be {profile['blender']['version']}")
    prepare_output_root(args.output_root, contract, args.resume)
    expected_by_key = {(record["cell_id"], record["view_index"]): record for record in requests}
    completed = load_completed_records(args.output_root, expected_by_key, contract) if args.resume else {}
    selected = requests[:args.limit] if args.limit is not None else requests
    manifest_path = args.output_root / "renderer_realization.jsonl"
    status_path = args.output_root / "renderer_status.jsonl"
    for index, request in enumerate(selected, start=1):
        key = (request["cell_id"], request["view_index"])
        if key in completed:
            print(f"[{index:03d}/{len(selected):03d}] resume skip {key}")
            continue
        print(f"[{index:03d}/{len(selected):03d}] render {key}")
        try:
            record = render_one(request, profile, contract, properties, args)
            append_jsonl(manifest_path, record)
            append_jsonl(status_path, {"cell_id": key[0], "view_index": key[1], "status": "ok"})
            completed[key] = record
        except Exception as exc:
            append_jsonl(status_path, {
                "cell_id": key[0], "view_index": key[1], "status": "failed",
                "error_type": type(exc).__name__, "error": str(exc),
            })
            raise
    partial_root = args.output_root / ".partial"
    if partial_root.is_dir() and not any(partial_root.iterdir()):
        partial_root.rmdir()
    state = "succeeded" if args.limit is None and len(completed) == 180 else "partial_smoke"
    result = {
        "status": state,
        "completed_count": len(completed),
        "selected_count": len(selected),
        "request_count": 180,
        "resume": args.resume,
        "limit": args.limit,
        "profile_id": profile["profile_id"],
        "render_contract_sha256": canonical_sha256(contract),
        "git_commit": git_commit(Path(__file__).resolve().parents[3]),
    }
    write_json(args.output_root / "renderer_status.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
