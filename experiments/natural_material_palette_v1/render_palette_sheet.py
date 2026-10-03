"""Render Palette roughness/normal maps with the existing three-light Cycles scene."""

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import bpy


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def set_material(obj, material, profile):
    if not obj.data.uv_layers:
        raise ValueError("CLEVR sphere has no UV coordinates for spatial maps")
    shader = bpy.data.materials.new("MaterialPaletteSpatialPreview")
    shader.use_nodes = True
    nodes = shader.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = profile["material"]["base_color_linear_rgba"]
    bsdf.inputs["Metallic"].default_value = material["renderer_metallic_assumption"]
    rough = nodes.new("ShaderNodeTexImage")
    rough.image = bpy.data.images.load(material["roughness_map_path"], check_existing=False)
    rough.image.colorspace_settings.name = "Non-Color"
    rough.extension = "REPEAT"
    normal = nodes.new("ShaderNodeTexImage")
    normal.image = bpy.data.images.load(material["normal_map_path"], check_existing=False)
    normal.image.colorspace_settings.name = "Non-Color"
    normal.extension = "REPEAT"
    normal_map = nodes.new("ShaderNodeNormalMap")
    normal_map.space = "TANGENT"
    normal_map.inputs["Strength"].default_value = 1.0
    links = shader.node_tree.links
    links.new(rough.outputs["Color"], bsdf.inputs["Roughness"])
    links.new(normal.outputs["Color"], normal_map.inputs["Color"])
    links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    obj.data.materials.clear()
    obj.data.materials.append(shader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--material-json", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-scene-blendfile", type=Path, required=True)
    parser.add_argument("--sphere-blendfile", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    material = json.loads(args.material_json.read_text())
    profile = json.loads(args.profile.read_text())
    if material["metallic"] is not None or material["renderer_metallic_assumption"] != 0.0:
        raise ValueError("Metallic must remain unestimated; preview assumption is zero")
    for key in ("roughness_map_path", "normal_map_path"):
        if not Path(material[key]).is_file():
            raise FileNotFoundError(material[key])
    if tuple(profile["blender"]["version"]) != bpy.app.version:
        raise RuntimeError("Blender profile version mismatch")
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    source = args.project_root / "experiments/natural_material_supermat_v1/render_calibration.py"
    spec = importlib.util.spec_from_file_location("calibration_renderer", source)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    args.output_root.mkdir(parents=True)
    records = []
    for name, condition in profile["lighting_conditions"].items():
        bpy.ops.wm.open_mainfile(filepath=str(args.base_scene_blendfile.resolve()))
        scene, lights, devices = base.configure_scene(profile, profile["blender"]["seed"], condition)
        for obj in list(scene.objects):
            if obj.type == "MESH" and obj.name != "Ground":
                bpy.data.objects.remove(obj, do_unlink=True)
        obj = base.append_sphere(args.sphere_blendfile.resolve(), profile)
        set_material(obj, material, profile)
        camera = base.configure_camera(obj, profile)
        folder = args.output_root / name
        folder.mkdir()
        base.configure_mask(folder)
        image = folder / "image.png"
        scene.render.filepath = str(image)
        bpy.ops.render.render(write_still=True)
        generated = list(folder.glob("__mask_*.png"))
        rgba = list(folder.glob("__input_rgba_*.png"))
        if len(generated) != 1 or len(rgba) != 1:
            raise RuntimeError("Expected one object mask and one RGBA image")
        generated[0].rename(folder / "object_mask.png")
        rgba[0].rename(folder / "input_rgba.png")
        record = {"light": name, "seed": profile["blender"]["seed"],
                  "camera": camera, "lights": lights, "devices": devices,
                  "roughness_map_sha256": sha(material["roughness_map_path"]),
                  "normal_map_sha256": sha(material["normal_map_path"]),
                  "base_color_linear_rgba": profile["material"]["base_color_linear_rgba"],
                  "metallic_assumption": 0.0, "normal_map_strength": 1.0,
                  "normal_map_space_assumption": "TANGENT",
                  "renderer_profile_sha256": sha(args.profile),
                  "material_json_sha256": sha(args.material_json),
                  "base_scene_sha256": sha(args.base_scene_blendfile),
                  "sphere_asset_sha256": sha(args.sphere_blendfile),
                  "renderer_script_sha256": sha(__file__),
                  "project_git_commit": subprocess.check_output(
                      ["git", "rev-parse", "HEAD"], cwd=args.project_root, text=True).strip(),
                  "image_sha256": sha(image),
                  "object_mask_sha256": sha(folder / "object_mask.png")}
        write_json(folder / "metadata.json", record)
        records.append(record)
    write_json(args.output_root / "manifest.json", {"image_count": len(records), "records": records})


if __name__ == "__main__":
    main()
