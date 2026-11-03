"""Compare two extracted scalar materials with the unchanged Material8 recipe.

Blender --python this_file -- --supermat material.json --trellis material.json
--output fresh_directory [--gpu N]. Omit --gpu for four-thread CPU rendering.
Use ordinary Python with --grid renders/manifest.json to assemble the PNG grid.
"""

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.texture_material_disentangled_v1.render import sha256

PROTOCOL = ROOT / "experiments/texture_material_disentangled_v1/protocol.json"
METHODS = ("supermat", "trellis")


def mask_pixel_sha256(path, bpy=None):
    """Hash dimensions and exact top-down RGBA8 pixels, excluding PNG metadata."""
    if bpy is None:
        from PIL import Image
        with Image.open(path) as opened:
            image = opened.convert("RGBA")
            width, height = image.size
            pixels = image.tobytes()
    else:
        import numpy as np
        image = bpy.data.images.load(str(path), check_existing=False)
        try:
            image.colorspace_settings.name = "Non-Color"
            width, height = image.size
            values = np.empty(width * height * 4, dtype=np.float32)
            image.pixels.foreach_get(values)
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite decoded mask pixels")
            rgba = np.rint(np.clip(values, 0, 1) * 255).astype(np.uint8).reshape(height, width, 4)
            pixels = rgba[::-1].tobytes()  # Blender stores rows from bottom to top.
        finally:
            bpy.data.images.remove(image)
    return hashlib.sha256(b"RGBA8\0" + struct.pack(">II", width, height) + pixels).hexdigest()


def read_material(path):
    material = json.loads(Path(path).read_text(encoding="utf8"))
    for name in ("roughness", "metallic"):
        value = material.get(name)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError(f"{path}: require finite {name} in [0, 1]")
    return material


def comparison_rows(protocol):
    dataset = protocol["dataset"]
    rows = []
    for index, (shape, color, light) in enumerate(itertools.product(
            dataset["train_shapes"], dataset["train_colors"], dataset["train_lights"])):
        for method in METHODS:
            rows.append({"id": f"{shape}__{color}__{light}__{method}", "method": method,
                         "material_id": method, "shape": shape, "color": color,
                         "light": light, "seed": protocol["seed"] + index})
    return rows


def configure_cpu_scene(profile, seed, condition):
    """Same scene settings as render_calibration.configure_scene, without CUDA."""
    import bpy

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = profile["blender"]["cycles_samples"]
    scene.cycles.seed = seed
    scene.render.threads_mode, scene.render.threads = "FIXED", 4
    scene.render.resolution_x, scene.render.resolution_y = profile["blender"]["resolution"]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.display_settings.display_device = "sRGB"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.render.film_transparent = False
    scene.world.use_nodes = True
    background = next(node for node in scene.world.node_tree.nodes if node.type == "BACKGROUND")
    background.inputs["Color"].default_value = profile["background"]["world_rgba"]
    background.inputs["Strength"].default_value = condition["world_strength"]
    ground = bpy.data.objects["Ground"]
    material = bpy.data.materials.new("CalibrationNeutralGround")
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = profile["background"]["ground_rgba"]
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Roughness"].default_value = 1.0
    ground.data.materials.clear()
    ground.data.materials.append(material)
    lights = {}
    for name, spec in condition["lights"].items():
        light = bpy.data.objects[name]
        if light.type != "LIGHT":
            raise ValueError(f"Expected Blender light: {name}")
        light.location = [a + b for a, b in zip(light.location, spec["offset_xyz"])]
        light.data.energy *= spec["energy_scale"]
        light.data.color = (1.0, 1.0, 1.0)
        lights[name] = {"location_xyz": list(light.location), "energy": light.data.energy,
                        "type": light.data.type, "color": list(light.data.color)}
    return scene, lights, []


def verify_pair(first, second):
    for key in ("shape", "color", "light", "seed", "camera", "lights", "mask_pixel_sha256",
                "uniform_base_color_linear_rgba", "world_strength", "object_scale",
                "resolution", "cycles_samples", "color_management", "cycles_device"):
        if first[key] != second[key]:
            raise ValueError(f"Paired render control changed: {key}")


def render(supermat, trellis, output, gpu=None):
    output = Path(output).resolve()
    if output.exists() or output.is_relative_to(ROOT):
        raise ValueError("Use a fresh output directory outside the source checkout")
    if gpu is not None and os.environ.get("CUDA_VISIBLE_DEVICES") != gpu:
        raise ValueError("GPU argument and CUDA_VISIBLE_DEVICES disagree")
    inputs = {method: {"path": str(Path(path).resolve()), "sha256": sha256(path),
                       "material": read_material(path)}
              for method, path in zip(METHODS, (supermat, trellis))}
    protocol = json.loads(PROTOCOL.read_text(encoding="utf8"))
    protocol["materials"] = {method: record["material"] for method, record in inputs.items()}
    renderer = protocol["renderer"]
    profile_path = ROOT / renderer["profile"]
    profile = json.loads(profile_path.read_text(encoding="utf8"))
    profile["blender"].update(cuda_visible_devices=gpu, cycles_samples=renderer["cycles_samples"],
                              resolution=[renderer["resolution"]] * 2)
    if sha256(renderer["base_scene"]) != renderer["base_scene_sha256"]:
        raise ValueError("Frozen Material8 base scene changed")
    import bpy
    from experiments.material_uniform_albedo_v1.render import render_one
    from experiments.natural_material_threeway_v1.render_grid import ASSETS, append_shape
    from experiments.natural_material_supermat_v1 import render_calibration as base
    from experiments.natural_texture_material_v1.render_dataset import shader_audit

    if tuple(bpy.app.version) != (4, 2, 11):
        raise ValueError("Expected Blender4.2.11")
    backend = base if gpu is not None else SimpleNamespace(
        configure_scene=configure_cpu_scene, configure_camera=base.configure_camera,
        configure_mask=base.configure_mask, write_json=base.write_json)
    rows = comparison_rows(protocol)
    evidence = {"status": "running", "training_launched": False, "inputs": inputs,
                "expected_images": len(rows), "records": [], "git_commit": base.git_commit(),
                "protocol_sha256": sha256(PROTOCOL), "profile_sha256": sha256(profile_path),
                "base_scene_sha256": sha256(renderer["base_scene"]), "script_sha256": sha256(__file__),
                "reused_render_sha256": sha256(render_one.__code__.co_filename),
                "scene_configuration_sha256": sha256(base.__file__),
                "shape_sha256": {shape: sha256(Path(renderer["shape_dir"]) / f"{ASSETS[shape]}.blend")
                                 for shape in protocol["dataset"]["train_shapes"]}}
    output.mkdir(parents=True)
    try:
        for row in rows:
            record = render_one(row, output, protocol, profile, bpy, backend, append_shape)
            scene = bpy.context.scene
            record.update(shader_audit(bpy, "material"), cycles_device=scene.cycles.device,
                          mask_pixel_sha256=mask_pixel_sha256(record["mask"], bpy),
                          cpu_threads=scene.render.threads if gpu is None else None,
                          color_management={"display_device": scene.display_settings.display_device,
                              "view_transform": scene.view_settings.view_transform,
                              "look": scene.view_settings.look, "exposure": scene.view_settings.exposure,
                              "gamma": scene.view_settings.gamma})
            base.write_json(Path(record["metadata"]), {key: value for key, value in record.items()
                                                if key not in ("metadata", "metadata_sha256")})
            record["metadata_sha256"] = sha256(record["metadata"])
            evidence["records"].append(record)
            if row["method"] == "trellis":
                verify_pair(evidence["records"][-2], record)
            base.write_json(output / "manifest.json", evidence)
            print(f"MATERIAL_COMPARE {len(evidence['records'])}/{len(rows)} {row['id']}", flush=True)
        if any(sha256(record["path"]) != record["sha256"] for record in inputs.values()):
            raise ValueError("Input material file changed during rendering")
        evidence.update(status="complete", image_count=len(evidence["records"]))
    except BaseException as error:
        evidence.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        base.write_json(output / "manifest.json", evidence)
    return evidence


def make_grid(manifest_path, output=None):
    from PIL import Image, ImageDraw

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    records = manifest["records"]
    if manifest["status"] != "complete" or len(records) != manifest["expected_images"]:
        raise ValueError("Require completed paired renders")
    output = Path(output) if output else manifest_path.with_name("material_compare.png")
    if output.exists():
        raise FileExistsError(output)
    tile, header = 384, 66
    canvas = Image.new("RGB", (tile * 2, (tile + header) * (len(records) // 2)), "white")
    draw = ImageDraw.Draw(canvas)
    for index in range(0, len(records), 2):
        first, second = records[index:index + 2]
        verify_pair(first, second)
        if any(sha256(row["mask"]) != row["mask_sha256"] or
               mask_pixel_sha256(row["mask"]) != row["mask_pixel_sha256"] for row in (first, second)):
            raise ValueError("Mask file or decoded pixels differ from the render receipt")
        y = (index // 2) * (tile + header)
        draw.text((8, y + 5), f"{first['shape']} / {first['color']} / {first['light']}", fill="black")
        for column, row in enumerate((first, second)):
            if row["method"] != METHODS[column] or sha256(row["image"]) != row["image_sha256"]:
                raise ValueError("Method order or rendered image changed")
            label = "SuperMat" if column == 0 else "TRELLIS.2"
            draw.text((column * tile + 8, y + 26),
                      f"{label}   M={row['metallic']:.5f}   R={row['roughness']:.5f}", fill="black")
            with Image.open(row["image"]) as image:
                canvas.paste(image.convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS),
                             (column * tile, y + header))
    canvas.save(output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supermat", type=Path)
    parser.add_argument("--trellis", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu")
    parser.add_argument("--grid", type=Path)
    args = parser.parse_args(argv)
    if args.grid:
        print(make_grid(args.grid, args.output))
    elif args.supermat and args.trellis and args.output:
        render(args.supermat, args.trellis, args.output, args.gpu)
    else:
        parser.error("Require --supermat, --trellis and --output, or --grid")


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:])
