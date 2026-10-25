"""Render a fixed-scene, uniform-Albedo Cycles material reference bank.

Run inside Blender: --background --python-exit-code 1 --python this_file --
--plan plan.json --output NEW_EXTERNAL_DIRECTORY --gpu 2 [--resume].
"""

import argparse
from array import array
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mask_pixels_sha256(pixels):
    """Hash decoded samples; Blender PNG timestamps are not silhouette changes."""
    return hashlib.sha256(array("f", pixels).tobytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf8")
    temporary.replace(path)


def load_plan(path):
    plan = json.loads(Path(path).read_text(encoding="utf8"))
    renderer = plan["renderer"]
    for key in ("resolution", "cycles_samples", "seed"):
        value = renderer[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if key == "seed" else 1):
            raise ValueError(f"Invalid renderer {key}")
    color = plan["albedo_linear_rgb"]
    if len(color) != 3 or any(isinstance(v, bool) or not isinstance(v, (int, float))
                              or not math.isfinite(v) or not 0 <= v <= 1 for v in color):
        raise ValueError("Albedo must contain three finite linear RGB values in [0,1]")
    seen = set()
    if not plan["records"]:
        raise ValueError("Reference bank must not be empty")
    for row in plan["records"]:
        identifier = row["id"]
        if (not isinstance(identifier, str) or not identifier or identifier in seen
                or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in identifier)):
            raise ValueError(f"Unsafe or duplicate reference ID: {identifier!r}")
        seen.add(identifier)
        for key in ("roughness", "metallic"):
            value = row[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid {key}: {identifier}")
    paths = {"profile": Path(plan["profile"]), "base_scene": Path(renderer["base_scene"]),
             "sphere_path": Path(renderer["sphere_path"])}
    paths = {key: path if path.is_absolute() else ROOT / path for key, path in paths.items()}
    profile = json.loads(paths["profile"].read_text(encoding="utf8"))
    if renderer["light"] not in profile["lighting_conditions"]:
        raise ValueError("Unknown fixed lighting condition")
    locks = {str(path.resolve()): sha256(path) for path in (*paths.values(), Path(path), Path(__file__))}
    return plan, profile, paths, locks


def checked_record(record, output, expected):
    if any(record[key] != expected[key] for key in ("id", "roughness", "metallic")):
        raise ValueError("Saved reference parameters differ from the plan")
    for name in ("image", "mask", "input_rgba", "metadata"):
        path = Path(record[name]).resolve()
        if not path.is_relative_to(output) or not path.is_file() or sha256(path) != record[name + "_sha256"]:
            raise ValueError(f"Saved reference changed or missing: {name}/{record['id']}")
    return record


def open_output(output, locks, resume):
    receipt = output / "manifest.json"
    if output.exists():
        if not resume:
            raise FileExistsError(f"Use a fresh output, or --resume with the identical plan: {output}")
        if not receipt.is_file():
            raise ValueError("Existing output has no manifest; preserve it and choose a new output directory")
        manifest = json.loads(receipt.read_text(encoding="utf8"))
        if manifest["source_locks"] != locks:
            raise ValueError("Cannot resume: plan, renderer or immutable scene assets changed")
        return manifest
    output.mkdir(parents=True)
    manifest = {"schema": "zest_material_reference_bank/v1", "status": "running", "source_locks": locks, "records": []}
    write(receipt, manifest)
    return manifest


def render_one(row, output, plan, profile, paths, bpy, base):
    folder = output / row["id"]
    folder.mkdir(exist_ok=True)
    # A failed attempt is retained intact. New attempts never overwrite its images.
    index = 1
    while (folder / f"attempt{index:03d}").exists():
        index += 1
    attempt = folder / f"attempt{index:03d}"
    attempt.mkdir()
    bpy.ops.wm.open_mainfile(filepath=str(paths["base_scene"].resolve()))
    renderer = plan["renderer"]
    scene, lights, devices = base.configure_scene(profile, renderer["seed"], profile["lighting_conditions"][renderer["light"]])
    for existing in list(scene.objects):
        if existing.type == "MESH" and existing.name != "Ground":
            bpy.data.objects.remove(existing, do_unlink=True)
    obj = base.append_sphere(paths["sphere_path"].resolve(), profile)
    base.set_material(obj, row, profile)
    camera = base.configure_camera(obj, profile)
    geometry = hashlib.sha256()
    for vertex in obj.data.vertices:
        geometry.update(struct.pack("<3f", *vertex.co))
    for polygon in obj.data.polygons:
        geometry.update(struct.pack("<I", len(polygon.vertices)))
        geometry.update(struct.pack("<" + "I" * len(polygon.vertices), *polygon.vertices))
    bsdf = next(node for node in obj.data.materials[0].node_tree.nodes if node.type == "BSDF_PRINCIPLED")
    if any(node.type == "TEX_IMAGE" for node in obj.data.materials[0].node_tree.nodes):
        raise ValueError("Reference bank must have no texture image nodes")
    base.configure_mask(attempt)
    image = attempt / "image.png"
    scene.render.filepath = str(image)
    bpy.ops.render.render(write_still=True)
    for prefix, name in (("__mask_", "object_mask.png"), ("__input_rgba_", "input_rgba.png")):
        files = list(attempt.glob(prefix + "*.png"))
        if len(files) != 1:
            raise RuntimeError(f"Expected one {prefix} output: {attempt}")
        files[0].rename(attempt / name)
    record = {**row, "geometry_sha256": geometry.hexdigest(), "object_scale": profile["object"]["scale"],
              "albedo_linear_rgb": plan["albedo_linear_rgb"], "texture_image_nodes": 0,
              "normal_input_linked": bool(bsdf.inputs["Normal"].is_linked), "shader": "Principled BSDF",
              "ior": float(bsdf.inputs["IOR"].default_value), "camera": camera, "lights": lights,
              "world_strength": profile["lighting_conditions"][renderer["light"]]["world_strength"],
              "seed": scene.cycles.seed, "cycles_samples": scene.cycles.samples, "cuda_devices": devices,
              "resolution": [scene.render.resolution_x, scene.render.resolution_y], "blender_version": list(bpy.app.version),
              "color_management": {"view_transform": scene.view_settings.view_transform, "look": scene.view_settings.look,
                                   "exposure": scene.view_settings.exposure, "gamma": scene.view_settings.gamma}}
    for key, name in (("image", "image.png"), ("mask", "object_mask.png"), ("input_rgba", "input_rgba.png")):
        path = attempt / name
        record.update({key: str(path), key + "_sha256": sha256(path)})
    decoded_mask = bpy.data.images.load(record["mask"], check_existing=False)
    record["mask_pixels_sha256"] = mask_pixels_sha256(decoded_mask.pixels[:])
    bpy.data.images.remove(decoded_mask)
    metadata = attempt / "metadata.json"
    write(metadata, record)
    return {**record, "metadata": str(metadata), "metadata_sha256": sha256(metadata)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output == ROOT or output.is_relative_to(ROOT):
        raise ValueError("Use an external result directory")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu:
        raise ValueError("GPU argument and CUDA_VISIBLE_DEVICES differ")
    plan, profile, paths, locks = load_plan(args.plan)
    profile["blender"].update(cuda_visible_devices=args.gpu, cycles_samples=plan["renderer"]["cycles_samples"],
                              resolution=[plan["renderer"]["resolution"]] * 2)
    profile["material"]["base_color_linear_rgba"] = [*plan["albedo_linear_rgb"], 1.0]
    from experiments.natural_material_supermat_v1 import render_calibration as base
    import bpy
    if tuple(bpy.app.version) != tuple(profile["blender"]["version"]):
        raise ValueError("Blender version differs from the frozen renderer profile")
    locks[str(Path(base.__file__).resolve())] = sha256(base.__file__)
    manifest = open_output(output, locks, args.resume)
    manifest.update(status="running")
    manifest.pop("error", None)
    try:
        expected = {row["id"]: row for row in plan["records"]}
        completed = {}
        for record in manifest["records"]:
            identifier = record["id"]
            if identifier not in expected or identifier in completed:
                raise ValueError("Unexpected or duplicated saved reference")
            completed[identifier] = checked_record(record, output, expected[identifier])
        for row in plan["records"]:
            if row["id"] not in completed:
                record = render_one(row, output, plan, profile, paths, bpy, base)
                if manifest["records"]:
                    first = manifest["records"][0]
                    for key in ("geometry_sha256", "object_scale", "albedo_linear_rgb", "camera", "lights", "world_strength", "seed", "cycles_samples", "resolution", "mask_pixels_sha256"):
                        if record[key] != first[key]:
                            raise ValueError(f"Reference bank fixed-scene invariant changed: {key}")
                manifest["records"].append(record)
                write(output / "manifest.json", manifest)
                print(f"PBR_REFERENCE_RENDERED {len(manifest['records'])}/{len(expected)} {row['id']}", flush=True)
        if any(sha256(path) != digest for path, digest in locks.items()):
            raise ValueError("Immutable render inputs changed while generating the bank")
        manifest.update(status="complete", image_count=len(manifest["records"]))
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write(output / "manifest.json", manifest)


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else None)
