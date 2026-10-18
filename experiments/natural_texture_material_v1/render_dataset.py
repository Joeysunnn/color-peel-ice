"""Render the natural T/M factorial with the two existing Phase 1 renderers."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.natural_texture_material_v1 import dataset_protocol as schema


def revision():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    if git("status", "--porcelain"):
        raise ValueError("Render checkout must be clean")
    return {"commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current")}


def verify_locks(locks):
    if any(schema.sha256(Path(path)) != digest for path, digest in locks.items()):
        raise ValueError("Accepted extraction or its source inputs changed")


def validate_plan(plan, protocol):
    if (plan.get("status") != "complete" or plan.get("protocol_sha256") != schema.protocol_sha256(protocol)
            or plan["records"] != schema.records(protocol) or len(plan["records"]) != 72):
        raise ValueError("Plan differs from the natural72-image factorial")
    verify_locks(protocol["source_locks"])
    return plan["records"]


def shader_audit(bpy, branch):
    objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH" and obj.name != "Ground"]
    if len(objects) != 1 or len(objects[0].data.materials) != 1:
        raise ValueError("Expected one controlled object and one material")
    nodes = objects[0].data.materials[0].node_tree.nodes
    bsdf = next(node for node in nodes if node.type == "BSDF_PRINCIPLED")
    result = {"texture_image_nodes": sum(node.type == "TEX_IMAGE" for node in nodes),
              **{key + "_input_linked": bool(bsdf.inputs[socket].is_linked) for key, socket in
                 (("roughness", "Roughness"), ("metallic", "Metallic"), ("normal", "Normal"),
                  ("base_color", "Base Color"))}}
    expected = {"texture_image_nodes": 1 if branch == "texture" else 0,
                "roughness_input_linked": False, "metallic_input_linked": False,
                "normal_input_linked": False, "base_color_input_linked": branch == "texture"}
    if result != expected:
        raise ValueError("Actual shader nodes violate the independent T/M definition")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(ROOT):
        raise ValueError("Use a new external render directory")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu:
        raise ValueError("GPU argument and CUDA_VISIBLE_DEVICES differ")
    plan = json.loads(args.plan.read_text(encoding="utf8"))
    protocol_path = Path(plan["protocol_path"])
    protocol = json.loads(protocol_path.read_text(encoding="utf8"))
    rows, git = validate_plan(plan, protocol), revision()
    if plan["git"] != git:
        raise ValueError("Plan and render Git revisions differ")
    renderer = protocol["renderer"]
    scene, profile_path = Path(renderer["base_scene"]), ROOT / renderer["profile"]
    if schema.sha256(scene) != renderer["base_scene_sha256"]:
        raise ValueError("Frozen base scene changed")
    profile = json.loads(profile_path.read_text(encoding="utf8"))
    profile["blender"].update(cuda_visible_devices=args.gpu, cycles_samples=renderer["cycles_samples"],
                              resolution=[renderer["resolution"]] * 2)
    from experiments.natural_material_threeway_v1.render_grid import ASSETS, append_shape
    shape_hashes = {shape: schema.sha256(Path(renderer["shape_dir"]) / f"{ASSETS[shape]}.blend")
                    for shape in sorted({row["shape"] for row in rows})}
    from experiments.natural_material_supermat_v1 import render_calibration as base
    from experiments.texture_material_disentangled_v1.render import render_one as texture_render
    from experiments.material_uniform_albedo_v1.render import render_one as material_render
    import bpy
    if tuple(bpy.app.version) != (4, 2, 11):
        raise ValueError("Expected Blender4.2.11")
    evidence = {"status": "running", "protocol_path": str(protocol_path),
                "protocol_sha256": schema.protocol_sha256(protocol), "plan_path": str(args.plan.resolve()),
                "plan_sha256": schema.sha256(args.plan), "source_locks": protocol["source_locks"],
                "profile_sha256": schema.sha256(profile_path), "base_scene_sha256": schema.sha256(scene),
                "shape_sha256": shape_hashes, "git": git, "git_commit": git["commit"], "expected_images": 72,
                "reused_renderer_sha256": {str(Path(function.__code__.co_filename)): schema.sha256(
                    Path(function.__code__.co_filename)) for function in (texture_render, material_render)},
                "records": []}
    output.mkdir(parents=True)
    try:
        for row in rows:
            render_one = texture_render if row["branch"] == "texture" else material_render
            record = render_one(row, output, protocol, profile, bpy, base, append_shape)
            record.update(shader_audit(bpy, row["branch"]))
            metadata = Path(record["metadata"])
            base.write_json(metadata, {k: v for k, v in record.items() if k not in ("metadata", "metadata_sha256")})
            record["metadata_sha256"] = schema.sha256(metadata)
            evidence["records"].append(record)
            print(f"NATURAL_DATASET_RENDERED {len(evidence['records'])}/72 {row['id']}", flush=True)
            base.write_json(output / "manifest.json", evidence)
        verify_locks(protocol["source_locks"])
        if (revision() != git or schema.sha256(args.plan) != evidence["plan_sha256"]
                or schema.protocol_sha256(json.loads(protocol_path.read_text())) != evidence["protocol_sha256"]):
            raise ValueError("Checkout or plan changed during rendering")
        evidence.update(status="complete", image_count=len(evidence["records"]))
    except BaseException as error:
        evidence.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        base.write_json(output / "manifest.json", evidence)


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
