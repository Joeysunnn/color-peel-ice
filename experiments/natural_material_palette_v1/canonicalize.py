"""Archive Palette A/R/N maps and remove chromatic Albedo from the concept."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stats(values):
    q05, q25, q50, q75, q95 = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {"median": float(q50), "iqr": float(q75 - q25),
            "q05": float(q05), "q95": float(q95),
            "minimum": float(values.min()), "maximum": float(values.max())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-id", choices=("mailbox", "spoon_positive_control"), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    sample = args.run_root / "samples" / args.sample_id
    decompose_manifest = sample / "decompose_manifest.json"
    if not decompose_manifest.is_file():
        raise FileNotFoundError(decompose_manifest)
    checkpoint = next((sample / "weights/material/an_object_with_azertyuiop_texture").glob("checkpoint-*"))
    output_root = sample / "canonical"
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir()
    summary = []
    for variant, directory in (("raw", "outputs"), ("renorm", "out_renorm")):
        folder = checkpoint / directory
        maps = {}
        for name in ("albedo", "normals", "roughness"):
            paths = list(folder.glob(f"*_{name}.png"))
            if len(paths) != 1:
                raise ValueError(f"Expected one {variant}/{name} map: {paths}")
            maps[name] = paths[0]
        albedo = np.asarray(Image.open(maps["albedo"]).convert("RGB"), dtype=np.float32) / 255
        roughness_rgb = np.asarray(Image.open(maps["roughness"]).convert("RGB"), dtype=np.float32) / 255
        normal_rgb = np.asarray(Image.open(maps["normals"]).convert("RGB"), dtype=np.float32) / 255
        if albedo.shape != roughness_rgb.shape or albedo.shape != normal_rgb.shape:
            raise ValueError(f"A/R/N shape mismatch: {variant}")
        if not np.allclose(roughness_rgb[..., 0], roughness_rgb[..., 1], atol=1/255):
            raise ValueError("Official roughness PNG channels differ")
        gray = np.rint(np.dot(albedo, [0.2126, 0.7152, 0.0722]) * 255).astype(np.uint8)
        subdir = output_root / variant
        subdir.mkdir()
        neutral = subdir / "neutral_albedo.png"
        Image.fromarray(np.repeat(gray[..., None], 3, axis=2)).save(neutral)
        normals = normal_rgb * 2 - 1
        normals /= np.maximum(np.linalg.norm(normals, axis=2, keepdims=True), 1e-8)
        tilt = np.degrees(np.arccos(np.clip(normals[..., 2], -1, 1)))
        record = {"sample_id": args.sample_id, "variant": variant,
                  "representation": "spatial roughness and normal maps; chromatic albedo excluded",
                  "albedo_source_path": str(maps["albedo"].resolve()),
                  "roughness_map_path": str(maps["roughness"].resolve()),
                  "normal_map_path": str(maps["normals"].resolve()),
                  "neutral_albedo_path": str(neutral.resolve()),
                  "roughness": stats(roughness_rgb[..., 0].ravel()),
                  "normal_tilt_degrees": stats(tilt.ravel()),
                  "metallic": None, "metallic_status": "not estimated by Material Palette",
                  "renderer_base_color": "neutral gray fixed by renderer",
                  "renderer_metallic_assumption": 0.0,
                  "normal_encoding": "official N in RGB as (unit vector + 1) / 2; tangent-space use is a preview assumption",
                  "map_sha256": {key: sha(path) for key, path in maps.items()},
                  "neutral_albedo_sha256": sha(neutral),
                  "decompose_manifest_sha256": sha(decompose_manifest),
                  "script_sha256": sha(__file__),
                  "project_git_commit": subprocess.check_output(
                      ["git", "rev-parse", "HEAD"], cwd=args.project_root, text=True).strip()}
        (subdir / "material.json").write_text(json.dumps(record, indent=2) + "\n")
        summary.append({"variant": variant, "roughness": record["roughness"],
                        "normal_tilt_degrees": record["normal_tilt_degrees"]})
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
