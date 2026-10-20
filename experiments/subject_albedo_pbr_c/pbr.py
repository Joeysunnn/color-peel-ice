"""Build and audit a fixed-view Cycles mailbox scene; no training is available."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

from experiments.subject_albedo_pbr_c import geometry, pbr_data
from experiments.subject_albedo_pbr_c.review import _font
from experiments.sd15_token_local_kv_v1.contract import read_json, sha256
from experiments.sd_native_comparison_v1.generate import write_json
from scripts.launch.colorpeel_run import git_info, resolve_run_dir
from src.methods.colorpeel_ice.gamut_aware_lch_injection import _srgb_to_linear

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = Path(__file__).with_name("pbr_protocol.json")


def review(output, rows):
    colors = list(dict.fromkeys(row["color"] for row in rows))
    sheet = Image.new("RGB", (1536, 285 + len(colors) * 280), "white")
    draw, font = ImageDraw.Draw(sheet), _font(15)
    draw.text((10, 8), "Actual Cycles RGB candidates; manually fitted single-view geometry; fixed A / scene; only M/R vary", font=font, fill="black")
    draw.text((10, 30), "Below: identical object crops from the full 512px images. Not yet visually accepted or used for training.", font=font, fill="darkred")
    for index, (name, path) in enumerate([
            ("Original RGB", output / "source/original.jpg"),
            ("Actual fitted geometry / clay", output / "clay.png"),
            ("Actual mesh normals", output / "normal.png"),
            ("UV diagnostic / NOT training", output / "uv.png")]):
        image = Image.open(path).convert("RGB")
        image.thumbnail((235, 210))
        sheet.paste(image, (index * 300 + 8, 72))
        draw.text((index * 300 + 8, 52), name, font=font, fill="black")
    for row_index, color in enumerate(colors):
        selected = sorted((row for row in rows if row["color"] == color), key=lambda row: (row["metallic"], row["roughness"]))
        for column, row in enumerate(selected):
            x, y = column * 256, 285 + row_index * 280
            draw.text((x + 5, y), f"{color} M={row['metallic']:g} R={row['roughness']:g}", font=font, fill="black")
            with Image.open(row["image"]) as image:
                crop = image.crop((120, 270, 470, 512)).resize((250, 173), Image.Resampling.LANCZOS)
                sheet.paste(crop, (x + 3, y + 25))
            draw.text((x + 5, y + 205), f"clip={row['qc']['clipped_pixel_fraction']:.1%}", font=font, fill="black")
            draw.text((x + 5, y + 227), "fixed A / fixed original background", font=font, fill="black")
    sheet.save(output / "contact_sheet.png")
    (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Subject C Cycles candidates</title>'
        '<h1>Fixed-view mailbox: true foreground PBR, manually fitted 2.5D geometry</h1>'
        '<p>Not measured geometry or recovered lighting. Original background shadows stay fixed. No training started.</p>'
        '<p>All images are candidates pending visual acceptance. Crops below reference complete 512px RGB files.</p>'
        '<a href="status.json">Status</a> | <a href="plan.json">Frozen scene and inputs</a> | '
        '<a href="dataset_manifest.jsonl">Images, captions and numeric checks</a>'
        '<p><a href="contact_sheet.png"><img style="max-width:100%" src="contact_sheet.png"></a></p>'
        + ''.join(f'<p><a href="images/{row["id"]}.png">{row["id"]}</a></p>' for row in rows), encoding="utf-8")


def run(preview, output, gpu, full_grid=False):
    preview, output = Path(preview).resolve(), resolve_run_dir(Path(output))
    if (output.exists() or output.is_relative_to(ROOT) or output.is_relative_to(preview)
            or preview.is_relative_to(output) or type(gpu) is not int or gpu < 0):
        raise ValueError("select an isolated GPU and fresh external output outside all sources")
    git = git_info()
    rows, locks = pbr_data.verify_preview(preview)
    if not full_grid:
        rows = [row for row in rows if row["color"] == "green"]
    protocol = read_json(PROTOCOL)
    protocol["geometry"] = geometry.DEFAULT_GEOMETRY
    locks[str(PROTOCOL)] = sha256(PROTOCOL)
    output.mkdir(parents=True)
    (output / "source").mkdir()
    for name in ("original.jpg", "object_mask.png"):
        shutil.copyfile(preview / "source" / name, output / "source" / name)
    original = np.asarray(Image.open(output / "source/original.jpg").convert("RGB"))
    mask = np.asarray(Image.open(output / "source/object_mask.png").convert("L"))
    mesh = geometry.build_mesh(mask, protocol["geometry"])
    arrays = {name: np.asarray(mesh.pop(name), dtype=np.int32 if name == "faces" else np.float32)
              for name in ("vertices", "faces", "uv")}
    np.savez_compressed(output / "geometry.npz", **arrays)
    mesh["array_content_sha256"] = hashlib.sha256(b"".join(value.tobytes() for value in arrays.values())).hexdigest()
    write_json(output / "geometry_audit.json", mesh)
    protocol_sha = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    plan = {"git": git, "preview": str(preview), "protocol": protocol, "protocol_sha256": protocol_sha,
            "mesh": str(output / "geometry.npz"), "mesh_sha256": sha256(output / "geometry.npz"),
            "rows": rows, "source_locks": locks}
    write_json(output / "plan.json", plan)
    command = [protocol["renderer"]["blender"], "--background", "--python-exit-code", "1",
               "--python", str(Path(__file__).with_name("render_pbr.py")), "--",
               "--plan", str(output / "plan.json"), "--output", str(output / "renders")]
    write_json(output / "command.json", {"command": command, "cwd": str(ROOT), "CUDA_VISIBLE_DEVICES": str(gpu)})
    state = {"status": "rendering", "git": git, "mode": "full30" if full_grid else "geometry_and_green6",
             "expected_rgb": len(rows), "generated_rgb": 0, "training_started": False,
             "training_ready": False, "geometry": mesh["fit_description"]}
    write_json(output / "status.json", state)
    try:
        with (output / "render.log").open("x") as log:
            result = subprocess.run(command, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
                "PYTHONDONTWRITEBYTECODE": "1"}, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError("Cycles render failed; see render.log")
        rendered = read_json(output / "renders/renders.json")
        if len(rendered) != len(rows) or [item["id"] for item in rendered] != [row["id"] for row in rows]:
            raise ValueError("rendered A/M/R grid is incomplete")
        uv = pbr_data.read_exr(output / "renders/uv_diagnostic.exr")
        albedo = np.asarray(Image.open(rows[0]["maps"]["albedo"]["path"]).convert("RGB"))
        core = (mask == 255) & (uv[:, :, 3] > .999)
        error = np.abs(uv[:, :, :3] - _srgb_to_linear(albedo) * uv[:, :, 3:4])
        uv_p95 = float(np.percentile(error[core], 95))
        predicted = uv[:, :, 3] > .5
        iou = float(np.count_nonzero(predicted & (mask > 0)) / np.count_nonzero(predicted | (mask > 0)))
        if uv_p95 > .01 or iou < .995:
            raise ValueError(f"UV/alpha validation failed: p95={uv_p95}, mask IoU={iou}")
        write_json(output / "projection_audit.json", {"UV_linear_error_p95": uv_p95, "mask_IoU": iou,
            "limitations": "mask fit proves projection only, not true recovered geometry or normals"})
        uv_image, _ = pbr_data.composite(uv, original, mask)
        Image.fromarray(uv_image).save(output / "uv.png")
        clay, _ = pbr_data.composite(pbr_data.read_exr(output / "renders/clay_diagnostic.exr"), original, mask)
        Image.fromarray(clay).save(output / "clay.png")
        import cv2
        normal = cv2.imread(str(next((output / "renders").glob("normal_*.exr"))), cv2.IMREAD_UNCHANGED)[:, :, :3][:, :, ::-1]
        if not np.isfinite(normal).all() or np.median(np.linalg.norm(normal[core], axis=1)) < .98:
            raise ValueError("actual foreground normal pass is nonfinite or not approximately unit length")
        display_normal = np.rint(np.clip((normal + 1) / 2, 0, 1) * 255).astype(np.uint8)
        display_normal[mask == 0] = 255
        Image.fromarray(display_normal).save(output / "normal.png")
        (output / "images").mkdir()
        generated = []
        for row, receipt in zip(rows, rendered):
            if (receipt["sha256"] != sha256(receipt["path"]) or receipt["mesh_sha256"] != plan["mesh_sha256"]
                    or receipt["protocol_sha256"] != protocol_sha or receipt["albedo_sha256"] != row["maps"]["albedo"]["sha256"]):
                raise ValueError("render scene or Albedo provenance changed")
            image, qc = pbr_data.composite(pbr_data.read_exr(receipt["path"]), original, mask)
            path = output / "images" / f"{row['id']}.png"
            Image.fromarray(image).save(path)
            generated.append({**row, "image": str(path), "image_sha256": sha256(path),
                "mask": str(output / "source/object_mask.png"), "caption": row["proposed_caption"],
                "render_status": "true_cycles_pbr", "training_eligible": False, "qc": qc,
                "scene_sha256": protocol_sha, "geometry_content_sha256": mesh["array_content_sha256"]})
        (output / "dataset_manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in generated), encoding="utf-8")
        review(output, generated)
        if git_info() != git or any(sha256(path) != value for path, value in locks.items()):
            raise ValueError("sources or checkout changed during fixed-scene rendering")
        state.update(status="pbr_candidates_numeric_verified_pending_visual_review", generated_rgb=len(generated),
                     projection_audit=read_json(output / "projection_audit.json"))
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "status.json", state)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--full-grid", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.preview, args.output, args.gpu, args.full_grid)))
