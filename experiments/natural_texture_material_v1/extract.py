"""Extract independent natural Texture and scalar Material from reviewed SuperMat maps."""

import argparse
import html
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from experiments.natural_material_fullmaps_v1.prepare_maps import source_paths
from experiments.natural_material_supermat_v1.material_pipeline import map_values
from experiments.texture_material_disentangled_v1 import data
from scripts.launch.colorpeel_run import git_info, resolve_run_dir


MATERIALS = ("mailbox", "metal_spoon", "wood_spoon")
ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def inspect_sources(source):
    """Resolve the reviewed regions, verify prior receipts, and lock every input."""
    manifest_path = source / "maps/manifest.json"
    manifest = read(manifest_path)
    if manifest.get("status") != "complete" or set(manifest["materials"]) != set(MATERIALS):
        raise ValueError("Expected the completed three-source full-map extraction")
    previous = Path(manifest["previous_run"])
    locks, records = {str(manifest_path): data.sha256(manifest_path)}, {}
    for name in MATERIALS:
        recorded = manifest["materials"][name]
        canonical_path, mask_path, maps, candidate = source_paths(previous, name)
        canonical = read(canonical_path)
        receipt_canonical = canonical
        paths = {"canonical": canonical_path, "mask": mask_path,
                 **{channel: maps / f"{channel}.png" for channel in ("albedo", "roughness", "metallic")}}
        if (str(canonical_path.resolve()) != recorded["source_canonical_path"]
                or data.sha256(canonical_path) != recorded["source_canonical_sha256"]
                or str(mask_path.resolve()) != recorded["mask_path"]
                or data.sha256(mask_path) != recorded["mask_sha256"]
                or canonical["selected_region"] != recorded["selected_region"]):
            raise ValueError(f"Reviewed canonical or mask changed: {name}")
        if "source_canonical" in canonical:
            upstream = Path(canonical["source_canonical"])
            if data.sha256(upstream) != canonical["source_canonical_sha256"]:
                raise ValueError(f"Upstream canonical changed: {name}")
            paths["upstream_canonical"] = upstream
            receipt_canonical = read(upstream)
        mask = Image.open(mask_path).convert("L")
        if mask.size != (512, 512):
            raise ValueError("Expected a 512px reviewed SuperMat mask")
        distributions = {}
        for channel in ("albedo", "roughness", "metallic"):
            path, receipt = paths[channel], recorded["maps"][channel]
            if (str(path.resolve()) != receipt["original_path"]
                    or data.sha256(path) != receipt["original_sha256"]
                    or Image.open(path).size != (512, 512)):
                raise ValueError(f"Source SuperMat map changed: {name}/{channel}")
            if channel != "albedo":
                if data.sha256(path) != candidate["measurements"][channel]["map_sha256"]:
                    raise ValueError(f"Canonical map hash changed: {name}/{channel}")
                distributions[channel] = map_values(path, mask)
                if abs(distributions[channel]["median"] - canonical[channel]) > 1e-12:
                    raise ValueError(f"Canonical R/M disagrees with reviewed region: {name}/{channel}")
        input_name = "spoon_positive_control" if name == "metal_spoon" else name
        paths["input"] = maps.parent.parent / "supermat_inputs" / f"{input_name}.png"
        paths["supermat_manifest"] = maps.parent / "inference_manifest.json"
        inference = read(paths["supermat_manifest"])
        expected_receipt = receipt_canonical.get("supermat_manifest_sha256")
        if expected_receipt and data.sha256(paths["supermat_manifest"]) != expected_receipt:
            raise ValueError(f"Frozen SuperMat inference receipt changed: {name}")
        matches = [row for row in inference["outputs"] if Path(row["input"]).stem == input_name]
        if (inference.get("status") != "complete" or len(matches) != 1
                or inference.get("exit_code") != 0 or inference.get("image_size") != 512
                or inference.get("seed") != 42
                or Path(matches[0]["input"]).resolve() != paths["input"].resolve()
                or data.sha256(paths["input"]) != matches[0]["input_sha256"]):
            raise ValueError(f"SuperMat input receipt differs: {name}")
        for channel in ("albedo", "roughness", "metallic"):
            if data.sha256(paths[channel]) != matches[0]["maps"][channel]["sha256"]:
                raise ValueError(f"SuperMat output receipt differs: {name}/{channel}")
        if Image.open(paths["input"]).size != (512, 512):
            raise ValueError("Expected a 512px natural-image SuperMat input")
        locks.update({str(path): data.sha256(path) for path in paths.values()})
        records[name] = {"paths": paths, "canonical": canonical, "distributions": distributions}
    return records, locks


def texture_patch(gray, mask):
    """Crop the reviewed region and fill only missing pixels; retain aspect ratio."""
    valid = np.asarray(mask.convert("L")) >= 128
    ys, xs = np.nonzero(valid)
    box = (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1))
    inside = valid[box[1]:box[3], box[0]:box[2]]
    pixels = np.asarray(gray)[box[1]:box[3], box[0]:box[2]].copy()
    filled = cv2.inpaint(pixels, (~inside).astype("uint8") * 255, 3, cv2.INPAINT_TELEA)
    low, high = int(pixels[inside].min()), int(pixels[inside].max())
    filled[~inside] = np.clip(filled[~inside], low, high)
    if not np.array_equal(filled[inside], pixels[inside]):
        raise ValueError("Texture completion altered valid-region pixels")
    return Image.fromarray(filled), {"crop_xyxy": list(box), "valid_pixels": int(inside.sum()),
        "inpainted_pixels": int((~inside).sum()), "fill_fraction": float((~inside).mean()),
        "fill_method": "OpenCV TELEA radius3 outside reviewed mask only",
        "filled_pixel_clip_range_uint8": [low, high],
        "pixel_aspect_preserved": True, "physical_texture_scale": "unknown from single image"}


def displayed_input(path):
    image = Image.open(path).convert("RGBA")
    return Image.alpha_composite(Image.new("RGBA", image.size, (128, 128, 128, 255)), image).convert("RGB")


def overview(output, records):
    labels = ("Natural image / gray alpha composite", "Reviewed measurement region", "SuperMat Albedo (RGB)",
              "Albedo linear luminance", "Texture: grayscale + robust normalize", "SuperMat roughness map", "SuperMat metallic map")
    width, height, header = 280, 336, 68
    sheet = Image.new("RGB", (width * len(labels), header + height * 3), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 8), "Phase2 extraction only: Texture = spatial albedo pattern; Material = two scalar R/M medians.", fill="black")
    draw.text((10, 26), "R/M panels are original spatial SuperMat outputs for audit; Material training will use scalar values only.", fill="black")
    for row, name in enumerate(MATERIALS):
        folder, record = output / name, records[name]
        natural = displayed_input(folder / "input.png")
        valid = np.asarray(Image.open(folder / "mask.png").convert("L")) >= 128
        overlay = np.asarray(natural).copy()
        overlay[valid] = (overlay[valid] * .65 + np.array([255, 0, 0]) * .35).astype("uint8")
        images = [natural, Image.fromarray(overlay), Image.open(folder / "albedo.png"),
                  Image.open(folder / "luminance.png"), Image.open(folder / "texture.png"),
                  Image.open(folder / "roughness.png"), Image.open(folder / "metallic.png")]
        y = header + row * height
        for col, (label, image) in enumerate(zip(labels, images)):
            thumbnail = ImageOps.contain(image.convert("RGB"), (264, 264))
            sheet.paste(thumbnail, (col * width + (width - thumbnail.width) // 2, y + 34))
            draw.text((col * width + 8, y + 6), label, fill="black")
        draw.text((8, y + 304), f"{name}: R={record['roughness']:.6f}; M={record['metallic']:.6f}; region={record['selected_region']}", fill="black")
        draw.text((8, y + 320), f"Albedo luminance q98-q02={record['texture_statistics']['luminance_span']:.6f}; fill={record['texture_completion']['fill_fraction']:.1%}; semantic={record['semantic_status']}", fill="black")
    sheet.save(output / "overview.png")
    links = []
    for name in MATERIALS:
        links.append(f"<h2>{name}</h2><p>R={records[name]['roughness']:.6f}; M={records[name]['metallic']:.6f}</p>")
        links.extend(f'<a href="{name}/{path.name}">{html.escape(path.name)}</a> ' for path in sorted((output / name).iterdir()))
    (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><h1>Natural Texture / Material extraction</h1>'
        '<p>Extraction only. No renderer or concept training has run. See manifest.json for limitations and provenance.</p>'
        '<a href="manifest.json">Manifest</a> | <a href="source_locks.json">Source hashes</a>'
        '<p><a href="overview.png"><img src="overview.png" style="max-width:100%"></a></p>' + "\n".join(links), encoding="utf8")


def run(source, output):
    source, output = Path(source).resolve(), resolve_run_dir(Path(output))
    if output.exists():
        raise FileExistsError(output)
    if output == ROOT or output.is_relative_to(ROOT) or output == source or output.is_relative_to(source):
        raise ValueError("Use a new external output directory, separate from source artifacts")
    revision, protocol = git_info(), data.read_protocol()
    sources, locks = inspect_sources(source)
    protocol_hash = data.protocol_sha256()
    output.mkdir(parents=True)
    data.write_json(output / "source_locks.json", locks)
    extraction_protocol = {"phase": 2, "stage": "extraction_only", "texture_extraction": protocol["texture_extraction"],
        "material_extraction": "SuperMat separate 8bit R/M maps divided by255; reviewed-region median",
        "phase1_protocol_sha256": protocol_hash}
    data.write_json(output / "protocol.json", extraction_protocol)
    manifest = {"schema": "natural_texture_material/extraction_v1", "phase": 2, "status": "running", "git": revision,
        "source_fullmaps_run": str(source), "phase1_protocol_sha256": protocol_hash, "materials": {},
        "protocol_sha256": data.sha256(output / "protocol.json"), "script_sha256": data.sha256(Path(__file__)),
        "limitations": ["Single-image SuperMat albedo may retain shading or highlights; grayscale does not guarantee reflectance removal.",
            "Robust normalization can amplify weak texture, residual shading, or noise; raw luminance spans are recorded.",
            "Texture patch is a single-view rectangular completion, without recovered physical scale or multiview refinement.",
            "Irregular-mask rectangular completion may retain silhouette or inpainting structure; review fill fractions and patches visually.",
            "SuperMat R/M medians are estimator outputs, not measured BRDF ground truth; known SuperMat-to-Cycles bias is uncorrected.",
            "Wood matte-semantic conflict is preserved; no VLM-driven numeric override."]}
    data.write_json(output / "manifest.json", manifest)
    try:
        for name, source_record in sources.items():
            folder, paths = output / name, source_record["paths"]
            folder.mkdir()
            for key in ("input", "mask", "albedo", "roughness", "metallic"):
                shutil.copyfile(paths[key], folder / f"{key}.png")
            mask = Image.open(folder / "mask.png").convert("L")
            albedo = Image.open(folder / "albedo.png").convert("RGB")
            gray, stats = data.extract_texture(albedo, mask)
            gray.save(folder / "gray_full.png")
            rgb = np.asarray(albedo, dtype=np.float64) / 255
            linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
            luminance = linear @ np.array([.2126, .7152, .0722])
            Image.fromarray(np.rint(luminance * 255).astype("uint8")).save(folder / "luminance.png")
            patch, completion = texture_patch(gray, mask)
            patch.save(folder / "texture.png")
            stats["luminance_span"] = stats["upper_luminance"] - stats["lower_luminance"]
            canonical = source_record["canonical"]
            material = {"representation": "scalar roughness/metallic only; no albedo, spatial R/M, bump or normal",
                "roughness": source_record["distributions"]["roughness"]["median"],
                "metallic": source_record["distributions"]["metallic"]["median"],
                "distributions": source_record["distributions"], "selected_region": canonical["selected_region"],
                "semantic_status": canonical["semantic_status"], "source_canonical": str(paths["canonical"])}
            data.write_json(folder / "material.json", material)
            record = {**material, "texture_statistics": stats, "texture_completion": completion,
                "files": {path.name: {"path": str(path), "sha256": data.sha256(path)} for path in sorted(folder.iterdir())}}
            data.write_json(folder / "extraction.json", record)
            manifest["materials"][name] = record
        if (git_info() != revision or data.protocol_sha256() != protocol_hash
                or any(data.sha256(Path(path)) != digest for path, digest in locks.items())):
            raise ValueError("Source inputs, protocol or checkout changed during extraction")
        overview(output, manifest["materials"])
        manifest.update(status="complete", overview_sha256=data.sha256(output / "overview.png"),
                        source_locks_sha256=data.sha256(output / "source_locks.json"))
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        data.write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-fullmaps-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.source_fullmaps_run, args.output)
    print(json.dumps({"status": result["status"], "materials": list(result["materials"])}))


if __name__ == "__main__":
    main()
