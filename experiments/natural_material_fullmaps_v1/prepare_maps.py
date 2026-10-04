"""Preserve masked SuperMat A/R/M maps as renderable texture patches."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


MATERIALS = ("mailbox", "metal_spoon", "wood_spoon")
CHANNELS = ("albedo", "roughness", "metallic")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_paths(previous, name):
    canonical_path = previous / "canonical" / f"{name}.json"
    canonical = json.loads(canonical_path.read_text())
    selected = canonical["selected_region"]
    if name == "wood_spoon":
        candidate = next(row for row in canonical["candidates"] if row["name"] == selected)
        return canonical_path, Path(candidate["mask"]), previous / "supermat_outputs" / name, candidate
    old = json.loads(Path(canonical["source_canonical"]).read_text())
    candidate = next(row for row in old["candidates"] if row["name"] == selected)
    maps = Path(canonical["source_canonical"]).parent.parent / "supermat_outputs"
    maps /= "mailbox" if name == "mailbox" else "spoon_positive_control"
    return canonical_path, Path(candidate["mask_path"]), maps, candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    overview = Image.new("RGB", (512 * 3, 512 * 3), "white")
    draw = ImageDraw.Draw(overview)
    records = {}
    for row, name in enumerate(MATERIALS):
        canonical_path, mask_path, maps_dir, candidate = source_paths(args.previous_run, name)
        mask = np.asarray(Image.open(mask_path).convert("L")) >= 128
        if mask.shape != (512, 512) or mask.sum() < 1000:
            raise ValueError(f"Invalid material mask: {mask_path}")
        ys, xs = np.nonzero(mask)
        left, top, right, bottom = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        inside = mask[top:bottom, left:right]
        missing = (~inside).astype("uint8") * 255
        source_dir = args.output / "source" / name
        patch_dir = args.output / "textures" / name
        source_dir.mkdir(parents=True)
        patch_dir.mkdir(parents=True)
        shutil.copyfile(mask_path, source_dir / "region_mask.png")
        output_maps = {}
        for col, channel in enumerate(CHANNELS):
            original = maps_dir / f"{channel}.png"
            image = Image.open(original).convert("RGB" if channel == "albedo" else "L")
            if image.size != (512, 512):
                raise ValueError(f"Unexpected map size: {original}")
            if channel != "albedo":
                expected = candidate["measurements"][channel]["map_sha256"]
                if sha(original) != expected:
                    raise ValueError(f"Map changed since canonicalization: {original}")
            copied = source_dir / original.name
            shutil.copyfile(original, copied)
            crop = np.asarray(image)[top:bottom, left:right].copy()
            filled = cv2.inpaint(crop, missing, 3, cv2.INPAINT_TELEA)
            if not np.array_equal(filled[inside], crop[inside]):
                raise ValueError(f"Inpainting altered valid material pixels: {name}/{channel}")
            patch = patch_dir / f"{channel}.png"
            Image.fromarray(filled).save(patch)
            output_maps[channel] = {
                "original_path": str(original.resolve()),
                "original_sha256": sha(original),
                "copied_sha256": sha(copied),
                "texture_sha256": sha(patch),
                "mode": image.mode,
            }
            shown = Image.fromarray(filled).convert("RGB").resize((512, 512))
            overview.paste(shown, (col * 512, row * 512))
            draw.text((col * 512 + 8, row * 512 + 8), f"{name} {channel}", fill="white")
        records[name] = {
            "source_canonical_path": str(canonical_path.resolve()),
            "source_canonical_sha256": sha(canonical_path),
            "selected_region": json.loads(canonical_path.read_text())["selected_region"],
            "mask_path": str(mask_path.resolve()),
            "mask_sha256": sha(mask_path),
            "crop_xyxy": [int(left), int(top), int(right), int(bottom)],
            "valid_pixels": int(inside.sum()),
            "inpainted_pixels": int(missing.astype(bool).sum()),
            "maps": output_maps,
        }
    overview.save(args.output / "texture_overview.png")
    (args.output / "manifest.json").write_text(json.dumps({
        "status": "complete", "method": "crop reviewed region and inpaint outside mask only",
        "previous_run": str(args.previous_run.resolve()),
        "script_sha256": sha(Path(__file__)),
        "texture_overview_sha256": sha(args.output / "texture_overview.png"),
        "materials": records,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
