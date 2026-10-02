"""Aggregate SuperMat maps for the independent polished-metal control."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


LIGHTS = ("soft_front", "side_directional", "top_environment")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measure(maps, mask):
    mask_image = Image.open(mask).convert("L") if isinstance(mask, Path) else mask
    region = np.asarray(mask_image) >= 128
    if region.sum() < 1000:
        raise ValueError("Measurement region has fewer than 1000 pixels")
    result = {"pixel_count": int(region.sum())}
    for channel in ("roughness", "metallic"):
        path = maps / f"{channel}.png"
        image = Image.open(path).convert("L")
        if image.size != mask_image.size:
            raise ValueError(f"Size mismatch: {path}")
        values = np.asarray(image, dtype=np.float64)[region] / 255.0
        result[channel] = float(np.median(values))
        result[f"{channel}_iqr"] = float(np.quantile(values, 0.75) - np.quantile(values, 0.25))
        result[f"{channel}_map_sha256"] = sha256(path)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("source", "roundtrip"))
    parser.add_argument("--maps", type=Path, help="Source SuperMat output directory")
    parser.add_argument("--region", type=Path, help="Source bowl material ROI")
    parser.add_argument("--material", type=Path, help="Canonical source material JSON")
    parser.add_argument("--render-root", type=Path)
    parser.add_argument("--roundtrip-root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.phase == "source":
        if args.maps is None or args.region is None:
            parser.error("source requires --maps and --region")
        data = measure(args.maps, args.region)
        data.update({
            "control": "real_photo_bare_stainless_steel_spoon_bowl",
            "aggregation": "median_of_8bit_SuperMat_maps_divided_by_255",
            "measurement_region_sha256": sha256(args.region),
            "albedo_map_sha256": sha256(args.maps / "albedo.png"),
        })
    else:
        if args.material is None or args.render_root is None or args.roundtrip_root is None:
            parser.error("roundtrip requires --material, --render-root, --roundtrip-root")
        target = json.loads(args.material.read_text())
        rows = []
        for light in LIGHTS:
            mask_path = args.render_root / light / "object_mask.png"
            interior = Image.open(mask_path).convert("L").filter(ImageFilter.MinFilter(11))
            result = measure(args.roundtrip_root / light / "input_rgba", interior)
            result.update({
                "light": light,
                "render_mask_sha256": sha256(mask_path),
                "roughness_delta": result["roughness"] - target["roughness"],
                "metallic_delta": result["metallic"] - target["metallic"],
            })
            rows.append(result)
        data = {
            "control": target["control"],
            "source_material_sha256": sha256(args.material),
            "source_roughness": target["roughness"],
            "source_metallic": target["metallic"],
            "roundtrip": rows,
            "interpretation": "Same-estimator consistency check, not a physical ground-truth measurement",
        }
    args.output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
