"""Pad square natural images and material masks to IID's 640x480 canvas."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from PIL import Image


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pad(image, side, fill, resample):
    if image.width != image.height:
        raise ValueError("This pilot expects the existing square natural crops")
    resized = image.resize((side, side), resample)
    canvas = Image.new(image.mode, (640, 480), fill)
    canvas.paste(resized, ((640 - side) // 2, 0))
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    project_root, output_root = args.project_root.resolve(), args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    if config["iid_resolution_hw"] != [480, 640]:
        raise ValueError("Expected IID's released 480x640 inference resolution")
    (output_root / "inputs").mkdir(parents=True)
    (output_root / "masks").mkdir()
    (output_root / "vlm_priors").mkdir()
    records = []
    for sample in config["samples"]:
        image_path = project_root / sample["image"]
        mask_path = project_root / sample["region_mask"]
        prior_path = Path(sample["vlm_prior_source"])
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if image.size != mask.size or mask.getbbox() is None:
            raise ValueError(f"Invalid material mask for {sample['id']}")
        image_out = output_root / "inputs" / f"{sample['id']}.png"
        mask_out = output_root / "masks" / f"{sample['id']}.png"
        prior_out = output_root / "vlm_priors" / f"{sample['id']}.json"
        pad(image, 480, (127, 127, 127), Image.Resampling.LANCZOS).save(image_out)
        pad(mask, 480, 0, Image.Resampling.NEAREST).save(mask_out)
        shutil.copy2(prior_path, prior_out)
        records.append({"id": sample["id"], "image_origin": str(image_path),
                        "image_origin_sha256": sha(image_path),
                        "material_mask_origin": str(mask_path),
                        "material_mask_origin_sha256": sha(mask_path),
                        "vlm_prior_origin": str(prior_path),
                        "vlm_prior_origin_sha256": sha(prior_path),
                        "iid_input_sha256": sha(image_out),
                        "iid_mask_sha256": sha(mask_out),
                        "preprocessing": "square resized to 480x480 and centered with 80 gray pixels on each side"})
    manifest = {"config_sha256": sha(args.config), "script_sha256": sha(__file__),
                "project_git_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=project_root, text=True).strip(),
                "samples": records}
    (output_root / "input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
