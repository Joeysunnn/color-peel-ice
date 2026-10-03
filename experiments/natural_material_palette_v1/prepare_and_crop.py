"""Copy natural samples into the official layout and run its crop function."""

import argparse
import hashlib
import importlib.util
import json
import random
import shutil
import subprocess
from pathlib import Path

from PIL import Image


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def commit(root):
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config["schema_version"] != 1 or args.run_root.exists():
        raise ValueError("Expected schema 1 and a new run directory")
    spec = importlib.util.spec_from_file_location("material_palette_crop", args.official_root / "concept/crop.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = args.run_root.resolve()
    (root / "samples").mkdir(parents=True)
    records = []
    for sample in config["samples"]:
        source = args.project_root / sample["image"]
        mask_source = args.project_root / sample["mask"]
        image = Image.open(source)
        mask = Image.open(mask_source)
        if image.size != mask.size or mask.convert("L").getbbox() is None:
            raise ValueError(f"Invalid source mask: {sample['id']}")
        folder = root / "samples" / sample["id"]
        (folder / "masks").mkdir(parents=True)
        image_out = folder / f"source{source.suffix.lower()}"
        mask_out = folder / "masks/material.png"
        shutil.copy2(source, image_out)
        shutil.copy2(mask_source, mask_out)
        random.seed(config["seed"])
        crop_dir = module.main(folder, patch_sizes=sample["patch_sizes"],
                               threshold=config["crop_threshold"]) / "material"
        crops = sorted(crop_dir.glob("*.png")) if crop_dir.is_dir() else []
        if len(crops) < 2:
            raise RuntimeError(f"Official crop found fewer than two patches: {sample['id']}")
        records.append({"sample_id": sample["id"], "source": str(source),
                        "source_sha256": sha(source), "mask": str(mask_source),
                        "mask_sha256": sha(mask_source), "copied_source_sha256": sha(image_out),
                        "copied_mask_sha256": sha(mask_out), "image_size": image.size,
                        "patch_sizes": sample["patch_sizes"], "threshold": config["crop_threshold"],
                        "crop_note": sample["crop_note"],
                        "crops": [{"path": str(p.relative_to(root)), "sha256": sha(p)} for p in crops]})
    manifest = {"status": "crops_complete", "seed": config["seed"],
                "config_sha256": sha(args.config), "script_sha256": sha(__file__),
                "official_crop_sha256": sha(args.official_root / "concept/crop.py"),
                "official_git_commit": commit(args.official_root),
                "project_git_commit": commit(args.project_root), "samples": records}
    (root / "input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps([{"sample": r["sample_id"], "crop_count": len(r["crops"])}
                      for r in records], indent=2))


if __name__ == "__main__":
    main()
