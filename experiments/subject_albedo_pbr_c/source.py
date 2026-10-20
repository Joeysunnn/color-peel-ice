"""Resolve immutable mailbox Albedo/PBR maps and their original whole-object alpha."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.sd15_token_local_kv_v1.contract import sha256

ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path("experiments/natural_material_supermat_v1/assets")
CONFIG = Path("experiments/natural_material_supermat_v1/configs/subject_eroded_material_v1.json")
REPAIRED_SUBJECT_MASK_SHA256 = "b4d119d110ec356fca9daa9003c8505ba04f7ee694d0d8785e9c5281f63a8cf7"


def load_source(fullmaps):
    """Reuse existing SuperMat inference; do not crop, inpaint, recolor or rerun it."""
    # OpenCV is only a dependency of the existing resolver's unrelated texture CLI.
    # Keep module import lightweight; source resolution uses the original audited API.
    from experiments.natural_texture_material_v1.extract import inspect_sources

    fullmaps = Path(fullmaps).resolve()
    records, upstream_locks = inspect_sources(fullmaps)
    mailbox = records["mailbox"]
    selected = mailbox["paths"]
    config_path = ROOT / CONFIG
    config = json.loads(config_path.read_text(encoding="utf8"))
    paths = {"original": ROOT / ASSETS / "source_mailbox.jpg", "mask": ROOT / ASSETS / "object_mask.png",
             **{name: Path(selected[name]).resolve() for name in ("input", "albedo", "roughness", "metallic")}}
    for name, field in (("original", "source_image_sha256"), ("mask", "original_object_mask_sha256"),
                        ("input", "supermat_input_sha256")):
        if sha256(paths[name]) != config[field]:
            raise ValueError(f"frozen original mailbox {name} differs")
    for name, mode in (("original", "RGB"), ("mask", "L"), ("input", "RGBA"), ("albedo", "RGB")):
        with Image.open(paths[name]) as image:
            if image.size != (512, 512) or image.mode != mode:
                raise ValueError(f"mailbox {name} must be 512px {mode}")
    with Image.open(paths["original"]) as image:
        original = np.asarray(image).copy()
    with Image.open(paths["mask"]) as image:
        mask = np.asarray(image).copy()
    with Image.open(paths["input"]) as image:
        rgba = np.asarray(image).copy()
    if set(np.unique(mask)) != {0, 255}:
        raise ValueError("original mailbox object mask must be nonempty binary foreground/background")
    if not np.array_equal(rgba[:, :, :3], original) or not np.array_equal(rgba[:, :, 3], mask):
        raise ValueError("SuperMat RGBA RGB/alpha differ from the original photo/object mask")
    for name in ("roughness", "metallic"):
        with Image.open(paths[name]) as image:
            pixels = np.asarray(image)
            if (image.size != (512, 512) or image.mode not in ("L", "RGB")
                    or pixels.dtype != np.uint8
                    or image.mode == "RGB" and not (np.array_equal(pixels[:, :, 0], pixels[:, :, 1])
                                                    and np.array_equal(pixels[:, :, 0], pixels[:, :, 2]))):
                raise ValueError(f"SuperMat {name} must be a 512px 8bit scalar map")
    locks = {**upstream_locks, **{str(path.resolve()): sha256(path) for path in
                                 (paths["original"], paths["mask"], config_path)}}
    if any(not Path(path).is_absolute() or sha256(Path(path)) != digest for path, digest in locks.items()):
        raise ValueError("immutable SuperMat source locks changed during resolution")
    receipt = json.loads(Path(selected["supermat_manifest"]).read_text(encoding="utf8"))
    record = {"source_id": "mailbox", "extraction": "reuse_existing_supermat_no_rerun",
              "source_fullmaps_run": str(fullmaps),
              "source_chain": {key: str(Path(selected[key]).resolve()) for key in
                               ("canonical", "upstream_canonical", "supermat_manifest", "mask") if key in selected},
              "reviewed_rm_region": mailbox["canonical"]["selected_region"],
              "mask_policy": "original whole-object binary mask exactly equals existing SuperMat RGBA alpha; not the small R/M measurement region",
              "mask_sha256": sha256(paths["mask"]), "foreground_pixels": int((mask == 255).sum()),
              "experiment_b_mask_difference": {"repaired_subject_mask_sha256": REPAIRED_SUBJECT_MASK_SHA256,
                  "added_bottom_pixels": 1450, "used_here": False,
                  "reason": "B repaired mask extends beyond the original SuperMat alpha; no extraction is assumed for those extra pixels"},
              "supermat": {key: receipt.get(key) for key in ("seed", "image_size", "supermat_git_commit",
                  "checkpoint_sha256", "base_model_index_sha256", "command")},
              "map_policy": "original full-size Albedo RGB and separate scalar R/M; no crop, inpainting, normalization or semantic overrides",
              "files": {name: {"path": str(path.resolve()), "sha256": sha256(path)} for name, path in paths.items()}}
    return paths, locks, record
