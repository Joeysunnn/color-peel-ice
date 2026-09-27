"""Read-only source checks for the two matched Subject training cohorts."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def one_png(directory: Path, expected_sha256: str) -> None:
    files = list(directory.iterdir())
    if (len(files) != 1 or files[0].suffix.lower() != ".png"
            or not files[0].is_file() or sha256(files[0]) != expected_sha256):
        raise ValueError(f"training image or mask differs: {directory}")


def verify_training_data(cohort: str, concepts_path: Path, manifest_path: Path,
                         concepts_sha256: str, manifest_sha256: str) -> dict:
    if (concepts_path.parent != manifest_path.parent
            or sha256(concepts_path) != concepts_sha256
            or sha256(manifest_path) != manifest_sha256):
        raise ValueError("frozen B0 concepts or asset manifest differs")
    concepts = json.loads(concepts_path.read_text(encoding="utf-8"))
    if cohort == "original":
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("schema") != "natural_subject_counterfactual_v2_training_staging_manifest/v1"
                or manifest.get("record_count") != 25 or len(concepts) != 25):
            raise ValueError("original B0 must have 25 exposure-matched rows")
        records = manifest["records"]
        colors = Counter()
        for concept, record in zip(concepts, records):
            if (concept.get("instance_prompt") != [record["prompt"]]
                    or Path(concept["instance_data_dir"]).parent.name != record["record_id"]):
                raise ValueError("original B0 row or prompt differs")
            image_dir = Path(concept["instance_data_dir"])
            mask_dir = Path(concept["instance_mask_dir"])
            one_png(image_dir, record["image_sha256"])
            one_png(mask_dir, record["mask_sha256"])
            colors[record["color_name"]] += 1
        if colors != dict.fromkeys(("red", "green", "cyan", "blue", "magenta"), 5):
            raise ValueError("original B0 color exposure differs")
    elif cohort == "balanced_aligned":
        records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
        if len(records) != 10 or len(concepts) != 10:
            raise ValueError("balanced aligned B0 must have ten material rows")
        captions = {"red": "pink", "green": "green", "cyan": "cyan", "blue": "blue", "magenta": "purple"}
        cells = Counter()
        for concept, record in zip(concepts, records):
            color, material = record["color"], record["material"]
            suffix = "glossy painted metal" if material == "metal" else "matte plastic"
            prompt = f"a photo of <S*> mailbox in {captions[color]} color made of {suffix}"
            if (concept.get("instance_prompt") != [prompt]
                    or Path(concept["instance_data_dir"]) != Path(record["image"]).parent
                    or Path(concept["instance_mask_dir"]) != Path(record["mask"]).parent):
                raise ValueError("balanced aligned B0 row or prompt differs")
            one_png(Path(concept["instance_data_dir"]), record["image_sha256"])
            one_png(Path(concept["instance_mask_dir"]), record["mask_sha256"])
            cells[(color, material)] += 1
        if cells != {(color, material): 1 for color in captions for material in ("metal", "matte")}:
            raise ValueError("balanced aligned B0 color/material cells differ")
    else:
        raise ValueError(f"unknown Subject cohort: {cohort}")
    return {"cohort": cohort, "concepts_sha256": concepts_sha256,
            "asset_manifest_sha256": manifest_sha256, "row_count": len(concepts)}
