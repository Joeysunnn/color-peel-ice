"""Read-only verification of the accepted thirty-image Experiment C dataset."""

import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.sd15_token_local_kv_v1.contract import read_json, sha256
from experiments.subject_albedo_pbr_c import pbr_data

ROOT = Path(__file__).resolve().parents[2]


def _jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]


def verify_dataset(dataset):
    """Check acceptance, source receipts and every actual training pair; write nothing."""
    dataset = Path(dataset).resolve()
    audit = read_json(dataset / "dataset_audit.json")
    names = ("concepts.json", "training_assets_manifest.jsonl", "source_locks.json", "visual_review.json")
    if (audit.get("schema") != "subject_c_visually_accepted_pbr_dataset/v1"
            or audit.get("status") != "verified" or audit.get("rows") != 30
            or audit.get("training_eligible_images") != 30 or audit.get("training_started") is not False
            or audit.get("files") != {name: sha256(dataset / name) for name in names}):
        raise ValueError("accepted Experiment C dataset audit or metadata hashes differ")
    locks = read_json(dataset / "source_locks.json")
    pbr_data._check_locks(locks)
    manifests = [Path(path) for path, digest in locks.items()
                 if Path(path).name == "dataset_manifest.jsonl" and digest == audit["candidate_manifest_sha256"]]
    if len(manifests) != 1:
        raise ValueError("accepted candidate manifest must be uniquely bound to source locks")
    candidate_manifest = manifests[0]
    originals = _jsonl(candidate_manifest)
    if len(originals) != 30 or len({row["id"] for row in originals}) != 30:
        raise ValueError("accepted candidate manifest must contain thirty unique images")
    originals = {row["id"]: row for row in originals}
    review = read_json(dataset / "visual_review.json")
    decisions = review.get("decisions", [])
    if (sha256(dataset / "visual_review.json") != audit.get("visual_review_sha256")
            or review.get("candidate_manifest_sha256") != audit["candidate_manifest_sha256"]
            or not review.get("reviewer") or review.get("limitations") != audit.get("limitations")
            or len(decisions) != 30 or any(item.get("accepted") is not True for item in decisions)
            or len({item.get("id") for item in decisions}) != 30
            or {item.get("id") for item in decisions} != set(originals)):
        raise ValueError("explicit per-image accepted visual review differs")
    rows = _jsonl(dataset / "training_assets_manifest.jsonl")
    expected_grid = set(itertools.product(pbr_data.COLORS, pbr_data.METALLIC, pbr_data.ROUGHNESS))
    if (len(rows) != 30 or len({row["id"] for row in rows}) != 30
            or {(row["color"], row["metallic_state"], row["roughness_state"]) for row in rows} != expected_grid):
        raise ValueError("training dataset must retain the complete five-by-two-by-three factorial")
    original_path = candidate_manifest.parent / "source/original.jpg"
    if locks.get(str(original_path)) != sha256(original_path):
        raise ValueError("original background photo must be locked")
    with Image.open(original_path) as image:
        if image.mode != "RGB" or image.size != (512, 512):
            raise ValueError("original background must remain 512px RGB")
        background = np.asarray(image).copy()
    common_mask, albedos, concepts = None, {}, []
    for row in rows:
        identity = f"{row['color']}__{row['metallic_state']}__{row['roughness_state']}"
        source = originals.get(row["id"])
        if source is None:
            raise ValueError("training image is absent from the accepted candidates")
        image_path = dataset / "assets" / identity / "images/image.png"
        mask_path = dataset / "assets" / identity / "masks/image.png"
        expected = {**source, "source_image": source["image"], "source_image_sha256": source["image_sha256"],
                    "source_mask": source["mask"], "source_mask_sha256": locks.get(source["mask"]),
                    "image": str(image_path), "image_sha256": source["image_sha256"],
                    "mask": str(mask_path), "mask_sha256": locks.get(source["mask"]),
                    "training_eligible": True, "render_mask": audit.get("render_mask")}
        if (row != expected or row["id"] != identity
                or row["metallic"] != pbr_data.METALLIC[row["metallic_state"]]
                or row["roughness"] != pbr_data.ROUGHNESS[row["roughness_state"]]
                or row["hue_degrees"] != pbr_data.COLORS[row["color"]]
                or row["render_status"] != "true_cycles_pbr"
                or row["scene_sha256"] != audit["scene_sha256"]
                or row["geometry_content_sha256"] != audit["geometry_content_sha256"]
                or row["caption"] != row["proposed_caption"] or row["caption"].count("<S*>") != 1
                or any(token in row["caption"] for token in ("<M*>", "<T*>", "<C*>"))):
            raise ValueError("training pair, accepted source, caption or fixed PBR factors differ")
        for field, path in (("image", image_path), ("mask", mask_path)):
            if (path.is_symlink() or set(path.parent.iterdir()) != {path}
                    or sha256(path) != row[field + "_sha256"]
                    or locks.get(row["source_" + field]) != row["source_" + field + "_sha256"]):
                raise ValueError("actual training image/mask member or frozen byte hash differs")
        with Image.open(mask_path) as image:
            mask = np.asarray(image).copy()
            if image.mode != "L" or image.size != (512, 512) or set(np.unique(mask)) != {0, 255}:
                raise ValueError("training mask must remain nonempty binary 512px")
        if common_mask is None:
            common_mask = mask
        if not np.array_equal(mask, common_mask):
            raise ValueError("training images must use one shared Subject mask")
        with Image.open(image_path) as image:
            if (image.mode != "RGB" or image.size != (512, 512)
                    or not np.array_equal(np.asarray(image)[mask == 0], background[mask == 0])):
                raise ValueError("actual training RGB or fixed outside-mask background differs")
        albedo = row["maps"]["albedo"]
        if (locks.get(albedo["path"]) != albedo["sha256"]
                or (row["color"] in albedos and albedos[row["color"]] != albedo["sha256"])):
            raise ValueError("one frozen Albedo must be retained across every M/R state of each color")
        albedos[row["color"]] = albedo["sha256"]
        concepts.append({"instance_prompt": [row["caption"]], "instance_data_dir": str(image_path.parent),
                         "instance_mask_dir": str(mask_path.parent)})
    if len(set(albedos.values())) != 5 or read_json(dataset / "concepts.json") != concepts:
        raise ValueError("five distinct Albedos or exact training concepts differ")
    pbr_data._check_locks(locks)
    return audit
