"""Copy the reviewed mailbox pairs into an independent surface-caption ablation."""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from src.methods.colorpeel_ice import prepare_mailbox_matte_counterfactual as legacy


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = Path(__file__).with_name("protocol.json")
COLORS = ("pink", "green", "cyan", "blue", "purple")
MATERIALS = ("metallic", "matte")
SOURCE_COLORS = ("red", "green", "cyan", "blue", "magenta")
sha256 = legacy.sha256
read_json = legacy.read_json
write_json = legacy.write_json


def protocol_sha256():
    encoded = json.dumps(read_json(PROTOCOL), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _arrays(image, mask):
    with Image.open(image) as handle:
        if handle.mode != "RGB" or handle.size != (512, 512):
            raise ValueError("mailbox RGB image format differs")
        rgb = np.asarray(handle).copy()
    with Image.open(mask) as handle:
        binary = np.asarray(handle)
        if handle.mode != "L" or handle.size != (512, 512) or set(np.unique(binary)) != {0, 255}:
            raise ValueError("mailbox binary mask differs")
        return rgb, binary.copy()


def _sources(original_root, reviewed_root):
    specification = read_json(PROTOCOL)
    original_root, reviewed_root = Path(original_root).resolve(), Path(reviewed_root).resolve()
    protocol = ROOT / specification["legacy_protocol"]
    caption_protocol = ROOT / specification["caption_protocol"]
    if (sha256(protocol) != specification["legacy_protocol_sha256"]
            or sha256(caption_protocol) != specification["caption_protocol_sha256"]):
        raise ValueError("frozen legacy protocols differ")
    value = legacy.protocol(protocol)
    legacy.validated_preview(protocol, reviewed_root.parent.parent, reviewed_root / "preview")
    preview_manifest = reviewed_root / "preview/preview_manifest.json"
    review_path = reviewed_root / "preview_review.json"
    review = read_json(review_path)
    if (sha256(preview_manifest) != specification["preview_manifest_sha256"]
            or sha256(review_path) != specification["review_sha256"]
            or review.get("verdict") != "pass" or not review.get("reviewer") or not review.get("reviewed_at")
            or review.get("preview_manifest_sha256") != sha256(preview_manifest)):
        raise ValueError("matching accepted matte preview review is required")
    aligned = reviewed_root / "caption_aligned_staged"
    provenance_path = aligned / "staging_provenance.json"
    concepts_path = aligned / "balanced_concepts.json"
    ledger_path = aligned / "training_assets_manifest.jsonl"
    provenance = read_json(provenance_path)
    if (sha256(provenance_path) != specification["reviewed_provenance_sha256"]
            or sha256(concepts_path) != specification["reviewed_concepts_sha256"]
            or sha256(ledger_path) != specification["reviewed_manifest_sha256"]
            or provenance["balanced_concepts_sha256"] != sha256(concepts_path)
            or provenance["training_assets_manifest_sha256"] != sha256(ledger_path)
            or provenance["caption_protocol_sha256"] != sha256(caption_protocol)):
        raise ValueError("reviewed aligned staging differs")
    old = read_json(ROOT / value["source_training_protocol"])
    category = original_root / "category"
    raw_concepts, raw_manifest = category / "concepts.json", category / "staging_manifest.json"
    raw_rows = read_json(raw_manifest).get("records", [])
    if (sha256(raw_concepts) != specification["original_concepts_sha256"]
            or sha256(raw_manifest) != specification["original_manifest_sha256"]
            or [row.get("name") for row in raw_rows] != list(SOURCE_COLORS)
            or len(read_json(raw_concepts)) != 5):
        raise ValueError("original metallic source coverage differs")
    ledger, old_concepts = _jsonl(ledger_path), read_json(concepts_path)
    if [(row.get("color"), row.get("material")) for row in ledger] != [
            (color, material) for color in SOURCE_COLORS for material in ("metal", "matte")]:
        raise ValueError("reviewed five-by-two material pairing differs")
    preview_rows = read_json(preview_manifest)["records"]
    locked = [protocol, caption_protocol, ROOT / value["source_training_protocol"],
              ROOT / value["transform"]["reference_path"], preview_manifest, review_path,
              provenance_path, concepts_path, ledger_path, raw_concepts, raw_manifest]
    records, background, common_mask = [], None, None
    for index, source_color in enumerate(SOURCE_COLORS):
        raw_image = category / source_color / "images/image.png"
        raw_mask = category / source_color / "masks/image.png"
        expected_image = old["training_data"]["expected_image_sha256"][source_color]
        expected_mask = old["source_pilot"]["repaired_mask_sha256"]
        if (sha256(raw_image) != expected_image or sha256(raw_mask) != expected_mask
                or raw_rows[index].get("image_sha256") != expected_image
                or raw_rows[index].get("mask_sha256") != expected_mask):
            raise ValueError("original metallic image or mask differs")
        expected_raw = {"instance_prompt": [f"a photo of <S*> mailbox in {source_color} color"],
                        "instance_data_dir": str(raw_image.parent), "instance_mask_dir": str(raw_mask.parent)}
        if read_json(raw_concepts)[index] != expected_raw:
            raise ValueError("original metallic concepts differ")
        locked.extend([raw_image, raw_mask])
        for material_index, material in enumerate(MATERIALS):
            source = ledger[2 * index + material_index]
            image, mask = Path(source["image"]), Path(source["mask"])
            if (not image.resolve().is_relative_to(aligned) or not mask.resolve().is_relative_to(aligned)
                    or sha256(image) != source["image_sha256"] or sha256(mask) != source["mask_sha256"]
                    or sha256(mask) != expected_mask):
                raise ValueError("reviewed RGB or mask differs")
            if material == "metallic" and sha256(image) != expected_image:
                raise ValueError("reviewed metallic image differs from original")
            if material == "matte":
                candidate = reviewed_root / "preview" / preview_rows[index]["matte_image"]
                if (preview_rows[index]["color"] != source_color
                        or sha256(candidate) != source["image_sha256"]):
                    raise ValueError("reviewed matte image differs from accepted preview")
                locked.append(candidate)
            color = COLORS[index]
            old_caption = f"a photo of <S*> mailbox in {color} color made of " + (
                "glossy painted metal" if material == "metallic" else "matte plastic")
            if old_concepts[2 * index + material_index] != {
                    "instance_prompt": [old_caption], "instance_data_dir": str(image.parent),
                    "instance_mask_dir": str(mask.parent)}:
                raise ValueError("reviewed material caption differs")
            rgb, binary = _arrays(image, mask)
            if common_mask is None:
                common_mask, background = binary, rgb[binary == 0]
            if not np.array_equal(binary, common_mask) or not np.array_equal(rgb[binary == 0], background):
                raise ValueError("paired mailbox mask or outside-mask background differs")
            records.append({"color": color, "source_color": source_color, "material": material,
                            "source_image": str(image), "source_mask": str(mask)})
            locked.extend([image, mask])
    return records, {"original_root": str(original_root), "reviewed_root": str(reviewed_root),
                     "protocol_sha256": protocol_sha256(),
                     "files": {str(path): sha256(path) for path in locked}}


def _record(source, output):
    color, material = source["color"], source["material"]
    root = output / "assets" / color / material
    return {"id": f"{color}_{material}", "color": color, "source_color": source["source_color"],
            "material": material,
            "caption": read_json(PROTOCOL)["caption_template"].format(color=color, material=material),
            "image": str(root / "images/image.png"), "mask": str(root / "masks/image.png"),
            "image_sha256": sha256(Path(source["source_image"])),
            "mask_sha256": sha256(Path(source["source_mask"])),
            "source_image": source["source_image"],
            "source_image_sha256": sha256(Path(source["source_image"]))}


def _concept(row):
    return {"instance_prompt": [row["caption"]],
            "instance_data_dir": str(Path(row["image"]).parent),
            "instance_mask_dir": str(Path(row["mask"]).parent)}


def _audit(output):
    return {"schema": "subject_experiment_b_dataset_audit/v1", "status": "verified", "rows": 10,
            "colors": list(COLORS), "materials": list(MATERIALS), "source_pixels_unchanged": True,
            "shared_binary_mask": True, "outside_mask_changed_pixels": 0,
            "protocol_sha256": protocol_sha256(),
            "files": {name: sha256(output / name) for name in (
                "concepts.json", "training_assets_manifest.jsonl", "source_locks.json")},
            "comparison_scope": read_json(PROTOCOL)["comparison_scope"],
            "limitations": read_json(PROTOCOL)["limitations"]}


def stage(original_root: Path, reviewed_root: Path, output: Path):
    output = Path(output).resolve()
    protected = [ROOT, Path(original_root).resolve(), Path(reviewed_root).resolve()]
    if output.exists():
        raise FileExistsError(output)
    if any(output.is_relative_to(root) or root.is_relative_to(output) for root in protected):
        raise ValueError("new dataset must be outside the repository and old source directories")
    sources, locks = _sources(original_root, reviewed_root)
    output.mkdir(parents=True)
    rows = [_record(source, output) for source in sources]
    for source, row in zip(sources, rows):
        for field in ("image", "mask"):
            target = Path(row[field])
            target.parent.mkdir(parents=True)
            shutil.copy2(source["source_" + field], target)
            if sha256(target) != row[field + "_sha256"]:
                raise ValueError("dataset copy changed source bytes")
    write_json(output / "concepts.json", [_concept(row) for row in rows])
    (output / "training_assets_manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    write_json(output / "source_locks.json", locks)
    write_json(output / "dataset_audit.json", _audit(output))
    verify_dataset(output)
    return rows


def verify_dataset(output: Path):
    output = Path(output).resolve()
    locks = read_json(output / "source_locks.json")
    sources, current_locks = _sources(Path(locks["original_root"]), Path(locks["reviewed_root"]))
    if locks != current_locks:
        raise ValueError("immutable source locks differ")
    rows = _jsonl(output / "training_assets_manifest.jsonl")
    expected = [_record(source, output) for source in sources]
    if rows != expected or read_json(output / "concepts.json") != [_concept(row) for row in expected]:
        raise ValueError("dataset row, source pairing or material-conditioned caption differs")
    background, common_mask = None, None
    for row in rows:
        for field in ("image", "mask"):
            path = Path(row[field])
            if (path.is_symlink() or set(path.parent.iterdir()) != {path}
                    or sha256(path) != row[field + "_sha256"]):
                raise ValueError("dataset directory member or image hash differs")
        rgb, mask = _arrays(row["image"], row["mask"])
        if common_mask is None:
            common_mask, background = mask, rgb[mask == 0]
        if not np.array_equal(mask, common_mask) or not np.array_equal(rgb[mask == 0], background):
            raise ValueError("dataset paired masks or background pixels differ")
    audit = _audit(output)
    if read_json(output / "dataset_audit.json") != audit:
        raise ValueError("dataset audit receipt differs")
    return audit
