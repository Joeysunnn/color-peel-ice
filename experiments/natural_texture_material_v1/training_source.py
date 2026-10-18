"""Read-only verification of the accepted 72-image Phase 2 training source."""

import json
from pathlib import Path

from PIL import Image, ImageChops

from experiments.natural_texture_material_v1 import dataset_protocol as schema
from experiments.natural_texture_material_v1 import dataset_stage


def verify_dataset(source_root):
    source = Path(source_root).resolve()
    read = dataset_stage.read
    state = read(source / "status.json")
    if (state.get("status") != "dataset_verified" or state.get("training_launched") is not False
            or state.get("phase") != 2 or state.get("stage") != "counterfactual_dataset_only"):
        raise ValueError("Expected the completed, unmodified dataset-only source")
    protocol = read(source / "plan/protocol.json")
    if protocol != schema.build_protocol(protocol["source_extraction"]):
        raise ValueError("Dataset differs from the accepted extraction and frozen protocol")
    audit = dataset_stage.verify_staging(source / "staging")
    rendered_protocol, rows, validated, manifest_path = dataset_stage.validate_renders(source / "renders")
    digest = schema.protocol_sha256(protocol)
    if (read(source / "staging/protocol.json") != protocol or rendered_protocol != protocol
            or state.get("extraction") != protocol["source_extraction"] or state.get("dataset") != audit
            or audit.get("phase") != 2 or audit.get("stage") != "dataset_only"
            or audit.get("training_started") is not False
            or tuple(audit.get(key) for key in ("train_images", "texture_images", "material_images")) != (72, 48, 24)
            or audit.get("protocol_sha256") != digest
            or audit.get("source_render_manifest") != str(manifest_path)
            or audit.get("source_render_manifest_sha256") != schema.sha256(manifest_path)):
        raise ValueError("Dataset status, counts, protocols or render provenance disagree")
    plan_path = source / "plan/plan.json"
    plan, manifest = read(plan_path), read(manifest_path)
    if (plan.get("status") != "complete" or plan.get("training_launched") is not False
            or plan.get("source_locks") != protocol["source_locks"]
            or plan.get("protocol_path") != str(source / "plan/protocol.json")
            or manifest.get("plan_path") != str(plan_path)
            or manifest.get("protocol_path") != str(source / "plan/protocol.json")):
        raise ValueError("Render manifest does not use this dataset's frozen plan")
    locks = dict(protocol["source_locks"])
    paths = [source / name for name in ("status.json", "plan/plan.json", "plan/protocol.json",
             "renders/manifest.json", "staging/audit.json", "staging/protocol.json",
             "staging/training_references.json", "staging/index.html", "index.html")]
    by_id = {row["id"]: row for row in rows}
    references = read(source / "staging/training_references.json")
    if set(references) != set(audit["datasets"]):
        raise ValueError("Training-reference concept membership changed")
    for row in rows:
        paths.extend(validated[row["id"]].values())
        paths.append(Path(row["metadata"]))
    for concept, receipt in audit["datasets"].items():
        folder = source / "staging" / concept
        ledger = folder / "training_assets_manifest.jsonl"
        assets = [json.loads(line) for line in ledger.read_text(encoding="utf8").splitlines()]
        if receipt.get("source_render_manifest_sha256") != schema.sha256(manifest_path):
            raise ValueError("Dataset receipt references another render manifest")
        reference = references[concept]
        expected_refs = [{key: row[key] for key in ("id", "image", "image_sha256", "prompt", "shape",
                         "color", "light", "roughness", "metallic", "seed")} for row in assets]
        if (reference.get("training_assets_manifest") != str(ledger)
                or reference.get("image_count") != len(assets) or reference.get("images") != expected_refs):
            raise ValueError("Contact-sheet references differ from actual training RGB")
        paths.extend(folder / name for name in ("dataset_receipt.json", "concepts.json", "training_assets_manifest.jsonl"))
        paths.append(source / "staging/contact_sheets" / (concept + ".jpg"))
        for row in assets:
            original = by_id[row["id"]]
            raw = validated[row["id"]]
            if (row.get("source_image") != str(raw["image"]) or row.get("source_mask") != str(raw["mask"])
                    or row.get("source_image_sha256") != original["image_sha256"]
                    or row.get("source_mask_sha256") != original["mask_sha256"]
                    or row["image_sha256"] != original["image_sha256"]):
                raise ValueError("Staged training RGB or mask does not come from its expected render")
            with Image.open(raw["mask"]) as raw_mask, Image.open(row["mask"]) as mask:
                binary = raw_mask.convert("L").point(lambda value: 255 if value >= 128 else 0)
                if ImageChops.difference(binary, mask.convert("L")).getbbox() is not None:
                    raise ValueError("Training mask differs from thresholded source render")
            paths.extend((Path(row["image"]), Path(row["mask"])))
    locks.update({str(path): schema.sha256(path) for path in paths})
    if any(schema.sha256(path) != value for path, value in locks.items()):
        raise ValueError("Dataset or extraction changed while verifying training sources")
    return protocol, audit, locks
