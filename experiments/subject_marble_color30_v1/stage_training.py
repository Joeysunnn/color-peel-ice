"""Freeze approved MARBLE candidates or their pink/cyan subset for training."""

import argparse
import json
from pathlib import Path
import shutil

from PIL import Image

from experiments.sd15_token_local_kv_v1.contract import read_json, sha256
from experiments.subject_marble_color30_v1 import data
from experiments.subject_multicolor_multimaterial_b.training import ROOT, _write


def _locks(hashes):
    if not hashes or any(sha256(Path(path)) != digest for path, digest in hashes.items()):
        raise ValueError("approved candidate or training source changed")


def stage(candidates, output, pink_cyan_only=False):
    candidates, output = Path(candidates).resolve(), Path(output).resolve()
    if (output.exists() or output.is_relative_to(ROOT) or output.is_relative_to(candidates.parent)
            or candidates.is_relative_to(output)):
        raise ValueError("stage into a new external directory, preserving the candidate run")
    plan = data.verify_prepared(candidates.parent)
    generation = read_json(candidates.parent / "generation_manifest.json")
    manifest = read_json(candidates / "manifest.json")
    foreground = plan["config"].get("schema") == "marble_foreground_only/v1"
    if pink_cyan_only and not foreground:
        raise ValueError("pink/cyan selection requires the foreground-only candidate dataset")
    expected_status = ("generated35_candidates_and5_diagnostics_pending_review" if foreground
                       else "generated35_pending_user_review")
    rows = manifest["records"]
    expected = {row["id"]: row for row in plan["rows"]}
    originals = {f"{color}__original" for color in plan["sources"]}
    if (generation.get("status") != expected_status
            or len(rows) != 35 or len({row["id"] for row in rows}) != 35
            or {row["id"] for row in rows} != set(expected) | originals
            or len(generation["records"]) != 30
            or {row["id"] for row in generation["records"]} != set(expected)):
        raise ValueError("require the complete generated thirty-plus-five candidate dataset")
    if foreground:
        diagnostics = generation.get("identity_records", [])
        if (len(diagnostics) != 5 or {row["color"] for row in diagnostics} != set(plan["sources"])
                or any(row["controls"] or row["id"] in set(expected) | originals for row in diagnostics)
                or plan["config"].get("neutral_rgb") != [127, 127, 127]):
            raise ValueError("foreground data must retain five separate no-attribute diagnostics and neutral padding")
    locks = {str(path): sha256(path) for path in (candidates / "manifest.json",
        candidates / "concepts.json", candidates.parent / "generation_manifest.json",
        candidates.parent / "prepare_receipt.json", candidates.parent / "plan.json")}
    if foreground:
        original_mask = candidates.parent / "source/object_mask512.png"
        locks[str(original_mask)] = sha256(original_mask)
    for row in generation["records"]:
        if any(row.get(key) != value for key, value in expected[row["id"]].items()):
            raise ValueError("generated controls or caption differ from the frozen plan")
        locks[str(candidates.parent / row["raw_image"])] = row["raw_sha256"]
    for row in rows:
        if row["id"] in expected:
            if any(row.get(key) != value for key, value in expected[row["id"]].items()):
                raise ValueError("candidate caption or requested controls changed")
        else:
            color = row["color"]
            original = candidates.parent / plan["sources"][color]["original"]
            expected_kind = "original_recolor_cutout" if foreground else "original_recolor"
            if (row["kind"] != expected_kind or row["image_sha256"] != sha256(original)
                    or row["caption"] != f"a photo of <S*> mailbox in {color} color made of glossy painted metal"):
                raise ValueError("the five additional original recolors must remain byte-identical")
        for key in ("image", "mask"):
            source = (candidates / row[key]).resolve()
            if not source.is_relative_to(candidates):
                raise ValueError("candidate path escapes its dataset")
            locks[str(source)] = row[key + "_sha256"]
    _locks(locks)
    selected_colors = ["pink", "cyan"] if pink_cyan_only else [color["name"] for color in plan["config"]["colors"]]
    rows = [row for row in rows if row["color"] in selected_colors]
    count = 14 if pink_cyan_only else 35
    if len(rows) != count:
        raise ValueError("selected colors must each retain one original plus all six M/R variants")
    output.mkdir(parents=True)
    training_rows, concepts = [], []
    for row in rows:
        paths = {key: output / "assets" / row["id"] / ("images" if key == "image" else "masks") / "image.png"
                 for key in ("image", "mask")}
        record = {**row, "training_eligible": True, "review_status": "user_authorized_training",
                  "caption_status": "user-approved requested-control supervision; physical accuracy unverified"}
        for key, path in paths.items():
            source = (candidates / row[key]).resolve()
            path.parent.mkdir(parents=True)
            shutil.copyfile(source, path)
            record.update({key: str(path), "source_" + key: str(source),
                           "source_" + key + "_sha256": row[key + "_sha256"]})
        training_rows.append(record)
        concepts.append({"instance_prompt": [row["caption"]], "instance_data_dir": str(paths["image"].parent),
                         "instance_mask_dir": str(paths["mask"].parent)})
    _write(output / "concepts.json", concepts)
    (output / "training_assets_manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in training_rows), encoding="utf8")
    _write(output / "source_locks.json", locks)
    _write(output / "approval.json", {"request": "只使用pink和cyan两种颜色对应的去背景原图以及反事实数据进行训练推理" if pink_cyan_only else "生成完成了，开始训练",
        "scope": "pink/cyan: twelve generated plus two neutral originals" if pink_cyan_only else "all thirty generated plus five original recolors",
        "selected_colors": selected_colors,
        "not_claimed": "No per-image physical-material or disentanglement validation is implied."})
    names = ("concepts.json", "training_assets_manifest.jsonl", "source_locks.json", "approval.json")
    _write(output / "dataset_audit.json", {"schema": "marble_subject14_training/v1" if pink_cyan_only else "marble_subject35_training/v1", "status": "verified",
        "rows": count, "training_eligible_images": count, "candidates": str(candidates), "selected_colors": selected_colors,
        "source_mode": "foreground_only" if foreground else "original_scene",
        "files": {name: sha256(output / name) for name in names},
        "sampling": f"unchanged sequential DataLoader; seven images per color; {1000 // count} complete cycles plus first{1000 % count} at1000 updates",
        "limitations": "Requested relative M/R and color captions are supervision labels, not measured physical properties."})
    return verify_dataset(output)


def verify_dataset(dataset):
    dataset = Path(dataset).resolve()
    audit = read_json(dataset / "dataset_audit.json")
    names = ("concepts.json", "training_assets_manifest.jsonl", "source_locks.json", "approval.json")
    subset = audit.get("schema") == "marble_subject14_training/v1"
    count = 14 if subset else 35
    if (audit.get("schema") not in ("marble_subject35_training/v1", "marble_subject14_training/v1") or audit.get("status") != "verified"
            or audit.get("rows") != count or audit.get("training_eligible_images") != count
            or audit.get("files") != {name: sha256(dataset / name) for name in names}):
        raise ValueError("approved training dataset receipt changed")
    locks = read_json(dataset / "source_locks.json")
    _locks(locks)
    candidate_root = Path(audit["candidates"])
    source_plan = read_json(candidate_root.parent / "plan.json")
    foreground = source_plan["config"].get("schema") == "marble_foreground_only/v1"
    if audit.get("source_mode", "original_scene") != ("foreground_only" if foreground else "original_scene"):
        raise ValueError("training source mode differs from the frozen generation plan")
    selected_colors = ["pink", "cyan"] if subset else [color["name"] for color in source_plan["config"]["colors"]]
    approval = read_json(dataset / "approval.json")
    if ((subset and not foreground) or audit.get("selected_colors", selected_colors) != selected_colors
            or (subset and approval.get("selected_colors") != selected_colors)):
        raise ValueError("training color selection differs from the approved pink/cyan subset")
    source_rows = {row["id"]: row for row in read_json(Path(audit["candidates"]) / "manifest.json")["records"]}
    source_rows = {identity: row for identity, row in source_rows.items() if row["color"] in selected_colors}
    expected_ids = {row["id"] for row in source_plan["rows"] if row["color"] in selected_colors} | {
        color + "__original" for color in selected_colors}
    rows = [json.loads(line) for line in (dataset / "training_assets_manifest.jsonl").read_text(encoding="utf8").splitlines()]
    if (len(rows) != count or len({row["id"] for row in rows}) != count
            or {row["id"] for row in rows} != expected_ids or set(source_rows) != expected_ids):
        raise ValueError("training must retain every approved candidate once")
    concepts = []
    for row in rows:
        source = source_rows[row["id"]]
        if (not row["training_eligible"] or row["caption"] != source["caption"]
                or row["caption"].count("<S*>") != 1):
            raise ValueError("approved per-image Subject caption changed")
        for key in ("image", "mask"):
            path = Path(row[key])
            if (not path.is_relative_to(dataset) or path.is_symlink() or set(path.parent.iterdir()) != {path}
                    or sha256(path) != source[key + "_sha256"] or row[key + "_sha256"] != source[key + "_sha256"]
                    or locks.get(row["source_" + key]) != row[key + "_sha256"]):
                raise ValueError("training image/mask pairing or byte identity changed")
            with Image.open(path) as image:
                if image.size != (512, 512) or image.mode != ("RGB" if key == "image" else "L"):
                    raise ValueError("training image/mask must be aligned 512px RGB/L")
                if key == "mask" and {value for _, value in image.getcolors(256)} != {0, 255}:
                    raise ValueError("require a nonempty binary Subject mask")
        if foreground:
            if row["mask_sha256"] != locks[str(candidate_root.parent / "source/object_mask512.png")]:
                raise ValueError("foreground-only training must retain its frozen Subject mask")
            with Image.open(row["image"]) as rgb, Image.open(row["mask"]) as mask:
                isolated = Image.composite(rgb, Image.new("RGB", rgb.size, (127, 127, 127)), mask)
                if isolated.tobytes() != rgb.tobytes():
                    raise ValueError("foreground-only training cannot contain original-scene pixels outside its mask")
        concepts.append({"instance_prompt": [row["caption"]], "instance_data_dir": str(Path(row["image"]).parent),
                         "instance_mask_dir": str(Path(row["mask"]).parent)})
    if read_json(dataset / "concepts.json") != concepts:
        raise ValueError("one-image-one-caption concepts differ")
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pink-cyan-only", action="store_true")
    args = parser.parse_args()
    print(stage(args.candidates, args.output, args.pink_cyan_only))
