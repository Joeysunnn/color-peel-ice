"""Reference-only crop diagnostic; preserve the frozen ZeST ablation and raw RGB."""

import argparse
import gc
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from experiments.sd15_token_local_kv_v1.contract import sha256
from experiments.zest_material_counterfactual_v1 import assets, official
from experiments.zest_material_counterfactual_v1.render_bank import write
from experiments.zest_material_counterfactual_v1.run import ROOT, load_plan, read
from scripts.launch.colorpeel_run import git_info


def focus_reference(reference, mask, padding=.1):
    if reference.size != (1024, 1024):
        raise ValueError("require the same1024 reference supplied to the frozen generator")
    aligned = mask.convert("L").resize(reference.size, Image.Resampling.NEAREST)
    aligned = aligned.point(lambda value: 255 if value >= 128 else 0)
    bbox = aligned.getbbox()
    if bbox is None:
        raise ValueError("reference object mask is empty")
    side = min(1024, math.ceil(max(bbox[2]-bbox[0], bbox[3]-bbox[1]) * (1+2*padding)))
    x = min(1024-side, max(0, math.floor((bbox[0]+bbox[2]-side)/2)))
    y = min(1024-side, max(0, math.floor((bbox[1]+bbox[3]-side)/2)))
    box = (x, y, x+side, y+side)
    focused = reference.crop(box).resize((1024, 1024), Image.Resampling.BICUBIC)
    focused_mask = aligned.crop(box).resize((1024, 1024), Image.Resampling.NEAREST)
    return focused, {"mask_bbox1024": list(bbox), "crop_box1024": list(box),
        "padding_per_side": padding, "mask_resize": "nearest; threshold128",
        "foreground_fraction_full": float((np.asarray(aligned) > 0).mean()),
        "foreground_fraction_focused": float((np.asarray(focused_mask) > 0).mean())}


def copy_prepared(source, destination):
    receipt = read(source / "prepare_receipt.json")
    for name, digest in receipt["output_hashes"].items():
        original, copied = (source / name).resolve(), (destination / name).resolve()
        if not original.is_relative_to(source) or not copied.is_relative_to(destination) or sha256(original) != digest:
            raise ValueError("prepared artifact changed or escaped its run")
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, copied)
    shutil.copyfile(source / "prepare_receipt.json", destination / "prepare_receipt.json")


def picture(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def sheet(folder, target, full_ref, focused_ref, full_output, focus_output, title):
    canvas = Image.new("RGB", (1300, 340), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    draw.text((8, 8), title + " | reference crop only | raw RGB | QC unverified; NO training", fill="black", font=font)
    for index, (label, image) in enumerate(zip(("Original", "Full reference", "Focused reference", "Full output", "Focused output"),
                                              (target, full_ref, focused_ref, full_output, focus_output))):
        draw.text((index*260+6, 45), label, fill="black", font=font)
        image = image.copy()
        image.thumbnail((250, 250), Image.Resampling.LANCZOS)
        canvas.paste(image, (index*260+5, 75))
    canvas.save(folder / "comparison.png")


def run(existing, asset_root, gpu, variants=("baseline", "combined")):
    existing, asset_root = Path(existing).resolve(), Path(asset_root).resolve()
    if existing.is_relative_to(ROOT) or gpu < 0:
        raise ValueError("select an external prepared run and a GPU")
    plan, generated = load_plan(existing), read(existing / "generation_manifest.json")
    bank = {row["id"]: row for row in read(existing / "bank/manifest.json")["records"]}
    selected = []
    for subject in plan["subjects"]:
        for variant in variants:
            for reference_id in ("self", "estimated_original"):
                matches = [row for row in generated["records"] if row["subject_id"] == subject["id"]
                    and row["stage"] == "identity" and row["variant"] == variant and row["reference_id"] == reference_id]
                if len(matches) != 1:
                    raise ValueError("require exactly one completed full-reference control for every requested case")
                selected.append((subject, matches[0]))
    locks = {str(existing / name): digest for name, digest in read(existing / "prepare_receipt.json")["output_hashes"].items()}
    for path in (Path(__file__), Path(official.__file__), existing/"prepare_receipt.json",
                 existing/"bank/manifest.json", asset_root/"assets.json"):
        locks[str(path)] = sha256(path)
    for subject, row in selected:
        reference = Path(row["reference_image"])
        reference = reference if reference.is_absolute() else existing / reference
        mask = existing / subject["source_files"]["object_mask"] if row["reference_id"] == "self" else Path(bank[row["reference_id"]]["mask"])
        for path, digest in ((reference, row["reference_sha256"]), (existing/row["raw_image"], row["raw_sha256"]),
                             (mask, sha256(mask) if row["reference_id"] == "self" else bank[row["reference_id"]]["mask_sha256"])):
            if sha256(path) != digest:
                raise ValueError("saved full-reference input/output changed")
            locks[str(path)] = digest
    parent = existing / "diagnostics"
    parent.mkdir(exist_ok=True)
    number = 1
    while (parent / f"reference_focus{number:03d}").exists():
        number += 1
    output = parent / f"reference_focus{number:03d}"
    output.mkdir()
    copy_prepared(existing, output)
    write(output / "source_generation_snapshot.json", generated)
    locks[str(output / "source_generation_snapshot.json")] = sha256(output / "source_generation_snapshot.json")
    state = {"status": "generating", "git": git_info(), "gpu": gpu, "source_run": str(existing), "source_locks": locks,
        "records": [], "training_started": False, "training_eligible": False,
        "purpose": "reference framing diagnostic; separate from the frozen44-image ablation; no identity/material approval"}
    write(output / "generation_manifest.json", state)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    receipt = assets.read(asset_root)
    try:
        for variant in variants:
            adapter = official.setup(receipt, with_edges=variant in ("texture", "combined"))
            write(output / f"runtime_{variant}.json", official.compatibility(adapter))
            for subject, row in selected:
                if row["variant"] != variant:
                    continue
                source = subject["source_files"]
                target = picture(output / source["target"])
                reference_path = Path(row["reference_image"])
                reference_path = reference_path if reference_path.is_absolute() else existing/reference_path
                reference = picture(reference_path).resize((1024, 1024), Image.Resampling.BICUBIC)
                mask_path = output/source["object_mask"] if row["reference_id"] == "self" else Path(bank[row["reference_id"]]["mask"])
                focused, crop = focus_reference(reference, picture(mask_path))
                folder = output / "cases" / row["id"] / "attempt001"
                folder.mkdir(parents=True)
                reference.save(folder / "full_reference.png")
                focused.save(folder / "focused_reference.png")
                start = time.monotonic()
                result = official.generate(adapter, target, picture(output/source["edit_mask"]), picture(output/source["control_depth"]), focused,
                    init_image=picture(output/source["initialization_color"]) if variant in ("color", "combined") else None,
                    edge_image=picture(output/source["albedo_edges"]) if variant in ("texture", "combined") else None, **plan["inference"])
                result.save(folder / "raw.png")
                record = {**row, "reference_image": (folder/"focused_reference.png").relative_to(output).as_posix(),
                    "reference_sha256": sha256(folder/"focused_reference.png"), "reference_crop": crop,
                    "full_reference_image": (folder/"full_reference.png").relative_to(output).as_posix(),
                    "full_reference_sha256": sha256(folder/"full_reference.png"), "full_control": row,
                    "raw_image": (folder/"raw.png").relative_to(output).as_posix(), "raw_sha256": sha256(folder/"raw.png"),
                    "actual_denoising_steps": adapter.pipe.num_timesteps, "seconds": time.monotonic()-start,
                    "training_eligible": False, "quality_status": "not_assessed"}
                write(folder / "record.json", record)
                state["records"].append(record)
                write(output / "generation_manifest.json", state)
                sheet(folder, target, reference, focused, picture(existing/row["raw_image"]), result,
                      f"{subject['id']} / {variant} / {row['reference_id']}")
                print("ZEST_REFERENCE_FOCUS", len(state["records"]), row["id"], flush=True)
            del adapter
            gc.collect()
            import torch
            torch.cuda.empty_cache()
        assets.verify(receipt)
        if any(sha256(path) != digest for path, digest in locks.items()):
            raise ValueError("read-only diagnostic inputs changed")
        current = read(existing / "generation_manifest.json")["records"]
        for subject, row in selected:
            if [candidate for candidate in current if candidate["id"] == row["id"]] != [row]:
                raise ValueError("selected full-reference control record changed")
        state["status"] = "generated_pending_quality_review"
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write(output / "generation_manifest.json", state)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--variants", nargs="+", choices=("baseline", "combined"), default=["baseline", "combined"])
    args = parser.parse_args()
    print(run(args.run, args.assets, args.gpu, args.variants))
