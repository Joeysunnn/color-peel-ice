"""Prepare an independent normalized-Albedo edge diagnostic; never overwrite a run."""

import argparse
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image, ImageDraw

from experiments.zest_material_counterfactual_v1.focus_reference import copy_prepared
from experiments.zest_material_counterfactual_v1.render_bank import ROOT, sha256, write
from experiments.zest_material_counterfactual_v1.run import load_plan, read


def edges(albedo, valid, edit, old):
    gray512 = albedo.convert("L")
    valid512 = np.asarray(valid.convert("L")) >= 128
    effective = np.asarray(edit.convert("L").resize((1024, 1024), Image.Resampling.NEAREST)) >= 128
    valid1024 = np.asarray(valid.convert("L").resize((1024, 1024), Image.Resampling.NEAREST)) >= 128
    support = effective & valid1024
    measure = valid512 & np.asarray(Image.fromarray(effective).resize(gray512.size, Image.Resampling.NEAREST))
    if int(measure.sum()) < 1000:
        raise ValueError("Need at least1000 valid effective Albedo pixels")
    gray = np.asarray(gray512, dtype=np.float64)
    low, high = np.quantile(gray[measure], [.02, .98])
    normalized = (gray - low) / (high - low) if high > low else np.full_like(gray, .5)
    clipped = (normalized < 0) | (normalized > 1)
    normalized = Image.fromarray(np.rint(np.clip(normalized, 0, 1) * 255).astype(np.uint8))
    normalized1024 = normalized.resize((1024, 1024), Image.Resampling.BICUBIC)
    blurred = cv2.GaussianBlur(np.asarray(normalized1024), (0, 0), sigmaX=1., sigmaY=1.)
    interior3 = cv2.erode(support.astype(np.uint8), np.ones((7, 7), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    interior25 = cv2.erode(support.astype(np.uint8), np.ones((51, 51), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    comparison12 = cv2.erode(effective.astype(np.uint8), np.ones((25, 25), np.uint8), borderType=cv2.BORDER_REPLICATE) > 0
    new = cv2.Canny(blurred, 40, 80)
    new[~interior3] = 0
    previous = np.asarray(old.convert("L").resize((1024, 1024), Image.Resampling.NEAREST)) > 0
    stats = {"quantiles": [.02, .98], "gray_q02_255": float(low), "gray_q98_255": float(high),
        "gray_span_255": float(high - low), "normalization_pixels512": int(measure.sum()),
        "clipped_fraction": float(clipped[measure].mean()), "gaussian_sigma1024": 1., "canny_thresholds": [40, 80],
        "normalization": "PIL estimated-Albedo grayscale; valid effective q02/q98; clipped to[0,255]; constant span maps to128",
        "resize": "normalize512 then BICUBIC1024, Gaussian then Canny; mask nearest; no source RGB edges",
        "support_pixels1024": int(support.sum()), "interior3_support_pixels": int(interior3.sum()),
        "edge_assessability": "candidate_condition_only" if high > low else "unassessable_constant_Albedo_span",
        "erosion_radii_pixels1024": {"internal_support": 3, "strict_internal_support": 25, "prior_edit_mask_comparison": 12},
        "prior_comparison_border": "replicate, matching PIL MinFilter25 at image borders; new edge support uses constant-zero erosion",
        "prior_edit_kernel25_support_pixels": int(comparison12.sum()),
        "old_prior_edit_kernel25_edge_count": int((previous & comparison12).sum()),
        "new_prior_edit_kernel25_edge_count": int(((new > 0) & comparison12).sum()),
        "interior25_support_pixels": int(interior25.sum()), "old_edge_count": int(previous.sum()), "new_edge_count": int((new > 0).sum()),
        "old_interior3_edge_count": int((previous & interior3).sum()), "new_interior3_edge_count": int(((new > 0) & interior3).sum()),
        "old_interior25_edge_count": int((previous & interior25).sum()), "new_interior25_edge_count": int(((new > 0) & interior25).sum()),
        "boundary_edge_count_new": int(((new > 0) & ~interior3).sum()),
        "interpretation": "Condition-signal diagnostic; more edges do not establish true texture, preservation or counterfactual quality"}
    return {"albedo_gray": gray512.resize((1024, 1024), Image.Resampling.BICUBIC), "normalized_gray": normalized1024,
            "edges": Image.fromarray(new).convert("RGB"), "support_interior3": Image.fromarray(interior3.astype(np.uint8) * 255)}, stats


def run(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Keep existing results intact; choose a fresh edge diagnostic directory")
    if output == ROOT or output.is_relative_to(ROOT) or source.is_relative_to(output):
        raise ValueError("Require a new external output separate from source ancestors")
    plan = load_plan(source)
    prepared = read(source / "prepare_receipt.json")
    locks = {str((source / relative).resolve()): digest for relative, digest in prepared["output_hashes"].items()}
    locks.update({str(path): sha256(path) for path in (source / "prepare_receipt.json", source / "bank/manifest.json", Path(__file__).resolve())})
    output.mkdir(parents=True)
    copy_prepared(source, output)
    (output / "bank").mkdir(exist_ok=True)
    shutil.copyfile(source / "bank/manifest.json", output / "bank/manifest.json")
    write(output / "source_locks.json", locks)
    plan["input_hashes"].update(locks)
    plan["input_hashes"][str(output / "source_locks.json")] = sha256(output / "source_locks.json")
    report = {}
    for subject in plan["subjects"]:
        files = subject["source_files"]
        def image(key):
            path = (output / files[key]).resolve()
            if not path.is_relative_to(output):
                raise ValueError("Prepared source path escapes new output")
            with Image.open(path) as value:
                return value.copy()
        old = image("albedo_edges")
        pictures, stats = edges(image("original_albedo"), image("valid_mask"), image("edit_mask"), old)
        folder = output / "edge_diagnostic" / subject["id"]
        folder.mkdir(parents=True)
        for name, picture in pictures.items():
            picture.save(folder / f"{name}.png")
        pictures["edges"].save(output / files["albedo_edges"])
        subject["edge_policy"] = "Independent normalized-Albedo edge diagnostic: q02/q98 valid effective; Gaussian sigma1; Canny40/80; erode3px1024 internal support"
        subject["edge_diagnostic"] = stats
        write(folder / "metrics.json", stats)
        panels = [("Original Albedo gray", pictures["albedo_gray"]), ("Normalized Albedo gray", pictures["normalized_gray"]),
                  ("Old Canny100/200", old), ("New Canny40/80", pictures["edges"]), ("Interior support erode3", pictures["support_interior3"])]
        sheet = Image.new("RGB", (1500, 380), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 6), f"{subject['id']} | independent edge-condition diagnostic | identity / material / Albedo quality unverified; NO training", fill="black")
        draw.text((8, 23), f"Edit kernel25 old/new edges: {stats['old_prior_edit_kernel25_edge_count']}/{stats['new_prior_edit_kernel25_edge_count']} | q02/q98: {stats['gray_q02_255']:.2f}/{stats['gray_q98_255']:.2f} | clipped: {stats['clipped_fraction']:.2%}", fill="black")
        for index, (label, picture) in enumerate(panels):
            draw.text((index * 300 + 8, 48), label, fill="black")
            sheet.paste(picture.convert("RGB").resize((290, 290), Image.Resampling.LANCZOS), (index * 300 + 5, 75))
        sheet.save(folder / "input_check.png")
        report[subject["id"]] = stats
    plan["edge_diagnostic"] = {"source_run": str(source), "only_changed_generation_input": "source_files.albedo_edges",
        "unchanged": "all sampling, thresholds, references, depth, masks, prompt, seeds and initializations", "training_started": False}
    write(output / "plan.json", plan)
    write(output / "edge_diagnostic.json", report)
    if any(sha256(path) != digest for path, digest in locks.items()):
        raise ValueError("Original source changed during copy; preserve diagnostic and investigate")
    hashes = {p.relative_to(output).as_posix(): sha256(p) for p in output.rglob("*") if p.is_file() and p.name != "prepare_receipt.json"}
    write(output / "prepare_receipt.json", {"status": "complete", "input_hashes": plan["input_hashes"], "output_hashes": hashes})
    load_plan(output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.run, args.output))
