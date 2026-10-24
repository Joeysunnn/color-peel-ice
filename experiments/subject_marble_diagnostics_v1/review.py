"""Verify completed diagnostic RGBs and make a compact comparison; no diffusion."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.subject_marble_diagnostics_v1.run import ROOT, read_json, sha256, sheet, write
from scripts.launch.colorpeel_run import git_info, resolve_run_dir


def run(source, output):
    source, output = Path(source).resolve(), resolve_run_dir(Path(output))
    if output.exists() or output.is_relative_to(ROOT) or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("use a fresh separate external review directory")
    revision = git_info()
    state = read_json(source / "status.json")
    if state["status"] != "generated_pending_visual_review" or state["generated_images"] != 18:
        raise ValueError("review requires the completed18-case mailbox diagnostic run")
    rows = [json.loads(line) for line in (source / "manifest.jsonl").read_text(encoding="utf8").splitlines()]
    if len(rows) != 18 or any(sha256(source / row["raw_image"]) != row["sha256"] for row in rows):
        raise ValueError("diagnostic RGB manifest mismatch")
    by_id = {row["id"]: row for row in rows}
    mask = np.asarray(Image.open(source / "source/mask.png").convert("L")) >= 128
    comparisons = []

    def compare(first, second, kind):
        a, b = [np.asarray(Image.open(source / by_id[name]["raw_image"]), dtype=float) for name in (first, second)]
        comparisons.append({"first": first, "second": second, "kind": kind,
            "foreground_MAE_255": float(np.abs(a - b)[mask].mean()),
            "background_MAE_255": float(np.abs(a - b)[~mask].mean()),
            "identical_pixels": bool(np.array_equal(a, b))})

    for reference in ("full", "masked_gray"):
        compare(reference + "__none", reference + "__joint_zero", "empty_MLP_vs_joint_zero")
    for row in rows:
        if row["reference"] == "full" and row["target_size"] == 1024:
            compare(row["id"], row["id"].replace("full__", "masked_gray__", 1), "reference_background_only")
    output.mkdir(parents=True)
    selected = [{**by_id[reference + "__" + name],
                 "raw_image": str(source / by_id[reference + "__" + name]["raw_image"])}
                for reference in ("full", "masked_gray") for name in ("none", "M-20", "R+1")]
    pair_metrics = read_json(source / "size_pair_metrics.json")
    sheet(output, selected, [("Original input", source / "source/original512.jpg"),
          ("Full photo reference", source / "source/reference_full.png"),
          ("Masked-gray reference", source / "source/reference_masked_gray.png")],
          "reference_comparison.png", crop=tuple(pair_metrics["fixed_review_crop"]))
    for name in ("full", "masked_gray"):
        group = [{**row, "raw_image": str(source / row["raw_image"])} for row in rows
                 if row["reference"] == name and row["target_size"] == 1024]
        sources = [("Original input", source / "source/original512.jpg"),
                   ("Actual material reference", source / f"source/reference_{name}.png"),
                   ("Aligned initialization", source / "source/initialization.png")]
        sheet(output, group, sources, f"subject_crop_{name}.png", crop=tuple(pair_metrics["fixed_review_crop"]))
    write(output / "review_metrics.json", {"generation_git": state["git"], "review_git": revision,
          "source_run": str(source), "verified_raw_images": len(rows), "comparisons": comparisons,
          "size_pairs": pair_metrics["pairs"], "training_eligible_images": 0,
          "interpretation": "RGB differences only; not physical M/R or identity scores. All18 full results remain in the source run."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.input, args.output)
