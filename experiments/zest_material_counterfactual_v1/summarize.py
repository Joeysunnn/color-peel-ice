"""Read-only secondary audit; save a fresh report without changing QC or eligibility."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.zest_material_counterfactual_v1.render_bank import sha256, write


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def local(run, name):
    path = (run / name).resolve()
    if not path.is_relative_to(run):
        raise ValueError(f"Artifact escapes run: {name}")
    return path


def mask(path, size):
    with Image.open(path) as image:
        return np.asarray(image.convert("L").resize(size, Image.Resampling.NEAREST)) >= 128


def rgb(path):
    with Image.open(path) as image:
        if image.size != (1024, 1024) or image.mode != "RGB":
            raise ValueError(f"Require raw1024 RGB: {path}")
        return np.asarray(image).copy()


def qc_counts(rows, scores):
    selected = [scores[row["id"]] for row in rows if row["id"] in scores]
    statuses = Counter(score["status"] for score in selected)
    not_rejected = [score for score in selected if score["status"] in ("accepted", "pending_manual_review")]
    fully_assessable = [score for score in not_rejected
                        if all(reason == "manual_review_required" for reason in score["reasons"])]
    return {"assessed": len(selected), "not_assessed": len(rows) - len(selected),
            "status_counts": {key: statuses[key] for key in ("invalid", "rejected", "pending_manual_review", "accepted")},
            "automatic_thresholds_not_rejected": len(not_rejected),
            "automatic_fully_assessable_not_rejected": len(fully_assessable),
            "not_rejected_but_not_fully_assessable": len(not_rejected) - len(fully_assessable),
            "interpretation": "Not rejected is not acceptance. Missing QC, uncalibrated or unassessable metrics and pending manual review do not establish dataset quality."}


def response_matrices(rows, scores):
    result = []
    metallic_axis, roughness_axis = [0., .5, 1.], [.2, .5, .8]
    groups = sorted({(row["subject_id"], row["variant"]) for row in rows if row["stage"] == "grid"})
    for subject, variant in groups:
        matrices = {key: [[None] * 3 for _ in range(3)] for key in (
            "case_ids", "estimated_metallic_median", "estimated_roughness_median", "quality_status")}
        for row in rows:
            if row["stage"] != "grid" or (row["subject_id"], row["variant"]) != (subject, variant):
                continue
            target = row["reference_target"]
            i, j = metallic_axis.index(target["metallic"]), roughness_axis.index(target["roughness"])
            if matrices["case_ids"][i][j] is not None:
                raise ValueError("Duplicate reference target in response matrix")
            score = scores.get(row["id"], {})
            material = score.get("metrics", {}).get("estimated_material", {})
            matrices["case_ids"][i][j] = row["id"]
            matrices["quality_status"][i][j] = score.get("status", "not_assessed")
            for channel in ("metallic", "roughness"):
                matrices["estimated_" + channel + "_median"][i][j] = material.get(channel, {}).get("generated", {}).get("median")
        result.append({"subject_id": subject, "variant": variant, "reference_metallic_rows": metallic_axis,
                       "reference_roughness_columns": roughness_axis, **matrices,
                       "interpretation": "Axes are reference-sphere PBR values; cells are approximate generated-image SuperMat estimates, not physical output ground truth. Missing cases/estimates remain null."})
    return result


def summarize(run):
    run = Path(run).resolve()
    plan = read(run / "plan.json")
    generation = read(run / "generation_manifest.json")
    rows = generation["records"]
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate generation IDs")
    hashes = {}
    def lock(path):
        path = Path(path).resolve()
        hashes[path.relative_to(run).as_posix()] = sha256(path)
        return hashes[path.relative_to(run).as_posix()]
    lock(run / "plan.json")
    lock(run / "generation_manifest.json")
    prepared = read(run / "prepare_receipt.json")
    lock(run / "prepare_receipt.json")
    for relative, expected in prepared["output_hashes"].items():
        if lock(local(run, relative)) != expected:
            raise ValueError(f"Prepared input changed: {relative}")
    scores, quality_path, identity_gate = {}, None, {}
    if (run / "quality/latest.json").exists():
        latest = read(run / "quality/latest.json")
        lock(run / "quality/latest.json")
        quality_path = local(run, latest["report"])
        report = read(quality_path)
        lock(quality_path)
        if len({row["id"] for row in report["records"]}) != len(report["records"]):
            raise ValueError("Duplicate QC IDs")
        scores = {row["id"]: row for row in report["records"]}
        identity_gate = report["identity_gate"]
        if not set(scores).issubset({row["id"] for row in rows}):
            raise ValueError("QC contains cases absent from generation manifest")
    subjects = {subject["id"]: subject for subject in plan["subjects"]}
    coverage, original_images, background_masks = {}, {}, {}
    for identifier, subject in subjects.items():
        files = subject["source_files"]
        for relative in files.values():
            lock(local(run, relative))
        obj = mask(local(run, files["object_mask"]), (512, 512))
        edit = mask(local(run, files["edit_mask"]), (512, 512))
        valid = mask(local(run, files["valid_mask"]), (512, 512))
        protected = mask(local(run, files["protected_mask"]), (512, 512))
        intersection = edit & valid & obj & ~protected
        if not edit.any() or obj.all():
            raise ValueError("Empty edit region or missing background support")
        coverage[identifier] = {"object_pixels512": int(obj.sum()), "edit_pixels512": int(edit.sum()),
                                "valid_alpha_pixels512": int(valid.sum()), "qc_valid_edit_pixels512": int(intersection.sum()),
                                "valid_fraction_of_edit": float(intersection.sum() / edit.sum()),
                                "edited_pixels_outside_valid_alpha": int((edit & ~valid).sum()),
                                "protected_pixels512": int(protected.sum()),
                                "protected_status": "configured" if protected.any() else "not_configured",
                                "interpretation": "Intrinsic/texture/depth ROI scores do not establish preservation in edited pixels outside valid alpha. Empty protected mask is not a successful text/logo protection test."}
        original_images[identifier] = rgb(local(run, files["target"]))
        background_masks[identifier] = ~mask(local(run, files["object_mask"]), (1024, 1024))
    background, pairs = [], []
    indexed = {(row["stage"], row["subject_id"], row["variant"], row["reference_id"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate stage/subject/variant/reference cases")
    for row in rows:
        path = local(run, row["raw_image"])
        if lock(path) != row["raw_sha256"]:
            raise ValueError(f"Raw output changed: {row['id']}")
        score = scores.get(row["id"])
        if score and score["raw_sha256"] != row["raw_sha256"]:
            raise ValueError(f"Stale QC/raw pairing: {row['id']}")
        generated = rgb(path)
        pixel_l1 = np.abs(generated.astype(float) - original_images[row["subject_id"]].astype(float)).mean(axis=2)
        values = pixel_l1[background_masks[row["subject_id"]]]
        background.append({"id": row["id"], "background_pixels1024": int(values.size),
                           "pixel_L1_mean255": float(values.mean()), "pixel_L1_p95_255": float(np.quantile(values, .95)),
                           "pixel_L1_p99_255": float(np.quantile(values, .99)),
                           "pixel_L1_gt10_fraction": float((values > 10).mean()),
                           "interpretation": "Per-pixel mean absolute RGB-channel difference, raw1024; descriptive only, no new rejection threshold or color-leakage verdict."})
        compared_variant = {"baseline": "color", "texture": "combined"}.get(row["variant"])
        other = indexed.get((row["stage"], row["subject_id"], compared_variant, row["reference_id"]))
        if other:
            other_path = local(run, other["raw_image"])
            if sha256(other_path) != other["raw_sha256"]:
                raise ValueError("Compared raw output changed")
            difference = np.abs(generated.astype(float) - rgb(other_path).astype(float))
            edit = mask(local(run, subjects[row["subject_id"]]["source_files"]["edit_mask"]), (1024, 1024))
            pairs.append({"stage": row["stage"], "subject_id": row["subject_id"], "reference_id": row["reference_id"],
                          "first_id": row["id"], "second_id": other["id"],
                          "same_png_sha256": row["raw_sha256"] == other["raw_sha256"],
                          "rgb_exactly_equal": bool(not difference.any()),
                          "RGB_MAE255": float(difference.mean()), "edit_RGB_MAE255": float(difference[edit].mean()),
                          "max_channel_difference255": float(difference.max()),
                          "changed_pixel_fraction": float(np.any(difference > 0, axis=2).mean())})
    eligibility = {"status": "not_available", "manifest_grid_records": None,
                   "training_eligible": None, "interpretation": "No dataset eligibility manifest was available; no approvals inferred."}
    if (run / "dataset_manifest.json").exists():
        dataset = read(run / "dataset_manifest.json")
        lock(run / "dataset_manifest.json")
        if any(not isinstance(row.get("training_eligible"), bool) for row in dataset["records"]):
            raise ValueError("Dataset eligibility must be explicit booleans")
        dataset_ids = {row["id"] for row in dataset["records"]}
        grid_ids = {row["id"] for row in rows if row["stage"] == "grid"}
        if len(dataset_ids) != len(dataset["records"]) or not dataset_ids.issubset(grid_ids):
            raise ValueError("Dataset contains duplicate or nonexistent grid IDs")
        eligibility = {"status": "recorded", "manifest_grid_records": len(dataset["records"]),
                       "training_eligible": sum(row["training_eligible"] for row in dataset["records"]),
                       "stored_assessment": dataset["assessment"],
                       "matches_latest_qc": bool(quality_path and dataset["assessment"] == quality_path.relative_to(run).as_posix()),
                       "generated_grid_images_missing_from_dataset_manifest": len(grid_ids - dataset_ids),
                       "interpretation": "Copied from existing dataset manifest; this audit does not approve or reclassify samples."}
    stages = Counter(row["stage"] for row in rows)
    expected_identity, expected_grid = len(subjects) * len(plan["variants"]) * 2, len(subjects) * len(plan["variants"]) * 9
    summary = {"source_qc_report": quality_path.relative_to(run).as_posix() if quality_path else None,
               "raw_counts": {"total": len(rows), "identity": stages["identity"], "grid": stages["grid"],
                              "expected_identity": expected_identity, "expected_grid": expected_grid,
                              "generation_complete": stages["identity"] == expected_identity and stages["grid"] == expected_grid},
               "qc": qc_counts(rows, scores), "qc_by_variant": {
                   variant: qc_counts([row for row in rows if row["variant"] == variant], scores) for variant in plan["variants"]},
               "identity_gate": identity_gate, "dataset_eligibility": eligibility,
               "coverage512": coverage, "reference_response_matrices": response_matrices(rows, scores),
               "raw_background1024": background, "initialization_variant_pairs": pairs,
               "pair_rgb_duplicate_count": sum(row["rgb_exactly_equal"] for row in pairs),
               "input_sha256": hashes, "audit_script_sha256": sha256(__file__), "training_started_by_audit": False,
               "interpretation": "Secondary descriptive audit only. SuperMat estimates and monocular depth are not physical ground truth. No changed thresholds, QC statuses or training eligibility."}
    # Refuse a mixed snapshot if generation, latest-QC or any image changed while reading.
    if any(sha256(local(run, relative)) != expected for relative, expected in hashes.items()):
        raise ValueError("Audit inputs changed during summarization; no summary written")
    analysis = run / "analysis"
    analysis.mkdir(exist_ok=True)
    number = 1
    while (analysis / f"analysis{number:03d}").exists():
        number += 1
    output = analysis / f"analysis{number:03d}"
    output.mkdir()
    write(output / "summary.json", summary)
    counts, qc = summary["raw_counts"], summary["qc"]
    lines = ["# Secondary ZeST audit", "", f"Raw generation: {counts['total']} images ({counts['identity']} Identity / {counts['grid']} grid).",
             f"Expected: {expected_identity} Identity / {expected_grid} grid; generation complete: {counts['generation_complete']}.", "",
             f"QC assessed: {qc['assessed']}; not assessed: {qc['not_assessed']}; statuses: {json.dumps(qc['status_counts'], sort_keys=True)}.",
             f"Automatic thresholds did not reject {qc['automatic_thresholds_not_rejected']} assessed cases; "
             f"{qc['automatic_fully_assessable_not_rejected']} had no outstanding automatic-assessability reason. This is not acceptance.",
             (f"Recorded training eligibility: {eligibility['training_eligible']} / {eligibility['manifest_grid_records']} grid records."
              if eligibility["status"] == "recorded" else "Dataset training eligibility: not available; no approval inferred."), "",
             "Reference M/R matrices contain approximate SuperMat output estimates; reference PBR values are not output ground truth.",
             "Raw1024 background L1 percentiles and fractions above10 are descriptive statistics, not new filters.",
             f"Initialization comparison pairs: {len(pairs)}; exactly equal RGB pairs: {summary['pair_rgb_duplicate_count']}.", ""]
    for identifier, record in coverage.items():
        lines.append(f"- {identifier}: valid/edit coverage {record['valid_fraction_of_edit']:.4f}; "
                     f"edited pixels outside valid alpha {record['edited_pixels_outside_valid_alpha']}; "
                     f"protected region {record['protected_status']} ({record['protected_pixels512']} pixels).")
    lines += ["", "This audit changes no QC status, threshold, image or training eligibility. See summary.json for per-case statistics and input hashes."]
    (output / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    print(summarize(args.run))
