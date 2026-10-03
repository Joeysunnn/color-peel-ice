"""Freeze three Scheme 3 material estimates with explicit semantic decisions."""

import argparse
import hashlib
import json
from pathlib import Path

from experiments.natural_material_semantic_constraint_v1.canonicalize import measurements


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme3-run", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    out = args.run_root / "canonical"
    out.mkdir(exist_ok=False)
    for name, source_name in (("mailbox", "mailbox"), ("metal_spoon", "spoon_positive_control")):
        path = args.scheme3_run / "canonical" / f"{source_name}.json"
        source = json.loads(path.read_text())
        if source["status"] != "accepted":
            raise ValueError(f"Existing Scheme 3 source was not accepted: {path}")
        save(out / f"{name}.json", {
            "material_id": name, "roughness": source["roughness"],
            "metallic": source["metallic"], "semantic_prior": source["semantic_prior"],
            "semantic_status": "accepted_in_previous_scheme3_run",
            "selected_region": source["selected_region"],
            "source_canonical": str(path.resolve()), "source_canonical_sha256": sha(path),
            "parameter_origin": "SuperMat map medians; VLM selected/validated region",
        })
    prior_path = args.run_root / "semantic_prior/wood_spoon/prior.json"
    prior = json.loads(prior_path.read_text())["prior"]
    map_dir = args.run_root / "supermat_outputs/wood_spoon"
    candidates = []
    for name, filename in (("bowl_interior", "bowl_region.png"),
                           ("eroded_object", "eroded_object_region.png")):
        mask = args.run_root / "source/wood_prepared" / filename
        stats = measurements(map_dir, mask, 1000)
        r, m = stats["roughness"]["median"], stats["metallic"]["median"]
        checks = {
            "wood_metallic_max_0.15": m <= 0.15,
            "matte_roughness_min_0.45": r >= 0.45 if prior["gloss"] == "matte" else None,
            "roughness_iqr_max_0.12": stats["roughness"]["iqr"] <= 0.12,
            "metallic_iqr_max_0.20": stats["metallic"]["iqr"] <= 0.20,
        }
        candidates.append({"name": name, "mask": str(mask.resolve()),
                           "measurements": stats, "semantic_checks": checks})
    selected = min(candidates, key=lambda c: (
        c["measurements"]["roughness"]["iqr"] +
        c["measurements"]["metallic"]["iqr"], c["name"]))
    checks = selected["semantic_checks"]
    status = "accepted" if all(value is not False for value in checks.values()) else "semantic_gloss_conflict"
    save(out / "wood_spoon.json", {
        "material_id": "wood_spoon", "roughness": selected["measurements"]["roughness"]["median"],
        "metallic": selected["measurements"]["metallic"]["median"],
        "semantic_prior": prior, "semantic_status": status,
        "selected_region": selected["name"], "candidates": candidates,
        "prior_sha256": sha(prior_path),
        "supermat_manifest_sha256": sha(args.run_root / "supermat_outputs/inference_manifest.json"),
        "parameter_origin": "SuperMat map medians only; VLM used as a diagnostic gate",
        "continuation_policy": "Keep raw SuperMat R/M for requested end-to-end pilot despite gloss disagreement"
        if status != "accepted" else "Scheme 3 accepted",
    })
    save(out / "summary.json", {
        name: {"roughness": json.loads((out / f"{name}.json").read_text())["roughness"],
               "metallic": json.loads((out / f"{name}.json").read_text())["metallic"],
               "semantic_status": json.loads((out / f"{name}.json").read_text())["semantic_status"]}
        for name in ("mailbox", "metal_spoon", "wood_spoon")
    })


if __name__ == "__main__":
    main()
