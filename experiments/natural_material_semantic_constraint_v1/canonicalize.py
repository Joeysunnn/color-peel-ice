"""Select a SuperMat material region using categorical consistency gates."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measurements(map_dir, mask_path, minimum_pixels):
    mask = np.asarray(Image.open(mask_path).convert("L")) >= 128
    result = {}
    for key in ("roughness", "metallic"):
        image = np.asarray(Image.open(map_dir / f"{key}.png").convert("L"))
        if image.shape != mask.shape:
            raise ValueError(f"Map and mask dimensions differ: {map_dir}, {mask_path}")
        values = image[mask].astype(np.float64) / 255.0
        if len(values) < minimum_pixels:
            raise ValueError(f"Too few valid pixels: {mask_path}")
        q05, q25, q50, q75, q95 = np.quantile(values, [0.05, 0.25, 0.50, 0.75, 0.95])
        result[key] = {"median": float(q50), "iqr": float(q75 - q25),
                       "q05": float(q05), "q95": float(q95),
                       "map_sha256": sha(map_dir / f"{key}.png")}
    result["pixel_count"] = int(mask.sum())
    result["mask_sha256"] = sha(mask_path)
    return result


def gate(sample, prior, config):
    checks = {}
    quality = config["quality_gate"]
    checks["roughness_spread"] = sample["roughness"]["iqr"] <= quality["maximum_roughness_iqr"]
    checks["metallic_spread"] = sample["metallic"]["iqr"] <= quality["maximum_metallic_iqr"]
    for field in ("surface_family", "gloss"):
        rule = config["semantic_gates"][field][prior[field]]
        for name, limit in rule.items():
            value = sample["roughness" if name.startswith("roughness") else "metallic"]["median"]
            checks[f"{field}_{name}"] = value >= limit if name.endswith("_min") else value <= limit
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    project_root, run_root = args.project_root.resolve(), args.run_root.resolve()
    output_dir = run_root / "canonical"
    output_dir.mkdir(exist_ok=False)
    summary = []
    for sample in config["samples"]:
        sample_id = sample["id"]
        prior_path = run_root / "semantic_prior" / sample_id / "prior.json"
        prior = json.loads(prior_path.read_text())["prior"]
        map_dir = run_root / "supermat_outputs" / sample_id
        candidates = []
        for name, relative_path in sample["candidate_masks"].items():
            mask_path = project_root / relative_path
            stats = measurements(map_dir, mask_path, config["quality_gate"]["minimum_pixels"])
            checks = gate(stats, prior, config)
            candidates.append({"name": name, "mask_path": str(mask_path),
                               "measurements": stats, "checks": checks,
                               "accepted": all(checks.values())})
        accepted = [candidate for candidate in candidates if candidate["accepted"]]
        selected = min(accepted, key=lambda item: (
            item["measurements"]["roughness"]["iqr"] +
            item["measurements"]["metallic"]["iqr"], item["name"])) if accepted else None
        result = {
            "sample_id": sample_id, "status": "accepted" if selected else "rejected",
            "semantic_prior": prior, "opacity_check": "not_testable_without_transmission_map",
            "selected_region": selected["name"] if selected else None,
            "roughness": selected["measurements"]["roughness"]["median"] if selected else None,
            "metallic": selected["measurements"]["metallic"]["median"] if selected else None,
            "parameter_origin": "SuperMat map medians only; VLM used for validation/selection",
            "candidates": candidates,
            "supermat_manifest_sha256": sha(run_root / "supermat_outputs/inference_manifest.json"),
            "semantic_prior_sha256": sha(prior_path), "config_sha256": sha(args.config),
            "code_sha256": sha(__file__),
            "project_git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=project_root, text=True).strip(),
        }
        (output_dir / f"{sample_id}.json").write_text(json.dumps(result, indent=2) + "\n")
        summary.append({"sample_id": sample_id, "status": result["status"],
                        "selected_region": result["selected_region"],
                        "roughness": result["roughness"], "metallic": result["metallic"]})
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
