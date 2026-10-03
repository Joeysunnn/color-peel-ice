"""Apply a fixed VLM semantic gate to IID hypotheses, then aggregate R/M."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def distribution(values):
    if not values:
        return None
    q05, q25, q50, q75, q95 = np.quantile(values, [0.05, 0.25, 0.50, 0.75, 0.95])
    return {"count": len(values), "median": float(q50), "q05": float(q05),
            "q25": float(q25), "q75": float(q75), "q95": float(q95),
            "iqr": float(q75 - q25), "minimum": float(min(values)),
            "maximum": float(max(values))}


def checks_for(roughness, metallic, prior, gates):
    checks = {"roughness_in_range": 0 <= roughness <= 1,
              "metallic_in_range": 0 <= metallic <= 1}
    for field in ("surface_family", "gloss"):
        for rule, limit in gates[field][prior[field]].items():
            value = roughness if rule.startswith("roughness") else metallic
            checks[f"{field}_{rule}"] = value >= limit if rule.endswith("_min") else value <= limit
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    run_root, project_root = args.run_root.resolve(), args.project_root.resolve()
    candidate_manifest = run_root / "candidates/manifest.json"
    if not candidate_manifest.is_file():
        raise FileNotFoundError(candidate_manifest)
    output_root = run_root / "canonical"
    output_root.mkdir(exist_ok=False)
    summary = []
    for sample in config["samples"]:
        sample_id = sample["id"]
        mask = np.asarray(Image.open(run_root / "masks" / f"{sample_id}.png").convert("L")) >= 128
        if mask.sum() < 1000:
            raise ValueError(f"Material region too small: {sample_id}")
        prior_path = run_root / "vlm_priors" / f"{sample_id}.json"
        prior = json.loads(prior_path.read_text())["prior"]
        candidates = []
        for index, seed in enumerate(config["iid_candidate_seeds"]):
            folder = run_root / "candidates" / sample_id / f"candidate_{index:02d}"
            metadata_path = folder / "metadata.json"
            metadata = json.loads(metadata_path.read_text())
            if metadata["seed"] != seed:
                raise ValueError(f"Candidate seed mismatch: {folder}")
            maps_path = folder / "raw_maps.npz"
            if sha(maps_path) != metadata["raw_maps_sha256"]:
                raise ValueError(f"Candidate hash mismatch: {maps_path}")
            with np.load(maps_path) as maps:
                roughness_map = maps["roughness"]
                metallic_map = maps["metallic"]
            if roughness_map.shape != mask.shape or metallic_map.shape != mask.shape:
                raise ValueError(f"Candidate/mask size mismatch: {folder}")
            roughness_values, metallic_values = roughness_map[mask], metallic_map[mask]
            if not np.isfinite(roughness_values).all() or not np.isfinite(metallic_values).all():
                raise ValueError(f"Non-finite material values: {folder}")
            r_stats = distribution(roughness_values.tolist())
            m_stats = distribution(metallic_values.tolist())
            checks = checks_for(r_stats["median"], m_stats["median"], prior,
                                config["semantic_gates"])
            candidates.append({"index": index, "seed": seed, "accepted": all(checks.values()),
                               "checks": checks, "roughness": r_stats, "metallic": m_stats,
                               "roughness_outside_fraction": float(np.mean(
                                   (roughness_values < 0) | (roughness_values > 1))),
                               "metallic_outside_fraction": float(np.mean(
                                   (metallic_values < 0) | (metallic_values > 1))),
                               "raw_maps_sha256": sha(maps_path),
                               "metadata_sha256": sha(metadata_path)})
        accepted = [row for row in candidates if row["accepted"]]
        all_r = distribution([row["roughness"]["median"] for row in candidates])
        all_m = distribution([row["metallic"]["median"] for row in candidates])
        accepted_r = distribution([row["roughness"]["median"] for row in accepted])
        accepted_m = distribution([row["metallic"]["median"] for row in accepted])
        representative = min(accepted, key=lambda row: (
            abs(row["roughness"]["median"] - accepted_r["median"]) +
            abs(row["metallic"]["median"] - accepted_m["median"]), row["index"])) if accepted else None
        result = {
            "sample_id": sample_id, "status": "accepted" if accepted else "rejected",
            "roughness": accepted_r["median"] if accepted else None,
            "metallic": accepted_m["median"] if accepted else None,
            "accepted_count": len(accepted), "total_count": len(candidates),
            "representative_candidate_index": representative["index"] if representative else None,
            "all_candidate_uncertainty": {"roughness": all_r, "metallic": all_m},
            "accepted_candidate_uncertainty": {"roughness": accepted_r, "metallic": accepted_m},
            "semantic_prior": prior, "opacity_check": "not_testable_without_transmission_map",
            "parameter_origin": "median of accepted IID hypothesis region medians; VLM gates only",
            "material_region_mask_sha256": sha(run_root / "masks" / f"{sample_id}.png"),
            "vlm_prior_sha256": sha(prior_path),
            "candidate_manifest_sha256": sha(candidate_manifest),
            "config_sha256": sha(args.config), "script_sha256": sha(__file__),
            "project_git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=project_root, text=True).strip(),
            "candidates": candidates,
        }
        (output_root / f"{sample_id}.json").write_text(json.dumps(result, indent=2) + "\n")
        summary.append({key: result[key] for key in (
            "sample_id", "status", "roughness", "metallic", "accepted_count", "total_count")})
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
