"""Deterministic, provisional D65 Lab target-color measurements; no approval gates."""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import subprocess
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw, __version__ as pillow_version

from src.methods.colorpeel_ice.natural_image_cohort import (
    NaturalImageCohortError, load_and_validate_cohort, parse_mask_run_pairs,
    select_downstream_records, select_measurement_records,
)
from src.methods.colorpeel_ice.natural_image_masks import canonical_sha256, file_sha256


REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_RELPATH = "experiments/natural_image_subject_color_pilot/configs/d1_target_color_audit_protocol_v1.json"
COHORT_RELPATH = "experiments/natural_image_subject_color_pilot/configs/d1_target_color_audit_cohort_v1.json"
FROZEN_PROTOCOL_CANONICAL_SHA256 = "99131f3345c9f7ec732554ba477cb04b6696bc35a6d62df8b93fe05f8c50fe06"


class TargetColorAuditError(ValueError):
    """Invalid audit inputs; no measurement output should be created."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TargetColorAuditError(message)


def read_json(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TargetColorAuditError(f"Cannot read JSON {path}: {exc}") from exc
    require(isinstance(result, dict), f"JSON must be an object: {path}")
    return result


def load_protocol(path: Path, cohort_path: Path) -> dict[str, Any]:
    """Accept only the frozen repository paths and complete protocol semantics."""
    require(path.resolve() == (REPO_ROOT / PROTOCOL_RELPATH).resolve(),
            "protocol path must be the frozen repository protocol")
    require(cohort_path.resolve() == (REPO_ROOT / COHORT_RELPATH).resolve(),
            "cohort path must be the frozen repository cohort")
    protocol = read_json(path)
    require(canonical_sha256(protocol) == FROZEN_PROTOCOL_CANONICAL_SHA256,
            "repository protocol hash drift")
    require(isinstance(protocol.get("cohort"), dict), "protocol cohort must be an object")
    digest = protocol["cohort"].get("sha256")
    require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
            "protocol cohort sha256 must be lowercase SHA256")
    require(digest == file_sha256(cohort_path), "cohort hash drift against protocol")
    return protocol


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def rgb_uint8_to_lab(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb)
    require(rgb.dtype == np.uint8 and rgb.shape[-1] == 3, "RGB input must be uint8 with 3 channels")
    linear = srgb_to_linear(rgb.astype(np.float64) / 255.0)
    matrix = np.array([[0.4124564, 0.3575761, 0.1804375],
                       [0.2126729, 0.7151522, 0.0721750],
                       [0.0193339, 0.1191920, 0.9503041]], dtype=np.float64)
    xyz = (linear @ matrix.T) / np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), ((24389 / 27) * xyz + 16) / 116)
    return np.stack((116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])), axis=-1)


def ab_to_lch(a: float, b: float) -> dict[str, float | None]:
    chroma = float(np.hypot(a, b))
    return {"C": chroma, "h_degrees": None if chroma <= 1e-8 else float(np.degrees(np.arctan2(b, a)) % 360)}


def filter_lightness(lab: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    lightness = lab[..., 0][mask]
    area = int(mask.sum())
    require(area > 0, "color mask must be nonempty")
    dark = int(np.count_nonzero(lightness <= 5.0))
    light = int(np.count_nonzero(lightness >= 95.0))
    eligible = mask & (lab[..., 0] > 5.0) & (lab[..., 0] < 95.0)
    count = int(eligible.sum())
    return eligible, {
        "color_area_px": area, "eligible_px": count, "dark_px": dark, "light_px": light,
        "valid_ratio": count / area, "dark_ratio": dark / area, "light_ratio": light / area,
        "l_percentiles": {f"p{p:02d}": float(np.percentile(lightness, p)) for p in (1, 5, 50, 95, 99)},
    }


def _point_order(lab: np.ndarray, yx: np.ndarray) -> np.ndarray:
    return np.lexsort((yx[:, 1], yx[:, 0], lab[:, 2], lab[:, 1]))


def _assign(ab: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.argmin(np.sum((ab[:, None, :] - centers[None, :, :]) ** 2, axis=2), axis=1)


def cluster_lab(lab: np.ndarray, yx: np.ndarray, max_iterations: int = 100) -> tuple[dict[str, Any], np.ndarray | None]:
    """Cluster all pixels; restore labels to caller order after sorted reduction."""
    lab = np.asarray(lab, dtype=np.float64)
    yx = np.asarray(yx)
    review = {"status": "REVIEW", "reason": "cluster_degenerate", "iterations": 0, "clusters": [],
              "provisional_target": None, "dominant_cluster": None}
    if len(lab) < 3 or len(np.unique(lab[:, 1:3], axis=0)) < 3:
        return review, None
    order = _point_order(lab, yx)
    points = lab[order]
    ab = points[:, 1:3]
    centers = [ab[(len(ab) - 1) // 2].copy()]
    for _ in range(2):
        distance = np.min(np.sum((ab[:, None, :] - np.asarray(centers)[None, :, :]) ** 2, axis=2), axis=1)
        centers.append(ab[int(np.argmax(distance))].copy())
    centers = np.asarray(centers)
    previous = None
    for iteration in range(1, max_iterations + 1):
        labels = _assign(ab, centers)
        counts = np.bincount(labels, minlength=3)
        if np.any(counts == 0):
            return dict(review, reason="empty_cluster", iterations=iteration), None
        centers = np.stack([ab[labels == index].mean(axis=0) for index in range(3)])
        if previous is not None and np.array_equal(previous, labels):
            break
        previous = labels.copy()
    else:
        return dict(review, reason="nonconvergence", iterations=max_iterations), None
    canonical = sorted(range(3), key=lambda index: (centers[index, 0], centers[index, 1], index))
    remap = np.empty(3, dtype=int)
    remap[canonical] = np.arange(3)
    labels = remap[labels]
    clusters = []
    for index, original in enumerate(canonical):
        members = points[labels == index]
        distances = np.linalg.norm(members[:, 1:3] - centers[original], axis=1)
        clusters.append({
            "cluster": index, "original_index": original, "population": len(members),
            "ratio_over_eligible": len(members) / len(points), "mean_ab": centers[original].tolist(),
            "median_lab": np.median(members, axis=0).tolist(),
            "compactness": {"rms": float(np.sqrt(np.mean(distances ** 2))),
                            "median": float(np.median(distances)), "p90": float(np.percentile(distances, 90))},
        })
    dominant = max(range(3), key=lambda index: clusters[index]["population"])
    observed_l, a, b = clusters[dominant]["median_lab"]
    restored = np.empty(len(labels), dtype=int)
    restored[order] = labels
    return {"status": "MEASURED", "reason": None, "iterations": iteration, "clusters": clusters,
            "dominant_cluster": dominant, "dominant_ratio_over_eligible": clusters[dominant]["ratio_over_eligible"],
            "provisional_target": {"a": a, "b": b, **ab_to_lch(a, b), "observed_median_L": observed_l,
                                   "lightness_is_fixed_target_token": False}}, restored


def safe_input_path(root: Path, relative: Any) -> Path:
    require(isinstance(relative, str) and bool(relative), "artifact path must be safe relative")
    posix, windows = PurePosixPath(relative), PureWindowsPath(relative)
    require(not posix.is_absolute() and not windows.is_absolute() and not windows.drive
            and ".." not in posix.parts and ".." not in windows.parts and "\\" not in relative
            and ":" not in relative, f"artifact path must be safe relative: {relative}")
    resolved = (root / relative).resolve()
    require(resolved.is_relative_to(root.resolve()), f"artifact path escapes run root: {relative}")
    require(resolved.is_file(), f"artifact file missing: {resolved}")
    return resolved


def validate_artifacts(root: Path, source: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    try:
        arrays, provenance = {}, {}
        for key, mode in (("raw_image", "RGB"), ("color_mask", "L")):
            path = safe_input_path(root, source["outputs"][key])
            data = path.read_bytes()
            with Image.open(BytesIO(data)) as image:
                image.load()
                require(image.mode == mode, f"{key} mode must be {mode}")
                metadata = source["output_metadata"][key]
                require(metadata["mode"] == image.mode and metadata["size"] == list(image.size),
                        f"{key} output_metadata mode/size mismatch")
                arrays[key] = np.asarray(image).copy()
            provenance[key] = {"path": str(path), "run_relative_path": source["outputs"][key],
                               "current_sha256": hashlib.sha256(data).hexdigest()}
        expected = source["source"]["image"]["sha256"]
        require(source["raw_copy"]["image_bytes_identical"] is True, "raw_copy.image_bytes_identical must be true")
        require(provenance["raw_image"]["current_sha256"] == expected, "raw image SHA256 differs from source image")
        provenance["raw_image"].update(expected_sha256=expected, expected_hash_available=True)
        provenance["color_mask"].update(expected_sha256=None, expected_hash_available=False,
                                       hash_note="Historical manifest does not declare a derived color-mask SHA256; current hash recorded only")
        rgb, mask = arrays["raw_image"], arrays["color_mask"]
        require(rgb.shape[:2] == mask.shape, "raw image and color mask shape mismatch")
        require(np.isin(mask, [0, 255]).all(), "color mask must be binary {0,255}")
        area = int(np.count_nonzero(mask))
        require(area > 0, "color mask must be nonempty")
        require(type(source["geometry"]["color_area_px"]) is int and area == source["geometry"]["color_area_px"],
                "color mask area differs from manifest geometry.color_area_px")
        return rgb, mask == 255, provenance
    except (KeyError, TypeError, OSError) as exc:
        raise TargetColorAuditError(f"Invalid artifact manifest or image: {exc}") from exc


def validate_output_dir(output: Path, runs: dict[str, Path], repo: Path = REPO_ROOT) -> Path:
    output = output.resolve()
    require(not output.exists(), "output directory already exists; never overwrite")
    require(not output.is_relative_to(repo.resolve()), "output directory must be outside repository")
    for root in runs.values():
        root = root.resolve()
        require(not output.is_relative_to(root) and not root.is_relative_to(output),
                "output directory and input run roots must not contain one another")
    return output


def git_commit(expected: str) -> str:
    actual = subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()
    require(expected == actual, f"expected git commit mismatch: expected {expected}, actual {actual}")
    status = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain=v1", "--untracked-files=all"], text=True,
    )
    require(not status.strip(), "repository must be clean, including untracked files")
    return actual


def measure(rgb: np.ndarray, mask: np.ndarray) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    lab = rgb_uint8_to_lab(rgb)
    eligible, metrics = filter_lightness(lab, mask)
    yx = np.argwhere(eligible)
    clustered, labels = cluster_lab(lab[eligible], yx)
    dominant_mask = np.zeros(mask.shape, dtype=bool)
    if labels is not None:
        dominant_mask[eligible] = labels == clustered["dominant_cluster"]
        clustered["dominant_ratio_over_color_mask"] = int(dominant_mask.sum()) / metrics["color_area_px"]
    else:
        clustered["dominant_ratio_over_eligible"] = None
        clustered["dominant_ratio_over_color_mask"] = None
    return {**metrics, **clustered, "qc_decision": "PENDING_HUMAN", "approved_for_downstream": False}, eligible, dominant_mask, lab


def sample_qc(rgb: np.ndarray, mask: np.ndarray, eligible: np.ndarray, dominant: np.ndarray,
              lab: np.ndarray, row: dict[str, Any]) -> Image.Image:
    """Pillow-only display; deterministic scatter sampling never affects clustering."""
    sheet = Image.new("RGB", (768, 672), "white")
    draw = ImageDraw.Draw(sheet)
    overlay = rgb.copy()
    overlay[mask] = np.rint(0.6 * rgb[mask] + 0.4 * np.array([255, 0, 255])).astype(np.uint8)
    views = [rgb, overlay, np.where(eligible[..., None], rgb, 0), np.where(dominant[..., None], rgb, 0)]
    for index, (title, pixels) in enumerate(zip(["raw RGB", "color mask overlay", "eligible pixels", "dominant pixels"], views)):
        x, y = (index % 3) * 256, (index // 3) * 336
        draw.text((x + 4, y + 4), title, fill="black")
        image = Image.fromarray(pixels)
        image.thumbnail((256, 256))
        sheet.paste(image, (x, y + 24))
    x0, y0 = 256, 360
    draw.text((260, 340), "a,b scatter [-128,128]", fill="black")
    draw.rectangle((x0, y0, x0 + 255, y0 + 255), outline="black")
    draw.line((x0 + 128, y0, x0 + 128, y0 + 255), fill="#bbbbbb")
    draw.line((x0, y0 + 128, x0 + 255, y0 + 128), fill="#bbbbbb")
    points = lab[eligible]
    if len(points):
        order = _point_order(points, np.argwhere(eligible))
        selected = order[np.linspace(0, len(order) - 1, min(4096, len(order)), dtype=int)]
        colors = rgb[eligible]
        for index in selected:
            a, b = points[index, 1:3]
            px = x0 + int(np.clip((a + 128) * 255 / 256, 0, 255))
            py = y0 + int(np.clip((128 - b) * 255 / 256, 0, 255))
            draw.point((px, py), fill=tuple(int(c) for c in colors[index]))
    target = row["provisional_target"]
    lines = [f"#{row['cohort_rank']} {row['stable_id']}", f"role: {row['role']}", row["status"],
             row["reason"] or "algorithm converged", f"eligible {row['eligible_px']}/{row['color_area_px']}",
             f"valid ratio {row['valid_ratio']:.4f}", f"dark/light {row['dark_ratio']:.4f}/{row['light_ratio']:.4f}"]
    if target:
        cluster = row["clusters"][row["dominant_cluster"]]
        hue = target["h_degrees"]
        lines += [f"dominant eligible {row['dominant_ratio_over_eligible']:.4f}",
                  f"dominant full {row['dominant_ratio_over_color_mask']:.4f}",
                  f"observed L {target['observed_median_L']:.3f}", f"a,b {target['a']:.3f}, {target['b']:.3f}",
                  f"C,h {target['C']:.3f}, {hue:.3f}" if hue is not None else f"C {target['C']:.3f}; h null",
                  f"RMS {cluster['compactness']['rms']:.3f}", "L is observed, not fixed target L"]
    else:
        lines += ["No provisional target", "Dominant mask is empty"]
    lines += ["PENDING_HUMAN", "NOT APPROVED FOR DOWNSTREAM"]
    draw.multiline_text((516, 340), "\n".join(lines), fill="black", spacing=5)
    return sheet


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def flatten(prefix: str, value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {key: item for name, nested in value.items()
                for key, item in flatten(f"{prefix}_{name}" if prefix else name, nested).items()}
    if isinstance(value, list):
        return {key: item for index, nested in enumerate(value)
                for key, item in flatten(f"{prefix}_{index}", nested).items()}
    return {prefix: value}


def run_audit(cohort_path: Path, protocol_path: Path, pairs: Sequence[Sequence[str]],
              output: Path, expected_commit: str) -> dict[str, Any]:
    runs = {alias: root.resolve() for alias, root in parse_mask_run_pairs(pairs).items()}
    output = validate_output_dir(output, runs, REPO_ROOT)
    commit = git_commit(expected_commit)
    config_hashes = {"cohort": file_sha256(cohort_path), "protocol": file_sha256(protocol_path)}
    protocol = load_protocol(protocol_path, cohort_path)
    # Resolve manifest paths explicitly before TC-1 reads them, including symlinks.
    cohort_input = read_json(cohort_path)
    require(set(runs) == set(cohort_input.get("source_runs", {})), "mask_runs aliases differ")
    for alias, spec in cohort_input["source_runs"].items():
        safe_input_path(runs[alias], spec["manifest_relpath"])
    cohort = load_and_validate_cohort(cohort_path, runs)
    records = select_measurement_records(cohort)
    downstream = select_downstream_records(cohort)
    manifests = {alias: read_json(safe_input_path(root, cohort["source_runs"][alias]["manifest_relpath"]))
                 for alias, root in runs.items()}
    inputs = []
    for row in records:
        matches = [source for source in manifests[row["source_alias"]]["samples"] if source["stable_id"] == row["stable_id"]]
        require(len(matches) == 1, "stable_id must match exactly one source row")
        inputs.append(validate_artifacts(runs[row["source_alias"]], matches[0]))
    require(config_hashes == {"cohort": file_sha256(cohort_path), "protocol": file_sha256(protocol_path)},
            "config hash drift during input validation")
    for alias, root in runs.items():
        require(file_sha256(root / cohort["source_runs"][alias]["manifest_relpath"])
                == cohort["source_runs"][alias]["manifest_sha256"], "source manifest hash drift during input validation")
    # All input bytes have been decoded and validated before the first write.
    output.mkdir(parents=True, exist_ok=False)
    for folder in ("manifests", "metrics", "qc", "reports", "samples"):
        (output / folder).mkdir()
    result_rows, output_hashes, thumbnails = [], {}, []
    for row, (rgb, mask, provenance) in zip(records, inputs):
        metrics, eligible, dominant, lab = measure(rgb, mask)
        result = {**deepcopy(row), **metrics, "inputs": provenance}
        sample_dir = output / "samples" / f"{row['cohort_rank']:02d}_{row['sample_id']}"
        (sample_dir / "qc").mkdir(parents=True)
        paths = {"valid_mask": sample_dir / "valid_mask.png", "dominant_mask": sample_dir / "dominant_mask.png",
                 "sample_qc": sample_dir / "qc" / "sample_qc.png"}
        Image.fromarray(eligible.astype(np.uint8) * 255).save(paths["valid_mask"])
        Image.fromarray(dominant.astype(np.uint8) * 255).save(paths["dominant_mask"])
        qc = sample_qc(rgb, mask, eligible, dominant, lab, result)
        qc.save(paths["sample_qc"])
        # Summary tiles retain a readable sample identifier above the scaled sheet.
        tile = Image.new("RGB", (384, 336), "white")
        thumb = qc.resize((360, 315))
        tile.paste(thumb, (12, 21))
        ImageDraw.Draw(tile).text((4, 3), f"#{row['cohort_rank']} {row['stable_id']} {row['role']} {result['status']}", fill="black")
        thumbnails.append(tile)
        result["outputs"] = {key: path.relative_to(output).as_posix() for key, path in paths.items()}
        result["output_sha256"] = {key: file_sha256(path) for key, path in paths.items()}
        output_hashes.update({result["outputs"][key]: digest for key, digest in result["output_sha256"].items()})
        result_rows.append(result)
    contact = Image.new("RGB", (384 * 3, 336 * ((len(thumbnails) + 2) // 3)), "white")
    for index, tile in enumerate(thumbnails):
        contact.paste(tile, ((index % 3) * 384, (index // 3) * 336))
    contact.save(output / "qc" / "target_color_audit_contact_sheet.png")
    flat_rows = [flatten("", row) for row in result_rows]
    fields = list(dict.fromkeys(key for row in flat_rows for key in row))
    with (output / "metrics" / "target_color_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat_rows)
    summary = {"measurement_count": len(result_rows), "downstream_primary_count": len(downstream),
               "measured_count": sum(row["status"] == "MEASURED" for row in result_rows),
               "review_count": sum(row["status"] == "REVIEW" for row in result_rows),
               "pending_human_count": len(result_rows), "approved_count": 0}
    report = ("# Provisional target-color audit\n\n"
              "All measurements require human review. No downstream targets are approved.\n\n"
              f"Measured cohort: {len(result_rows)} active records; downstream candidate IDs: {len(downstream)} primary only.\n"
              f"Algorithm MEASURED: {summary['measured_count']}; REVIEW: {summary['review_count']}.\n\n"
              "L* <= 5 and L* >= 95 are provisionally excluded. No dominant-ratio, compactness, or valid-ratio gate is set.\n"
              "Dominant a,b medians are provisional targets; L is observed lightness, not fixed target-token lightness.\n"
              "Historical derived color-mask hashes are unavailable; current hashes are recorded without claiming historical comparison.\n"
              "Review raw/masked imagery, eligible area, cluster compactness and dominance in each full-resolution sample QC.\n"
              "The contact sheet is an overview; the manifest and CSV contain all cluster statistics.\n"
              "No recoloring, calibration, training, or approved target configuration is produced.\n")
    (output / "reports" / "01_target_color_audit.md").write_text(report, encoding="utf-8")
    (output / "README.md").write_text(report + "\nSee manifests/target_color_audit_manifest.json, metrics/target_color_audit.csv, and qc/.\n", encoding="utf-8")
    for relative in ("qc/target_color_audit_contact_sheet.png", "metrics/target_color_audit.csv", "README.md", "reports/01_target_color_audit.md"):
        output_hashes[relative] = file_sha256(output / relative)
    manifest = {"schema": "natural_image_target_color_audit/v1", "git_commit": commit,
                "protocol": {"path": str(protocol_path.resolve()), "sha256": config_hashes["protocol"], "parameters": protocol},
                "cohort": {"path": str(cohort_path.resolve()), "sha256": config_hashes["cohort"], "cohort_id": cohort["cohort_id"]},
                "source_runs": {alias: {**cohort["source_runs"][alias], "root": str(root),
                                        "current_manifest_sha256": cohort["source_runs"][alias]["manifest_sha256"]}
                                for alias, root in runs.items()},
                "versions": {"numpy": np.__version__, "Pillow": pillow_version}, "records": result_rows, "summary": summary,
                "downstream_candidate_ids": [row["stable_id"] for row in downstream],
                "qc_decision": "PENDING_HUMAN", "approved_for_downstream": False, "output_sha256": output_hashes}
    write_json(output / "manifests" / "target_color_audit_manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-config", required=True, type=Path)
    parser.add_argument("--protocol-config", required=True, type=Path)
    parser.add_argument("--mask-run", required=True, action="append", nargs=2, metavar=("ALIAS", "RUN_ROOT"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-git-commit", required=True)
    args = parser.parse_args(argv)
    try:
        result = run_audit(args.cohort_config, args.protocol_config, args.mask_run, args.output_dir, args.expected_git_commit)
    except (TargetColorAuditError, NaturalImageCohortError, OSError, subprocess.SubprocessError) as exc:
        parser.exit(2, f"target-color audit aborted: {exc}\n")
    print(json.dumps(result["summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
