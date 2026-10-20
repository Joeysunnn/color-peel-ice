"""Validate and visualize Subject C parameter interventions; never render or train."""

import argparse
import itertools
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from experiments.sd15_token_local_kv_v1.contract import read_json, sha256
from experiments.sd_native_comparison_v1.generate import write_json
from experiments.subject_albedo_pbr_c.source import load_source
from scripts.launch.colorpeel_run import git_info, resolve_run_dir
from src.methods.colorpeel_ice.natural_subject_recolor_pilot import recolor, hue_distance_degrees
from src.methods.colorpeel_ice import gamut_aware_lch_injection as gamut

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = Path(__file__).with_name("protocol.json")


def edit_albedo(albedo, mask, hue, qc_protocol):
    edited, metrics, _ = recolor(albedo, mask, mask.astype(np.uint16) * 257, hue)
    inside = mask == 255
    before = gamut.linear_rgb_to_lab(gamut._srgb_to_linear(albedo[inside]))
    after = gamut.linear_rgb_to_lab(gamut._srgb_to_linear(edited[inside]))
    chroma = np.hypot(before[:, 1], before[:, 2])
    support = (chroma >= qc_protocol["minimum_chromatic_support_C"]) & (
        np.hypot(after[:, 1], after[:, 2]) >= qc_protocol["minimum_chromatic_support_C"])
    measured_hue = np.degrees(np.arctan2(after[support, 2], after[support, 1])) % 360
    delta = np.abs(after[:, 0] - before[:, 0])
    metrics.update(delta_L_p95=float(np.percentile(delta, 95)),
                   delta_L_max=float(delta.max()), source_mean_chroma=float(chroma.mean()),
                   chromatic_support_fraction=float(support.mean()),
                   chromatic_hue_error_p95=float(np.percentile(hue_distance_degrees(measured_hue, hue), 95))
                   if support.any() else None)
    reasons = []
    if metrics["outside_mask_changed_pixel_count"] or metrics["rgb_clipping_pixel_count"]:
        reasons.append("outside-mask change or RGB clipping")
    if metrics["delta_L_p95"] > qc_protocol["max_delta_L_p95"]:
        reasons.append("Albedo lightness structure changed beyond quantization tolerance")
    if not support.any():
        reasons.append("no chromatic support; fixed-hue editing cannot color an achromatic source")
    metrics.update(status="rejected" if reasons else "accepted_parameter_edit", reasons=reasons,
                   structural_rgb_check="not applicable: this is an Albedo map, not rendered RGB",
                   intrinsic_validity="pending visual review; residual shading/highlights may remain")
    return edited, metrics


def parameter_map(original, mask, value, output):
    """Store exact float-map interventions separately from 8-bit display previews."""
    original = np.asarray(original, dtype=np.float32) / 255
    inside = mask == 255
    edited = original.copy()
    edited[inside] = value
    np.save(output.with_suffix(".npy"), edited, allow_pickle=False)
    preview = np.rint(edited * 255).astype(np.uint8)
    Image.fromarray(preview).save(output.with_suffix(".png"))
    saved = np.load(output.with_suffix(".npy"), allow_pickle=False)
    accepted = (saved.dtype == np.float32 and np.isfinite(saved).all()
                and saved.min() >= 0 and saved.max() <= 1
                and np.array_equal(saved[~inside], original[~inside])
                and np.all(saved[inside] == np.float32(value)))
    return {"path": str(output.with_suffix(".npy")), "sha256": sha256(output.with_suffix(".npy")),
            "preview": str(output.with_suffix(".png")), "preview_sha256": sha256(output.with_suffix(".png")),
            "encoding": "float32 linear; PNG preview only", "value": value,
            "preview_foreground_value": float(preview[inside][0] / 255),
            "qc": "accepted_parameter_edit" if accepted else "rejected"}


def renderer_readiness():
    return {"status": "missing_original_subject_recomposition", "rendered_rgb": 0,
            "training_ready": False,
            "available": ["verified original RGB and object alpha", "SuperMat A/M/R estimates",
                          "existing Blender Cycles PBR renderer for generic geometry"],
            "missing": ["original mailbox mesh/UV or validated same-view geometry and normals",
                        "image-space A/M/R correspondence to that geometry",
                        "matched camera intrinsics, projection and pose",
                        "estimated or explicitly calibrated fixed lighting and visibility",
                        "foreground compositing and shadow policy with fixed background",
                        "identity and material-response validation against the source"],
            "rgb_quality_gate": "not run; no RGB was rendered and no sample is training-eligible",
            "fallback": "none; neither proxy objects nor RGB recoloring are original-subject PBR"}


def run(source_fullmaps, output):
    from experiments.subject_albedo_pbr_c.review import write_review

    source_fullmaps, output = Path(source_fullmaps).resolve(), resolve_run_dir(Path(output))
    if (output.exists() or output.is_relative_to(ROOT) or output.is_relative_to(source_fullmaps)
            or source_fullmaps.is_relative_to(output)):
        raise ValueError("use a fresh external output outside source artifacts and the repository")
    revision, protocol = git_info(), read_json(PROTOCOL)
    paths, locks, source_record = load_source(source_fullmaps)
    locks[str(PROTOCOL)] = sha256(PROTOCOL)
    if any(output.is_relative_to(Path(path).parent) for path in locks):
        raise ValueError("output must be separate from all immutable source directories")
    (output / "source").mkdir(parents=True)
    write_json(output / "source_locks.json", locks)
    write_json(output / "source_receipt.json", source_record)
    write_json(output / "protocol.json", protocol)
    write_json(output / "renderer_readiness.json", renderer_readiness())
    state = {"status": "preparing", "git": revision, "training_started": False,
             "rendered_rgb": 0, "training_eligible_images": 0, "parameter_configurations": 0}
    write_json(output / "status.json", state)
    try:
        names = {"original": "original.jpg", "mask": "object_mask.png"}
        for key, path in paths.items():
            shutil.copyfile(path, output / "source" / names.get(key, f"{key}.png"))
        mask = np.asarray(Image.open(paths["mask"]).convert("L"))
        albedo = np.asarray(Image.open(paths["albedo"]).convert("RGB"))
        maps, quality = {}, {}
        for channel in ("albedo", "metallic", "roughness"):
            directory = output / "maps" / channel
            directory.mkdir(parents=True)
            maps[channel] = {}
            if channel == "albedo":
                for color, hue in protocol["colors"].items():
                    image, qc = edit_albedo(albedo, mask, hue, protocol["qc"])
                    path = directory / f"{color}.png"
                    Image.fromarray(image).save(path)
                    maps[channel][color] = {"path": str(path), "sha256": sha256(path),
                                            "encoding": protocol["albedo_encoding"]}
                    quality[color] = qc
            else:
                original = np.asarray(Image.open(paths[channel]).convert("L"))
                for name, value in protocol[channel].items():
                    maps[channel][name] = parameter_map(original, mask, value, directory / name)
        write_json(output / "albedo_qc.json", quality)
        rows = []
        for color, metallic, roughness in itertools.product(
                protocol["colors"], protocol["metallic"], protocol["roughness"]):
            selected = {"albedo": maps["albedo"][color], "metallic": maps["metallic"][metallic],
                        "roughness": maps["roughness"][roughness]}
            accepted = (quality[color]["status"] == "accepted_parameter_edit"
                        and all(selected[channel]["qc"] == "accepted_parameter_edit" for channel in ("metallic", "roughness")))
            material = "non-metallic" if metallic == "nonmetallic" else "metallic"
            caption = f"a photo of <S*> mailbox in {color} color with a {protocol['roughness_words'][roughness]} {material} surface"
            rows.append({"id": f"{color}__{metallic}__{roughness}", "color": color,
                "hue_degrees": protocol["colors"][color], "metallic_state": metallic,
                "metallic": protocol["metallic"][metallic], "roughness_state": roughness,
                "roughness": protocol["roughness"][roughness], "proposed_caption": caption,
                "maps": selected, "qc": "accepted_parameter_edit" if accepted else "rejected_parameter_edit",
                "render_status": "not_rendered", "rgb_image": None, "training_eligible": False})
        (output / "parameter_manifest.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        if len(rows) != 30 or len({row["id"] for row in rows}) != 30:
            raise ValueError("incomplete A x M x R parameter grid")
        state.update(parameter_configurations=len(rows),
                     accepted_parameter_edits=sum(row["qc"] == "accepted_parameter_edit" for row in rows),
                     rejected_parameter_edits=sum(row["qc"] == "rejected_parameter_edit" for row in rows))
        state["grids"] = write_review(output, rows)
        if (git_info() != revision or any(sha256(Path(path)) != digest for path, digest in locks.items())):
            raise ValueError("immutable source or checkout changed during parameter editing")
        state["status"] = "parameter_preview_verified_renderer_missing"
        if state["rejected_parameter_edits"]:
            state["status"] = "parameter_preview_has_rejected_edits_renderer_missing"
        # A file receipt allows later review to detect edits without touching old assets.
        write_json(output / "output_hashes.json", {
            str(path): sha256(path) for path in sorted(output.rglob("*"))
            if path.is_file() and path.name not in ("status.json", "output_hashes.json")})
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "status.json", state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-fullmaps-run", type=Path, default=read_json(PROTOCOL)["source_fullmaps_run"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.source_fullmaps_run, args.output)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
