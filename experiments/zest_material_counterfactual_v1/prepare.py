"""Freeze ZeST subject inputs and a separate PBR reference-bank plan; no inference."""

import argparse
import json
from pathlib import Path
import re
import shutil

import cv2
import numpy as np
from PIL import Image

from experiments.zest_material_counterfactual_v1.official import grayscale_init
from experiments.zest_material_counterfactual_v1.render_bank import ROOT, sha256, write


def resolve(path):
    path = Path(path)
    return (path if path.is_absolute() else ROOT / path).resolve()


def rgb(path):
    with Image.open(path) as image:
        if image.size != (512, 512) or image.mode != "RGB":
            raise ValueError(f"Expected 512px RGB: {path}")
        return np.asarray(image).copy()


def mask(path, allow_empty=False):
    with Image.open(path) as image:
        pixels = np.asarray(image)
        if image.size != (512, 512) or image.mode != "L" or not set(np.unique(pixels)).issubset({0, 255}):
            raise ValueError(f"Expected binary 512px L mask: {path}")
    result = pixels == 255
    if not allow_empty and not result.any():
        raise ValueError(f"Empty mask: {path}")
    return result


def linear(srgb):
    srgb = np.asarray(srgb, dtype=np.float64) / 255
    return np.where(srgb <= .04045, srgb / 12.92, ((srgb + .055) / 1.055) ** 2.4)


def encode(value):
    value = np.clip(value, 0, 1)
    return np.rint(np.where(value <= .0031308, value * 12.92, 1.055 * value ** (1 / 2.4) - .055) * 255).astype(np.uint8)


def color_initialization(target, gray, albedo, valid_edit):
    """Retain source linear luminance with estimated Albedo chromaticity in valid edit pixels."""
    original, intrinsic = linear(target), linear(albedo)
    weights = np.array([.2126, .7152, .0722])
    luminance, intrinsic_luminance = original @ weights, intrinsic @ weights
    chromaticity = np.divide(intrinsic, intrinsic_luminance[:, :, None],
                             out=np.ones_like(intrinsic), where=intrinsic_luminance[:, :, None] > 1e-8)
    candidate = luminance[:, :, None] * chromaticity
    result = np.asarray(gray).copy()
    result[valid_edit] = encode(candidate)[valid_edit]
    clipped = np.any((candidate < 0) | (candidate > 1), axis=2)
    return Image.fromarray(result), {"method": "linear_luminance_original_times_estimated_Albedo_chromaticity",
        "valid_edit_pixels": int(valid_edit.sum()), "clipped_pixels": int((clipped & valid_edit).sum()),
        "clipped_fraction": float(clipped[valid_edit].mean()),
        "near_black_Albedo_pixels": int(((intrinsic_luminance <= 1e-8) & valid_edit).sum()),
        "outside_valid_edit_changed_pixels": int(np.any(result != np.asarray(gray), axis=2)[~valid_edit].sum()),
        "interpretation": "Initialization adaptation only; no generated-RGB overlay or exact Albedo guarantee"}


def inspect_subject(subject):
    identifier = subject["id"]
    if not isinstance(identifier, str) or re.fullmatch(r"[A-Za-z0-9_-]+", identifier) is None:
        raise ValueError(f"Unsafe subject ID: {identifier!r}")
    paths = {key: resolve(subject[key]) for key in ("image", "object_mask", "valid_mask", "material_region", "supermat_input")}
    paths.update({"original_" + key: resolve(subject["original_maps"][key]) for key in ("albedo", "roughness", "metallic")})
    if subject.get("protected_mask"):
        paths["protected_mask"] = resolve(subject["protected_mask"])
    if subject.get("control_depth"):
        paths["control_depth"] = resolve(subject["control_depth"])
    locks = {str(path): sha256(path) for path in paths.values()}
    for path, expected in subject.get("expected_sha256", {}).items():
        path = resolve(path)
        if sha256(path) != expected:
            raise ValueError(f"Expected immutable source hash differs: {path}")
        locks[str(path)] = expected
    original, albedo = rgb(paths["image"]), rgb(paths["original_albedo"])
    object_mask, valid = mask(paths["object_mask"]), mask(paths["valid_mask"])
    region = mask(paths["material_region"])
    protected = mask(paths["protected_mask"], True) if "protected_mask" in paths else np.zeros(valid.shape, dtype=bool)
    if np.any(valid & ~object_mask) or np.any(region & ~valid):
        raise ValueError("Original valid alpha must be inside object mask; material region must be inside valid alpha")
    with Image.open(paths["supermat_input"]) as image:
        if (image.size != (512, 512) or image.mode != "RGBA"
                or not np.array_equal(np.asarray(image)[:, :, :3], original)
                or not np.array_equal(np.asarray(image)[:, :, 3], valid.astype(np.uint8) * 255)):
            raise ValueError("Original SuperMat RGBA must exactly match source RGB and valid alpha")
    effective = object_mask & ~protected
    measurement = region & valid & effective
    if measurement.sum() < 1000:
        raise ValueError("Material measurement needs at least 1000 valid, unprotected pixels")
    estimates = {}
    for channel in ("roughness", "metallic"):
        with Image.open(paths["original_" + channel]) as image:
            pixels = np.asarray(image)
            if (image.size != (512, 512) or image.mode not in ("L", "RGB")
                    or image.mode == "RGB" and not np.all(pixels == pixels[:, :, :1])):
                raise ValueError(f"Expected scalar 512px 8bit {channel} map")
            pixels = pixels if pixels.ndim == 2 else pixels[:, :, 0]
            estimates["estimated_" + channel] = float(np.median(pixels[measurement].astype(np.float64) / 255))
    if "control_depth" in paths:
        with Image.open(paths["control_depth"]) as image:
            if image.size != (1024, 1024) or image.mode not in ("L", "RGB"):
                raise ValueError("Frozen control depth must be 1024px L/RGB")
    estimates["albedo_linear_rgb"] = np.median(linear(albedo)[measurement], axis=0).tolist()
    return paths, locks, original, albedo, effective, valid, protected, estimates


def prepare(config_path, output, resume=False):
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    if output == ROOT or output.is_relative_to(ROOT) or ROOT.is_relative_to(output):
        raise ValueError("Use a fresh external experiment directory")
    config = json.loads(config_path.read_text(encoding="utf8"))
    subjects = config["subjects"]
    ids = [subject["id"] for subject in subjects]
    if len(set(ids)) != len(ids) or not ids or config["bank_source"] not in ids:
        raise ValueError("Require unique subjects and a valid bank_source")
    inspected = [inspect_subject(subject) for subject in subjects]
    locks = {str(config_path): sha256(config_path)}
    for record in inspected:
        locks.update(record[1])
    receipt_path = output / "prepare_receipt.json"
    if output.exists():
        if not resume:
            raise FileExistsError(f"Use a fresh output or --resume with identical inputs: {output}")
        if not receipt_path.is_file():
            raise ValueError("Preparation is incomplete; preserve it and choose a new output directory")
        receipt = json.loads(receipt_path.read_text(encoding="utf8"))
        if receipt["input_hashes"] != locks:
            raise ValueError("Cannot resume preparation with changed config or source files")
        for relative, digest in receipt["output_hashes"].items():
            path = (output / relative).resolve()
            if not path.is_relative_to(output) or sha256(path) != digest:
                raise ValueError(f"Prepared artifact changed: {relative}")
        return json.loads((output / "plan.json").read_text(encoding="utf8"))
    if any(output == Path(path) or output.is_relative_to(Path(path).parent) for path in locks if Path(path) != config_path):
        raise ValueError("Do not place new output inside an existing source-asset directory")
    output.mkdir(parents=True)
    shutil.copyfile(config_path, output / "config.json")
    plan = {"schema": "zest_material_counterfactual/prepared_v1", "subjects": [], "bank_plan": "bank_plan.json",
            "config": "config.json", "inference": config["inference"], "variants": config["variants"],
            "thresholds": config["thresholds"], "supermat": config["supermat"],
            "training_started": False, "input_hashes": locks}
    for subject, inspected_subject in zip(subjects, inspected):
        paths, subject_locks, original, albedo, effective, valid, protected, estimates = inspected_subject
        folder = output / "source" / subject["id"]
        folder.mkdir(parents=True)
        files = {}
        for key in ("image", "object_mask", "valid_mask", "material_region", "original_albedo", "original_roughness", "original_metallic"):
            destination = folder / f"{key}.png"
            if key == "image":
                Image.fromarray(original).save(destination)
            else:
                shutil.copyfile(paths[key], destination)
            files[key] = destination.relative_to(output).as_posix()
        Image.fromarray(protected.astype(np.uint8) * 255).save(folder / "protected_mask.png")
        target = Image.fromarray(original).resize((1024, 1024), Image.Resampling.BICUBIC)
        edit_mask = Image.fromarray(effective.astype(np.uint8) * 255).convert("RGB").resize(target.size, Image.Resampling.BICUBIC)
        protected1024 = np.asarray(Image.fromarray(protected).resize(target.size, Image.Resampling.NEAREST))
        edit_pixels = np.asarray(edit_mask).copy()
        edit_pixels[protected1024] = 0
        edit_mask = Image.fromarray(edit_pixels)
        gray = grayscale_init(target, edit_mask)
        support = np.asarray(Image.fromarray(valid & effective).resize(target.size, Image.Resampling.NEAREST))
        albedo1024 = Image.fromarray(albedo).resize(target.size, Image.Resampling.BICUBIC)
        colored, stats = color_initialization(target, gray, albedo1024, support)
        # Canny sees estimated Albedo luminance, never source RGB highlights.
        edge_gray = np.asarray(albedo1024.convert("L"))
        edges = cv2.Canny(edge_gray, 100, 200)
        edges[~support] = 0
        for key, image in (("target", target), ("edit_mask", edit_mask), ("initialization_gray", gray),
                           ("initialization_color", colored), ("albedo_edges", Image.fromarray(edges).convert("RGB"))):
            image.save(folder / f"{key}.png")
            files[key] = (folder / f"{key}.png").relative_to(output).as_posix()
        files["protected_mask"] = (folder / "protected_mask.png").relative_to(output).as_posix()
        if "control_depth" in paths:
            destination = folder / "control_depth.png"
            shutil.copyfile(paths["control_depth"], destination)
            files["control_depth"] = destination.relative_to(output).as_posix()
        plan["subjects"].append({"id": subject["id"], "source_files": files, **estimates,
            "color_init_stats": stats, "input_hashes": subject_locks,
            "edge_policy": "Canny(100,200) from estimated Albedo grayscale, zero outside valid effective edit domain",
            "mask_policy": "existing object minus protected; RGB BICUBIC1024 resize; nearest protected pixels forced zero",
            "control_depth_policy": "adapted ZeST frozen-existing-DPT; not an exact official DPT preprocessing reproduction" if "control_depth" in paths else "not supplied",
            "estimation_policy": "original reviewed material region intersect valid effective edit; SuperMat approximate, not PBR ground truth"})
    source = next(subject for subject in plan["subjects"] if subject["id"] == config["bank_source"])
    bank_plan = {"profile": config["renderer"]["profile"],
                 "renderer": {key: value for key, value in config["renderer"].items() if key != "profile"},
                 "albedo_linear_rgb": source["albedo_linear_rgb"],
                 "records": [{"id": f"M{m:.1f}_R{r:.1f}".replace(".", "p"), "metallic": m, "roughness": r}
                             for m in (0., .5, 1.) for r in (.2, .5, .8)] + [
                     {"id": "estimated_original", "metallic": source["estimated_metallic"], "roughness": source["estimated_roughness"]}]}
    write(output / "bank_plan.json", bank_plan)
    write(output / "plan.json", plan)
    if any(sha256(path) != digest for path, digest in locks.items()):
        raise ValueError("Immutable sources changed during preparation; preserve output and choose a new run")
    hashes = {path.relative_to(output).as_posix(): sha256(path) for path in output.rglob("*") if path.is_file()}
    write(receipt_path, {"status": "complete", "input_hashes": locks, "output_hashes": hashes})
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = prepare(args.config, args.output, args.resume)
    print(json.dumps({"status": "prepared", "subjects": [s["id"] for s in result["subjects"]]}))


if __name__ == "__main__":
    main()
