"""Frozen, source-verified natural counterfactual plan; standard-library only."""

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

PARENT = Path(__file__).resolve().parents[1] / "texture_material_disentangled_v1/protocol.json"
PARENT_SHA256 = "3563eb6c263e9d4c4b6c57f834a789e3a7df1687cfc3d768fd033024e62cf116"
SOURCES = ("mailbox", "metal_spoon", "wood_spoon")
NUISANCE_MATERIALS = ("smooth_dielectric", "rough_metal")
ROOT = Path(__file__).resolve().parents[2]


def git_info():
    from scripts.launch.colorpeel_run import git_info as read_git_info
    return read_git_info()


def resolve_run_dir(path):
    from scripts.launch.colorpeel_run import resolve_run_dir as resolve
    return resolve(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol_sha256(protocol):
    return hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf8")).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def build_protocol(extraction_root):
    extraction_root = Path(extraction_root).resolve()
    protocol = _read(PARENT)
    if protocol_sha256(protocol) != PARENT_SHA256:
        raise ValueError("Frozen Phase 1 protocol changed")
    manifest_path = extraction_root / "manifest.json"
    manifest = _read(manifest_path)
    if (manifest.get("schema") != "natural_texture_material/extraction_v2"
            or manifest.get("phase") != 2 or manifest.get("status") != "complete"
            or manifest.get("phase1_protocol_sha256") != PARENT_SHA256
            or set(manifest["materials"]) != set(SOURCES)):
        raise ValueError("Expected the completed three-source extraction_v2")
    locks_path, extraction_protocol = extraction_root / "source_locks.json", extraction_root / "protocol.json"
    if (sha256(locks_path) != manifest["source_locks_sha256"]
            or sha256(extraction_protocol) != manifest["protocol_sha256"]
            or sha256(extraction_root / "overview.png") != manifest["overview_sha256"]):
        raise ValueError("Accepted extraction output receipt changed")
    extracted = _read(extraction_protocol)
    if (extracted.get("phase") != 2 or extracted.get("stage") != "extraction_only"
            or extracted.get("phase1_protocol_sha256") != PARENT_SHA256
            or extracted["texture_extraction"].get("patch_sampling", {}).get("no_inpainting") is not True):
        raise ValueError("Expected accepted all-valid texture extraction protocol")
    locks = _read(locks_path)
    if not isinstance(locks, dict) or not locks:
        raise ValueError("Extraction must retain its upstream source locks")
    for path, digest in locks.items():
        if not Path(path).is_absolute() or sha256(path) != digest:
            raise ValueError(f"Upstream extraction source changed: {path}")
    for path in (manifest_path, locks_path, extraction_protocol, extraction_root / "overview.png"):
        locks[str(path)] = sha256(path)
    sources, textures = {}, {}
    materials = {name: spec for name, spec in protocol["materials"].items() if spec["role"] == "nuisance"}
    expected_files = {"input.png", "mask.png", "albedo.png", "roughness.png", "metallic.png",
                      "gray_full.png", "luminance.png", "texture_inpaint_audit.png", "texture.png", "material.json"}
    for name in SOURCES:
        folder, record = extraction_root / name, manifest["materials"][name]
        if set(record["files"]) != expected_files:
            raise ValueError(f"Extraction output membership changed: {name}")
        sampling = record["texture_sampling"]
        if (sampling.get("no_inpainting") is not True
                or sampling.get("method") != "maximum_area_all_valid_axis_aligned_rectangle"
                or record["texture_statistics"].get("patch_sampling") != sampling):
            raise ValueError(f"Texture must be the accepted all-valid interior crop: {name}")
        for filename, receipt in record["files"].items():
            path = folder / filename
            if Path(receipt["path"]).resolve() != path or sha256(path) != receipt["sha256"]:
                raise ValueError(f"Accepted extraction file changed: {name}/{filename}")
            locks[str(path)] = receipt["sha256"]
        if _read(folder / "extraction.json") != record:
            raise ValueError(f"Source extraction record disagrees with manifest: {name}")
        locks[str(folder / "extraction.json")] = sha256(folder / "extraction.json")
        material = _read(folder / "material.json")
        if any(record.get(key) != value for key, value in material.items()):
            raise ValueError(f"Scalar material disagrees with accepted extraction: {name}")
        for channel in ("roughness", "metallic"):
            value = material[channel]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid scalar {channel}: {name}")
        sources[name] = {"input": str(folder / "input.png"), "texture": str(folder / "texture.png"),
                         "material": str(folder / "material.json"), "roughness": material["roughness"],
                         "metallic": material["metallic"], "texture_sampling": sampling,
                         "semantic_status": material["semantic_status"]}
        textures[name + "_texture"] = {"role": "target", "source_id": name,
            "path": str(folder / "texture.png"), "sha256": locks[str(folder / "texture.png")]}
        materials[name + "_material"] = {"role": "target", "source_id": name,
            "roughness": material["roughness"], "metallic": material["metallic"]}
    protocol.update(study="natural_texture_material_v1", phase=2, parent_protocol_sha256=PARENT_SHA256,
                    source_extraction=str(extraction_root), extraction_manifest_sha256=sha256(manifest_path),
                    source_locks=locks, sources=sources, textures=textures, materials=materials,
                    representation={"material": {"albedo_mode": "uniform", "albedo_level": 0.5,
                        "texture_image_nodes": 0, "normal_bump_displacement": False},
                        "texture": "accepted interior texture.png only; fixed pattern, varied shape/color/material/light"})
    protocol["dataset"].update(expected_train_images=72, expected_per_concept={"texture": 16, "material": 8})
    protocol["training"].update(steps=1000, checkpoint_steps=[500, 750, 1000])
    protocol["phases"]["phase2"] = "accepted extraction; counterfactual dataset only; no training launched"
    return protocol


def records(protocol):
    dataset, rows = protocol["dataset"], []
    nuisances = NUISANCE_MATERIALS
    for branch in ("texture", "material"):
        conditions = list(itertools.product(dataset["train_shapes"], dataset["train_colors"],
                          dataset["train_lights"], nuisances if branch == "texture" else [None]))
        for source in SOURCES:
            concept = source + "_" + branch
            for index, (shape, color, light, nuisance) in enumerate(conditions):
                identifier = f"train__{branch}__{source}__{shape}__{color}__{light}"
                row = {"id": identifier + ("__" + nuisance if nuisance else ""), "split": "train",
                       "branch": branch, "source_id": source, "concept_id": concept,
                       "material_id": nuisance or concept, "shape": shape, "color": color,
                       "light": light, "seed": protocol["seed"] + index}
                if branch == "texture":
                    texture = protocol["textures"][concept]
                    row.update(texture_id=concept, texture_path=texture["path"], texture_sha256=texture["sha256"])
                else:
                    row.update(albedo_mode="uniform", albedo_level=0.5)
                rows.append(row)
    return rows


def caption(row, protocol):
    light = protocol["dataset"]["light_phrases"][row["light"]]
    prefix = f'a photo of a {row["color"]} {row["shape"]}'
    if row["branch"] == "texture":
        material = protocol["materials"][row["material_id"]]["literal"]
        return f"{prefix} with <T*> texture made of {material} under {light}"
    return f"{prefix} made of <M*> under {light}"


def make_plan(extraction, output):
    extraction, output = Path(extraction).resolve(), resolve_run_dir(Path(output))
    if output.exists():
        raise FileExistsError(output)
    if output == ROOT or output.is_relative_to(ROOT) or output == extraction or output.is_relative_to(extraction):
        raise ValueError("Use a new external output separate from extraction artifacts")
    revision, protocol = git_info(), build_protocol(extraction)
    plan = {"status": "complete", "phase": 2, "stage": "counterfactual_dataset_only", "git": revision,
            "protocol_path": str(output / "protocol.json"), "protocol_sha256": protocol_sha256(protocol),
            "source_locks": protocol["source_locks"], "train_images": 72, "eval_images": 0,
            "training_launched": False, "records": records(protocol)}
    if git_info() != revision or any(sha256(path) != digest for path, digest in protocol["source_locks"].items()):
        raise ValueError("Checkout or extraction inputs changed during planning")
    output.mkdir(parents=True)
    for name, value in (("protocol.json", protocol), ("plan.json", plan)):
        (output / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf8")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extraction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = make_plan(args.extraction, args.output)
    print(json.dumps({"status": result["status"], "train_images": result["train_images"]}))


if __name__ == "__main__":
    main()
