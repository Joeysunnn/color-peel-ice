"""Fresh SuperMat/output-depth QC with immutable, resumable estimate receipts."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
from PIL import Image

from experiments.zest_material_counterfactual_v1 import assets, quality
from experiments.zest_material_counterfactual_v1.render_bank import ROOT, sha256, write


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def local(run, name):
    path = (run / name).resolve()
    if not path.is_relative_to(run):
        raise ValueError(f"Artifact escapes run directory: {name}")
    return path


def fresh(parent, prefix):
    parent.mkdir(parents=True, exist_ok=True)
    number = 1
    while (parent / f"{prefix}{number:03d}").exists():
        number += 1
    result = parent / f"{prefix}{number:03d}"
    result.mkdir()
    return result


def load_mask(path, size=(512, 512)):
    with Image.open(path) as image:
        return np.asarray(image.convert("L").resize(size, Image.Resampling.NEAREST)) >= 128


def inspect(run, plan, receipt, supermat_root=None):
    assets.verify(receipt)
    prepared = read(run / "prepare_receipt.json")
    for path, digest in plan["input_hashes"].items():
        if sha256(path) != digest:
            raise ValueError(f"Immutable preparation source changed: {path}")
    for relative, digest in prepared["output_hashes"].items():
        if sha256(local(run, relative)) != digest:
            raise ValueError(f"Prepared run input changed: {relative}")
    config = dict(plan["supermat"])
    if supermat_root is not None:
        config["root"] = str(Path(supermat_root).resolve())
    assets.source_state(config["root"], config["official_commit"])
    if sha256(config["checkpoint"]) != config["checkpoint_sha256"]:
        raise ValueError("SuperMat checkpoint changed")
    files = [Path(__file__), Path(quality.__file__),
             ROOT / "experiments/natural_material_supermat_v1/run_supermat_batch.py",
             Path(config["root"]) / "inference_supermat.py",
             Path(config["base_model"]) / "model_index.json"]
    return {"plan_sha256": sha256(run / "plan.json"), "supermat": config,
            "script_hashes": {str(path): sha256(path) for path in files},
            "asset_hashes": receipt["sha256"], "depth_model": receipt["paths"]["depth"],
            "preprocessing": "fresh original/output RGBA512, identical frozen valid alpha; independent DPT RGB1024 inputs to float512 predictions"}


def job(run, key, rgb_path, depth_image, alpha_path):
    if re.fullmatch(r"[A-Za-z0-9_:.-]+", key) is None:
        raise ValueError(f"Unsafe estimation ID: {key}")
    return {"id": key, "rgb_source": rgb_path.relative_to(run).as_posix(),
            "rgb_source_sha256": sha256(rgb_path),
            "depth_image": depth_image.relative_to(run).as_posix(),
            "depth_image_sha256": sha256(depth_image),
            "alpha_source": alpha_path.relative_to(run).as_posix(),
            "alpha_source_sha256": sha256(alpha_path)}


def validate_cache(run, cache, jobs):
    for key, entry in cache["records"].items():
        for name, digest in entry.get("hashes", {}).items():
            if sha256(local(run, name)) != digest:
                raise ValueError(f"Cached estimate changed: {key}/{name}")
        if key in jobs and entry["input"] != jobs[key]:
            raise ValueError(f"Cannot resume changed RGB/alpha/depth source: {key}")


def estimate_materials(run, jobs, cache, config, env):
    missing = [item for key, item in jobs.items() if "maps" not in cache["records"].get(key, {})]
    for start in range(0, len(missing), 8):
        batch = fresh(run / "estimates", "supermat_batch")
        inputs = batch / "inputs"
        inputs.mkdir()
        names = {}
        for index, item in enumerate(missing[start:start + 8]):
            name = f"item{index:02d}"
            image_path = local(run, item["rgb_source"])
            with Image.open(image_path) as image:
                rgb = image.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
            rgba = rgb.convert("RGBA")
            alpha = Image.fromarray(load_mask(local(run, item["alpha_source"])).astype(np.uint8) * 255)
            rgba.putalpha(alpha)
            rgba.save(inputs / (name + ".png"))
            names[name] = item
        command = [sys.executable, "-B", "-m", "experiments.natural_material_supermat_v1.run_supermat_batch",
                   "--python", config["python"], "--supermat-root", config["root"],
                   "--checkpoint", config["checkpoint"], "--base-model", config["base_model"],
                   "--input-dir", str(inputs), "--output-dir", str(batch / "outputs"),
                   "--device", "cuda:0", "--seed", "42"]
        write(batch / "request.json", {"command": command, "inputs": names, "status": "started"})
        with (batch / "wrapper.log").open("w", encoding="utf8") as log:
            status = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        if status.returncode:
            raise RuntimeError(f"SuperMat failed; preserved attempt {batch}, exit {status.returncode}")
        manifest = read(batch / "outputs/inference_manifest.json")
        if manifest["status"] != "complete" or manifest["seed"] != 42 or manifest["image_size"] != 512:
            raise ValueError(f"Incomplete SuperMat receipt: {batch}")
        records = {Path(row["input"]).stem: row for row in manifest["outputs"]}
        if set(records) != set(names):
            raise ValueError("SuperMat output batch IDs differ")
        for name, item in names.items():
            maps, hashes = {}, {}
            for channel in ("albedo", "roughness", "metallic"):
                record = records[name]["maps"][channel]
                path = Path(record["path"]).resolve()
                relative = path.relative_to(run).as_posix()
                if sha256(path) != record["sha256"]:
                    raise ValueError("SuperMat output map hash differs")
                maps[channel], hashes[relative] = relative, record["sha256"]
            input_path = inputs / (name + ".png")
            if sha256(input_path) != records[name]["input_sha256"]:
                raise ValueError("SuperMat RGBA input changed")
            input_relative = input_path.relative_to(run).as_posix()
            manifest_relative = (batch / "outputs/inference_manifest.json").relative_to(run).as_posix()
            hashes.update({input_relative: sha256(input_path), manifest_relative: sha256(run / manifest_relative)})
            entry = cache["records"].setdefault(item["id"], {"input": item, "hashes": {}})
            entry.update(maps=maps, rgba_input=input_relative, supermat_manifest=manifest_relative)
            entry["hashes"].update(hashes)
        write(run / "estimates_manifest.json", cache)


def estimate_depths(run, jobs, cache, receipt):
    missing = [item for key, item in jobs.items() if "depth" not in cache["records"].get(key, {})]
    if not missing:
        return
    import torch
    from transformers import DPTForDepthEstimation, DPTImageProcessor

    model_path = receipt["paths"]["depth"]
    processor = DPTImageProcessor.from_pretrained(model_path, local_files_only=True)
    model = DPTForDepthEstimation.from_pretrained(model_path, local_files_only=True).eval().to("cuda:0")
    attempt = fresh(run / "estimates", "depth_attempt")
    for index, item in enumerate(missing):
        with Image.open(local(run, item["depth_image"])) as image:
            if image.size != (1024, 1024):
                raise ValueError("Independent QC depth expects1024 RGB input")
            pixels = processor(images=image.convert("RGB"), return_tensors="pt").pixel_values.to("cuda:0")
        with torch.inference_mode():
            prediction = model(pixel_values=pixels).predicted_depth
            prediction = torch.nn.functional.interpolate(prediction.unsqueeze(1), size=(512, 512),
                mode="bicubic", align_corners=False)[0, 0].float().cpu().numpy()
        if not np.isfinite(prediction).all():
            raise ValueError("Nonfinite independent QC depth")
        path = attempt / f"item{index:04d}.npy"
        np.save(path, prediction, allow_pickle=False)
        relative = path.relative_to(run).as_posix()
        entry = cache["records"].setdefault(item["id"], {"input": item, "hashes": {}})
        entry["depth"] = relative
        entry["hashes"][relative] = sha256(path)
        write(run / "estimates_manifest.json", cache)
    del model
    torch.cuda.empty_cache()


def load_maps(run, estimate):
    result = {}
    for channel, relative in estimate["maps"].items():
        with Image.open(local(run, relative)) as image:
            if image.size != (512, 512):
                raise ValueError("SuperMat map dimensions differ from512")
            if channel != "albedo" and image.mode == "RGB":
                pixels = np.asarray(image)
                if not np.all(pixels == pixels[:, :, :1]):
                    raise ValueError("SuperMat scalar map has unequal RGB channels")
            result[channel] = np.asarray(image.convert("RGB") if channel == "albedo" else image.convert("L"))
        if channel != "albedo":
            result[channel] = result[channel].astype(float) / 255
    return result


def rgb512(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)).copy()


def run_assessment(run, asset_root, gpu=2, stage="all", resume=False, manual_review=None, supermat_root=None):
    run = Path(run).resolve()
    plan, manifest = read(run / "plan.json"), read(run / "generation_manifest.json")
    receipt = assets.read(asset_root)
    protocol = inspect(run, plan, receipt, supermat_root)
    cache_path = run / "estimates_manifest.json"
    if cache_path.exists() and not resume:
        raise FileExistsError("Estimation cache exists; use --resume to validate and reuse it")
    cache = read(cache_path) if cache_path.exists() else {"protocol": protocol, "records": {}}
    if cache["protocol"] != protocol:
        raise ValueError("Assessment scripts, plan, source models or extraction protocol changed; use a new run")
    subjects = {subject["id"]: subject for subject in plan["subjects"]}
    all_records = manifest["records"]
    if len({row["id"] for row in all_records}) != len(all_records):
        raise ValueError("Duplicate generated IDs")
    review_bytes = Path(manual_review).read_bytes() if manual_review else None
    approvals = json.loads(review_bytes) if manual_review else {}
    review_hash = hashlib.sha256(review_bytes).hexdigest() if manual_review else None
    if not isinstance(approvals, dict) or any(key not in {r["id"] for r in all_records} or not isinstance(value, bool)
                                             for key, value in approvals.items()):
        raise ValueError("Manual review must map existing case IDs to explicit boolean approvals")
    selected = [row for row in all_records if stage == "all" or row["stage"] == stage or row["stage"] == "identity"]
    if not selected:
        raise ValueError("No generated records for selected assessment stage")
    jobs = {}
    for row in selected:
        raw = local(run, row["raw_image"])
        if sha256(raw) != row["raw_sha256"]:
            raise ValueError(f"Generated image changed: {row['id']}")
        subject = subjects[row["subject_id"]]
        files = subject["source_files"]
        for key, rgb_file, depth_file in (("original:" + subject["id"], files["image"], files["target"]),
                                           ("generated:" + row["id"], row["raw_image"], row["raw_image"])):
            jobs[key] = job(run, key, local(run, rgb_file), local(run, depth_file), local(run, files["valid_mask"]))
    validate_cache(run, cache, jobs)
    write(cache_path, cache)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    estimate_materials(run, jobs, cache, protocol["supermat"], env)
    estimate_depths(run, jobs, cache, receipt)
    validate_cache(run, cache, jobs)
    records = []
    for row in selected:
        files = subjects[row["subject_id"]]["source_files"]
        original_estimate = cache["records"]["original:" + row["subject_id"]]
        generated_estimate = cache["records"]["generated:" + row["id"]]
        thresholds = dict(plan["thresholds"], require_manual_review=True, manual_review_passed=approvals.get(row["id"], False))
        if row["stage"] == "grid" and row.get("reference_target"):
            thresholds.update({"target_" + channel: row["reference_target"][channel] for channel in ("metallic", "roughness")})
        else:
            thresholds.pop("target_metallic", None)
            thresholds.pop("target_roughness", None)
        assessment = quality.evaluate(rgb512(local(run, files["image"])), rgb512(local(run, row["raw_image"])),
            load_mask(local(run, files["object_mask"])), load_mask(local(run, files["edit_mask"])),
            load_mask(local(run, files["protected_mask"])), load_mask(local(run, files["valid_mask"])),
            load_mask(local(run, files["material_region"])), load_maps(run, original_estimate), load_maps(run, generated_estimate),
            np.load(local(run, original_estimate["depth"]), allow_pickle=False),
            np.load(local(run, generated_estimate["depth"]), allow_pickle=False), thresholds)
        records.append({**row, **assessment, "source_original_estimate": original_estimate,
                        "generated_estimate": generated_estimate, "thresholds": thresholds})
    gates = {}
    for subject_id in subjects:
        baseline = [row for row in records if row["subject_id"] == subject_id and row["stage"] == "identity"
                    and row["variant"] == "baseline" and row["reference_id"] == "self"]
        statuses = {row["status"] for row in baseline}
        gates[subject_id] = ("invalid" if "invalid" in statuses else "rejected" if "rejected" in statuses
                              else "accepted" if baseline and statuses == {"accepted"} else "pending_manual_review")
    by_id = {row["id"]: row for row in records}
    dataset = []
    for row in all_records:
        if row["stage"] != "grid":
            continue
        assessed = by_id.get(row["id"])
        eligible = bool(assessed and assessed["training_eligible"] and gates[row["subject_id"]] == "accepted")
        dataset.append({**row, "quality_status": assessed["status"] if assessed else "not_assessed",
                        "identity_gate": gates[row["subject_id"]], "training_eligible": eligible})
    if inspect(run, plan, receipt, supermat_root) != protocol:
        raise ValueError("Assessment scripts or extraction protocol changed during execution")
    validate_cache(run, cache, jobs)
    output = fresh(run / "quality", "assessment")
    report = {"records": records, "identity_gate": gates, "stage": stage,
              "interpretation": "Reference targets are known renderer inputs; generated SuperMat estimates are approximate and not those target parameters.",
              "training_started": False, "estimates_manifest_sha256": sha256(cache_path)}
    if manual_review and sha256(manual_review) != review_hash:
        raise ValueError("Manual review file changed during assessment")
    report["manual_review"] = {"path": str(Path(manual_review).resolve()), "sha256": review_hash} if manual_review else None
    write(output / "report.json", report)
    dataset_record = {"records": dataset, "assessment": (output / "report.json").relative_to(run).as_posix(),
                      "training_eligible_images": sum(row["training_eligible"] for row in dataset), "training_started": False}
    write(output / "dataset_manifest.json", dataset_record)
    write(run / "dataset_manifest.json", dataset_record)
    write(run / "quality/latest.json", {"report": (output / "report.json").relative_to(run).as_posix()})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--stage", choices=("identity", "grid", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--manual-review", type=Path, help="User-authored JSON: case ID -> boolean approval; thresholds still apply")
    parser.add_argument("--supermat-root", type=Path, help="Independent clean checkout at the same pinned commit; recorded in the immutable QC protocol")
    args = parser.parse_args()
    result = run_assessment(args.run, args.assets, args.gpu, args.stage, args.resume, args.manual_review, args.supermat_root)
    print(json.dumps({"records": len(result["records"]), "identity_gate": result["identity_gate"]}))


if __name__ == "__main__":
    main()
