"""Single photograph -> object instances -> independent four-branch initializers.

This source stage does not synthesize training datasets or launch training.
Qwen and SAM run sequentially in their existing, separate environments.
"""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from .vlm_initialization import REVISION, sha256, write_json

ROOT = Path(__file__).resolve().parents[3]
ENTRY = ROOT / "scripts/methods/colorpeel_ice/prepare_from_image.py"
ENVIRONMENT = {"PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1",
               "TRANSFORMERS_OFFLINE": "1", "DIFFUSERS_OFFLINE": "1"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def check_locks(locks):
    for path, digest in locks.items():
        if sha256(path) != digest:
            raise ValueError(f"Source changed: {path}")


def merge_locks(target, incoming):
    if any(path in target and target[path] != digest for path, digest in incoming.items()):
        raise ValueError("Conflicting source hashes")
    check_locks(incoming)
    target.update(incoming)


def run_stage(command, log, gpu):
    from experiments.sd21_qwen_initializers_v1.run import require_idle
    require_idle((gpu,))
    with Path(log).open("x", encoding="utf8") as stream:
        subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                       stdout=stream, stderr=subprocess.STDOUT, check=True,
                       env={**os.environ, **ENVIRONMENT, "CUDA_VISIBLE_DEVICES": str(gpu)})


def run(args):
    from scripts.launch.colorpeel_run import git_info, resolve_run_dir
    image = args.image.resolve()
    output = resolve_run_dir(args.output)
    protected = (ROOT, args.qwen_model_dir.resolve(), args.sd_model_dir.resolve())
    if (output.exists() or image.is_relative_to(output)
            or any(output.is_relative_to(root) or root.is_relative_to(output) for root in protected)):
        raise ValueError("Require a fresh external output separate from source and models")
    revision = git_info()
    source_hash = sha256(image)
    output.mkdir(parents=True)
    state = {"schema": "colorpeel_image_intake/v1", "status": "started", "git": revision,
             "source": {"image": str(image), "image_sha256": source_hash},
             "objects": [], "commands": [], "training_started": False,
             "scope": "Masks and per-instance initializers; derived datasets still required",
             "source_locks": {str(image): source_hash}}

    def execute(command, name):
        state["commands"].append({"argv": command, "physical_gpu": args.gpu, "log": str(output / name)})
        write_json(output / "status.json", state)
        run_stage(command, output / name, args.gpu)

    try:
        execute([str(args.vlm_python), "-B", str(ENTRY), "--stage", "discover",
                 "--image", str(image), "--qwen-model-dir", str(args.qwen_model_dir),
                 "--output", str(output / "discovery")], "discovery.log")
        categories_path = output / "discovery/categories.json"
        categories = read_json(categories_path)
        if (categories["status"] != "verified" or categories["source"]["image_sha256"] != source_hash
                or Path(categories["source"]["image"]).resolve() != image):
            raise ValueError("Object discovery source/status mismatch")
        check_locks(categories["source_locks"])
        merge_locks(state["source_locks"], categories["source_locks"])
        state["source_locks"][str(categories_path)] = sha256(categories_path)
        masks_path = output / "segmentation/objects.json"
        try:
            execute([str(args.sam_python), "-B", str(ENTRY), "--stage", "segment",
                     "--image", str(image), "--categories-manifest", str(categories_path),
                     "--output", str(output / "segmentation")], "segmentation.log")
        except subprocess.CalledProcessError:
            if not masks_path.is_file() or read_json(masks_path).get("status") != "partial":
                raise
        masks = read_json(masks_path)
        if (masks["source"]["image_sha256"] != source_hash
                or Path(masks["source"]["image"]).resolve() != image):
            raise ValueError("Segmentation source mismatch")
        merge_locks(state["source_locks"], {
            masks["source"]["canonical_image"]: masks["source"]["canonical_image_sha256"]})
        state["segmentation_manifest"] = str(masks_path)
        state["source_locks"][str(masks_path)] = sha256(masks_path)
        if masks["status"] not in ("succeeded", "partial") or not masks["objects"]:
            raise ValueError("No usable object instances; see segmentation/status.json")
        for item in masks["objects"]:
            if re.fullmatch(r"object_[0-9]+", item["id"]) is None:
                raise ValueError("Invalid object ID")
            row = {"object_id": item["id"], "category": item["category"],
                   "segmentation": item, "status": "initializing"}
            state["objects"].append(row)
            try:
                merge_locks(state["source_locks"], {item[key]: item[key + "_sha256"]
                                                   for key in ("object_mask", "cutout", "overlay", "vlm_input")})
                execute([str(args.vlm_python), "-B", str(ROOT / "scripts/methods/colorpeel_ice/initialize_from_image.py"),
                         "--image", item["vlm_input"], "--model-dir", str(args.qwen_model_dir),
                         "--sd-model-dir", str(args.sd_model_dir),
                         "--output", str(output / "initializers" / item["id"])], f"{item['id']}_initializers.log")
                manifest_path = output / "initializers" / item["id"] / "initializers.json"
                manifest = read_json(manifest_path)
                if (manifest["status"] != "verified"
                        or manifest["source"]["image_sha256"] != item["vlm_input_sha256"]
                        or Path(manifest["source"]["image"]).resolve() != Path(item["vlm_input"]).resolve()
                        or set(manifest["branches"]) != {"subject", "color", "material", "texture"}):
                    raise ValueError("Per-instance initializer source/status mismatch")
                check_locks(manifest["source_locks"])
                merge_locks(state["source_locks"], manifest["source_locks"])
                state["source_locks"].update({str(manifest_path): sha256(manifest_path),
                                             item["object_mask"]: item["object_mask_sha256"]})
                row.update(status="ready_for_derived_data", initializer_manifest=str(manifest_path),
                           initializers={key: value["initializer"] for key, value in manifest["branches"].items()})
            except Exception as error:
                row.update(status="failed", error=f"{type(error).__name__}: {error}")
            write_json(output / "status.json", state)
        ready = sum(row["status"] == "ready_for_derived_data" for row in state["objects"])
        state["status"] = ("ready_for_derived_data" if ready == len(state["objects"]) and masks["status"] == "succeeded"
                           else "partial" if ready else "failed")
        check_locks(state["source_locks"])
        if git_info() != revision:
            raise ValueError("Checkout changed during image intake")
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "status.json", state)
        write_json(output / "intake.json", state)
    return output / "intake.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("all", "discover", "segment"), default="all")
    parser.add_argument("--qwen-model-dir", type=Path, default=Path(
        f"/home/r12user5/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/{REVISION}"))
    parser.add_argument("--sd-model-dir", type=Path, default=Path("/var/tmp/colorpeel-r12user5/assets/sd21_base_4e63672_common_checker"))
    parser.add_argument("--vlm-python", type=Path, default=Path("/home/r12user5/miniforge3/envs/ice-vlm/bin/python"))
    parser.add_argument("--sam-python", type=Path, default=Path("/home/r12user5/miniforge3/envs/ice/bin/python"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--device", default="cuda:0", help="Logical device for individual worker stages")
    parser.add_argument("--categories-manifest", type=Path)
    args = parser.parse_args(argv)
    if args.stage == "discover":
        from .object_discovery import run as discover
        path = discover(args.image, args.qwen_model_dir, args.output, args.device)
    elif args.stage == "segment":
        from .object_segmentation import run as segment
        if args.categories_manifest is None:
            parser.error("segment requires --categories-manifest")
        discovery = read_json(args.categories_manifest)
        if (discovery["status"] != "verified"
                or discovery["source"]["image_sha256"] != sha256(args.image)):
            raise ValueError("Category manifest does not describe this image")
        check_locks(discovery["source_locks"])
        path = segment(args.image, discovery["categories"], args.output, args.device)
    else:
        path = run(args)
    print(path)
    return 1 if read_json(path)["status"] in ("failed", "partial") else 0


if __name__ == "__main__":
    sys.exit(main())
