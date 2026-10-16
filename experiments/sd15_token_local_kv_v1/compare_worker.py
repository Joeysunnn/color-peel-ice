"""Generate one locked SD1.5 job and bundle its matched SD1.4 references."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.sd15_token_local_kv_v1.comparison_bundle import bundle_job
from experiments.sd15_token_local_kv_v1.compare import INFERENCE_PROTOCOL
from experiments.sd15_token_local_kv_v1.contract import read_json, sha256
from experiments.sd15_token_local_kv_v1.evaluate import attention_masks, load_pipeline
from experiments.sd_native_comparison_v1.generate import (
    git_info, package_versions, write_json,
)
from scripts.launch.colorpeel_run import resolve_run_dir


SAMPLING = {"num_inference_steps": 100, "guidance_scale": 3.5, "width": 512,
            "height": 512, "scheduler": "PNDM", "dtype": "float16",
            "generator": "torch.Generator(device).manual_seed(seed)", "safety_checker": True}


def verify_inputs(job):
    reference = job["reference"]
    root = Path(read_json(INFERENCE_PROTOCOL)["reference_root"])
    for name, digest in reference["files"].items():
        if sha256(root / reference["directory"] / name) != digest:
            raise ValueError("historical reference metadata changed")
    for selection in job["selections"]:
        for field in ("hashes", "source_hashes"):
            for path, digest in selection.get(field, {}).items():
                if sha256(Path(path)) != digest:
                    raise ValueError(f"selected checkpoint hash differs: {path}")
        if not selection.get("hashes"):
            raise ValueError("selected checkpoint requires hashes")
        receipt = Path(selection["checkpoint"]) / "receipt.json"
        recorded = read_json(receipt)
        if (not recorded or any(selection.get(key) != value for key, value in recorded.items())
                or any(key not in recorded for key in ("checkpoint", "step", "hashes", "source_hashes"))):
            raise ValueError("derived checkpoint receipt differs from selection")
        snapshot = Path(selection["source_checkpoint"]) / f"checkpoint-{selection['step']}"
        for name, field in (("pytorch_model.bin", "source_adapter_sha256"),
                            ("pytorch_model_1.bin", "source_text_encoder_sha256")):
            if sha256(snapshot / name) != selection[field]:
                raise ValueError("original training snapshot hash differs")
    for row in job["rows"]:
        if sha256(Path(row["reference_image"])) != row["reference_sha256"]:
            raise ValueError(f"SD1.4 reference image changed: {row['id']}")


def run(plan_path, job_id, device):
    plan_digest = sha256(plan_path)
    plan = read_json(plan_path)
    revision = git_info()
    if revision != plan["git"] or plan["sampling"] != SAMPLING:
        raise ValueError("worker revision or sampling differs from the locked plan")
    matches = [job for job in plan["jobs"] if job["id"] == job_id]
    if len(matches) != 1:
        raise ValueError("expected exactly one matching job")
    job = matches[0]
    rows, selections = job["rows"], job["selections"]
    tokens = [selection["token"] for selection in selections]
    if (tokens not in (["<S*>"], ["<M*>"], ["<S*>", "<M*>"])
            or len({selection["method"] for selection in selections}) != 1
            or not rows or len({row["id"] for row in rows}) != len(rows)):
        raise ValueError("job tokens, adapter methods or row IDs differ")
    for index, row in enumerate(rows):
        if (not isinstance(row["prompt"], str) or type(row["seed"]) is not int
                or type(row["reference_filtered"]) is not bool
                or not Path(row["reference_image"]).is_absolute()
                or row["image_path"] != f"images/{index:04d}.png"
                or row.get("num_inference_steps", 100) != 100
                or row.get("guidance_scale", 3.5) != 3.5
                or any(token in row["prompt"] for token in {"<S*>", "<M*>"} - set(tokens))):
            raise ValueError(f"job prompt, reference, image path or sampling differs: {row['id']}")
    verify_inputs(job)
    output = resolve_run_dir(Path(job["output"]))
    if output.exists() or output.is_relative_to(ROOT):
        raise ValueError("worker output must be new and outside the repository")
    if any(Path(row["reference_image"]).resolve().is_relative_to(output) for row in rows):
        raise ValueError("output must not contain SD1.4 references")
    output.mkdir(parents=True)
    provenance = {"status": "running", "job_id": job_id, "git": revision,
                  "plan": str(plan_path.resolve()), "plan_sha256": plan_digest,
                  "training": selections, "sampling": SAMPLING, "row_count": len(rows),
                  "device": device, "environment": package_versions()}
    provenance_path = output / "provenance.json"
    write_json(provenance_path, provenance)
    try:
        with (output / "generation_manifest.jsonl").open("x", encoding="utf-8") as stream:
            stream.writelines(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        import torch

        pipe = load_pipeline(selections, device)
        with torch.no_grad(), (output / "generation_status.jsonl").open(
                "x", encoding="utf-8") as ledger:
            for row in rows:
                if sha256(Path(row["reference_image"])) != row["reference_sha256"]:
                    raise ValueError(f"SD1.4 reference changed before generation: {row['id']}")
                result = pipe(
                    row["prompt"], num_inference_steps=100, guidance_scale=3.5,
                    width=512, height=512,
                    generator=torch.Generator(device=device).manual_seed(row["seed"]),
                    cross_attention_kwargs=attention_masks(pipe, row["prompt"], tokens, 3.5))
                flags = result.nsfw_content_detected
                if flags is None or len(flags) != 1 or len(result.images) != 1:
                    raise ValueError("expected one image and an enabled safety-checker result")
                image = output / row["image_path"]
                image.parent.mkdir(exist_ok=True)
                result.images[0].save(image)
                ledger.write(json.dumps({**row, "status": "safety_filtered" if flags[0] else "ok",
                                         "image_sha256": sha256(image)}, sort_keys=True) + "\n")
                ledger.flush()
        verify_inputs(job)
        if sha256(plan_path) != plan_digest or git_info() != revision:
            raise ValueError("plan or code changed during generation")
        provenance["status"] = "succeeded"
        write_json(provenance_path, provenance)
        bundle_job(output)
    except Exception as error:
        provenance.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(provenance_path, provenance)
        raise
    print(f"generated and bundled {len(rows)} matched pairs: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(args.plan, args.job, args.device)
