"""Generate one arm of the paired, adapter-free SD 1.4/1.5 comparison."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import traceback


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = Path(__file__).with_name("protocol.json")
MODEL_COMPONENTS = ("unet", "vae", "text_encoder", "tokenizer", "scheduler")
SD14_REVISION = "133a221b8aa7292a167afc5127cb63fb5005638b"
SD15_REVISION = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_protocol() -> dict:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    expected_models = {
        "sd14": {
            "repo": "CompVis/stable-diffusion-v1-4", "revision": SD14_REVISION,
            "path": "/home/r12user5/.cache/huggingface/hub/"
                    "models--CompVis--stable-diffusion-v1-4/snapshots/" + SD14_REVISION,
        },
        "sd15": {
            "repo": "stable-diffusion-v1-5/stable-diffusion-v1-5", "revision": SD15_REVISION,
            "path": "/home/r12user5/Documents/Jiawei/colorpeel-runs/"
                    "natural_material_palette_v1/deps/sd15",
        },
    }
    expected = {
        "schema": "sd_native_comparison/v1", "models": expected_models,
        "objects": ["chair", "mug", "vase", "sofa"],
        "prompt_template": "a photo of a {object}",
        "sampling": {
            "seeds": [42, 43, 44, 45, 46], "width": 512, "height": 512,
            "scheduler": "PNDM", "num_inference_steps": 100, "guidance_scale": 3.5,
            "dtype": "float16", "latent_policy": "cpu_float32_randn_then_float16_to_cuda",
            "latent_shape": [1, 4, 64, 64],
        },
        "safety": {"enabled": True, "source_model": "sd14",
                   "components": ["safety_checker", "feature_extractor"],
                   "shared_between_models": True},
        "adapters": False, "local_files_only": True,
    }
    if protocol != expected:
        raise ValueError("Native comparison protocol differs from the locked matrix")
    return protocol


def rows(protocol: dict, model: str) -> list[dict]:
    return [
        {"id": f"{model}__{obj}__{seed}", "pair_id": f"{obj}__{seed}",
         "model": model, "object": obj, "seed": seed,
         "prompt": protocol["prompt_template"].format(object=obj),
         "image_path": f"images/{obj}/seed{seed}.png"}
        for obj in protocol["objects"] for seed in protocol["sampling"]["seeds"]
    ]


def git_info() -> dict:
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    if git("status", "--porcelain"):
        raise RuntimeError("Refusing to generate from a dirty worktree")
    return {"commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
            "fork_url": git("remote", "get-url", "origin"), "clean_worktree": True}


def output_directory(path: Path) -> Path:
    value = os.environ.get("COLORPEEL_RUN_ROOT")
    if not value:
        raise EnvironmentError("COLORPEEL_RUN_ROOT must be set")
    run_root = Path(value).expanduser().resolve()
    output = path.expanduser().resolve()
    if (run_root == ROOT or ROOT in run_root.parents or output == run_root
            or run_root not in output.parents or output == ROOT or ROOT in output.parents):
        raise ValueError("Output must be below COLORPEEL_RUN_ROOT and outside the code checkout")
    if output.exists():
        raise FileExistsError(output)
    return output


def package_versions() -> dict:
    versions = {"python": sys.version, "executable": sys.executable,
                "platform": platform.platform()}
    for name in ("pip", "torch", "diffusers", "transformers", "accelerate", "numpy", "Pillow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def component_inventory(directory: Path, full_hash: bool) -> dict:
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    if not files:
        raise FileNotFoundError(f"Empty or missing model component: {directory}")
    return {path.relative_to(directory).as_posix(): {
        "bytes": path.stat().st_size,
        "sha256": sha256(path) if full_hash else None,
    } for path in files}


def model_inventory(protocol: dict, model: str, full_hash: bool) -> dict:
    selected = protocol["models"][model]
    base = Path(selected["path"])
    safety = Path(protocol["models"]["sd14"]["path"])
    if not (base / "model_index.json").is_file():
        raise FileNotFoundError(base / "model_index.json")
    result = {"model": model, "source": selected,
              "hashes_complete": full_hash,
              "hash_policy": "all component files" if full_hash else "deferred until execution",
              "model_index": component_file(base / "model_index.json", full_hash),
              "components": {name: component_inventory(base / name, full_hash)
                             for name in MODEL_COMPONENTS},
              "safety_source": protocol["models"]["sd14"],
              "safety_components": {name: component_inventory(safety / name, full_hash)
                                    for name in protocol["safety"]["components"]}}
    if model == "sd15":
        manifest_path = base / "download_manifest.json"
        download = json.loads(manifest_path.read_text(encoding="utf-8"))
        if download.get("repo") != selected["repo"] or download.get("revision") != selected["revision"]:
            raise ValueError("SD 1.5 download manifest repo/revision differs")
        result["download_manifest"] = component_file(manifest_path, full_hash)
    elif base.name != selected["revision"]:
        raise ValueError("SD 1.4 snapshot revision differs")
    return result


def component_file(path: Path, full_hash: bool) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": sha256(path) if full_hash else None}


def append_ledger(output: Path, row: dict) -> None:
    with (output / "ledger.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def execute(protocol: dict, model: str, device: str, output: Path, entries: list[dict],
            manifest: dict) -> None:
    import torch
    from diffusers import PNDMScheduler, StableDiffusionPipeline
    from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
    from transformers import CLIPImageProcessor

    if not torch.__version__.startswith("1.13."):
        raise RuntimeError("Use the existing colorpeel017 environment with PyTorch 1.13.x")
    if not importlib.metadata.version("diffusers").startswith("0.17."):
        raise RuntimeError("Use the existing colorpeel017 environment with Diffusers 0.17.x")
    if not device.startswith("cuda:") or not torch.cuda.is_available():
        raise RuntimeError("Native comparison requires a CUDA device")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    base = protocol["models"][model]["path"]
    safety_base = protocol["models"]["sd14"]["path"]
    checker = StableDiffusionSafetyChecker.from_pretrained(
        safety_base, subfolder="safety_checker", local_files_only=True, torch_dtype=torch.float16)
    extractor = CLIPImageProcessor.from_pretrained(
        safety_base, subfolder="feature_extractor", local_files_only=True)
    scheduler = PNDMScheduler.from_pretrained(base, subfolder="scheduler", local_files_only=True)
    pipe = StableDiffusionPipeline.from_pretrained(
        base, scheduler=scheduler, safety_checker=checker, feature_extractor=extractor,
        requires_safety_checker=True, torch_dtype=torch.float16,
        local_files_only=True, low_cpu_mem_usage=False,
    ).to(device)
    if pipe.safety_checker is None or not isinstance(pipe.scheduler, PNDMScheduler):
        raise RuntimeError("PNDM and the shared SD 1.4 safety checker are required")
    pipe.set_progress_bar_config(disable=True)
    manifest["runtime"] = {
        "cuda_version": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(device),
        "scheduler_config": dict(pipe.scheduler.config),
        "tf32": False, "cudnn_benchmark": False, "cudnn_deterministic": True,
    }
    write_json(output / "manifest.json", manifest)
    sampling = protocol["sampling"]
    for row in entries:
        record = {**row, "status": "running", "latent_sha256": None,
                  "image_sha256": None, "safety_filtered": None, "error": None}
        try:
            generator = torch.Generator(device="cpu").manual_seed(row["seed"])
            latent = torch.randn(tuple(sampling["latent_shape"]), generator=generator,
                                 device="cpu", dtype=torch.float32).to(dtype=torch.float16)
            record["latent_sha256"] = hashlib.sha256(latent.numpy().tobytes()).hexdigest()
            with torch.inference_mode():
                result = pipe(
                    row["prompt"], negative_prompt="", width=sampling["width"],
                    height=sampling["height"], num_inference_steps=sampling["num_inference_steps"],
                    guidance_scale=sampling["guidance_scale"], latents=latent.to(device),
                )
            flags = result.nsfw_content_detected
            if flags is None or len(flags) != 1 or len(result.images) != 1:
                raise RuntimeError("Expected one image and one explicit safety result")
            image = result.images[0]
            if image.size != (sampling["width"], sampling["height"]):
                raise RuntimeError("Generated image size differs")
            destination = output / row["image_path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            image.save(destination, format="PNG")
            record.update(status="succeeded", image_sha256=sha256(destination),
                          safety_filtered=bool(flags[0]))
        except Exception as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            append_ledger(output, record)
            raise
        append_ledger(output, record)
        print(f"{row['id']}: saved; safety_filtered={record['safety_filtered']}", flush=True)


def run(args: argparse.Namespace) -> Path:
    protocol = read_protocol()
    git = git_info()
    output = output_directory(args.output)
    output.mkdir(parents=True, exist_ok=False)
    entries = rows(protocol, args.model)
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    manifest = {
        "schema": "sd_native_comparison_run/v1", "status": "running", "started_at": utc_now(),
        "model": args.model, "model_source": protocol["models"][args.model],
        "git": git, "protocol_sha256": sha256(PROTOCOL),
        "script_sha256": sha256(Path(__file__)), "protocol": protocol,
        "command_argv": command, "cwd": os.getcwd(), "output": str(output), "device": args.device,
        "environment": {name: os.environ.get(name) for name in (
            "CONDA_DEFAULT_ENV", "CONDA_PREFIX", "CUDA_VISIBLE_DEVICES", "COLORPEEL_RUN_ROOT")},
        "dry_run": args.dry_run,
        "expected_rows": len(entries), "generated_rows": 0,
        "safety_source": protocol["models"]["sd14"], "adapters": False,
    }
    write_json(output / "manifest.json", manifest)
    try:
        manifest["versions"] = package_versions()
        write_json(output / "protocol.json", protocol)
        (output / "command.sh").write_text(
            "cd " + shlex.quote(os.getcwd()) + "\n" + shlex.join(command) + "\n", encoding="utf-8")
        (output / "planned_rows.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in entries), encoding="utf-8")
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], text=True,
                                capture_output=True, check=True)
        (output / "pip-freeze.txt").write_text(freeze.stdout, encoding="utf-8")
        inventory = model_inventory(protocol, args.model, full_hash=not args.dry_run)
        write_json(output / "model_inventory.json", inventory)
        manifest["model_inventory_sha256"] = sha256(output / "model_inventory.json")
        if args.dry_run:
            manifest["status"] = "dry_run"
            for row in entries:
                append_ledger(output, {**row, "status": "planned", "latent_sha256": None,
                                       "image_sha256": None, "safety_filtered": None, "error": None})
        else:
            execute(protocol, args.model, args.device, output, entries, manifest)
            ledger_path = output / "ledger.jsonl"
            ledger = ([json.loads(line) for line in ledger_path.read_text().splitlines()]
                      if ledger_path.exists() else [])
            if (len(ledger) != len(entries)
                    or [row["id"] for row in ledger] != [row["id"] for row in entries]
                    or any(row["status"] != "succeeded" or not row["image_sha256"]
                           or not row["latent_sha256"] or row["safety_filtered"] is None
                           for row in ledger)):
                raise RuntimeError("Incomplete generation ledger")
            manifest["generated_rows"] = len(ledger)
            manifest["safety_filtered_rows"] = sum(row["safety_filtered"] for row in ledger)
            manifest["status"] = "succeeded"
        manifest["returncode"] = 0
    except BaseException as exc:
        manifest.update(status="failed", returncode=1, error=f"{type(exc).__name__}: {exc}")
        (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        ledger_path = output / "ledger.jsonl"
        ledger = [json.loads(line) for line in ledger_path.read_text().splitlines()] if ledger_path.exists() else []
        manifest["generated_rows"] = sum(row["status"] == "succeeded" for row in ledger)
        attempted = {row["id"] for row in ledger}
        for row in entries:
            if row["id"] not in attempted:
                append_ledger(output, {**row, "status": "not_run", "latent_sha256": None,
                                       "image_sha256": None, "safety_filtered": None,
                                       "error": manifest["error"]})
        raise
    finally:
        manifest["finished_at"] = utc_now()
        write_json(output / "manifest.json", manifest)
    print(f"{manifest['status']}: {output}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("sd14", "sd15"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
