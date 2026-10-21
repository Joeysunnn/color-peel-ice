"""Fetch only pinned official MARBLE inference dependencies into a new asset root."""

import argparse
import json
from pathlib import Path
import subprocess

from experiments.sd15_token_local_kv_v1.contract import sha256

OFFICIAL_COMMIT = "0139f2563057e8ac7d8e82055ea891ffd7474bd5"
MODELS = {
    "base": {"repo": "stabilityai/stable-diffusion-xl-base-1.0", "revision": "462165984030d82259a11f4367a4eed129e94a7b",
             "patterns": ["model_index.json", "scheduler/scheduler_config.json", "tokenizer/*", "tokenizer_2/*",
                          "text_encoder/config.json", "text_encoder/model.safetensors", "text_encoder_2/config.json",
                          "text_encoder_2/model.safetensors", "unet/config.json", "unet/diffusion_pytorch_model.safetensors",
                          "vae/config.json", "vae/diffusion_pytorch_model.safetensors"]},
    "controlnet": {"repo": "diffusers/controlnet-depth-sdxl-1.0", "revision": "17bb97973f29801224cd66f192c5ffacf82648b4",
                   "patterns": ["config.json", "diffusion_pytorch_model.fp16.safetensors"]},
    "depth": {"repo": "Intel/dpt-hybrid-midas", "revision": "11eaf7a1cf4bd70740697dbc216f98980c0aeb03",
              "patterns": ["config.json", "preprocessor_config.json", "pytorch_model.bin"]},
    "adapter": {"repo": "h94/IP-Adapter", "revision": "018e402774aeeddd60609b4ecdb7e298259dc729",
                "patterns": ["models/image_encoder/config.json", "models/image_encoder/model.safetensors",
                             "sdxl_models/ip-adapter_sdxl_vit-h.bin"]}}


def official_state(path):
    commit = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True).strip()
    if commit != OFFICIAL_COMMIT or dirty:
        raise ValueError("require the pinned clean official MARBLE checkout")
    return commit


def run(output):
    from huggingface_hub import hf_hub_download, snapshot_download

    output = Path(output).resolve()
    root = Path(__file__).resolve().parents[2]
    if output.is_relative_to(root) or root.is_relative_to(output) or (output / "assets.json").exists():
        raise ValueError("use an external asset root without a completed receipt")
    output.mkdir(parents=True, exist_ok=True)
    official = output / "official"
    if not official.exists():
        subprocess.run(["git", "clone", "https://github.com/Stability-AI/marble.git", str(official)], check=True)
        subprocess.run(["git", "-C", str(official), "checkout", "--detach", OFFICIAL_COMMIT], check=True)
    official_state(official)
    cache = output / "hf_cache"
    paths = {}
    for name, spec in MODELS.items():
        print("Downloading pinned", name, spec["repo"], flush=True)
        if name == "adapter":
            for filename in spec["patterns"]:
                hf_hub_download(spec["repo"], filename, revision=spec["revision"],
                                local_dir=official, cache_dir=cache)
            paths[name] = str(official)
        else:
            paths[name] = snapshot_download(spec["repo"], revision=spec["revision"],
                allow_patterns=spec["patterns"], cache_dir=cache, max_workers=4)
    locks = {}
    for name, path in paths.items():
        if name == "adapter":
            files = [official / filename for filename in MODELS[name]["patterns"]]
        else:
            files = [file for file in Path(path).rglob("*") if file.is_file()]
        for file in files:
            locks[str(file)] = sha256(file)
    tracked = subprocess.check_output(["git", "-C", str(official), "ls-files"], text=True).splitlines()
    locks.update({str(official / file): sha256(official / file) for file in tracked})
    official_state(official)
    receipt = {"official": str(official), "official_commit": OFFICIAL_COMMIT,
               "models": MODELS, "paths": paths, "sha256": locks}
    (output / "assets.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf8")
    print("Assets verified", output / "assets.json", flush=True)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
