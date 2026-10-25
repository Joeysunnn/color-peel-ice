"""Pin ZeST/Tencent sources and reuse frozen MARBLE model files read-only."""

import argparse
import json
from pathlib import Path
import subprocess

from experiments.sd15_token_local_kv_v1.contract import read_json, sha256
from experiments.subject_marble_c_v1.assets import MODELS as SHARED_MODELS

OFFICIAL_COMMIT = "b71ea174c08c5385f2487cc0ee6d8302a4bdf8f3"
IP_ADAPTER_COMMIT = "62e4af9d0c1ac7d5f8dd386a0ccf2211346af1a2"
CANNY_MODEL = {
    "repo": "diffusers/controlnet-canny-sdxl-1.0",
    "revision": "eb115a19a10d14909256db740ed109532ab1483c",
    "patterns": ["config.json", "diffusion_pytorch_model.fp16.safetensors"],
}


def source_state(path, revision):
    head = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True).strip()
    if head != revision or dirty:
        raise ValueError(f"require clean pinned source {path} at {revision}")


def verify(receipt):
    if (receipt["official_commit"] != OFFICIAL_COMMIT
            or receipt["ip_adapter_commit"] != IP_ADAPTER_COMMIT
            or receipt["models"] != {**SHARED_MODELS, **({"canny": CANNY_MODEL} if receipt["with_canny"] else {})}):
        raise ValueError("asset source/model revisions differ from the frozen protocol")
    source_state(receipt["official"], OFFICIAL_COMMIT)
    source_state(receipt["ip_adapter_official"], IP_ADAPTER_COMMIT)
    for path, expected in receipt["sha256"].items():
        if sha256(path) != expected:
            raise ValueError(f"frozen asset changed: {path}")
    return receipt


def read(path):
    path = Path(path)
    return read_json(path / "assets.json" if path.is_dir() else path)


def clone(url, path, revision):
    if not path.exists():
        subprocess.run(["git", "clone", url, str(path)], check=True)
        subprocess.run(["git", "-C", str(path), "checkout", "--detach", revision], check=True)
    source_state(path, revision)


def run(output, shared_marble_assets, with_canny=False):
    output, shared = Path(output).resolve(), Path(shared_marble_assets).resolve()
    project = Path(__file__).resolve().parents[2]
    if (output.is_relative_to(project) or project.is_relative_to(output)
            or output.is_relative_to(shared) or shared.is_relative_to(output)):
        raise ValueError("use a separate external asset root outside project and shared assets")
    receipt_path = output / "assets.json"
    if receipt_path.exists():
        receipt = verify(read_json(receipt_path))
        if receipt["shared_marble_assets"] != str(shared) or (with_canny and not receipt["with_canny"]):
            raise ValueError("completed assets are immutable; use a new root for different inputs")
        print("Existing assets verified", receipt_path, flush=True)
        return receipt
    shared_receipt_path = shared / "assets.json"
    shared_receipt = read_json(shared_receipt_path)
    if shared_receipt["models"] != SHARED_MODELS:
        raise ValueError("shared MARBLE model revisions differ")
    locks = dict(shared_receipt["sha256"])
    locks[str(shared_receipt_path)] = sha256(shared_receipt_path)
    for path, expected in locks.items():
        if sha256(path) != expected:
            raise ValueError(f"shared asset changed: {path}")
    output.mkdir(parents=True, exist_ok=True)
    official, adapter = output / "official", output / "IP-Adapter"
    clone("https://github.com/ttchengab/zest_code.git", official, OFFICIAL_COMMIT)
    clone("https://github.com/tencent-ailab/IP-Adapter.git", adapter, IP_ADAPTER_COMMIT)
    paths = dict(shared_receipt["paths"])
    paths["image_encoder"] = str(Path(paths["adapter"]) / "models/image_encoder")
    paths["ip_checkpoint"] = str(Path(paths["adapter"]) / "sdxl_models/ip-adapter_sdxl_vit-h.bin")
    if with_canny:
        from huggingface_hub import snapshot_download

        paths["canny"] = snapshot_download(CANNY_MODEL["repo"], revision=CANNY_MODEL["revision"],
            allow_patterns=CANNY_MODEL["patterns"], cache_dir=output / "hf_cache", max_workers=2)
        for name in CANNY_MODEL["patterns"]:
            file = Path(paths["canny"]) / name
            locks[str(file)] = sha256(file)
    for checkout in (official, adapter):
        tracked = subprocess.check_output(["git", "-C", str(checkout), "ls-files"], text=True).splitlines()
        locks.update({str(checkout / name): sha256(checkout / name) for name in tracked})
    receipt = {"official": str(official), "official_commit": OFFICIAL_COMMIT,
        "ip_adapter_official": str(adapter), "ip_adapter_commit": IP_ADAPTER_COMMIT,
        "shared_marble_assets": str(shared), "shared_receipt": str(shared_receipt_path),
        "models": {**SHARED_MODELS, **({"canny": CANNY_MODEL} if with_canny else {})},
        "paths": paths, "sha256": locks, "with_canny": with_canny,
        "sharing": "Existing MARBLE receipt, weights and source files are read-only; no weight copies or environment changes."}
    verify(receipt)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf8")
    print("Assets verified", receipt_path, flush=True)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shared-marble-assets", type=Path, required=True)
    parser.add_argument("--with-canny", action="store_true")
    args = parser.parse_args()
    run(args.output, args.shared_marble_assets, args.with_canny)
