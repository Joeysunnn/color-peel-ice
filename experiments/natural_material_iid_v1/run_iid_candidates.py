"""Preserve each official IID material-diffusion sample before its CLI average."""

import argparse
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_png(array, path):
    image = np.clip(array, 0.0, 1.0)
    Image.fromarray(np.rint(image * 255).astype(np.uint8)).save(path)
    return sha(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--iid-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--openclip-weight", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    run_root, iid_root = args.run_root.resolve(), args.iid_root.resolve()
    checkpoint = args.checkpoint.resolve()
    openclip_weight = args.openclip_weight.resolve()
    output_root = run_root / "candidates"
    if output_root.exists():
        raise FileExistsError(output_root)
    if len(config["iid_candidate_seeds"]) != config["iid_samples_per_image"]:
        raise ValueError("Seed list does not match candidate count")
    if not openclip_weight.is_file():
        raise FileNotFoundError(openclip_weight)
    sys.path.insert(0, str(iid_root))
    import torch
    from omegaconf import OmegaConf
    from torchvision.transforms import Compose, Resize, ToTensor
    from iid.data import load_linear_image
    from iid.material_diffusion.iid import IntrinsicImageDiffusion

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Official IID material diffusion requires CUDA")
    torch.backends.cudnn.benchmark = False
    model_config_path = iid_root / "models/material_diffusion/config.yaml"
    model_config = OmegaConf.load(model_config_path)
    model = IntrinsicImageDiffusion(
        unet_config=model_config.unet_config,
        diffusion_config=model_config.diffusion_config,
        ddim_config=model_config.ddim_config)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    del state
    model = model.to(device).eval()
    output_root.mkdir()
    rows = []
    for sample in config["samples"]:
        sample_id = sample["id"]
        input_path = run_root / "inputs" / f"{sample_id}.png"
        image = Compose([ToTensor(), Resize(size=[480, 640])])(
            load_linear_image(str(input_path))).unsqueeze(0).to(device)
        if tuple(image.shape[-2:]) != (480, 640):
            raise ValueError(f"Unexpected IID input size: {image.shape}")
        sample_dir = output_root / sample_id
        sample_dir.mkdir()
        for index, seed in enumerate(config["iid_candidate_seeds"]):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            with torch.no_grad():
                prediction = model.sample(batch_size=1, conditioning_img=image)
                prediction = Resize(size=image.shape[-2:])(prediction)[0]
            maps = prediction.cpu().float().numpy()
            if maps.shape != (6, 480, 640) or not np.isfinite(maps).all():
                raise ValueError(f"Invalid IID sample {sample_id}/{index}: {maps.shape}")
            folder = sample_dir / f"candidate_{index:02d}"
            folder.mkdir()
            raw_path = folder / "raw_maps.npz"
            np.savez_compressed(raw_path, albedo=maps[:3], roughness=maps[3],
                                metallic=maps[4], brdf_aux=maps[5])
            png_hashes = {
                "albedo": save_png(np.transpose(maps[:3], (1, 2, 0)), folder / "albedo.png"),
                "roughness": save_png(maps[3], folder / "roughness.png"),
                "metallic": save_png(maps[4], folder / "metallic.png"),
            }
            record = {"sample_id": sample_id, "candidate_index": index, "seed": seed,
                      "raw_maps_sha256": sha(raw_path), "png_sha256": png_hashes,
                      "raw_min": float(maps.min()), "raw_max": float(maps.max()),
                      "brdf_aux_min": float(maps[5].min()),
                      "brdf_aux_max": float(maps[5].max()),
                      "out_of_range_fraction": float(np.mean((maps < 0) | (maps > 1))),
                      "input_sha256": sha(input_path)}
            (folder / "metadata.json").write_text(json.dumps(record, indent=2) + "\n")
            rows.append(record)
            print(f"{sample_id} candidate {index} seed {seed}", flush=True)
    manifest = {"status": "complete", "project_git_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=args.project_root, text=True).strip(),
                "iid_git_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=iid_root, text=True).strip(),
                "iid_model_config_sha256": sha(model_config_path),
                "ddim_steps": int(model_config.ddim_config.S),
                "ddim_eta": float(model_config.ddim_config.eta),
                "checkpoint_sha256": sha(checkpoint), "config_sha256": sha(args.config),
                "openclip_weight_sha256": sha(openclip_weight),
                "openclip_weight_path": str(openclip_weight),
                "script_sha256": sha(__file__), "device": str(device),
                "torch_version": torch.__version__,
                "input_policy": "official linear loader and 480x640 resize; padded RGB source",
                "candidate_policy": "one explicit seed per official model.sample call; retain all six channels before averaging; R/M are channels 3/4",
                "candidates": rows}
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
