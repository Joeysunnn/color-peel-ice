"""Run the official single-image SuperMat CLI on one immutable image batch."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(root):
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                   text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--supermat-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for name in ("python", "supermat_root", "checkpoint", "base_model", "input_dir", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    images = sorted(args.input_dir.glob("*.png"))
    if not images:
        raise ValueError("No PNG inputs")
    if len({path.stem for path in images}) != len(images):
        raise ValueError("Duplicate image stems")
    if not (args.base_model / "model_index.json").is_file():
        raise FileNotFoundError(args.base_model / "model_index.json")
    args.output_dir.mkdir(parents=True)
    command = [str(args.python), "inference_supermat.py", "--input", str(args.input_dir),
               "--output-dir", str(args.output_dir), "--checkpoint", str(args.checkpoint),
               "--base-model", str(args.base_model), "--device", args.device,
               "--image-size", "512", "--seed", str(args.seed), "--save-orm"]
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    log_path = args.output_dir / "inference.log"
    with log_path.open("w") as handle:
        result = subprocess.run(command, cwd=args.supermat_root, env=env,
                                stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"SuperMat exited {result.returncode}; see {log_path}")
    outputs = []
    for input_image in images:
        folder = args.output_dir / input_image.stem
        maps = {}
        for name in ("albedo", "roughness", "metallic", "orm"):
            path = folder / f"{name}.png"
            if not path.is_file():
                raise FileNotFoundError(path)
            maps[name] = {"path": str(path), "sha256": sha(path)}
        outputs.append({"input": str(input_image), "input_sha256": sha(input_image),
                        "maps": maps})
    project_root = Path(__file__).resolve().parents[2]
    manifest = {
        "schema_version": 1, "status": "complete", "command": command,
        "exit_code": result.returncode, "log_sha256": sha(log_path),
        "supermat_git_commit": git_commit(args.supermat_root),
        "project_git_commit": git_commit(project_root),
        "wrapper_sha256": sha(__file__),
        "checkpoint_sha256": sha(args.checkpoint),
        "base_model_index_sha256": sha(args.base_model / "model_index.json"),
        "seed": args.seed, "device": args.device, "image_size": 512,
        "outputs": outputs,
    }
    (args.output_dir / "inference_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
