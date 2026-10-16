"""Run native SD1.4/SD1.5 arms and bundle their results after completion."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.sd_native_comparison_v1.generate import (
    PROTOCOL, git_info, output_directory, sha256, utc_now, write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sd14-gpu", type=int, required=True)
    parser.add_argument("--sd15-gpu", type=int, required=True)
    args = parser.parse_args()
    if args.sd14_gpu < 0 or args.sd15_gpu < 0 or args.sd14_gpu == args.sd15_gpu:
        parser.error("select two distinct nonnegative GPU indices")
    revision = git_info()
    output = output_directory(args.output)
    output.mkdir(parents=True)
    receipt = {"status": "running", "git": revision, "protocol_sha256": sha256(PROTOCOL),
               "started_at": utc_now(), "children": {}}
    write_json(output / "manifest.json", receipt)
    children, logs = [], []
    try:
        for model, gpu in (("sd14", args.sd14_gpu), ("sd15", args.sd15_gpu)):
            environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
                           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                           "DIFFUSERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
            command = [sys.executable, "-B", str(Path(__file__).with_name("generate.py")),
                       "--model", model, "--output", str(output / model), "--device", "cuda:0"]
            log = (output / f"{model}.log").open("w", encoding="utf-8")
            logs.append(log)
            child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                     stderr=subprocess.STDOUT)
            children.append((model, child))
            receipt["children"][model] = {"pid": child.pid, "gpu": gpu, "command": command}
            write_json(output / "manifest.json", receipt)
        codes = {model: child.wait() for model, child in children}
        receipt["returncodes"] = codes
        if any(codes.values()):
            raise RuntimeError("a generation arm failed; preserve its log and failure ledger")
        command = [sys.executable, "-B", str(Path(__file__).with_name("bundle.py")),
                   "--sd14", str(output / "sd14"), "--sd15", str(output / "sd15"),
                   "--output", str(output / "review")]
        subprocess.run(command, cwd=ROOT, check=True)
        receipt.update(status="succeeded", review=str(output / "review"))
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        for log in logs:
            log.close()
        receipt["finished_at"] = utc_now()
        write_json(output / "manifest.json", receipt)


if __name__ == "__main__":
    main()
