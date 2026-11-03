"""Identify four initializers with pinned Qwen and controlled two-axis Material selection."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.methods.colorpeel_ice.vlm_initialization import run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--sd-model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    path = run(args.image, args.model_dir, args.sd_model_dir, args.output, args.device)
    print(path)


if __name__ == "__main__":
    main()
