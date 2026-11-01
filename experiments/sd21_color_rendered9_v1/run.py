"""Start the two finite rendered-background9 training and transfer queues."""

import argparse
from pathlib import Path

from experiments.sd21_color_gray9_v1 import run as shared
from experiments.sd21_color_rendered9_v1 import contract, inference


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpus', nargs='+', type=int, default=[2, 3])
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    shared.run(args.output, tuple(args.gpus), args.smoke, args.execute,
               contract=contract, inference=inference)
