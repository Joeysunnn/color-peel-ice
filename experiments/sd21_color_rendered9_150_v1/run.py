"""Fresh150-step training and transfer under the original experiment only."""

import argparse
from pathlib import Path

from experiments.sd21_color_gray9_v1 import run as shared
from experiments.sd21_color_rendered9_150_v1 import contract, inference


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpus', nargs='+', type=int, default=[2, 3])
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(contract.PARENT / 'supplemental150'):
        raise ValueError('use a fresh subdirectory of the original experiment/supplemental150')
    shared.run(args.output, tuple(args.gpus), execute=args.execute,
               contract=contract, inference=inference)
