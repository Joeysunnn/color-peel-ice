"""Color100/150 paired transfer: ten unchanged prompts, seeds30-45 inclusive."""

import argparse
from pathlib import Path

from experiments.sd21_color_gray9_v1 import inference as shared
from experiments.sd21_color_rendered9_150_v1 import contract

SEEDS = tuple(range(30, 46))
SAMPLING = {**shared.SAMPLING, 'seeds': list(SEEDS)}


def prompt_rows(method, step, smoke=False):
    return shared.prompt_rows(method, step, smoke, recipe=contract, seeds=SEEDS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', required=True, choices=contract.METHODS)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    shared.run(args.method, args.run, args.output, args.smoke, execute=args.execute,
               recipe=contract, seeds=SEEDS)
