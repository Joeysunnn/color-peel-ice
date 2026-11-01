"""800/1000 paired transfer, including matte, seeds30-40."""

import argparse
from pathlib import Path

from experiments.sd21_mailbox27_resume1000_v1 import contract
from experiments.sd21_mailbox_subject27_v1 import inference as parent

SEEDS, SAMPLING = parent.SEEDS, parent.SAMPLING


def prompt_rows(method, step):
    rows = parent.prompt_rows(method, step, experiment_contract=contract)
    if len([row for row in rows if row['label'] == 'matte']) != 11:
        raise ValueError('matte must be present for all11 seeds')
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', required=True, choices=contract.METHODS)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    for step in contract.STEPS:
        prompt_rows(args.method, step)
    parent.run(args.method, args.run, args.output, execute=args.execute, experiment_contract=contract)
