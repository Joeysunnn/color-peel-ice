"""The existing nine Mailbox transfer probes with paired500/1000/1500 exports."""

import argparse
from pathlib import Path

from experiments.sd21_mailbox_center11_v1 import contract
from experiments.sd21_mailbox_subject27_v1 import inference as shared

SEEDS, SAMPLING = shared.SEEDS, shared.SAMPLING


def prompt_rows(method, step):
    return shared.prompt_rows(method, step, experiment_contract=contract)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', required=True, choices=contract.METHODS)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    shared.run(args.method, args.run, args.output, execute=args.execute, experiment_contract=contract)
