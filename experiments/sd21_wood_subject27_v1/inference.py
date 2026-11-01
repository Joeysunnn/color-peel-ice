"""Use the established nine Subject probes with the frozen27-image contract."""

import argparse
from functools import partial
from pathlib import Path

from experiments.sd21_wood_branches_v1 import inference as shared, snapshots
from experiments.sd21_wood_subject27_v1 import contract


def run(train_run, output, method, execute=False, smoke=False):
    exporter = partial(snapshots.audit_export, experiment_contract=contract)
    return shared.run('subject', method, train_run, output, seeds=tuple(range(30, 61)),
                      smoke=smoke, execute=execute, snapshot_exporter=exporter)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--method', required=True, choices=contract.METHODS)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(args.run, args.output, args.method, args.execute, args.smoke)
