"""Run box-prompted SAM intake, then the six-object four-seed mesh batch."""

import argparse
import os
from pathlib import Path
import subprocess

from scripts.launch.colorpeel_run import git_info
from src.methods.colorpeel_ice.single_image_pipeline import OFFLINE, ROOT, RUNTIME, execute, fresh, write_json


def launch(selection, output, gpus):
    if not gpus or len(set(gpus)) != len(gpus) or any(gpu < 0 for gpu in gpus):
        raise ValueError('require distinct nonnegative GPUs')
    output = fresh(output)
    state = {'status': 'running', 'stage': 'sam_intake', 'git': git_info(),
        'expected_meshes': 96, 'expected_views': 384, 'commands': [],
        'automatic_retries': 0, 'quality_accepted_by_user': False}
    try:
        command = [RUNTIME['python'], '-B', '-m', 'experiments.sam_six_seed_v1.intake',
                   '--selection', str(Path(selection).resolve()), '--output', str(output / 'intake')]
        state['commands'].append(command)
        write_json(output / 'launch_status.json', state)
        execute(command, output / 'intake.log', gpus[0])
        # SAM exits before any TRELLIS worker checks GPU availability.
        command = [RUNTIME['python'], '-B', '-m', 'experiments.sam_six_seed_v1.run',
            '--inputs', str(output / 'intake/inputs.json'), '--output', str(output / 'meshes'),
            '--gpus', *map(str, gpus)]
        state.update(stage='meshes')
        state['commands'].append(command)
        write_json(output / 'launch_status.json', state)
        with (output / 'meshes.log').open('x') as stream:
            subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT, check=True, env={**os.environ, **OFFLINE})
        state.update(status='completed_pending_user_review', stage='completed')
    except Exception as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(output / 'launch_status.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpus', nargs='+', type=int, default=[0, 1, 2, 3])
    args = parser.parse_args()
    launch(args.selection, args.output, args.gpus)
