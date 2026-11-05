"""Two bounded reconstructions: exact hard-pink replay, then alpha-only intervention."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.mailbox_alpha_control_v1.prepare import prepare
from experiments.subject_pink_input_repro_v1.compare import compare
from scripts.launch.colorpeel_run import git_info
from src.methods.colorpeel_ice.single_image_pipeline import RUNTIME, execute, fresh, read, write_json
from src.methods.colorpeel_ice.single_image_subject import check_locks, sha256


def run(hard_run, soft_run, output, gpu):
    hard_run, soft_run = [Path(path).resolve() for path in (hard_run, soft_run)]
    output = Path(output).resolve()
    if any(output.is_relative_to(path) or path.is_relative_to(output) for path in (hard_run, soft_run)):
        raise ValueError('new output must be separate from both reference runs')
    locks = {str(root / name): sha256(root / name) for root in (hard_run, soft_run) for name in
             ('repro_outputs/status.json', 'pipeline.json', 'source_cutout512.png',
              'actual_condition.png', 'native_asset.pt')}
    revision = git_info()
    output = fresh(output)
    state = {'status': 'preparing', 'git': revision, 'gpu': gpu, 'commands': [], 'source_locks': locks,
             'training_started': False, 'quality_accepted_by_user': False}
    try:
        audit = prepare(read(hard_run / 'repro_outputs/status.json')['source_receipt']['path'],
                        read(soft_run / 'repro_outputs/status.json')['source_receipt']['path'],
                        output / 'inputs')
        crop = audit['fixed_crop_xyxy']
        reference_condition = np.asarray(Image.open(hard_run / 'actual_condition.png'))
        expected = np.asarray(Image.open(output / 'inputs/hard_replay/expected_condition.png'))
        if not np.array_equal(reference_condition, expected):
            raise ValueError('fixed-crop preprocessing does not reproduce previous actual condition')
        for name in ('hard_replay', 'soft_only'):
            check_locks(locks)
            state.update(status='running', stage=name)
            folder = output / name / 'trellis'
            command = [RUNTIME['trellis_python'], '-B', '-m', 'experiments.subject_trellis_counterfactual_v1.run',
                '--output', folder, '--encoder', RUNTIME['trellis_encoder'], '--source-receipt',
                output / 'inputs' / name / 'source_receipt.json', '--model-revisions',
                'experiments/subject_trellis_spoons_v1/model_revisions.json', '--seed', '42', '--skip-glb',
                '--fixed-crop', *crop]
            state['commands'].append({'argv': [str(value) for value in command], 'log': str(output / (name + '.log'))})
            write_json(output / 'status.json', state)
            execute(command, output / (name + '.log'), gpu)
            actual = np.asarray(Image.open(folder / 'actual_condition.png'))
            expected = np.asarray(Image.open(output / 'inputs' / name / 'expected_condition.png'))
            if not np.array_equal(actual, expected):
                raise ValueError('model conditioning differs from audited fixed-crop input')
            result = compare(hard_run, folder, output / (name + '_comparison.json'))
            if not all(result['configuration_equal'].values()):
                raise ValueError('generation configuration drifted')
            if name == 'hard_replay' and not result['native_output_exactly_reproduced']:
                raise ValueError('hard-pink replay differs; stop before blaming alpha')
        check_locks(locks)
        if git_info() != revision:
            raise ValueError('checkout changed during diagnostic')
        state.update(status='completed_pending_visual_review', stage='completed')
    except Exception as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(output / 'status.json', state)
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hard-run', required=True, type=Path)
    parser.add_argument('--soft-run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    run(args.hard_run, args.soft_run, args.output, args.gpu)
