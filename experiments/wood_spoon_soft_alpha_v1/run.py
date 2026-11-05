"""One fixed-SAM wood-spoon verification: fresh soft alpha, pink, then one mesh."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from src.methods.colorpeel_ice.single_image_pipeline import (
    ROOT, RUNTIME, execute, fresh, read, sha256, write_json)
from scripts.launch.colorpeel_run import git_info

SAM = Path('/home/r12user5/Documents/Jiawei/colorpeel-runs/wood_spoon_sam_pink_v1/'
           '20261105__current_sam_pink__7726b73__42/segmentation/objects.json')
OLD = Path('/var/tmp/colorpeel-r12user5/runs/subject_trellis_spoons_v1/'
           '20261028-013740__assets__2943bf6__42/inputs/wood_spoon/cutout512.png')
SAM_SHA = '12083c0af6f128393779b8fc9217544418fd7673c832e81dbdab74eb1609805a'
OLD_SHA = 'c54251c91afc98ba3e9f626a71ac234f1350dc239bc36aedfd846295da168842'
SOURCE_SHA = '19eaf8df994f7f077c01fe19487b6d741f2e1c637c8cb4cc58da13d0b8fd7e2d'
MASK_SHA = 'd77b0c7e3e58606952633ffb9555e577556dd927b18b9f067398db7e36a212fc'


def run(output, gpu=0):
    revision = git_info()
    if sha256(SAM) != SAM_SHA or sha256(OLD) != OLD_SHA or gpu < 0:
        raise ValueError('require fixed current SAM and historical comparison cutout')
    objects = read(SAM)
    if len(objects['objects']) != 1 or objects['objects'][0]['id'] != 'object_001':
        raise ValueError('require the one frozen wood-spoon SAM instance')
    row = objects['objects'][0]
    if sha256(row['source_image']) != SOURCE_SHA or sha256(row['object_mask']) != MASK_SHA:
        raise ValueError('frozen photograph or SAM mask changed')
    output = fresh(output)
    state = {'status': 'running', 'git': revision, 'physical_gpu': gpu, 'seed': 42,
        'source': row['source_image'], 'source_sha256': SOURCE_SHA,
        'sam_manifest': str(SAM), 'sam_manifest_sha256': SAM_SHA, 'sam_mask_sha256': MASK_SHA,
        'policy': 'Frozen current SAM for identity; freshly infer pinned BiRefNet soft alpha; '
                  'joint native RGBA resize; pink hue0; one fresh TRELLIS mesh.',
        'historical_cutout_usage': 'read-only comparison after fresh matting; never a fallback input',
        'commands': [], 'training_started': False, 'subject_dataset_created': False,
        'quality_accepted_by_user': False}

    def stage(command, name):
        log = output / (name + '.log')
        state['stage'] = name
        state['commands'].append({'argv': [str(value) for value in command], 'log': str(log), 'gpu': gpu})
        write_json(output / 'status.json', state)
        execute(command, log, gpu)

    try:
        stage([RUNTIME['python'], '-B', '-m', 'src.methods.colorpeel_ice.single_image_subject',
               '--stage', 'inputs', '--image', row['source_image'], '--mask', row['object_mask'],
               '--output', output / 'input'], 'inputs')
        stage([RUNTIME['trellis_python'], '-B', '-m', 'src.methods.colorpeel_ice.subject_alpha',
               '--source-receipt', output / 'input/source_receipt.json', '--objects-json', SAM,
               '--object-id', row['id'], '--output', output / 'trellis_alpha'], 'soft_alpha')
        matte = read(output / 'trellis_alpha/source_receipt.json')
        with Image.open(matte['cutout_path']) as image, Image.open(OLD) as old:
            current, historical = np.asarray(image), np.asarray(old)
        delta = np.abs(current.astype(np.int16) - historical.astype(np.int16))
        state['fresh_matte_vs_historical'] = {
            'reference': str(OLD), 'reference_sha256': OLD_SHA,
            'rgba_pixels_equal': bool(np.array_equal(current, historical)),
            'alpha_pixels_equal': bool(np.array_equal(current[..., 3], historical[..., 3])),
            'changed_rgba_pixels': int(np.any(delta != 0, axis=-1).sum()),
            'max_channel_delta': int(delta.max()),
            'association': matte['conditioning_alpha']['association']}
        write_json(output / 'input_comparison.json', state['fresh_matte_vs_historical'])
        stage([RUNTIME['python'], '-B', '-m', 'experiments.subject_pink_input_repro_v1.prepare',
               '--source-receipt', output / 'trellis_alpha/source_receipt.json',
               '--output', output / 'trellis_input', '--hue-degrees', '0'], 'pink')
        stage([RUNTIME['trellis_python'], '-B', '-m', 'experiments.subject_trellis_counterfactual_v1.run',
               '--source-receipt', output / 'trellis_input/source_receipt.json',
               '--output', output / 'trellis', '--encoder', RUNTIME['trellis_encoder'],
               '--model-revisions', ROOT / 'experiments/subject_trellis_spoons_v1/model_revisions.json',
               '--seed', '42', '--skip-glb'], 'trellis')
        generated = read(output / 'trellis/repro_outputs/status.json')
        if generated['status'] != 'generated_pending_user_review' or generated['generated_assets'] != 1:
            raise ValueError('require exactly one completed native mesh')
        if (sha256(SAM) != SAM_SHA or sha256(OLD) != OLD_SHA
                or sha256(row['source_image']) != SOURCE_SHA
                or sha256(row['object_mask']) != MASK_SHA or git_info() != revision):
            raise ValueError('source evidence or checkout changed during verification')
        state.update(status='completed_pending_user_review', stage='completed', generated_assets=1,
                     native_asset=str(output / 'trellis/native_asset.pt'),
                     review=str(output / 'trellis/review/review.json'))
    except Exception as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(output / 'status.json', state)
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    run(args.output, args.gpu)
