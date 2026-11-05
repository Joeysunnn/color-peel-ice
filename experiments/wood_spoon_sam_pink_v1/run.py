"""Current Scheme A masks and pink TRELLIS conditions for the full wood spoon."""

import argparse
from pathlib import Path

from src.methods.colorpeel_ice.single_image_pipeline import (
    ENTRY, ROOT, RUNTIME, execute, fresh, read, sha256, write_json)
from scripts.launch.colorpeel_run import git_info


def run(output, gpu=0):
    revision = git_info()
    source = ROOT / 'experiments/natural_material_threeway_v1/assets/wood_spoon_source.png'
    expected = 'b17fb667be977d2ed879f3c216011e6dde8324951a6f331136a7c52ab2bc4f9d'
    if sha256(source) != expected or gpu < 0:
        raise ValueError('require the frozen full wood-spoon photograph and a valid GPU')
    output = fresh(output)
    state = {'status': 'running', 'git': revision, 'source': str(source), 'source_sha256': expected,
        'scheme': 'A', 'conditioning': 'original photo -> LCh pink hue0; unchanged binary SAM alpha',
        'seed': 42, 'physical_gpu': gpu, 'objects': [], 'commands': [],
        'training_started': False, 'subject_dataset_created': False, 'quality_accepted_by_user': False}

    def stage(command, name):
        log = output / (name + '.log')
        state['stage'] = name
        state['commands'].append({'argv': [str(value) for value in command], 'log': str(log), 'gpu': gpu})
        write_json(output / 'status.json', state)
        execute(command, log, gpu)

    try:
        stage([RUNTIME['vlm_python'], '-B', ENTRY, '--worker', 'discover', '--image', source,
               '--output', output / 'discovery'], 'discovery')
        stage([RUNTIME['python'], '-B', ENTRY, '--worker', 'segment', '--image', source,
               '--prepared', output / 'discovery/categories.json', '--output', output / 'segmentation'], 'segmentation')
        segmented = read(output / 'segmentation/objects.json')
        if segmented['source']['image_sha256'] != expected or not segmented['objects']:
            raise ValueError('require fresh usable Scheme A masks for the full source')
        state['segmentation_status'] = segmented['status']
        state['segmentation_failures'] = segmented['failures']
        for record in segmented['objects']:
            folder = output / record['id']
            item = {'id': record['id'], 'category': record['category'], 'segmentation': record, 'status': 'preparing'}
            state['objects'].append(item)
            if (sha256(record['source_image']) != segmented['source']['canonical_image_sha256']
                    or sha256(record['object_mask']) != record['object_mask_sha256']):
                raise ValueError('canonical source or SAM mask changed')
            stage([RUNTIME['python'], '-B', '-m', 'src.methods.colorpeel_ice.single_image_subject',
                   '--stage', 'inputs', '--image', record['source_image'], '--mask', record['object_mask'],
                   '--output', folder / 'input'], record['id'] + '_inputs')
            stage([RUNTIME['python'], '-B', '-m', 'experiments.subject_pink_input_repro_v1.prepare',
                   '--source-receipt', folder / 'input/source_receipt.json', '--output', folder / 'trellis_input',
                   '--hue-degrees', '0'], record['id'] + '_pink')
            stage([RUNTIME['trellis_python'], '-B', '-m', 'experiments.subject_trellis_counterfactual_v1.run',
                   '--source-receipt', folder / 'trellis_input/source_receipt.json', '--output', folder / 'trellis',
                   '--encoder', RUNTIME['trellis_encoder'], '--model-revisions',
                   ROOT / 'experiments/subject_trellis_spoons_v1/model_revisions.json', '--seed', '42', '--skip-glb'],
                  record['id'] + '_trellis')
            generated = read(folder / 'trellis/repro_outputs/status.json')
            if generated['status'] != 'generated_pending_user_review' or generated['generated_assets'] != 1:
                raise ValueError('one complete native mesh is required per retained instance')
            item.update(status='generated_pending_user_review', native_asset=str(folder / 'trellis/native_asset.pt'),
                        review=str(folder / 'trellis/review/review.json'),
                        pink_receipt=str(folder / 'trellis_input/source_receipt.json'))
        if sha256(source) != expected or git_info() != revision:
            raise ValueError('source or checkout changed')
        state.update(status='completed_pending_user_review' if segmented['status'] == 'succeeded' else 'partial',
                     stage='completed', generated_assets=len(state['objects']))
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
