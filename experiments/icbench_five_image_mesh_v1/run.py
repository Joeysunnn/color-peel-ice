"""Discover all Scheme A instances in five sampled images, then compare eight meshes each."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import random
from threading import Lock

from scripts.launch.colorpeel_run import git_info
from src.methods.colorpeel_ice.single_image_pipeline import ENTRY, RUNTIME, execute, read, write_json
from src.methods.colorpeel_ice.single_image_subject import check_locks, fresh, prepare_inputs, sha256


def validate_selection(selection):
    pool = selection['eligible_image_ids']
    ids = [row['id'] for row in selection['images']]
    if (len(pool) != 29 or pool != sorted(set(pool)) or len(ids) != 5
            or random.Random(selection['sampling_seed']).sample(pool, 5) != ids):
        raise ValueError('require the recorded single five-image sample from the 29 eligible images')
    check_locks({row['path']: row['sha256'] for row in selection['images']})


def intake(selection_path, output, gpus):
    selection_path = Path(selection_path).resolve()
    selection = read(selection_path)
    validate_selection(selection)
    if not gpus or len(set(gpus)) != len(gpus) or any(type(gpu) is not int or gpu < 0 for gpu in gpus):
        raise ValueError('require distinct nonnegative physical GPUs')
    revision, output = git_info(), fresh(output)
    locks = {str(selection_path): sha256(selection_path),
             **{row['path']: row['sha256'] for row in selection['images']}}
    state = {'status': 'running', 'stage': 'intake', 'git': revision,
        'selection': selection, 'source_locks': locks, 'images': [],
        'training_started': False, 'automatic_retries': 0, 'quality_accepted_by_user': False}
    for index, source in enumerate(selection['images']):
        state['images'].append({'id': source['id'], 'source': source['path'],
            'source_sha256': source['sha256'], 'gpu': gpus[index % len(gpus)],
            'status': 'pending', 'objects': [], 'commands': []})
    lock = Lock()

    def update(row, **values):
        with lock:
            row.update(values)
            write_json(output / 'status.json', state)

    def worker(gpu):
        for row in (row for row in state['images'] if row['gpu'] == gpu):
            folder = output / f"image_{row['id']:03d}"
            folder.mkdir()

            def stage(command, name):
                command = [str(value) for value in command]
                log = folder / (name + '.log')
                with lock:
                    row['commands'].append({'argv': command, 'log': str(log), 'gpu': gpu})
                    write_json(output / 'status.json', state)
                execute(command, log, gpu)

            try:
                update(row, status='discovering')
                stage([RUNTIME['vlm_python'], '-B', ENTRY, '--worker', 'discover',
                    '--image', row['source'], '--output', folder / 'discovery'], 'discovery')
                update(row, status='segmenting')
                stage([RUNTIME['python'], '-B', ENTRY, '--worker', 'segment',
                    '--image', row['source'], '--prepared', folder / 'discovery/categories.json',
                    '--output', folder / 'segmentation'], 'segmentation')
                objects_path = folder / 'segmentation/objects.json'
                segmented = read(objects_path)
                if segmented['source']['image_sha256'] != row['source_sha256']:
                    raise ValueError('segmentation source differs from the sampled input')
                update(row, status='alpha', segmentation_status=segmented['status'],
                       segmentation_failures=segmented['failures'], segmentation_rejected=segmented['rejected'])
                for record in segmented['objects']:
                    key = f"image_{row['id']:03d}__{record['id']}"
                    directory = folder / 'objects' / record['id']
                    directory.mkdir(parents=True)
                    item = {'key': key, 'id': record['id'], 'category': record['category'],
                        'status': 'preparing', 'segmentation_record': record,
                        'matte_receipt': str(directory / 'trellis_alpha/source_receipt.json')}
                    with lock:
                        row['objects'].append(item)
                        write_json(output / 'status.json', state)
                    try:
                        prepare_inputs(record['source_image'], record['object_mask'], directory / 'input')
                        stage([RUNTIME['trellis_python'], '-B', '-m', 'src.methods.colorpeel_ice.subject_alpha',
                            '--source-receipt', directory / 'input/source_receipt.json',
                            '--objects-json', objects_path, '--object-id', record['id'],
                            '--output', directory / 'trellis_alpha'], record['id'] + '_alpha')
                        update(item, status='ready')
                    except Exception as error:
                        update(item, status='blocked', error=f'{type(error).__name__}: {error}')
                update(row, status='completed')
            except Exception as error:
                update(row, status='failed', error=f'{type(error).__name__}: {error}')

    write_json(output / 'status.json', state)
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        list(pool.map(worker, gpus))
    check_locks(locks)
    if git_info() != revision:
        raise ValueError('checkout changed during intake')
    objects = {}
    for row in state['images']:
        for item in row['objects']:
            sources = {'matte_receipt': item['matte_receipt'], 'source_rgb': True, 'reference_review': None}
            if item['status'] != 'ready':
                sources['blocked_reason'] = item['error']
            objects[item['key']] = sources
    write_json(output / 'factorial_objects.json', objects)
    locks[str(output / 'factorial_objects.json')] = sha256(output / 'factorial_objects.json')
    complete = all(row['status'] == 'completed' and row.get('segmentation_status') == 'succeeded'
                   and all(item['status'] == 'ready' for item in row['objects']) for row in state['images'])
    update(state, status='ready' if complete and objects else 'partial', stage='completed',
           retained_objects=len(objects), ready_objects=sum('blocked_reason' not in row for row in objects.values()),
           expected_meshes=8 * len(objects), objects_manifest=str(output / 'factorial_objects.json'))
    return state


def meshes(intake_path, output, gpus):
    from experiments.subject_conditioning_factorial_v1.run import run
    intake_path = Path(intake_path).resolve()
    state = read(intake_path / 'status.json')
    if state['stage'] != 'completed' or state['git'] != git_info():
        raise ValueError('require completed intake and the same unchanged Git')
    check_locks(state['source_locks'])
    objects = read(intake_path / 'factorial_objects.json')
    if not objects:
        raise ValueError('no retained A instances; preserve intake failures without resampling images')
    return run(output, gpus, objects=objects)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('intake', 'meshes'), required=True)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--intake', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3])
    args = parser.parse_args()
    if args.stage == 'intake':
        if args.selection is None:
            parser.error('--selection is required for intake')
        intake(args.selection, args.output, args.gpus)
    else:
        if args.intake is None:
            parser.error('--intake is required for meshes')
        meshes(args.intake, args.output, args.gpus)
