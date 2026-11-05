"""Exercise production dual-color mesh preparation for one frozen SAM instance."""

import argparse
from pathlib import Path

from scripts.launch.colorpeel_run import git_info
from src.methods.colorpeel_ice.single_image_pipeline import (
    RUNTIME, execute, fresh, prepare_subject_geometry, read, sha256, write_json)
from src.methods.colorpeel_ice.single_image_subject import prepare_inputs


def run(objects_json, object_id, output, gpu):
    revision = git_info()
    objects_json = Path(objects_json).resolve()
    manifest_hash = sha256(objects_json)
    objects = read(objects_json)
    matches = [row for row in objects['objects'] if row['id'] == object_id]
    if objects['status'] not in ('succeeded', 'partial') or len(matches) != 1:
        raise ValueError('require one retained SAM instance from a completed segmentation')
    row = matches[0]
    if (sha256(row['source_image']) != objects['source']['canonical_image_sha256']
            or sha256(row['object_mask']) != row['object_mask_sha256']):
        raise ValueError('frozen photograph or SAM mask changed')
    output = fresh(output)
    state = {'status': 'preparing', 'git': revision, 'objects_json': str(objects_json),
             'objects_json_sha256': manifest_hash, 'object_id': object_id, 'gpu': gpu,
             'commands': [], 'training_started': False, 'subject_dataset_created': False,
             'quality_accepted_by_user': False}

    def stage(command, name):
        log = output / (name + '.log')
        state['stage'] = name
        state['commands'].append({'argv': [str(value) for value in command], 'log': str(log)})
        write_json(output / 'status.json', state)
        execute(command, log, gpu)

    try:
        prepare_inputs(row['source_image'], row['object_mask'], output / 'input')
        state['conditioning'] = prepare_subject_geometry(output, objects_json, object_id, RUNTIME, stage)
        if sha256(objects_json) != manifest_hash or git_info() != revision:
            raise ValueError('segmentation manifest or checkout changed during preparation')
        state.update(status='selected_pending_user_review', stage='completed')
    except Exception as error:
        selection = output / 'trellis_selection/selection.json'
        state.update(status='blocked' if selection.exists() and read(selection)['status'] == 'blocked' else 'failed',
                     error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(output / 'status.json', state)
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--objects-json', required=True, type=Path)
    parser.add_argument('--object-id', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    run(args.objects_json, args.object_id, args.output, args.gpu)
