"""Generate one pink/GT-binary-alpha mesh and four views per selected ICBench object."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from html import escape
from pathlib import Path
from threading import Lock

from experiments.subject_pink_input_repro_v1.prepare import prepare as prepare_pink
from experiments.subject_trellis_counterfactual_v1.run import OFFICIAL
from scripts.launch.colorpeel_run import git_info
from src.methods.colorpeel_ice.single_image_pipeline import ROOT, RUNTIME, execute, fresh, read, write_json
from src.methods.colorpeel_ice.single_image_subject import check_locks, prepare_inputs, sha256

MODEL_REVISIONS = ROOT / 'experiments/subject_trellis_spoons_v1/model_revisions.json'


def validate_selection(selection):
    objects = selection['objects']
    keys = [row['object_key'] for row in objects]
    if (selection['schema'] != 'icbench_gt36_selection/v1' or len(objects) != 36
            or len(set(keys)) != 36 or len({row['sample_id'] for row in objects}) != 29):
        raise ValueError('require the workbook-selected 36 GT objects from 29 images')
    locks = {}
    for row in objects:
        sample, mask = row['sample_id'], row['mask_id']
        if (type(sample) is not int or type(mask) is not int or min(sample, mask) < 0
                or row['object_key'] != f'image_{sample:03d}__mask_{mask:03d}'
                or row['stable_id'] != f'D1GT:{sample}/{mask}.png'):
            raise ValueError('GT object identity differs from its sample/mask mapping')
        for kind, name in (('image', 'img.jpg'), ('mask', f'{mask}.png')):
            source = row[kind]
            path = Path(source['path'])
            if not path.is_absolute() or path.parts[-2:] != (str(sample), name):
                raise ValueError('source path differs from the recorded GT object identity')
            if str(path) in locks and locks[str(path)] != source['sha256']:
                raise ValueError('conflicting source hashes')
            locks[str(path)] = source['sha256']
    check_locks(locks)
    return locks


def gallery(output, state):
    cards = []
    for row in state['objects']:
        key = row['object_key']
        image = (f'<img loading="lazy" src="{key}/geometry_views/comparison_clay.png" '
                 f'alt="{key}: four mesh views">' if row['status'] == 'completed' else '')
        cards.append(f'<section><h2>{key} · {escape(row["status"])}</h2>'
                     f'<p>{escape(row.get("error", ""))}</p>{image}</section>')
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8">'
        '<title>ICBench 36 GT objects — pink / binary alpha</title>'
        '<style>body{font-family:sans-serif;margin:24px;background:#eee}section{background:white;'
        'padding:12px;margin:16px 0}img{width:100%;max-width:2048px}h2{font-size:18px}</style>'
        '<h1>Pink RGB + GT binary alpha · TRELLIS.2 seed 42</h1>'
        '<p>Reference, yaw 0, yaw +90, yaw -90. Geometry quality awaits user review.</p>'
        + ''.join(cards), encoding='utf8')


def run(selection_path, output, gpus):
    selection_path = Path(selection_path).resolve()
    selection = read(selection_path)
    locks = validate_selection(selection)
    if not gpus or len(set(gpus)) != len(gpus) or any(type(gpu) is not int or gpu < 0 for gpu in gpus):
        raise ValueError('require distinct nonnegative physical GPUs')
    revision, output = git_info(), fresh(output)
    locks.update({str(selection_path): sha256(selection_path), str(MODEL_REVISIONS): sha256(MODEL_REVISIONS)})
    pinned = read(MODEL_REVISIONS)
    state = {'status': 'running', 'git': revision, 'selection': selection, 'source_locks': locks,
        'seed': 42, 'expected_meshes': 36, 'expected_view_renders': 144,
        'protocol': 'pink hue=0; GT binary alpha 0/255; official alpha automatic crop',
        'automatic_retries': 0, 'training_started': False, 'quality_accepted_by_user': False,
        'objects': [{**row, 'gpu': gpus[index % len(gpus)], 'status': 'pending', 'commands': []}
                    for index, row in enumerate(selection['objects'])]}
    lock = Lock()

    def update(row, **values):
        with lock:
            row.update(values)
            write_json(output / 'status.json', state)
            gallery(output, state)

    def worker(gpu):
        for row in (row for row in state['objects'] if row['gpu'] == gpu):
            folder = output / row['object_key']
            folder.mkdir()

            def stage(argv, name):
                command = [str(value) for value in argv]
                log = folder / (name + '.log')
                with lock:
                    row['commands'].append({'argv': command, 'log': str(log), 'gpu': gpu})
                    write_json(output / 'status.json', state)
                execute(command, log, gpu)

            try:
                update(row, status='preparing')
                prepare_inputs(row['image']['path'], row['mask']['path'], folder / 'input',
                    segmentation=f'supplied ICBench GT binary mask {row["stable_id"]}; unchanged')
                prepare_pink(folder / 'input/source_receipt.json', folder / 'pink', hue_degrees=0.)
                receipt = folder / 'pink/source_receipt.json'
                trellis = folder / 'trellis'
                update(row, status='generating', source_receipt=str(receipt),
                       source_receipt_sha256=sha256(receipt))
                stage([RUNTIME['trellis_python'], '-B', '-m', 'experiments.subject_trellis_counterfactual_v1.run',
                    '--output', trellis, '--source-receipt', receipt, '--encoder', RUNTIME['trellis_encoder'],
                    '--model-revisions', MODEL_REVISIONS, '--seed', '42', '--skip-glb'], 'generation')
                generated = read(trellis / 'repro_outputs/status.json')
                if (generated['status'] != 'generated_pending_user_review' or generated['generated_assets'] != 1
                        or generated['git'] != revision
                        or generated['official_commit'] != OFFICIAL or generated['seed'] != 42
                        or generated['models']['revisions'] != pinned
                        or generated['source_receipt']['sha256'] != row['source_receipt_sha256']
                        or 'diagnostic_fixed_crop_xyxy' in generated):
                    raise ValueError('generation status or configuration differs from this experiment')
                update(row, status='rendering', native_asset=str(trellis / 'native_asset.pt'))
                views = folder / 'geometry_views'
                stage([RUNTIME['trellis_python'], '-B', '-m', 'experiments.mailbox_alpha_control_v1.render',
                    '--assets', trellis / 'native_asset.pt', '--labels', row['object_key'],
                    '--reference-review', trellis / 'review/review.json', '--output', views, '--gpu', gpu], 'render')
                rendered = read(views / 'status.json')
                if (rendered['status'] != 'completed_pending_visual_review' or rendered['git'] != revision
                        or rendered['completed_renders'] != 4):
                    raise ValueError('four-view rendering did not complete under the same Git')
                update(row, status='completed', comparison_clay=str(views / 'comparison_clay.png'))
            except Exception as error:
                update(row, status='failed', error=f'{type(error).__name__}: {error}')

    update(state)
    try:
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            list(pool.map(worker, gpus))
        check_locks(locks)
        if git_info() != revision:
            raise ValueError('checkout changed during the experiment')
        complete = all(row['status'] == 'completed' for row in state['objects'])
        update(state, status='completed_pending_user_review' if complete else 'partial',
               completed_objects=sum(row['status'] == 'completed' for row in state['objects']))
        if not complete:
            raise RuntimeError('GT mesh experiment is partial; failures retained in status.json')
    except Exception as error:
        update(state, status='partial' if state['status'] == 'partial' else 'failed',
               error=f'{type(error).__name__}: {error}')
        raise
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3])
    args = parser.parse_args()
    run(args.selection, args.output, args.gpus)
