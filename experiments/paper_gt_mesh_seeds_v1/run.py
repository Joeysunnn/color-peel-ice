"""Generate the sixteen user-requested GT mesh candidates with automatic cameras."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from html import escape
from pathlib import Path
from threading import Lock

from experiments.icbench_gt_alpha_four_v1.run import PREPARE_MODULE, VARIANTS, object_sheet
from experiments.icbench_gt_mesh_v1.run import MODEL_REVISIONS
from experiments.sam_six_seed_v1.run import compare_pre_alpha_rgb
from experiments.subject_trellis_counterfactual_v1.review_asset import PARAMETERS
from experiments.subject_trellis_counterfactual_v1.run import OFFICIAL
from scripts.launch.colorpeel_run import git_info
from src.methods.colorpeel_ice.single_image_pipeline import RUNTIME, execute, fresh, read, write_json
from src.methods.colorpeel_ice.single_image_subject import check_locks, prepare_inputs, sha256

OBJECT_KEY = 'image_000__mask_160'
SEEDS = (42, 43, 44, 45)


def validate_source(object_root):
    object_root = Path(object_root).resolve()
    recorded = read(object_root / 'REVIEW_STATE.json')
    if object_root.name != OBJECT_KEY or recorded['object_key'] != OBJECT_KEY:
        raise ValueError('require the reviewed image_000__mask_160 object')
    source = {key: recorded[key] for key in ('object_key', 'source_image', 'source_image_sha256',
                                            'target_gt_mask', 'target_gt_mask_sha256')}
    locks = {}
    for key, name in (('source_image', 'img.jpg'), ('target_gt_mask', '160.png')):
        path = Path(source[key])
        if not path.is_absolute() or path.parts[-2:] != ('0', name):
            raise ValueError('source path differs from the requested GT object')
        locks[str(path)] = source[key + '_sha256']
    check_locks(locks)
    return source, locks


def check_camera(trellis, rendered=None):
    review_path = Path(trellis) / 'review/review.json'
    trajectory_path = Path(trellis) / 'review/camera_fit_trajectory.json'
    review, trajectory = read(review_path), read(trajectory_path)
    camera = review['camera']
    for record in (camera, trajectory):
        if (record['fitting_mode'] != 'coarse72_then_powell'
                or record['optimizer_budget'] != {'maxfev': 350, 'maxiter': 6}):
            raise ValueError('require the existing automatic camera fit without warm start')
    evaluations = trajectory['evaluations']
    best = min(evaluations, key=lambda row: (row['loss'], row['parameters']))
    if (camera['evaluations'] != len(evaluations) or len(evaluations) < 72
            or camera['loss'] != best['loss']
            or [camera['parameters'][key] for key in PARAMETERS] != best['parameters']):
        raise ValueError('camera differs from the automatically selected trajectory minimum')
    if rendered is not None:
        if (rendered['reference_review'] != str(review_path)
                or any(rendered['cameras']['reference'][key] != camera[key]
                       for key in ('extrinsics', 'intrinsics'))):
            raise ValueError('render must use this mesh own automatic reference camera')
    return {'review': str(review_path), 'review_sha256': sha256(review_path),
            'trajectory': str(trajectory_path), 'trajectory_sha256': sha256(trajectory_path),
            'fitting_mode': camera['fitting_mode'], 'optimizer_budget': camera['optimizer_budget'],
            'evaluations': camera['evaluations'], 'parameters': camera['parameters'],
            'source_mask_iou': review['metrics']['original512_mask_iou']}


def gallery(output, state):
    cards = []
    for variant in state['variants']:
        statuses = '; '.join(f"seed {item['seed']}: {item['status']}" for item in variant['runs'])
        image = (f'<img src="variants/{variant["name"]}/comparison_clay.png" loading="lazy">'
                 if variant.get('comparison_clay') else '')
        cards.append(f'<section><h2>{variant["name"]}</h2><p>{escape(statuses)}</p>{image}</section>')
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8">'
        '<title>image_000__mask_160 GT16 mesh review</title>'
        '<style>body{font-family:sans-serif;margin:24px}img{width:100%;max-width:2048px}</style>'
        '<h1>image_000__mask_160: original / pink x GT hard / BiRef soft x seeds42-45</h1>'
        '<p>Rows: seeds42,43,44,45. Columns: automatic source view, yaw0, yaw+90, yaw-90. '
        'Each camera is fitted to that condition on the original image canvas. '
        'Mesh selection awaits user review.</p>' + ''.join(cards), encoding='utf8')


def run(object_root, output, gpus):
    object_root, output = Path(object_root).resolve(), Path(output).resolve()
    source, locks = validate_source(object_root)
    if not output.is_relative_to(object_root / 'mesh_candidates'):
        raise ValueError('output must remain under this object mesh_candidates folder')
    if not gpus or len(set(gpus)) != len(gpus) or any(type(gpu) is not int or gpu < 0 for gpu in gpus):
        raise ValueError('require distinct nonnegative physical GPUs')
    revision, output = git_info(), fresh(output)
    write_json(output / 'source_state.json', source)
    locks.update({str(MODEL_REVISIONS): sha256(MODEL_REVISIONS),
                  str(output / 'source_state.json'): sha256(output / 'source_state.json')})
    pinned = read(MODEL_REVISIONS)
    state = {'status': 'running', 'git': revision, 'source': source, 'source_locks': locks,
        'seeds': list(SEEDS), 'expected_meshes': 16, 'expected_view_renders': 64,
        'automatic_retries': 0, 'training_started': False, 'quality_accepted_by_user': False,
        'camera_policy': 'existing coarse72_then_powell; own condition original512 canvas; no warm start',
        'commands': [], 'preparation': {kind: {'status': 'pending'} for kind in ('hard', 'soft')},
        'variants': [{'name': name, 'gpu': gpus[index % len(gpus)], 'status': 'pending',
            'runs': [{'name': f'seed_{seed}', 'seed': seed, 'status': 'pending', 'generated': False,
                      'commands': []} for seed in SEEDS]} for index, name in enumerate(VARIANTS)]}
    lock, configuration = Lock(), []

    def update(record, **values):
        with lock:
            record.update(values)
            write_json(output / 'status.json', state)
            gallery(output, state)

    def stage(argv, log, gpu, record):
        command = [str(value) for value in argv]
        with lock:
            record['commands'].append({'argv': command, 'log': str(log), 'gpu': gpu})
            write_json(output / 'status.json', state)
        execute(command, log, gpu)

    def worker(gpu):
        for variant in (row for row in state['variants'] if row['gpu'] == gpu):
            folder = output / 'variants' / variant['name']
            folder.mkdir(parents=True)
            if variant['status'] == 'ready':
                for item in variant['runs']:
                    directory = folder / item['name']
                    directory.mkdir()
                    trellis, views = directory / 'trellis', directory / 'geometry_views'
                    try:
                        update(item, status='generating', trellis_run=str(trellis),
                               source_receipt=variant['source_receipt'],
                               source_receipt_sha256=variant['source_receipt_sha256'])
                        check_locks({variant['source_receipt']: variant['source_receipt_sha256']})
                        stage([RUNTIME['trellis_python'], '-B', '-m', 'experiments.subject_trellis_counterfactual_v1.run',
                            '--output', trellis, '--source-receipt', variant['source_receipt'],
                            '--encoder', RUNTIME['trellis_encoder'], '--model-revisions', MODEL_REVISIONS,
                            '--seed', item['seed'], '--skip-glb'], directory / 'generation.log', gpu, item)
                        generated = read(trellis / 'repro_outputs/status.json')
                        if (generated['status'] != 'generated_pending_user_review' or generated['generated_assets'] != 1
                                or generated['git'] != revision or generated['official_commit'] != OFFICIAL
                                or generated['seed'] != item['seed'] or generated['models']['revisions'] != pinned
                                or generated['source_receipt']['sha256'] != variant['source_receipt_sha256']
                                or 'diagnostic_fixed_crop_xyxy' in generated):
                            raise ValueError('generation configuration, seed or source differs from this experiment')
                        config = {key: generated[key] for key in ('models', 'pipeline_type', 'encoder_receipt_sha256',
                            'sampling', 'flow_dtype', 'sparse_algorithm')}
                        pipeline = read(trellis / 'pipeline.json')['args']
                        config.update({key: pipeline[key] for key in ('shape_slat_normalization', 'tex_slat_normalization')})
                        with lock:
                            if configuration and config != configuration[0]:
                                raise ValueError('non-seed generation settings differ across candidates')
                            if not configuration:
                                configuration.append(config)
                                state['generation_config'] = config
                        native = trellis / 'native_asset.pt'
                        update(item, status='rendering', generated=True, native_asset=str(native),
                               native_asset_sha256=sha256(native), camera=check_camera(trellis))
                        stage([RUNTIME['trellis_python'], '-B', '-m', 'experiments.mailbox_alpha_control_v1.render',
                            '--assets', native, '--labels', variant['name'] + '_' + item['name'],
                            '--reference-review', trellis / 'review/review.json', '--output', views, '--gpu', gpu],
                            directory / 'render.log', gpu, item)
                        rendered = read(views / 'status.json')
                        if (rendered['status'] != 'completed_pending_visual_review' or rendered['git'] != revision
                                or rendered['completed_renders'] != 4):
                            raise ValueError('four-view rendering did not complete under the same Git')
                        update(item, status='completed', completed_view_renders=4, camera=check_camera(trellis, rendered),
                               comparison_clay=str(views / 'comparison_clay.png'))
                    except Exception as error:
                        update(item, status='failed', error=f'{type(error).__name__}: {error}')
                update(variant, status='completed' if all(item['status'] == 'completed' for item in variant['runs']) else 'partial')
            update(variant, **object_sheet(output / 'variants', {'object_key': variant['name'], 'variants': variant['runs']}))

    update(state)
    try:
        prepare_inputs(source['source_image'], source['target_gt_mask'], output / 'input',
                       segmentation='supplied ICBench GT binary mask D1GT:0/160.png; unchanged')
        for kind in ('hard', 'soft'):
            selected = [variant for variant in state['variants'] if variant['name'].endswith('__' + kind)]
            prepared = output / (kind + '_inputs')
            try:
                update(state['preparation'][kind], status='preparing', output=str(prepared))
                stage([RUNTIME['trellis_python' if kind == 'soft' else 'python'], '-B', '-m', PREPARE_MODULE,
                    '--source-receipt', output / 'input/source_receipt.json', '--output', prepared,
                    '--mask-kind', 'GT', *(['--soft'] if kind == 'soft' else [])],
                    output / (kind + '_prepare.log'), gpus[0], state)
                if kind == 'soft' and state['preparation']['hard']['status'] == 'prepared':
                    parity = compare_pre_alpha_rgb(output)
                    update(state, rgb_parity=parity)
                    locks.update(parity['source_locks'])
                    if parity['status'] != 'passed':
                        raise ValueError('hard/soft pre-alpha RGB differs')
                for variant in selected:
                    receipt = str(prepared / variant['name'] / 'source_receipt.json')
                    locks[receipt] = sha256(receipt)
                    update(variant, status='ready', source_receipt=receipt, source_receipt_sha256=locks[receipt])
                update(state['preparation'][kind], status='prepared')
            except Exception as error:
                reason = f'{kind} preparation: {type(error).__name__}: {error}'
                update(state['preparation'][kind], status='failed', error=reason)
                for variant in selected:
                    update(variant, status='blocked', error=reason)
                    for item in variant['runs']:
                        update(item, status='blocked', error=reason)
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            list(pool.map(worker, gpus))
        check_locks(locks)
        if git_info() != revision:
            raise ValueError('checkout changed during the experiment')
        runs = [item for variant in state['variants'] for item in variant['runs']]
        complete = all(item['status'] == 'completed' for item in runs)
        update(state, status='completed_pending_user_review' if complete else 'partial',
            generated_meshes=sum(item['generated'] for item in runs),
            completed_mesh_runs=sum(item['status'] == 'completed' for item in runs),
            completed_view_renders=sum(item.get('completed_view_renders', 0) for item in runs))
        if not complete:
            raise RuntimeError('GT16 experiment is partial; failed/blocked candidates retained')
    except Exception as error:
        update(state, status='partial' if state['status'] == 'partial' else 'failed',
               error=f'{type(error).__name__}: {error}')
        raise
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3])
    args = parser.parse_args()
    run(args.object_root, args.output, args.gpus)
