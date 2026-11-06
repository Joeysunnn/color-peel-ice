"""Compare four fixed seeds for four conditions of six already prepared SAM instances."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from html import escape
from pathlib import Path
from threading import Lock

from PIL import Image

from experiments.icbench_gt_alpha_four_v1.run import PREPARE_MODULE, VARIANTS, object_sheet as four_row_sheet
from experiments.icbench_gt_mesh_v1.run import MODEL_REVISIONS
from experiments.subject_trellis_counterfactual_v1.run import OFFICIAL
from scripts.launch.colorpeel_run import git_info
from src.methods.colorpeel_ice.single_image_pipeline import RUNTIME, execute, fresh, read, write_json
from src.methods.colorpeel_ice.single_image_subject import check_locks, sha256

SEEDS = (42, 43, 44, 45)
OBJECT_KEYS = ('image_000__mask_160', 'image_001__mask_008', 'image_003__mask_207',
               'image_009__mask_205', 'image_046__mask_097', 'image_081__mask_130')


def validate_inputs(inputs, revision):
    objects = inputs['objects']
    if (inputs['schema'] != 'sam_six_seed_inputs/v1' or inputs['git'] != revision
            or len(objects) != 6 or {row['object_key'] for row in objects} != set(OBJECT_KEYS)):
        raise ValueError('require the six fixed SAM instances and the same input-producer Git')
    locks = dict(inputs.get('source_locks', {}))
    for row in objects:
        if 'blocked_reason' in row:
            if not isinstance(row['blocked_reason'], str) or not row['blocked_reason'].strip():
                raise ValueError('blocked SAM target requires an explicit reason')
            continue
        path = Path(row['source_receipt'])
        if not path.is_absolute():
            raise ValueError('semantic source receipt must be absolute')
        if str(path) in locks and locks[str(path)] != row['source_receipt_sha256']:
            raise ValueError('conflicting semantic receipt source lock')
        locks[str(path)] = row['source_receipt_sha256']
    check_locks(locks)
    return locks


def variant_sheet(output, row, variant):
    folder = output / row['object_key'] / 'variants'
    (folder / variant['name']).mkdir(parents=True, exist_ok=True)
    return four_row_sheet(folder, {'object_key': variant['name'], 'variants': variant['runs']})


def compare_pre_alpha_rgb(folder):
    colors, locks = {}, {}
    for color in ('original', 'pink'):
        paths = {kind: folder / (kind + '_inputs') / (color + '__' + kind) / 'pre_alpha_rgb512.png'
                 for kind in ('hard', 'soft')}
        hashes = {kind: sha256(path) for kind, path in paths.items()}
        with Image.open(paths['hard']) as hard, Image.open(paths['soft']) as soft:
            equal = hard.mode == soft.mode and hard.size == soft.size and hard.tobytes() == soft.tobytes()
            colors[color] = {'decoded_pixels_equal': equal, 'hard_mode': hard.mode, 'soft_mode': soft.mode,
                'hard_size': list(hard.size), 'soft_size': list(soft.size),
                'png_sha256': hashes, 'png_files_equal': hashes['hard'] == hashes['soft']}
        locks.update({str(path): hashes[kind] for kind, path in paths.items()})
    return {'status': 'passed' if all(row['decoded_pixels_equal'] for row in colors.values()) else 'mismatch',
            'comparison': 'PIL decoded mode, size and bytes; PNG encoding differences are allowed',
            'colors': colors, 'source_locks': locks}


def gallery(output, state):
    cards = []
    for row in state['objects']:
        parts = [f'<section><h2>{row["object_key"]} · {escape(row["status"])}</h2>']
        for variant in row['variants']:
            statuses = '; '.join(f"seed {run['seed']}: {run['status']}" for run in variant['runs'])
            parts.append(f'<h3>{variant["name"]} · {escape(variant["status"])}</h3><p>{escape(statuses)}</p>')
            if variant.get('comparison_clay'):
                parts.append(f'<img loading="lazy" src="{row["object_key"]}/variants/{variant["name"]}/comparison_clay.png" '
                             f'alt="{variant["name"]}: four seeds by four views">')
        cards.append(''.join(parts) + '</section>')
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8">'
        '<title>Six SAM instances: four conditions and four seeds</title>'
        '<style>body{font-family:sans-serif;margin:24px;background:#eee}section{background:white;'
        'padding:12px;margin:16px 0}img{width:100%;max-width:2048px}</style>'
        '<h1>Original / pink × SAM hard / BiRef soft × seeds42–45</h1>'
        '<p>Each image has rows42,43,44,45 and columns: own source reference, yaw0, yaw+90, yaw−90. '
        'Reference cameras are independently fitted; the three canonical views share fixed parameters. '
        'No quality ranking or seed selection is performed.</p>' + ''.join(cards), encoding='utf8')


def run(inputs_path, output, gpus):
    inputs_path = Path(inputs_path).resolve()
    inputs, revision = read(inputs_path), git_info()
    locks = validate_inputs(inputs, revision)
    if not gpus or len(set(gpus)) != len(gpus) or any(type(gpu) is not int or gpu < 0 for gpu in gpus):
        raise ValueError('require distinct nonnegative physical GPUs')
    output = fresh(output)
    locks.update({str(inputs_path): sha256(inputs_path), str(MODEL_REVISIONS): sha256(MODEL_REVISIONS)})
    pinned = read(MODEL_REVISIONS)
    by_key = {row['object_key']: row for row in inputs['objects']}
    state = {'status': 'running', 'git': revision, 'inputs': inputs, 'source_locks': locks,
        'seeds': list(SEEDS), 'expected_meshes': 96, 'expected_view_renders': 384,
        'automatic_retries': 0, 'training_started': False, 'quality_accepted_by_user': False,
        'objects': [{**by_key[key], 'gpu': gpus[index % len(gpus)], 'status': 'pending', 'commands': [],
            'preparation': {kind: {'status': 'pending'} for kind in ('hard', 'soft')},
            'variants': [{'name': name, 'status': 'pending', 'runs': [
                {'name': f'seed_{seed}', 'seed': seed, 'status': 'pending', 'generated': False, 'commands': []}
                for seed in SEEDS]} for name in VARIANTS]} for index, key in enumerate(OBJECT_KEYS)]}
    lock, configuration = Lock(), []

    def update(row, **values):
        with lock:
            row.update(values)
            write_json(output / 'status.json', state)
            gallery(output, state)

    def block(variant, reason):
        for item in variant['runs']:
            update(item, status='blocked', error=reason)
        update(variant, status='blocked', error=reason)

    def worker(gpu):
        for row in (row for row in state['objects'] if row['gpu'] == gpu):
            folder = output / row['object_key']
            folder.mkdir()

            def stage(argv, name, record):
                command, log = [str(value) for value in argv], folder / (name + '.log')
                with lock:
                    record['commands'].append({'argv': command, 'log': str(log), 'gpu': gpu})
                    write_json(output / 'status.json', state)
                execute(command, log, gpu)

            if row.get('blocked_reason'):
                for variant in row['variants']:
                    block(variant, row['blocked_reason'])
            else:
                update(row, status='preparing')
                for kind in ('hard', 'soft'):
                    selected = [variant for variant in row['variants'] if variant['name'].endswith('__' + kind)]
                    prepared = folder / (kind + '_inputs')
                    try:
                        update(row['preparation'][kind], status='preparing', output=str(prepared))
                        check_locks({row['source_receipt']: row['source_receipt_sha256']})
                        stage([RUNTIME['trellis_python' if kind == 'soft' else 'python'], '-B', '-m', PREPARE_MODULE,
                            '--source-receipt', row['source_receipt'], '--output', prepared, '--mask-kind', 'SAM',
                            *(['--soft'] if kind == 'soft' else [])], kind + '_prepare', row)
                        if kind == 'soft':
                            if row['preparation']['hard']['status'] == 'prepared':
                                parity = compare_pre_alpha_rgb(folder)
                                update(row, rgb_parity=parity)
                                with lock:
                                    locks.update(parity['source_locks'])
                                if parity['status'] != 'passed':
                                    raise ValueError('hard/soft pre-alpha RGB differs; soft blocked without changing either input')
                            else:
                                update(row, rgb_parity={'status': 'unavailable',
                                    'reason': 'Hard preparation failed; no hard counterpart is available to compare.'})
                        receipts = {str(prepared / variant['name'] / 'source_receipt.json'):
                                    sha256(prepared / variant['name'] / 'source_receipt.json') for variant in selected}
                        with lock:
                            locks.update(receipts)
                        for variant in selected:
                            receipt = str(prepared / variant['name'] / 'source_receipt.json')
                            update(variant, status='ready', source_receipt=receipt, source_receipt_sha256=receipts[receipt])
                        update(row['preparation'][kind], status='prepared')
                    except Exception as error:
                        reason = f'{kind} preparation: {type(error).__name__}: {error}'
                        update(row['preparation'][kind], status='failed', error=reason)
                        for variant in selected:
                            block(variant, reason)
            for variant in row['variants']:
                if variant['status'] == 'ready':
                    update(variant, status='processing')
                    for item in variant['runs']:
                        directory = folder / 'variants' / variant['name'] / item['name']
                        try:
                            directory.mkdir(parents=True)
                            trellis, views = directory / 'trellis', directory / 'geometry_views'
                            update(item, status='generating', trellis_run=str(trellis),
                                   source_receipt=variant['source_receipt'], source_receipt_sha256=variant['source_receipt_sha256'])
                            check_locks({variant['source_receipt']: variant['source_receipt_sha256']})
                            stage([RUNTIME['trellis_python'], '-B', '-m', 'experiments.subject_trellis_counterfactual_v1.run',
                                '--output', trellis, '--source-receipt', variant['source_receipt'],
                                '--encoder', RUNTIME['trellis_encoder'], '--model-revisions', MODEL_REVISIONS,
                                '--seed', item['seed'], '--skip-glb'], variant['name'] + '_' + item['name'] + '_generation', item)
                            generated = read(trellis / 'repro_outputs/status.json')
                            if (generated['status'] != 'generated_pending_user_review' or generated['generated_assets'] != 1
                                    or generated['git'] != revision or generated['official_commit'] != OFFICIAL
                                    or generated['seed'] != item['seed'] or generated['models']['revisions'] != pinned
                                    or generated['source_receipt']['sha256'] != variant['source_receipt_sha256']
                                    or 'diagnostic_fixed_crop_xyxy' in generated):
                                raise ValueError('generation configuration, seed or input differs from this experiment')
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
                            update(item, status='rendering', generated=True, native_asset=str(native), native_asset_sha256=sha256(native))
                            stage([RUNTIME['trellis_python'], '-B', '-m', 'experiments.mailbox_alpha_control_v1.render',
                                '--assets', native, '--labels', variant['name'] + '_' + item['name'],
                                '--reference-review', trellis / 'review/review.json', '--output', views, '--gpu', gpu],
                                variant['name'] + '_' + item['name'] + '_render', item)
                            rendered = read(views / 'status.json')
                            if (rendered['status'] != 'completed_pending_visual_review' or rendered['git'] != revision
                                    or rendered['completed_renders'] != 4):
                                raise ValueError('four-view rendering did not complete under the same Git')
                            update(item, status='completed', completed_view_renders=4,
                                   comparison_clay=str(views / 'comparison_clay.png'))
                        except Exception as error:
                            update(item, status='failed', error=f'{type(error).__name__}: {error}')
                    update(variant, status='completed' if all(item['status'] == 'completed' for item in variant['runs']) else 'partial')
                try:
                    update(variant, **variant_sheet(output, row, variant))
                except Exception as error:
                    update(variant, status='failed', error=f'Comparison sheet: {type(error).__name__}: {error}')
            update(row, status='completed' if all(variant['status'] == 'completed' for variant in row['variants']) else 'partial')

    update(state)
    try:
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            list(pool.map(worker, gpus))
        check_locks(locks)
        if git_info() != revision:
            raise ValueError('checkout changed during the experiment')
        variants = [variant for row in state['objects'] for variant in row['variants']]
        runs = [item for variant in variants for item in variant['runs']]
        complete = all(row['status'] == 'completed' for row in state['objects'])
        update(state, status='completed_pending_user_review' if complete else 'partial',
            generated_meshes=sum(item['generated'] for item in runs),
            completed_mesh_runs=sum(item['status'] == 'completed' for item in runs),
            completed_variants=sum(variant['status'] == 'completed' for variant in variants),
            completed_view_renders=sum(item.get('completed_view_renders', 0) for item in runs),
            completed_objects=sum(row['status'] == 'completed' for row in state['objects']))
        if not complete:
            raise RuntimeError('SAM six-instance seed experiment is partial; failed/blocked evidence retained')
    except Exception as error:
        update(state, status='partial' if state['status'] == 'partial' else 'failed',
               error=f'{type(error).__name__}: {error}')
        raise
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3])
    args = parser.parse_args()
    run(args.inputs, args.output, args.gpus)
