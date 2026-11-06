"""Generate original/pink x GT-hard/BiRef-soft meshes and four views per variant."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from html import escape
from pathlib import Path
from threading import Lock

from PIL import Image, ImageDraw

from experiments.icbench_gt_mesh_v1.run import MODEL_REVISIONS, validate_selection
from experiments.subject_trellis_counterfactual_v1.run import OFFICIAL
from scripts.launch.colorpeel_run import git_info
from src.methods.colorpeel_ice.single_image_pipeline import RUNTIME, execute, fresh, read, write_json
from src.methods.colorpeel_ice.single_image_subject import check_locks, prepare_inputs, sha256

VARIANTS = ('original__hard', 'pink__hard', 'original__soft', 'pink__soft')
PREPARE_MODULE = 'experiments.icbench_gt_alpha_four_v1.prepare'


def object_sheet(output, row):
    """Keep all four rows visible, including failed/blocked variants."""
    size = (2048, 544)  # Existing single-mesh renderer: four512 views and a32px label strip.
    sheet = Image.new('RGB', (size[0], size[1] * 4), '#eeeeee')
    draw = ImageDraw.Draw(sheet)
    for index, variant in enumerate(row['variants']):
        y = index * size[1]
        if variant['status'] == 'completed':
            with Image.open(variant['comparison_clay']) as image:
                if image.size != size:
                    raise ValueError('single-mesh four-view sheet dimensions changed')
                sheet.paste(image.convert('RGB'), (0, y))
        else:
            draw.text((20, y + 20), f"{variant['name']} / {variant['status']}", fill='black')
            draw.text((20, y + 55), variant.get('error', 'No completed four-view render')[:300], fill='black')
    path = output / row['object_key'] / 'comparison_clay.png'
    sheet.save(path)
    return {'comparison_clay': str(path), 'comparison_clay_sha256': sha256(path)}


def gallery(output, state):
    cards = []
    for row in state['objects']:
        key = row['object_key']
        image = (f'<img loading="lazy" src="{key}/comparison_clay.png" alt="{key}: four variants, four views each">'
                 if row.get('comparison_clay') else '')
        statuses = '; '.join(f"{variant['name']}: {variant['status']}" for variant in row['variants'])
        cards.append(f'<section><h2>{key} · {escape(row["status"])}</h2><p>{escape(statuses)}</p>'
                     f'<p>{escape(row.get("error", ""))}</p>{image}</section>')
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8">'
        '<title>ICBench GT36: original/pink and hard/soft alpha</title>'
        '<style>body{font-family:sans-serif;margin:24px;background:#eee}section{background:white;'
        'padding:12px;margin:16px 0}img{width:100%;max-width:2048px}h2{font-size:18px}</style>'
        '<h1>Original / pink × GT hard / BiRef soft · seed42</h1>'
        '<p>Rows: original hard, pink hard, original soft, pink soft. Each row: its own source reference, '
        'yaw0, yaw+90, yaw−90. Reference cameras are independently fitted; no quality ranking.</p>'
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
        'seed': 42, 'expected_meshes': 144, 'expected_view_renders': 576,
        'protocol': 'original/pink hue0 x GT-binary/BiRef-soft; official alpha automatic crop',
        'automatic_retries': 0, 'training_started': False, 'quality_accepted_by_user': False,
        'objects': [{**row, 'gpu': gpus[index % len(gpus)], 'status': 'pending', 'commands': [],
            'preparation': {kind: {'status': 'pending'} for kind in ('hard', 'soft')},
            'variants': [{'name': name, 'status': 'pending', 'generated': False, 'commands': []} for name in VARIANTS]}
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

            def stage(argv, name, record):
                command, log = [str(value) for value in argv], folder / (name + '.log')
                with lock:
                    record['commands'].append({'argv': command, 'log': str(log), 'gpu': gpu})
                    write_json(output / 'status.json', state)
                execute(command, log, gpu)

            try:
                update(row, status='preparing')
                prepare_inputs(row['image']['path'], row['mask']['path'], folder / 'input',
                    segmentation=f'supplied ICBench GT binary mask {row["stable_id"]}; unchanged')
            except Exception as error:
                reason = f'Semantic input: {type(error).__name__}: {error}'
                for variant in row['variants']:
                    update(variant, status='blocked', error=reason)
                update(row, status='failed', error=reason, **object_sheet(output, row))
                continue
            for kind in ('hard', 'soft'):
                selected = [variant for variant in row['variants'] if variant['name'].endswith('__' + kind)]
                inputs = folder / (kind + '_inputs')
                try:
                    update(row['preparation'][kind], status='preparing', output=str(inputs))
                    stage([RUNTIME['trellis_python' if kind == 'soft' else 'python'], '-B', '-m', PREPARE_MODULE,
                        '--source-receipt', folder / 'input/source_receipt.json', '--output', inputs,
                        *(['--soft'] if kind == 'soft' else [])], kind + '_prepare', row)
                    receipts = {str(inputs / variant['name'] / 'source_receipt.json'):
                                sha256(inputs / variant['name'] / 'source_receipt.json') for variant in selected}
                    with lock:
                        locks.update(receipts)
                    for variant in selected:
                        receipt = str(inputs / variant['name'] / 'source_receipt.json')
                        update(variant, status='ready', source_receipt=receipt, source_receipt_sha256=receipts[receipt])
                    update(row['preparation'][kind], status='prepared')
                except Exception as error:
                    reason = f'{kind} preparation: {type(error).__name__}: {error}'
                    update(row['preparation'][kind], status='failed', error=reason)
                    for variant in selected:
                        update(variant, status='blocked', error=reason)
            update(row, status='processing')
            for variant in row['variants']:
                if variant['status'] != 'ready':
                    continue
                directory = folder / 'variants' / variant['name']
                directory.mkdir(parents=True)
                trellis, views = directory / 'trellis', directory / 'geometry_views'
                try:
                    update(variant, status='generating', trellis_run=str(trellis))
                    check_locks({variant['source_receipt']: variant['source_receipt_sha256']})
                    stage([RUNTIME['trellis_python'], '-B', '-m', 'experiments.subject_trellis_counterfactual_v1.run',
                        '--output', trellis, '--source-receipt', variant['source_receipt'],
                        '--encoder', RUNTIME['trellis_encoder'], '--model-revisions', MODEL_REVISIONS,
                        '--seed', '42', '--skip-glb'], variant['name'] + '_generation', variant)
                    generated = read(trellis / 'repro_outputs/status.json')
                    if (generated['status'] != 'generated_pending_user_review' or generated['generated_assets'] != 1
                            or generated['git'] != revision or generated['official_commit'] != OFFICIAL
                            or generated['seed'] != 42 or generated['models']['revisions'] != pinned
                            or generated['source_receipt']['sha256'] != variant['source_receipt_sha256']
                            or 'diagnostic_fixed_crop_xyxy' in generated):
                        raise ValueError('generation status or configuration differs from this experiment')
                    native = trellis / 'native_asset.pt'
                    update(variant, status='rendering', generated=True,
                           native_asset=str(native), native_asset_sha256=sha256(native))
                    stage([RUNTIME['trellis_python'], '-B', '-m', 'experiments.mailbox_alpha_control_v1.render',
                        '--assets', native, '--labels', variant['name'], '--reference-review', trellis / 'review/review.json',
                        '--output', views, '--gpu', gpu], variant['name'] + '_render', variant)
                    rendered = read(views / 'status.json')
                    if (rendered['status'] != 'completed_pending_visual_review' or rendered['git'] != revision
                            or rendered['completed_renders'] != 4):
                        raise ValueError('four-view rendering did not complete under the same Git')
                    update(variant, status='completed', completed_view_renders=4,
                           comparison_clay=str(views / 'comparison_clay.png'))
                except Exception as error:
                    update(variant, status='failed', error=f'{type(error).__name__}: {error}')
            try:
                update(row, status='completed' if all(v['status'] == 'completed' for v in row['variants']) else 'partial',
                       **object_sheet(output, row))
            except Exception as error:
                update(row, status='failed', error=f'Comparison sheet: {type(error).__name__}: {error}')

    update(state)
    try:
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            list(pool.map(worker, gpus))
        check_locks(locks)
        if git_info() != revision:
            raise ValueError('checkout changed during the experiment')
        variants = [variant for row in state['objects'] for variant in row['variants']]
        complete = all(row['status'] == 'completed' for row in state['objects'])
        update(state, status='completed_pending_user_review' if complete else 'partial',
            generated_meshes=sum(variant['generated'] for variant in variants),
            completed_variants=sum(variant['status'] == 'completed' for variant in variants),
            completed_view_renders=sum(variant.get('completed_view_renders', 0) for variant in variants),
            completed_objects=sum(row['status'] == 'completed' for row in state['objects']))
        if not complete:
            raise RuntimeError('GT four-variant experiment is partial; failures retained in status.json')
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
