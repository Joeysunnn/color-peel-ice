"""Run eight fixed TRELLIS conditions per supplied object, without retries."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from pathlib import Path
from threading import Lock

from PIL import Image

from experiments.subject_trellis_counterfactual_v1.run import OFFICIAL
from scripts.launch.colorpeel_run import git_info
from src.methods.colorpeel_ice.single_image_pipeline import ROOT, RUNTIME, execute, fresh, read, sha256, write_json
from src.methods.colorpeel_ice.single_image_subject import check_locks

RUN_ROOT = Path('/home/r12user5/Documents/Jiawei/colorpeel-runs')
OBJECTS = {
    'mailbox': {
        'matte_receipt': RUN_ROOT / 'mailbox_soft_alpha_v1/20261105__sam_soft_pink__87efe3a__42/trellis_alpha/source_receipt.json',
        'reference_review': RUN_ROOT / 'subject_five_color_mesh_v1/20261105__sam_five_hues__6726dab__42/pink/trellis/review/review.json'},
    'wood_spoon': {
        'matte_receipt': RUN_ROOT / 'wood_spoon_soft_alpha_v1/20261105__sam_soft_pink__d5b9d15__42/trellis_alpha/source_receipt.json',
        'reference_review': RUN_ROOT / 'wood_spoon_soft_alpha_v1/20261105__sam_soft_original__d5b9d15__42/trellis/review/review.json'}}
VARIANT_ORDER = tuple('__'.join(values) for values in product(('original', 'pink'), ('sam', 'biref'), ('sam', 'biref')))
VARIANTS = set(VARIANT_ORDER)
MODEL_REVISIONS = ROOT / 'experiments/subject_trellis_spoons_v1/model_revisions.json'


def run(output, gpus, objects=None):
    from .prepare import prepare
    if not gpus or len(set(gpus)) != len(gpus) or any(type(gpu) is not int or gpu < 0 for gpu in gpus):
        raise ValueError('require distinct nonnegative physical GPUs')
    objects = OBJECTS if objects is None else objects
    if not objects:
        raise ValueError('require at least one object or explicit blocked-object record')
    revision, output = git_info(), fresh(output)
    pinned = read(MODEL_REVISIONS)
    state = {'status': 'running', 'stage': 'prepare', 'git': revision, 'gpus': list(gpus), 'seed': 42,
        'expected_jobs': 8 * len(objects), 'jobs': [], 'objects': {}, 'renders': [],
        'source_locks': {str(MODEL_REVISIONS): sha256(MODEL_REVISIONS)},
        'automatic_retries': 0, 'training_started': False, 'quality_accepted_by_user': False}
    lock, configuration = Lock(), []

    def update(record, **values):
        with lock:
            record.update(values)
            write_json(output / 'status.json', state)

    try:
        write_json(output / 'status.json', state)
        for name, sources in objects.items():
            reference = (Path(sources['reference_review']) if sources.get('reference_review') is not None else
                         output / name / 'variants/original__biref__biref/trellis/review/review.json')
            record = {'status': 'preparing', 'source_rgb': sources.get('source_rgb', False),
                'reference_review': str(reference), 'reference_policy': ('supplied_frozen_review'
                    if sources.get('reference_review') is not None else 'fixed_new_original__biref__biref')}
            state['objects'][name] = record
            jobs = []
            for variant in VARIANT_ORDER:
                run_path = output / name / 'variants' / variant / 'trellis'
                job = {'object': name, 'variant': variant, 'stage': 'pending', 'status': 'pending',
                    'gpu': gpus[len(state['jobs']) % len(gpus)], 'trellis_run': str(run_path),
                    'log': str(run_path.parent / 'generation.log')}
                jobs.append(job)
                state['jobs'].append(job)
            try:
                if sources.get('blocked_reason'):
                    raise ValueError(sources['blocked_reason'])
                manifest = prepare(sources['matte_receipt'], output / name / 'inputs',
                                   source_rgb=record['source_rgb'])
                if manifest['git'] != revision or {row['name'] for row in manifest['variants']} != VARIANTS or len(manifest['variants']) != 8:
                    raise ValueError('preparation must produce the eight fixed variants under the same full Git')
                manifest_path = output / name / 'prepare_manifest.json'
                write_json(manifest_path, manifest)
                state['source_locks'].update(manifest['source_locks'])
                paths = [Path(sources['matte_receipt']), manifest_path]
                if sources.get('reference_review') is not None:
                    paths.append(reference)
                state['source_locks'].update({str(path): sha256(path) for path in paths})
                record.update(manifest=str(manifest_path), manifest_sha256=sha256(manifest_path),
                              crop_policy=manifest['crop_policy'])
                variants = {row['name']: row for row in manifest['variants']}
                for job in jobs:
                    variant, run_path = variants[job['variant']], Path(job['trellis_run'])
                    run_path.parent.mkdir(parents=True)
                    job.update(source_receipt=variant['source_receipt'], source_receipt_sha256=sha256(variant['source_receipt']),
                        expected_condition=variant['expected_condition'], expected_condition_sha256=sha256(variant['expected_condition']),
                        auto_crop_xyxy=variant['auto_crop_xyxy'])
                    job['argv'] = [str(value) for value in [RUNTIME['trellis_python'], '-B', '-m',
                        'experiments.subject_trellis_counterfactual_v1.run', '--output', run_path,
                        '--source-receipt', job['source_receipt'], '--encoder', RUNTIME['trellis_encoder'],
                        '--model-revisions', MODEL_REVISIONS, '--seed', '42', '--skip-glb']]
                update(record, status='prepared')
            except Exception as error:
                reason = f'{type(error).__name__}: {error}'
                for job in jobs:
                    job.update(status='blocked', stage='prepare', error=reason)
                update(record, status='blocked', error=reason)
        check_locks(state['source_locks'])
        if git_info() != revision:
            raise ValueError('checkout changed during preparation')
        update(state, stage='generate')

        def worker(gpu):
            for job in (row for row in state['jobs'] if row['gpu'] == gpu and row['status'] == 'pending'):
                try:
                    update(job, status='running', stage='generate')
                    check_locks({job[key]: job[key + '_sha256'] for key in ('source_receipt', 'expected_condition')})
                    execute(job['argv'], Path(job['log']), gpu)
                    update(job, stage='verify')
                    directory = Path(job['trellis_run'])
                    generated = read(directory / 'repro_outputs/status.json')
                    if (generated['status'] != 'generated_pending_user_review' or generated['generated_assets'] != 1
                            or generated['git'] != revision or generated['official_commit'] != OFFICIAL
                            or generated['seed'] != 42 or generated['models']['revisions'] != pinned
                            or generated['source_receipt']['sha256'] != job['source_receipt_sha256']
                            or 'diagnostic_fixed_crop_xyxy' in generated
                            or not (directory / 'native_asset.pt').is_file()):
                        raise ValueError('generation status/configuration/input differs from the fixed experiment')
                    actual = directory / 'actual_condition.png'
                    with Image.open(actual) as a, Image.open(job['expected_condition']) as b:
                        if a.mode != b.mode or a.size != b.size or a.tobytes() != b.tobytes():
                            raise ValueError('actual TRELLIS conditioning pixels differ from the prepared expectation')
                    config = {key: generated[key] for key in ('official_commit', 'seed', 'models', 'pipeline_type',
                        'encoder_receipt_sha256', 'sampling', 'flow_dtype', 'sparse_algorithm')}
                    args = read(directory / 'pipeline.json')['args']
                    config.update({key: args[key] for key in ('shape_slat_normalization', 'tex_slat_normalization')})
                    with lock:
                        if configuration and config != configuration[0]:
                            raise ValueError('successful candidates do not share one generation configuration')
                        if not configuration:
                            configuration.append(config)
                    update(job, status='generated', stage='completed', actual_condition_sha256=sha256(actual),
                           condition_pixels_verified=True, native_asset=str(directory / 'native_asset.pt'))
                except Exception as error:
                    update(job, status='failed', error=f'{type(error).__name__}: {error}')

        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            for future in as_completed([pool.submit(worker, gpu) for gpu in gpus]):
                future.result()
        update(state, stage='render')
        for name, sources in objects.items():
            jobs = [job for job in state['jobs'] if job['object'] == name]
            view_output = output / name / 'geometry_views'
            row = {'object': name, 'stage': 'render', 'status': 'pending', 'gpu': gpus[0],
                   'output': str(view_output), 'log': str(output / name / 'geometry_views.log')}
            state['renders'].append(row)
            if any(job['status'] != 'generated' for job in jobs):
                update(row, status='blocked', reason='All eight verified native assets are required for this comparison.')
                continue
            row['argv'] = [str(value) for value in [RUNTIME['trellis_python'], '-B', '-m',
                'experiments.mailbox_alpha_control_v1.render', '--assets', *(job['native_asset'] for job in jobs),
                '--labels', *(job['variant'] for job in jobs), '--reference-review', state['objects'][name]['reference_review'],
                '--output', view_output, '--gpu', gpus[0]]]
            try:
                # A new reference camera is predetermined, but exists only after generation.
                reference = state['objects'][name]['reference_review']
                if reference not in state['source_locks']:
                    state['source_locks'][reference] = sha256(reference)
                check_locks({reference: state['source_locks'][reference]})
                row.update(reference_review=reference, reference_review_sha256=state['source_locks'][reference])
                update(row, status='running')
                execute(row['argv'], Path(row['log']), gpus[0])
                rendered = read(view_output / 'status.json')
                if (rendered['status'] != 'completed_pending_visual_review' or rendered['git'] != revision
                        or rendered['completed_renders'] != 32):
                    raise ValueError('shared-camera comparison did not complete under the same Git')
                update(row, status='completed')
            except Exception as error:
                update(row, status='failed', error=f'{type(error).__name__}: {error}')
        check_locks(state['source_locks'])
        if git_info() != revision:
            raise ValueError('checkout changed during the experiment')
        state['generated_jobs'] = sum(job['status'] == 'generated' for job in state['jobs'])
        complete = state['generated_jobs'] == state['expected_jobs'] and all(row['status'] == 'completed' for row in state['renders'])
        update(state, status='completed_pending_user_review' if complete else 'partial', stage='completed')
        if not complete:
            raise RuntimeError('Factorial experiment is partial; failed/blocked evidence retained in status.json.')
    except Exception as error:
        update(state, status='partial' if state['status'] == 'partial' else 'failed',
               error=f'{type(error).__name__}: {error}')
        raise
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3])
    args = parser.parse_args()
    run(args.output, args.gpus)
