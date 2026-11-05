"""Scheme A photograph intake, fresh per-object datasets, and independent SD2.1 jobs."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys

from .object_segmentation import sha256, write_json
from .single_image_training import BRANCHES, DEFAULT_STEPS, training_plan
from scripts.launch.colorpeel_run import git_info, read_config, resolve_run_dir

ROOT = Path(__file__).resolve().parents[3]
ENTRY = ROOT / 'scripts/methods/colorpeel_ice/learn_from_image.py'
RUNTIME = {
    'python': '/home/r12user5/miniforge3/envs/ice/bin/python',
    'vlm_python': '/home/r12user5/miniforge3/envs/ice-vlm/bin/python',
    'train_python': '/home/r12user5/miniforge3/envs/colorpeel017/bin/python',
    'trellis_python': '/var/tmp/colorpeel-r12user5/assets/trellis2_v1/env/bin/python',
    'trellis_encoder': '/var/tmp/colorpeel-r12user5/assets/trellis2_v1/dino_timm_v2/model',
    'supermat_python': '/home/r12user5/Documents/Jiawei/colorpeel-runs/natural_material_supermat_v1/deps/env/bin/python3.12',
    'supermat_root': '/home/r12user5/Documents/Jiawei/colorpeel-runs/natural_material_supermat_v1/deps/SuperMat',
    'supermat_base': '/home/r12user5/.cache/huggingface/hub/models--sd2-community--stable-diffusion-2-1/snapshots/bb2154823665391b4fb29b0b9cf82a198964ee05',
    'blender': '/home/r12user5/Documents/Jiawei/tools/blender-4.2.11-linux-x64/blender',
    'qwen': '/home/r12user5/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',
    'sd21': '/var/tmp/colorpeel-r12user5/assets/sd21_base_4e63672_common_checker',
}
OFFLINE = {'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_OFFLINE': '1',
           'TRANSFORMERS_OFFLINE': '1', 'DIFFUSERS_OFFLINE': '1'}
SUBJECT_CONDITIONING_POLICY = 'sam_semantics_biref_soft_alpha_original_pink_selection/v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def fresh(path):
    path = resolve_run_dir(Path(path))
    if path.exists() or path.is_relative_to(ROOT):
        raise ValueError('require a new output directory outside Git')
    path.mkdir(parents=True)
    return path


def execute(command, log, gpu):
    from experiments.sd21_qwen_initializers_v1.run import require_idle
    require_idle((gpu,))
    command = [str(value) for value in command]
    with Path(log).open('x', encoding='utf8') as stream:
        subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream,
            stderr=subprocess.STDOUT, check=True,
            env={**os.environ, **OFFLINE, 'CUDA_VISIBLE_DEVICES': str(gpu)})


def prepare_subject_geometry(folder, objects_json, object_id, runtime, stage):
    """Generate two fixed candidates independently and bind the selected receipt/run pair."""
    from .single_image_subject import check_locks
    folder = Path(folder).resolve()
    semantic = folder / 'input/source_receipt.json'
    matte, pink = folder / 'trellis_alpha', folder / 'trellis_input'
    candidates = folder / 'trellis_candidates'
    candidates.mkdir(parents=True, exist_ok=False)
    attempts_path = candidates / 'attempts.json'
    stage([runtime['trellis_python'], '-B', '-m', 'src.methods.colorpeel_ice.subject_alpha',
        '--source-receipt', semantic, '--objects-json', objects_json, '--object-id', object_id,
        '--output', matte], object_id + '_trellis_alpha')
    receipts = {'original': matte / 'source_receipt.json', 'pink': pink / 'source_receipt.json'}
    attempts = {name: {'source_receipt': str(path), 'trellis_run': str(candidates / name), 'status': 'pending'}
                for name, path in receipts.items()}
    try:
        stage([runtime['python'], '-B', '-m', 'experiments.subject_pink_input_repro_v1.prepare',
            '--source-receipt', receipts['original'], '--output', pink, '--hue-degrees', '0'],
            object_id + '_trellis_input')
    except Exception as error:
        attempts['pink'].update(status='input_failed', error=f'{type(error).__name__}: {error}')
    write_json(attempts_path, attempts)
    for name, receipt in receipts.items():
        if attempts[name]['status'] == 'input_failed':
            continue
        try:
            stage([runtime['trellis_python'], '-B', '-m', 'experiments.subject_trellis_counterfactual_v1.run',
                '--output', candidates / name, '--encoder', runtime['trellis_encoder'],
                '--source-receipt', receipt, '--model-revisions',
                ROOT / 'experiments/subject_trellis_spoons_v1/model_revisions.json', '--seed', '42', '--skip-glb'],
                object_id + '_trellis_' + name)
            attempts[name]['status'] = 'completed'
        except Exception as error:
            attempts[name].update(status='generation_failed', error=f'{type(error).__name__}: {error}')
        write_json(attempts_path, attempts)
    selection_path = folder / 'trellis_selection/selection.json'
    stage([runtime['trellis_python'], '-B', '-m', 'src.methods.colorpeel_ice.subject_mesh_selection',
        '--semantic-receipt', semantic, '--candidate', f'original={candidates / "original"}',
        '--candidate', f'pink={candidates / "pink"}', '--object-id', object_id, '--seed', '42',
        '--output', selection_path.parent], object_id + '_trellis_selection')
    selection = read(selection_path)
    selected = selection['selected']
    if selected is None:
        raise ValueError(f'No eligible Subject mesh; selection is blocked: {selection_path}')
    row = next(item for item in selection['candidates'] if item['name'] == selected)
    if (selected not in receipts or not row['eligible']
            or Path(row['source_receipt']).resolve() != receipts[selected]
            or Path(row['trellis_run']).resolve() != candidates / selected
            or attempts[selected]['status'] != 'completed'):
        raise ValueError('Subject selector returned an inconsistent receipt/run pair')
    receipt = receipts[selected]
    condition = read(receipt)
    locks = {**selection['source_locks'], str(selection_path): sha256(selection_path),
             str(attempts_path): sha256(attempts_path)}
    check_locks(locks)
    return {'policy': SUBJECT_CONDITIONING_POLICY, 'selected': selected, 'seed': 42,
        'source_receipt': str(receipt), 'source_receipt_sha256': sha256(receipt),
        'trellis_run': str(candidates / selected), 'selection_path': str(selection_path),
        'selection_sha256': locks[str(selection_path)], 'source_locks': locks,
        'semantic_receipt_path': condition['semantic_receipt_path'],
        'semantic_receipt_sha256': condition['semantic_receipt_sha256'],
        'conditioning_alpha_path': condition['conditioning_alpha_path'],
        'conditioning_alpha_sha256': condition['conditioning_alpha_sha256']}


def prepare(image, output, branches, gpu, runtime):
    from .single_image_subject import check_locks, prepare_inputs
    revision = git_info()
    image, output = Path(image).resolve(), fresh(output)
    state = {'schema': 'colorpeel_single_image_prepared/v1', 'status': 'preparing',
        'git': revision, 'source': str(image), 'source_sha256': sha256(image),
        'scheme': 'A', 'branches': list(branches), 'runtime': runtime,
        'subject_conditioning_policy': SUBJECT_CONDITIONING_POLICY,
        'objects': [], 'commands': [], 'training_started': False, 'quality_accepted_by_user': False}

    def stage(command, name):
        state['commands'].append({'argv': [str(x) for x in command], 'gpu': gpu,
                                  'log': str(output / (name + '.log'))})
        write_json(output / 'status.json', state)
        execute(command, output / (name + '.log'), gpu)

    def worker(python, action, *args):
        return [runtime[python], '-B', ENTRY, '--worker', action, *args]

    try:
        stage(worker('vlm_python', 'discover', '--image', image, '--output', output / 'discovery',
                     '--runtime-json', write_runtime(output, runtime)), 'discovery')
        stage(worker('python', 'segment', '--image', image, '--output', output / 'segmentation',
                     '--prepared', output / 'discovery/categories.json'), 'segmentation')
        segmented = read(output / 'segmentation/objects.json')
        if segmented['source']['image_sha256'] != state['source_sha256']:
            raise ValueError('segmentation source differs from input image')
        state['segmentation_status'] = segmented['status']
        state['segmentation_failures'] = segmented['failures']
        for raw_record in segmented['objects']:
            record = {**raw_record, 'source_image_sha256': segmented['source']['canonical_image_sha256']}
            folder = output / 'objects' / record['id']
            folder.mkdir(parents=True)
            item = {'id': record['id'], 'record': record, 'datasets': {}, 'failures': {}, 'status': 'preparing'}
            state['objects'].append(item)
            try:
                prepare_inputs(record['source_image'], record['object_mask'], folder / 'input')
                stage(worker('vlm_python', 'initialize', '--image', record['vlm_input'],
                    '--output', folder / 'initializers', '--branches', *branches,
                    '--runtime-json', output / 'runtime.json'), record['id'] + '_initializers')
                item['initializer_manifest'] = str(folder / 'initializers/initializers.json')
                initialized = read(item['initializer_manifest'])
                available = [branch for branch in branches if branch in initialized['selected_branches']]
                for branch in set(branches) - set(available):
                    item['failures'][branch] = 'initializer unavailable: ' + '; '.join(initialized['unavailable_branches'])
                if 'subject' in available:
                    try:
                        condition = prepare_subject_geometry(folder, output / 'segmentation/objects.json',
                                                             record['id'], runtime, stage)
                        item['trellis_conditioning'] = condition
                        stage([runtime['trellis_python'], '-B', '-m', 'src.methods.colorpeel_ice.single_image_subject',
                            '--stage', 'render', '--inputs', Path(condition['source_receipt']).parent,
                            '--trellis-run', condition['trellis_run'],
                            '--output', folder / 'subject', '--category', initialized['subject_category']],
                            record['id'] + '_subject11')
                        subject_data = read(folder / 'subject/subject.json')
                        subject_data['source_locks'].update(condition['source_locks'])
                        check_locks(subject_data['source_locks'])
                        subject_data['trellis_selection'] = {key: condition[key]
                            for key in ('selected', 'selection_path', 'selection_sha256')}
                        write_json(folder / 'subject/subject.json', subject_data)
                        item['datasets']['subject'] = subject_data
                    except Exception as error:
                        item['failures']['subject'] = f'{type(error).__name__}: {error}'
                cmt = [branch for branch in available if branch != 'subject']
                if cmt:
                    try:
                        module = [runtime['python'], '-B', '-m', 'src.methods.colorpeel_ice.single_image_cmt']
                        stage([*module, '--stage', 'inputs', '--prepared', folder / 'input', '--output', folder / 'cmt_input'],
                              record['id'] + '_cmt_input')
                        if set(cmt) & {'material', 'texture'}:
                            try:
                                super_root = Path(runtime['supermat_root'])
                                super_commit = subprocess.check_output(
                                    ['git', '-C', str(super_root), 'rev-parse', 'HEAD'], text=True).strip()
                                if super_commit != '4fe25bc6cb8cb7a3ed81ce74512f760dd33f80f4':
                                    raise ValueError('SuperMat source revision differs')
                                for diff in (['diff', '--exit-code'], ['diff', '--cached', '--exit-code']):
                                    subprocess.run(['git', '-C', str(super_root), *diff], check=True,
                                                   stdout=subprocess.DEVNULL)
                                if sha256(super_root / 'checkpoints/supermat.pth') != '21d171726e8170e1ed566985be68ef3c5009d762800926ad85c1f288747b7b29':
                                    raise ValueError('SuperMat checkpoint differs from the existing pinned model')
                                stage([runtime['python'], '-B', '-m', 'experiments.natural_material_supermat_v1.run_supermat_batch',
                                    '--python', runtime['supermat_python'], '--supermat-root', runtime['supermat_root'],
                                    '--checkpoint', Path(runtime['supermat_root']) / 'checkpoints/supermat.pth',
                                    '--base-model', runtime['supermat_base'], '--input-dir', folder / 'cmt_input/supermat_input',
                                    '--output-dir', folder / 'supermat', '--device', 'cuda:0', '--seed', '42'],
                                    record['id'] + '_supermat')
                            except Exception as error:
                                for branch in set(cmt) & {'material', 'texture'}:
                                    item['failures'][branch] = f'SuperMat: {type(error).__name__}: {error}'
                                cmt = [branch for branch in cmt if branch == 'color']
                        if cmt:
                            plan = [*module, '--stage', 'plan', '--prepared', folder / 'cmt_input',
                                    '--output', folder / 'cmt_plan', '--branches', *cmt]
                            if set(cmt) & {'material', 'texture'}:
                                plan += ['--maps', folder / 'supermat/input']
                            stage(plan, record['id'] + '_cmt_plan')
                            stage([runtime['blender'], '--background', '--python-exit-code', '1', '--python',
                                ROOT / 'src/methods/colorpeel_ice/single_image_cmt.py', '--', '--stage', 'render',
                                '--plan', folder / 'cmt_plan/plan.json', '--output', folder / 'cmt_renders', '--gpu', str(gpu)],
                                record['id'] + '_cmt_render')
                            stage([*module, '--stage', 'finalize', '--plan', folder / 'cmt_plan/plan.json',
                                '--renders', folder / 'cmt_renders', '--output', folder / 'cmt_data'], record['id'] + '_cmt_finalize')
                            derived = read(folder / 'cmt_data/datasets.json')
                            item['datasets'].update(derived['branches'])
                            item['failures'].update(derived['failures'])
                    except Exception as error:
                        for branch in cmt:
                            item['failures'].setdefault(branch, f'{type(error).__name__}: {error}')
            except Exception as error:
                for branch in branches:
                    item['failures'].setdefault(branch, f'{type(error).__name__}: {error}')
            for branch in branches:
                if branch not in item['datasets']:
                    item['failures'].setdefault(branch, 'requested dataset unavailable')
            item['status'] = 'partial' if item['failures'] else 'prepared'
            write_json(folder / 'object.json', item)
            item['object_manifest'] = str(folder / 'object.json')
            item['object_manifest_sha256'] = sha256(folder / 'object.json')
            write_json(output / 'status.json', state)
        if sha256(image) != state['source_sha256'] or git_info() != revision:
            raise ValueError('source or Git changed during preparation')
        state['status'] = ('prepared' if state['objects'] and segmented['status'] == 'succeeded'
                           and all(row['status'] == 'prepared' for row in state['objects']) else 'partial')
    except Exception as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(output / 'status.json', state)
        write_json(output / 'prepared.json', state)
    return output / 'prepared.json'


def write_runtime(output, runtime):
    path = output / 'runtime.json'
    write_json(path, runtime)
    return path


def train_object(object_manifest, output, branch, steps, gpu):
    from . import single_image_training as training
    item = read(object_manifest)
    output = fresh(output)
    state = {'status': 'preparing', 'object_id': item['id'], 'branch': branch,
             'max_steps': steps, 'completed_steps': None, 'gpu': gpu, 'best_metric': None,
             'best_checkpoint': None, 'quality_accepted_by_user': False}
    try:
        training.prepare(output / 'prepared', item['record'], item['initializer_manifest'],
                         item['datasets'], branches=[branch], steps={branch: steps})
        training.verify_prepared(output / 'prepared')
        config = training.build_config(output / 'prepared', branch, gpu=gpu)
        training._write(output / 'config.json', config)
        if read_config(output / 'config.json') != config:
            raise ValueError('serialized training config changes numeric types')
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        run_name = f"{stamp}__{config['run']['study']}__{config['run']['variant']}__{git_info()['commit'][:7]}__42"
        run_dir = output / 'runs' / run_name
        command = [sys.executable, '-u', '-B', str(ROOT / 'scripts/launch/colorpeel_run.py'),
                   '--config', str(output / 'config.json'), '--run-dir', str(run_dir)]
        state.update(status='training', command=command, run_dir=str(run_dir))
        write_json(output / 'status.json', state)
        # This worker owns one assigned GPU; launcher uses its explicit physical mapping.
        with (output / 'training.log').open('x') as stream:
            subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                           stdin=subprocess.DEVNULL, check=True, env={**os.environ, **OFFLINE})
        state.update(status='exporting', trainer_exit_code=0)
        write_json(output / 'status.json', state)
        exported = training.export(output / 'prepared', branch, run_dir, output / 'ckpts', gpu=gpu)
        state.update(status='trained', completed_steps=steps, checkpoint=exported['steps'][steps],
                     checkpoint_index=str(output / 'ckpts/index.json'), stop_reason='max_steps_reached')
    except Exception as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}', stop_reason='error')
        raise
    finally:
        write_json(output / 'status.json', state)
    return state


def train(prepared_path, output, plan, gpus, runtime):
    prepared_path, output = Path(prepared_path).resolve(), fresh(output)
    prepared = read(prepared_path)
    if prepared['status'] not in ('prepared', 'partial') or prepared['git'] != git_info():
        raise ValueError('require prepared data and the same unchanged Git for training')
    if set(plan) - set(prepared['branches']):
        raise ValueError('selected training branch was not requested during data preparation')
    state = {'status': 'training', 'prepared': str(prepared_path), 'prepared_sha256': sha256(prepared_path),
             'git': git_info(), 'steps': plan, 'jobs': [], 'quality_accepted_by_user': False}
    queue = Queue()
    for item in prepared['objects']:
        if sha256(item['object_manifest']) != item['object_manifest_sha256']:
            raise ValueError('prepared object manifest changed')
        for branch, steps in plan.items():
            job = {'object_id': item['id'], 'branch': branch, 'max_steps': steps,
                   'object_manifest': item['object_manifest'], 'status': 'pending'}
            state['jobs'].append(job)
            if branch in item['datasets']:
                queue.put(job)
            else:
                job.update(status='blocked', reason=item['failures'].get(branch, 'dataset unavailable'))
    (output / 'train_outputs').mkdir()
    write_json(output / 'status.json', state)

    def consume(gpu):
        finished = []
        while True:
            try:
                job = queue.get_nowait()
            except Empty:
                return finished
            folder = output / job['object_id'] / job['branch']
            folder.parent.mkdir(parents=True, exist_ok=True)
            command = [runtime['train_python'], '-u', '-B', ENTRY, '--worker', 'train-object',
                '--prepared', job['object_manifest'], '--output', folder, '--branches', job['branch'],
                '--steps', f"{job['branch']}={job['max_steps']}", '--gpus', str(gpu)]
            job.update(status='running', gpu=gpu, command=[str(x) for x in command], output=str(folder))
            try:
                execute(command, folder.parent / (job['branch'] + '.log'), gpu)
                job.update(read(folder / 'status.json'))
            except Exception as error:
                if (folder / 'status.json').exists():
                    job.update(read(folder / 'status.json'))
                job.update(status='failed', error=f'{type(error).__name__}: {error}')
            write_json(folder.parent / (job['branch'] + '_result.json'), job)
            finished.append(job)
            queue.task_done()

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        for result in as_completed([pool.submit(consume, gpu) for gpu in gpus]):
            result.result()
            write_json(output / 'status.json', state)
    state['status'] = 'succeeded' if state['jobs'] and all(job['status'] == 'trained' for job in state['jobs']) else 'partial'
    write_json(output / 'status.json', state)
    write_json(output / 'checkpoints.json', {'git': state['git'], 'status': state['status'], 'objects': state['jobs']})
    write_json(output / 'train_outputs/status.json', state)
    (output / 'train_outputs/COMMANDS.md').write_text('\n\n'.join(
        '```json\n' + json.dumps(job.get('command', [])) + '\n```' for job in state['jobs']), encoding='utf8')
    (output / 'train_outputs/SUMMARY.md').write_text(
        f"Status: {state['status']}. Completed {sum(job['status'] == 'trained' for job in state['jobs'])}"
        f" / {len(state['jobs'])} selected object/branch jobs. Final paired snapshots only; no quality acceptance.\n", encoding='utf8')
    (output / 'train_outputs/LOG.md').write_text('\n'.join(
        f"- {job['object_id']}/{job['branch']}: {job['status']}; "
        f"log: {job.get('output', 'not started')}/training.log" for job in state['jobs']) + '\n', encoding='utf8')
    (output / 'train_outputs/SCIENTIFIC_CHANGELOG.md').write_text(
        'New per-instance Scheme A masks, fresh TRELLIS geometry selected from original/pink hue0 '
        'candidates sharing BiRefNet soft alpha, unchanged SAM semantic masks, automatic eroded ROI and '
        'fresh SuperMat-derived targets replace historical reviewed Mailbox assets for this run. '
        'Original photo anchors, Qwen inputs and C/M/T inputs retain the original photograph. '
        'Selected-v1 initialization preserves raw evidence and permits at most one token repair. '
        'SD2.1 token-local LoRA64 optimizer recipes are inherited; steps are explicit per branch. '
        'No background-preservation objective, extra checkpoint resume, or quality tuning.\n', encoding='utf8')
    (output / 'train_outputs/COMPARABILITY_REPORT.md').write_text(
        'This is an engineering integration run, not an initializer-only ablation. Instance masks, '
        'geometry, ROI and derived data differ from historical center11/Material8/Texture16. '
        'Technical checkpoint validation does not establish disentanglement or background binding. '
        'No best checkpoint is selected: each completed job exports its requested final step.\n', encoding='utf8')
    return output / 'checkpoints.json'


def parse_steps(values):
    result = {}
    for value in values or []:
        name, count = value.split('=', 1)
        if name in result:
            raise ValueError('duplicate branch step override')
        result[name] = int(count)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('all', 'prepare', 'train'), default='all')
    parser.add_argument('--worker', choices=('discover', 'segment', 'initialize', 'train-object'))
    parser.add_argument('--image', type=Path)
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--branches', nargs='+', choices=BRANCHES, default=list(BRANCHES))
    parser.add_argument('--steps', nargs='+', help='Overrides such as subject=1000 material=200')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--runtime-json', type=Path, help='Optional explicit environment/path mapping')
    args = parser.parse_args(argv)
    plan = training_plan(args.branches, parse_steps(args.steps))
    if not args.gpus or len(set(args.gpus)) != len(args.gpus) or any(gpu < 0 for gpu in args.gpus):
        parser.error('GPUs must be distinct nonnegative indices')
    runtime = {**RUNTIME, **(read(args.runtime_json) if args.runtime_json else {})}
    if args.worker:
        if args.worker == 'discover':
            from .object_discovery import run
            run(args.image, runtime['qwen'], args.output)
        elif args.worker == 'segment':
            from .object_segmentation import run
            from .image_intake import check_locks
            categories = read(args.prepared)
            if categories['status'] != 'verified' or categories['source']['image_sha256'] != sha256(args.image):
                raise ValueError('discovery does not describe the input image')
            check_locks(categories['source_locks'])
            path = run(args.image, categories['categories'], args.output)
            if read(path)['status'] == 'failed':
                raise RuntimeError('no usable segmentation')
        elif args.worker == 'initialize':
            from .vlm_initialization import run
            run(args.image, runtime['qwen'], runtime['sd21'], args.output,
                selected_branches=args.branches, allow_partial=True)
        else:
            if len(plan) != 1:
                parser.error('training worker requires one branch')
            branch = next(iter(plan))
            train_object(args.prepared, args.output, branch, plan[branch], args.gpus[0])
        return 0
    if args.stage != 'train' and args.image is None:
        parser.error('image is required for preparation')
    if args.stage == 'train' and args.prepared is None:
        parser.error('prepared manifest is required for training')
    if args.stage == 'prepare':
        path = prepare(args.image, args.output, args.branches, args.gpus[0], runtime)
    elif args.stage == 'train':
        path = train(args.prepared, args.output, plan, args.gpus, runtime)
    else:
        output = fresh(args.output)
        write_json(output / 'request.json', {'image': str(args.image.resolve()), 'steps': plan,
                   'gpus': args.gpus, 'git': git_info(), 'runtime': runtime})
        prepared = prepare(args.image, output / 'prepared', args.branches, args.gpus[0], runtime)
        path = train(prepared, output / 'training', plan, args.gpus, runtime)
    print(path)
    return int(read(path)['status'] not in ('prepared', 'succeeded'))


if __name__ == '__main__':
    sys.exit(main())
