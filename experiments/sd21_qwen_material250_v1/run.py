"""Fresh Mailbox Material training with automatic transfer at the requested saved steps."""

import argparse
from datetime import datetime
import html
import json
import os
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace

from experiments.sd15_token_local_kv_v1.contract import read_json, sha256
from experiments.sd21_qwen_initializers_v1 import contract, data
from experiments.sd21_qwen_initializers_v1.inference import register_initial_input
from experiments.sd21_qwen_initializers_v1.run import ENVIRONMENT, execute, export, require_idle
from experiments.sd21_qwen_midpoint_transfer_v1 import protocol as midpoint
from experiments.sd21_qwen_midpoint_transfer_v1.run import collect, run_group
from experiments.sd21_wood_branches_v1 import snapshots
from experiments.subject_multicolor_multimaterial_b.training import _write
from scripts.launch.colorpeel_run import git_info, read_config, resolve_run_dir

ROOT = Path(__file__).resolve().parents[2]
STUDY = 'sd21_qwen_material250_v1'
SOURCE = midpoint.SOURCE
STUDIES = {(250,): STUDY, (150, 200): 'sd21_qwen_material150_200_v1'}


def build_config(source, gpu, steps=(250,)):
    study = STUDIES[steps]
    config = contract.build_config(source, 'material')
    config['args'].update(max_train_steps=steps[-1], checkpoint_steps=list(steps))
    config['run'].update(study=study, variant=f'material_fresh{steps[-1]}')
    config['environment']['CUDA_VISIBLE_DEVICES'] = str(gpu)
    return config


def recipe(source, gpu, steps=(250,)):
    def expected(branch, method, model_dir=contract.MODEL_DIR, smoke=False):
        if branch != 'material' or method != 'lora_r64' or Path(model_dir) != contract.MODEL_DIR or smoke:
            raise ValueError('only the declared fresh Material recipe is supported')
        return build_config(source, gpu, steps)
    return SimpleNamespace(BRANCHES=('material',), METHODS=('lora_r64',), STEPS=steps,
        TOKENS=contract.TOKENS, CONCEPT_IDS=contract.CONCEPT_IDS, build_config=expected,
        verify_dataset=lambda branch: data.verify_dataset(Path(source) / 'dataset', branch))


def plan(manifest, step=250):
    if step not in (150, 200, 250):
        raise ValueError('only requested Material checkpoints are supported')
    return [{**row, 'phase': f'step{step}', 'id': row['id'].replace('mid__', f'step{step}__', 1),
             'branch_steps': {'material': step}}
            for row in midpoint.plan(manifest) if row['mode'] == 'material_only']


def gallery(output, rows, generated, state, group='material_only'):
    step = rows[0]['branch_steps']['material']
    folder = output / 'groups' / group
    folder.mkdir(parents=True, exist_ok=True)
    records = {row['id']: row for row in generated}
    cards = []
    for row in rows:
        record = records.get(row['id'])
        picture = '<p>pending</p>' if record is None else (
            f'<a href="../../{row["image_path"]}"><img width="256" src="../../{row["image_path"]}"></a>'
            f'<p>{record["status"]}</p>')
        cards.append(f'<figure><b>{row["case"]} / seed{row["seed"]}</b>{picture}'
                     f'<small>{html.escape(row["prompt"])}</small></figure>')
    (folder / 'index.html').write_text('<!doctype html><meta charset="utf-8">'
        f'<title>Material{step} transfer</title><style>body{{font:16px sans-serif}}section{{display:flex;flex-wrap:wrap}}'
        f'figure{{width:256px;margin:12px}}small{{display:block}}</style><h1>Material{step} only</h1>'
        f'<p>{state["status"]}: {len(generated)}/44. Chair, vase, sofa, mug; seeds30–40. '
        'PNDM100 / 512px / FP16 / CFG3.5 / empty negative / safety checker enabled. '
        'User manual review; filtered images retained.</p><section>' + ''.join(cards) + '</section>', encoding='utf8')
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8">'
        '<meta http-equiv="refresh" content="0;url=groups/material_only/index.html">'
        f'<a href="groups/material_only/index.html">Material{step} transfer gallery</a>', encoding='utf8')


def inference_dir(output, steps, step):
    return output / 'inference' / f'step{step}' if len(steps) > 1 else output / 'inference'


def prepare(source, output, gpu, steps=(250,)):
    from transformers import CLIPTokenizer
    from experiments.sd21_wood_branches_v1.prepare_model import verify_model
    from src.train.initializer_token_utils import load_initializer_manifest

    source, output = Path(source).resolve(), resolve_run_dir(Path(output))
    if source != SOURCE.resolve() or output.exists() or any(
            output.is_relative_to(p) or p.is_relative_to(output) for p in (ROOT, source, contract.MODEL_DIR.resolve())):
        raise ValueError('require the pinned source and a fresh external output separate from protected inputs')
    prior = read_json(source / 'source_audit.json')
    prior_status = read_json(source / 'status.json')
    if (prior['new_training_git'] != midpoint.SOURCE_GIT or prior_status.get('status') != 'succeeded'
            or prior_status.get('new_training_git') != midpoint.SOURCE_GIT):
        raise ValueError('source training identity or completion differs')
    revision, model = git_info(), verify_model(contract.MODEL_DIR)
    if model['model'] != prior['base_model']:
        raise ValueError('SD2.1 base differs from the prior Material recipe')
    tokenizer = CLIPTokenizer.from_pretrained(str(contract.MODEL_DIR), subfolder='tokenizer', local_files_only=True)
    manifest_path = Path(prior['initializer_manifest'])
    manifest = load_initializer_manifest(manifest_path, tokenizer, contract.MODEL_DIR)
    if manifest != prior['initializers']:
        raise ValueError('initializer manifest changed')
    dataset = data.verify_dataset(source / 'dataset', 'material')
    locks = {**snapshots.model_locks(contract.MODEL_DIR, model), **dataset['source_locks'],
             **manifest['source_locks'], str(manifest_path): sha256(manifest_path),
             **{str(source / name): sha256(source / name) for name in ('source_audit.json', 'status.json')}}
    snapshots.check_locks(locks)
    config = build_config(source, gpu, steps)
    plans = {step: plan(manifest, step) for step in steps}
    if config['args']['lr_scheduler'] != 'constant' or config['args']['lr_warmup_steps'] != 0:
        raise ValueError('step-budget comparison requires the unchanged constant LR schedule')
    output.mkdir(parents=True)
    audit = {'study': STUDIES[steps], 'source_experiment': str(source), 'source_training_git': midpoint.SOURCE_GIT,
        'new_training_git': revision, 'inference_git': revision, 'initializers': manifest,
        'initializer_manifest': str(manifest_path), 'dataset': dataset, 'base_model': model['model'],
        'source_locks': locks, 'fresh_training': True, 'old_weights_reused': False, 'gpu': gpu}
    _write(output / 'source_audit.json', audit)
    _write(output / 'config.json', config)
    if read_config(output / 'config.json') != config:
        raise ValueError('serialized training config changed numeric types')
    for step, rows in plans.items():
        inference = inference_dir(output, steps, step)
        inference.mkdir(parents=True)
        (inference / 'generation_manifest.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf8')
        gallery(inference, rows, [], {'status': 'waiting_for_training'})
    if len(steps) > 1:
        (output / 'inference/index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Material checkpoints</h1>'
            + ''.join(f'<p><a href="step{step}/index.html">Material{step}:44 images</a></p>' for step in steps), encoding='utf8')
    return output, audit, config, plans


def run(source, output, gpu, steps=(250,)):
    if steps not in STUDIES:
        raise ValueError('supported checkpoint plans are [250] or [150,200]')
    os.environ.update(ENVIRONMENT, CUDA_VISIBLE_DEVICES=str(gpu))
    require_idle((gpu,))
    output, audit, config, plans = prepare(source, output, gpu, steps)
    maximum, expected_images = steps[-1], 44 * len(steps)
    report = output / 'train_outputs'
    report.mkdir()
    name = (datetime.now().strftime('%Y%m%d-%H%M%S') + f"__{config['run']['study']}__{config['run']['variant']}__"
            + audit['new_training_git']['commit'][:7] + '__42')
    train_run = output / 'training' / name
    command = [sys.executable, '-u', '-B', str(ROOT / 'scripts/launch/colorpeel_run.py'),
               '--config', str(output / 'config.json'), '--run-dir', str(train_run)]
    _write(output / 'command.json', {'training': command, 'gpu': gpu,
        'next': f'paired Material checkpoints{list(steps)}, then{expected_images} material-only transfer images'})
    (report / 'COMMANDS.md').write_text('```sh\n' + shlex.join(command) + '\n```\n'
        'The parent automatically exports the paired checkpoint and performs Material-only inference.\n', encoding='utf8')
    (report / 'LOG.md').write_text('Training log: `../training.log`. Launcher details: `../training/`. '
        'Inference ledgers: beneath `../inference/`, separately for each saved step.\n', encoding='utf8')
    (report / 'SCIENTIFIC_CHANGELOG.md').write_text(f'Fresh Material{maximum}, save{list(steps)}; no other branch or objective. '
        'Same8 Mailbox images, captions, painted initializer and LoRA64 recipe; no old weights reused.\n', encoding='utf8')
    (report / 'COMPARABILITY_REPORT.md').write_text('Compared with Material500/1000: shorter training budget only. '
        'Constant LR and warmup0 retain the same LR schedule; seed42, data, initialization and sampling unchanged. '
        'No claim of bitwise training equivalence. User performs manual visual review.\n', encoding='utf8')
    state = {'status': 'training', 'training_steps': maximum, 'max_steps': maximum, 'completed_steps': None,
             'stop_reason': None, 'checkpoint_steps': list(steps), 'training_run': str(train_run),
             'expected_images': expected_images, 'generated_images': 0, 'gpu': gpu, 'fresh_training': True,
             'git': audit['new_training_git'], 'quality_verified': False}

    def save():
        _write(output / 'status.json', state)
        _write(report / 'status.json', {**state, 'best_checkpoint': None, 'best_metric': None})
        (report / 'SUMMARY.md').write_text(f"Status: {state['status']}. Material{list(steps)} then{expected_images} transfer images. "
            f"Generated: {state['generated_images']}/{expected_images}. User manual review; no monitoring or rerolls.\n", encoding='utf8')

    save()
    try:
        require_idle((gpu,))
        execute(command, output / 'training.log', gpu)
        state.update(status='exporting', completed_steps=maximum)
        save()
        result = export(output, 'material', train_run, output / 'ckpts', False,
                        experiment_contract=recipe(source, gpu, steps))
        audit['source_locks'].update(result['source_locks'])
        for selected in result['steps'].values():
            audit['source_locks'].update(selected['hashes'])
        if len(steps) == 1:
            audit['selected_exports'] = {'material': result['steps'][steps[0]]}
        _write(output / 'exports.json', {'material': {str(step): result['steps'][step] for step in steps}})
        _write(output / 'source_audit.json', audit)
        state['status'] = 'inference'
        save()
        generated = []
        for step, rows in plans.items():
            inference = inference_dir(output, steps, step)
            selected_audit = {**audit, 'selected_exports': {'material': result['steps'][step]}}
            _write(inference / 'source_audit.json', selected_audit)
            state['active_checkpoint'] = step
            save()
            run_group(inference, 'material_only', rows, selected_audit, gpu, render_gallery=gallery)
            step_images = collect(inference, rows)
            if len(step_images) != 44 or read_json(inference / 'groups/material_only/status.json')['status'] != 'succeeded':
                raise ValueError(f'Material{step} transfer did not complete44 images')
            generated.extend(step_images)
            state['generated_images'] = len(generated)
            save()
        if len(steps) > 1:
            initial_inputs = {}
            for row in generated:
                register_initial_input(initial_inputs, row['seed'], row['initial_denoiser_input'])
            _write(output / 'inference/initial_input_audit.json', {
                'same_seed_equal_across_checkpoints': True, 'generated_images': len(generated), 'by_seed': initial_inputs})
        snapshots.check_locks(audit['source_locks'])
        if git_info() != audit['new_training_git']:
            raise ValueError('checkout changed during training or inference')
        state.update(status='succeeded', stop_reason='completed', generated_images=len(generated),
                     safety_filtered_images=sum(row['nsfw_content_detected'] for row in generated))
    except BaseException as error:
        state.update(status='failed', stop_reason='stage_failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        save()
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, choices=range(4), default=0)
    parser.add_argument('--checkpoint-steps', type=int, nargs='+', default=[250])
    args = parser.parse_args()
    run(SOURCE, args.output, args.gpu, tuple(args.checkpoint_steps))
