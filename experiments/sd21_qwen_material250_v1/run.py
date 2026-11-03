"""Fresh Mailbox Material250 training followed automatically by 44 transfer images."""

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
from experiments.sd21_qwen_initializers_v1.run import ENVIRONMENT, execute, export, require_idle
from experiments.sd21_qwen_midpoint_transfer_v1 import protocol as midpoint
from experiments.sd21_qwen_midpoint_transfer_v1.run import collect, run_group
from experiments.sd21_wood_branches_v1 import snapshots
from experiments.subject_multicolor_multimaterial_b.training import _write
from scripts.launch.colorpeel_run import git_info, read_config, resolve_run_dir

ROOT = Path(__file__).resolve().parents[2]
STUDY = 'sd21_qwen_material250_v1'
SOURCE = midpoint.SOURCE


def build_config(source, gpu):
    config = contract.build_config(source, 'material')
    config['args'].update(max_train_steps=250, checkpoint_steps=[250])
    config['run'].update(study=STUDY, variant='material_fresh250')
    config['environment']['CUDA_VISIBLE_DEVICES'] = str(gpu)
    return config


def recipe(source, gpu):
    def expected(branch, method, model_dir=contract.MODEL_DIR, smoke=False):
        if branch != 'material' or method != 'lora_r64' or Path(model_dir) != contract.MODEL_DIR or smoke:
            raise ValueError('only the declared fresh Material250 recipe is supported')
        return build_config(source, gpu)
    return SimpleNamespace(BRANCHES=('material',), METHODS=('lora_r64',), STEPS=(250,),
        TOKENS=contract.TOKENS, CONCEPT_IDS=contract.CONCEPT_IDS, build_config=expected,
        verify_dataset=lambda branch: data.verify_dataset(Path(source) / 'dataset', branch))


def plan(manifest):
    return [{**row, 'phase': 'step250', 'id': row['id'].replace('mid__', 'step250__', 1),
             'branch_steps': {'material': 250}}
            for row in midpoint.plan(manifest) if row['mode'] == 'material_only']


def gallery(output, rows, generated, state, group='material_only'):
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
        '<title>Material250 transfer</title><style>body{font:16px sans-serif}section{display:flex;flex-wrap:wrap}'
        'figure{width:256px;margin:12px}small{display:block}</style><h1>Material250 only</h1>'
        f'<p>{state["status"]}: {len(generated)}/44. Chair, vase, sofa, mug; seeds30–40. '
        'PNDM100 / 512px / FP16 / CFG3.5 / empty negative / safety checker enabled. '
        'User manual review; filtered images retained.</p><section>' + ''.join(cards) + '</section>', encoding='utf8')
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8">'
        '<meta http-equiv="refresh" content="0;url=groups/material_only/index.html">'
        '<a href="groups/material_only/index.html">Material250 transfer gallery</a>', encoding='utf8')


def prepare(source, output, gpu):
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
    config, rows = build_config(source, gpu), plan(manifest)
    if config['args']['lr_scheduler'] != 'constant' or config['args']['lr_warmup_steps'] != 0:
        raise ValueError('step-budget comparison requires the unchanged constant LR schedule')
    output.mkdir(parents=True)
    audit = {'study': STUDY, 'source_experiment': str(source), 'source_training_git': midpoint.SOURCE_GIT,
        'new_training_git': revision, 'inference_git': revision, 'initializers': manifest,
        'initializer_manifest': str(manifest_path), 'dataset': dataset, 'base_model': model['model'],
        'source_locks': locks, 'fresh_training': True, 'old_weights_reused': False, 'gpu': gpu}
    _write(output / 'source_audit.json', audit)
    _write(output / 'config.json', config)
    if read_config(output / 'config.json') != config:
        raise ValueError('serialized training config changed numeric types')
    inference = output / 'inference'
    inference.mkdir()
    (inference / 'generation_manifest.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf8')
    gallery(inference, rows, [], {'status': 'waiting_for_training'})
    return output, audit, config, rows


def run(source, output, gpu):
    os.environ.update(ENVIRONMENT, CUDA_VISIBLE_DEVICES=str(gpu))
    require_idle((gpu,))
    output, audit, config, rows = prepare(source, output, gpu)
    report = output / 'train_outputs'
    report.mkdir()
    name = (datetime.now().strftime('%Y%m%d-%H%M%S') + f'__{STUDY}__material_fresh250__'
            + audit['new_training_git']['commit'][:7] + '__42')
    train_run = output / 'training' / name
    command = [sys.executable, '-u', '-B', str(ROOT / 'scripts/launch/colorpeel_run.py'),
               '--config', str(output / 'config.json'), '--run-dir', str(train_run)]
    _write(output / 'command.json', {'training': command, 'gpu': gpu,
        'next': 'paired Material250 export, then44 material-only transfer images'})
    (report / 'COMMANDS.md').write_text('```sh\n' + shlex.join(command) + '\n```\n'
        'The parent automatically exports the paired checkpoint and performs Material-only inference.\n', encoding='utf8')
    (report / 'LOG.md').write_text('Training log: `../training.log`. Launcher details: `../training/`. '
        'Inference ledger: `../inference/groups/material_only/generation_status.jsonl`.\n', encoding='utf8')
    (report / 'SCIENTIFIC_CHANGELOG.md').write_text('Fresh Material250, save250; no other branch or objective. '
        'Same8 Mailbox images, captions, painted initializer and LoRA64 recipe; no old weights reused.\n', encoding='utf8')
    (report / 'COMPARABILITY_REPORT.md').write_text('Compared with Material500/1000: shorter training budget only. '
        'Constant LR and warmup0 retain the same LR schedule; seed42, data, initialization and sampling unchanged. '
        'No claim of bitwise training equivalence. User performs manual visual review.\n', encoding='utf8')
    state = {'status': 'training', 'training_steps': 250, 'max_steps': 250, 'completed_steps': None,
             'stop_reason': None, 'checkpoint_steps': [250], 'training_run': str(train_run),
             'expected_images': 44, 'generated_images': 0, 'gpu': gpu, 'fresh_training': True,
             'git': audit['new_training_git'], 'quality_verified': False}

    def save():
        _write(output / 'status.json', state)
        _write(report / 'status.json', {**state, 'best_checkpoint': None, 'best_metric': None})
        (report / 'SUMMARY.md').write_text(f"Status: {state['status']}. Material250 then44 transfer images. "
            f"Generated: {state['generated_images']}/44. User manual review; no monitoring or rerolls.\n", encoding='utf8')

    save()
    try:
        require_idle((gpu,))
        execute(command, output / 'training.log', gpu)
        state.update(status='exporting', completed_steps=250)
        save()
        result = export(output, 'material', train_run, output / 'ckpts', False,
                        experiment_contract=recipe(source, gpu))
        audit['source_locks'].update(result['source_locks'])
        selected = result['steps'][250]
        audit['source_locks'].update(selected['hashes'])
        audit['selected_exports'] = {'material': selected}
        _write(output / 'exports.json', {'material': {'250': selected}})
        _write(output / 'source_audit.json', audit)
        _write(output / 'inference/source_audit.json', audit)
        state['status'] = 'inference'
        save()
        run_group(output / 'inference', 'material_only', rows, audit, gpu, render_gallery=gallery)
        generated = collect(output / 'inference', rows)
        if len(generated) != 44 or read_json(output / 'inference/groups/material_only/status.json')['status'] != 'succeeded':
            raise ValueError('Material250 transfer did not complete44 images')
        snapshots.check_locks(audit['source_locks'])
        if git_info() != audit['new_training_git']:
            raise ValueError('checkout changed during training or inference')
        state.update(status='succeeded', stop_reason='completed', generated_images=44,
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
    args = parser.parse_args()
    run(SOURCE, args.output, args.gpu)
