"""Bounded integration check: four recipes, Material2 paired export, one transfer image."""

import argparse
from datetime import datetime
import os
from pathlib import Path
import shlex
import sys

from experiments.sd15_token_local_kv_v1.contract import read_json
from experiments.sd21_qwen_initializers_v1 import contract
from experiments.sd21_qwen_initializers_v1.run import ENVIRONMENT, execute, export, require_idle
from experiments.sd21_qwen_midpoint_transfer_v1.protocol import plan
from experiments.sd21_qwen_midpoint_transfer_v1.run import run_group, collect
from experiments.subject_multicolor_multimaterial_b.training import _write
from scripts.launch.colorpeel_run import git_info, validate_run_id
from src.methods.colorpeel_ice.material_initialization import SCHEMA

ROOT = Path(__file__).resolve().parents[2]


def training_directory(output, config, revision):
    run = config['run']
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    path = output / 'training' / f'{stamp}__{run["study"]}__{run["variant"]}__{revision["commit"][:7]}__{run["seed"]}'
    validate_run_id(path, config, revision['commit'])
    return path


def verify(experiment, gpu):
    os.environ.update(ENVIRONMENT, CUDA_VISIBLE_DEVICES=str(gpu))
    require_idle((gpu,))
    experiment = Path(experiment).resolve()
    audit = read_json(experiment / 'source_audit.json')
    if (audit['initializers']['schema'] != SCHEMA or audit['new_training_git'] != git_info()
            or read_json(experiment / 'status.json')['status'] != 'prepared'):
        raise ValueError('require a freshly prepared controlled-initializer experiment')
    output = experiment / 'verification'
    output.mkdir()
    configs = {branch: contract.build_config(experiment, branch, smoke=True) for branch in contract.BRANCHES}
    _write(output / 'four_branch_startup_configs.json', configs)
    # Existing contract assigns Material physical GPU2. Keep the exact audited recipe.
    if gpu != contract.GPUS['material']:
        raise ValueError('Material startup uses the unchanged recipe GPU assignment')
    config = configs['material']
    _write(output / 'config.json', config)
    training = training_directory(output, config, audit['new_training_git'])
    command = [sys.executable, '-u', '-B', str(ROOT / 'scripts/launch/colorpeel_run.py'),
               '--config', str(output / 'config.json'), '--run-dir', str(training)]
    report = output / 'train_outputs'
    report.mkdir()
    (report / 'COMMANDS.md').write_text('```sh\n' + shlex.join(command) + '\n```\n', encoding='utf8')
    (report / 'LOG.md').write_text('Training log: ../training.log; paired export: ../ckpts/; inference: ../inference/.\n', encoding='utf8')
    (report / 'SCIENTIFIC_CHANGELOG.md').write_text('Technical Material2 startup only. Controlled VLM initializer; unchanged Material8 and recipe.\n', encoding='utf8')
    (report / 'COMPARABILITY_REPORT.md').write_text('Not a quality or convergence experiment. No composition; no comparison to150/200-step results.\n', encoding='utf8')
    state = {'status': 'training', 'max_steps': 2, 'completed_steps': 0, 'training_run': str(training),
             'best_checkpoint': None, 'best_metric': None, 'stop_reason': None, 'quality_verified': False}

    def save():
        _write(output / 'status.json', state)
        _write(report / 'status.json', state)
        (report / 'SUMMARY.md').write_text(f'Status: {state["status"]}; technical integration only, no visual-quality claim.\n', encoding='utf8')

    save()
    try:
        execute(command, output / 'training.log', gpu)
        state.update(status='exporting', completed_steps=2)
        save()
        result = export(experiment, 'material', training, output / 'ckpts', True)
        _write(output / 'exports.json', result)
        selected = result['steps'][2]
        inference = output / 'inference'
        inference.mkdir()
        infer_audit = {**audit, 'inference_git': audit['new_training_git'],
                       'selected_exports': {'material': selected}}
        _write(inference / 'source_audit.json', infer_audit)
        row = next(row for row in plan(audit['initializers'])
                   if row['mode'] == 'material_only' and row['case'] == 'chair' and row['seed'] == 30)
        row.update(id='controlled_material_smoke2__chair__seed30', phase='smoke2', branch_steps={'material': 2})
        state['status'] = 'inference'
        save()
        _write(inference / 'generation_plan.json', [row])
        run_group(inference, 'material_only', [row], infer_audit, gpu,
                  render_gallery=lambda *args: None)
        generated = collect(inference, [row])
        if len(generated) != 1 or read_json(inference / 'groups/material_only/status.json')['status'] != 'succeeded':
            raise ValueError('single-image inference did not complete')
        state.update(status='succeeded', stop_reason='bounded_integration_complete', generated_images=1,
                     safety_filtered_images=sum(item['nsfw_content_detected'] for item in generated))
    except BaseException as error:
        state.update(status='failed', stop_reason='stage_failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=2, choices=(2,))
    args = parser.parse_args()
    verify(args.experiment, args.gpu)
