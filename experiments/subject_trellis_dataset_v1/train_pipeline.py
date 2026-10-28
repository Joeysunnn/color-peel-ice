"""Train the two approved Subject recipes, then test all three saved steps."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shlex
import subprocess

from experiments.subject_multicolor_multimaterial_b import run as baseline
from experiments.subject_trellis_dataset_v1 import inference, stage_training, training
from scripts.launch.colorpeel_run import git_info, resolve_run_dir


def commands(dataset, output, gpu):
    jobs = baseline.commands(dataset, output, gpu)
    for job in jobs:
        job['command'][3] = job['command'][3].replace(
            'subject_multicolor_multimaterial_b', 'subject_trellis_dataset_v1')
    return jobs


def report_state(output, state):
    training._write(output / 'status.json', state)
    training._write(output / 'train_outputs/status.json', state)
    (output / 'train_outputs/SUMMARY.md').write_text(
        f"Status: {state['status']}. Mode: full kickoff, no resume.\n"
        f"Completed audited training steps: {state['completed_steps']} ({state['max_steps_per_method']} per method).\n"
        f"Completed inference steps: {state['inference_steps']}; {state['expected_images_per_step']} images per step.\n"
        f"Stop reason: {state['stop_reason']}. Best metric/checkpoint: not selected.\n", encoding='utf8')


def run(candidates, output, gpu, execute=False, blue_green_only=False, *,
        stage_module=stage_training, training_module=training, inference_module=inference,
        command_builder=commands, subject_noun='mailbox', max_steps=1000):
    stage_training, training, inference = stage_module, training_module, inference_module
    candidates, output = Path(candidates).resolve(), resolve_run_dir(Path(output))
    if (type(gpu) is not int or gpu < 0 or output.exists() or output.is_relative_to(training.ROOT)
            or output.is_relative_to(candidates.parent) or candidates.is_relative_to(output)):
        raise ValueError('require a fresh external training run outside the candidate source')
    git = git_info()
    dataset = output / 'training_dataset'
    audit = stage_training.stage(candidates, dataset, blue_green_only)
    count = audit['rows']
    step_label = '/'.join(map(str, inference.STEPS))
    expected_images = inference.expected_images(dataset)
    training.build_configuration_bundle(dataset, output / 'training_plan')
    jobs = command_builder(dataset, output, gpu)
    report = output / 'train_outputs'
    report.mkdir()
    training._write(report / 'COMMANDS.json', jobs)
    (report / 'COMMANDS.md').write_text('\n\n'.join(
        f"{job['name']} (cwd {training.ROOT}):\n```sh\n"
        + ' '.join(f'{key}={shlex.quote(value)}' for key, value in job['environment'].items())
        + ' ' + shlex.join(job['command']) + '\n```' for job in jobs) + '\n', encoding='utf8')
    references = [json.loads(line) for line in (dataset / 'training_assets_manifest.jsonl').read_text().splitlines()]
    exposures = {key: dict(Counter(references[i % count][key] for i in range(max_steps)))
                 for key in ('color', 'view', 'light', 'material')}
    cycles, prefix = divmod(max_steps, count)
    training._write(report / 'EXPOSURES.json', {'updates_per_method': max_steps, 'images': count,
        'complete_cycles': cycles, 'prefix_images': prefix, 'exposures_per_image': [cycles, cycles + 1], 'counts': exposures})
    (report / 'LOG.md').write_text(
        'One serial training/inference job queue, no monitoring service. Fullrank runs first, then rank64, '
        f'then step{step_label} tests. See fullrank.log, lora_r64.log and step*.log; '
        'detailed launcher logs, finite-loss metrics, model locks and training audits are below training/{method}.\n', encoding='utf8')
    (report / 'SCIENTIFIC_CHANGELOG.md').write_text(
        ('Only data source, study/status metadata and paths change from the previous Subject recipes. '
         if subject_noun == 'mailbox' else
         f'Data source, subject metadata, paths and the initializer/class noun change from mailbox to {subject_noun}. ')
        +
        f'All{count} approved synthetic RGB/mask/caption triples from colors{audit["selected_colors"]} '
        'are copied byte-for-byte, in source order. '
        'No original photograph or extra candidate is added. '
        + ('SD1.5, mailbox initializer, S embedding, ' if subject_noun == 'mailbox' else
           f'The {subject_noun} initializer is used consistently; SD1.5, S embedding, ')
        +
        'full-rank token-local K/V or rank64/alpha64 token-local K/V, seed42, optimizer, precision, losses, '
        + ('1000 updates and paired500/750/1000 snapshots stay unchanged. No data regeneration or resume.\n'
           if max_steps == 1000 else f'Training dose is{max_steps} updates with paired{step_label} snapshots. No data regeneration or resume.\n'), encoding='utf8')
    (report / 'COMPARABILITY_REPORT.md').write_text(
        'CONFIG_COMPARISON.json records exact inherited arguments. Same five material prompts '
        '(metallic/matte/glossy plastic/ceramic/wood), seeds42/43,PNDM100,CFG3.5,512px,fp16,safety checker. '
        f'Each step produces{expected_images} outputs and sheets containing all{count} actual training images and captions. '
        f'Sequential DataLoader is unchanged:{max_steps} updates={cycles} complete{count}-image cycles+first{prefix}, '
        f'so each image has{cycles} or{cycles + 1} exposures; EXPOSURES.json records exact condition counts. '
        'This differs in dose per image, '
        'geometry, viewpoints, composition and reflectance controls from MARBLE35/14; it is not a pure '
        'training-method or data-generation ablation. Training approval does not establish photographic '
        'identity, wood/cloth supervision or successful disentanglement.\n', encoding='utf8')
    if blue_green_only:
        with (report / 'SCIENTIFIC_CHANGELOG.md').open('a', encoding='utf8') as handle:
            handle.write('Only blue/green72 candidates are selected; pink/cyan/purple are excluded. '
                'Add four independent context/color prompts: in the snow; surrounded by flowers; '
                'in white color; in orange color. No joint material/context/color cross product.\n')
        with (report / 'COMPARABILITY_REPORT.md').open('a', encoding='utf8') as handle:
            handle.write('The unchanged20 material outputs plus16 context/color outputs total36 images per step '
                'and108 overall. All probes use the same seeds/sampling and paired step-specific Subject weights.\n')
    state = {'status': 'planned', 'git': git, 'dataset': str(dataset), 'candidates': str(candidates),
        'source_mode': audit['source_mode'], 'training_images': count, 'selected_colors': audit['selected_colors'],
        'blue_green_only': blue_green_only, 'expected_images_per_step': expected_images,
        'expected_images_total': expected_images * len(inference.STEPS), 'gpu': gpu, 'execute': execute,
        'resume': False, 'max_steps_per_method': max_steps,
        'completed_steps': {method: 0 for method in training.METHODS},
        'checkpoint_steps': list(inference.STEPS), 'inference_steps': [], 'monitoring': False,
        'best_metric': None, 'best_checkpoint': None, 'stop_reason': None}
    if 'subject_id' in audit:
        state['subject_id'] = audit['subject_id']
    report_state(output, state)
    try:
        if not execute:
            state.update(status='dry_run_verified', stop_reason='execution not requested')
            return state
        selections = []
        for job in jobs:
            if git_info() != git:
                raise ValueError('checkout changed during TRELLIS training/inference')
            state.update(status=job['phase'], active_job=job['name'])
            report_state(output, state)
            with (report / f"{job['name']}.log").open('x', encoding='utf8') as log:
                result = subprocess.run(job['command'], cwd=training.ROOT,
                    env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', **job['environment']},
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"{job['name']} failed; see train_outputs/{job['name']}.log")
            if job['phase'] == 'training':
                directory = output / 'training' / job['name']
                status = training.read_json(directory / 'status.json')
                index = training.read_json(directory / 'training_index.json')
                runs = index.get('runs', [])
                if (status.get('status') != 'training_verified' or status.get('completed_steps') != max_steps
                        or index.get('git') != git or len(runs) != 1 or any(runs[0].get(key) != value
                        for key, value in {'status': 'verified', 'method': job['name'], 'branch': 'subject',
                                           'token': '<S*>', 'steps': max_steps}.items())):
                    raise ValueError('training did not produce a completed Subject audit')
                selections.extend(runs)
                state['completed_steps'][job['name']] = max_steps
                if len(selections) == 2:
                    training._write(output / 'training_index.json', {'git': git, 'runs': selections})
            else:
                step = int(job['name'][4:])
                provenance = training.read_json(output / job['name'] / 'provenance.json')
                if (provenance.get('status') != 'succeeded' or provenance.get('generated_images') != expected_images
                        or provenance.get('step') != step or provenance.get('sampling') != inference.SAMPLING
                        or provenance.get('training_images') != count
                        or len(provenance.get('contact_sheets', [])) != 3):
                    raise ValueError('incomplete Subject material-transfer test')
                state['inference_steps'].append(step)
            report_state(output, state)
        training.verify_dataset(dataset)
        if git_info() != git:
            raise ValueError('checkout changed during TRELLIS training/inference')
        state.update(status='execution_verified', active_job=None, stop_reason='all requested jobs completed')
        return state
    except BaseException as error:
        state.update(status='failed', stop_reason=f'{type(error).__name__}: {error}')
        raise
    finally:
        report_state(output, state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpu', required=True, type=int)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--blue-green-only', action='store_true')
    args = parser.parse_args()
    run(args.candidates, args.output, args.gpu, args.execute, args.blue_green_only)
