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


def run(candidates, output, gpu, execute=False):
    candidates, output = Path(candidates).resolve(), resolve_run_dir(Path(output))
    if (type(gpu) is not int or gpu < 0 or output.exists() or output.is_relative_to(training.ROOT)
            or output.is_relative_to(candidates.parent) or candidates.is_relative_to(output)):
        raise ValueError('require a fresh external training run outside the candidate source')
    git = git_info()
    dataset = output / 'training_dataset'
    audit = stage_training.stage(candidates, dataset)
    training.build_configuration_bundle(dataset, output / 'training_plan')
    jobs = commands(dataset, output, gpu)
    report = output / 'train_outputs'
    report.mkdir()
    training._write(report / 'COMMANDS.json', jobs)
    (report / 'COMMANDS.md').write_text('\n\n'.join(
        f"{job['name']} (cwd {training.ROOT}):\n```sh\n"
        + ' '.join(f'{key}={shlex.quote(value)}' for key, value in job['environment'].items())
        + ' ' + shlex.join(job['command']) + '\n```' for job in jobs) + '\n', encoding='utf8')
    references = [json.loads(line) for line in (dataset / 'training_assets_manifest.jsonl').read_text().splitlines()]
    exposures = {key: dict(Counter(references[i % 180][key] for i in range(1000)))
                 for key in ('color', 'view', 'light', 'material')}
    training._write(report / 'EXPOSURES.json', {'updates_per_method': 1000, 'images': 180,
        'complete_cycles': 5, 'prefix_images': 100, 'exposures_per_image': [5, 6], 'counts': exposures})
    (report / 'LOG.md').write_text(
        'One serial training/inference job queue, no monitoring service. Fullrank runs first, then rank64, '
        'then step500/750/1000 tests. See fullrank.log, lora_r64.log and step*.log; '
        'detailed launcher logs, finite-loss metrics, model locks and training audits are below training/{method}.\n', encoding='utf8')
    (report / 'SCIENTIFIC_CHANGELOG.md').write_text(
        'Only data source, study/status metadata and paths change from the previous Subject recipes. '
        'All180 approved synthetic RGB/mask/caption triples are copied byte-for-byte, in source order. '
        'No original photograph or extra candidate is added. SD1.5, mailbox initializer, S embedding, '
        'full-rank token-local K/V or rank64/alpha64 token-local K/V, seed42, optimizer, precision, losses, '
        '1000 updates and paired500/750/1000 snapshots stay unchanged. No data regeneration or resume.\n', encoding='utf8')
    (report / 'COMPARABILITY_REPORT.md').write_text(
        'CONFIG_COMPARISON.json records exact inherited arguments. Same five material prompts '
        '(metallic/matte/glossy plastic/ceramic/wood), seeds42/43,PNDM100,CFG3.5,512px,fp16,safety checker. '
        'Each step produces20 outputs and sheets containing all180 actual training images and captions. '
        'Sequential DataLoader is unchanged:1000 updates=5 complete180-image cycles+first100, so each image '
        'has5 or6 exposures; EXPOSURES.json records exact condition counts. This differs in dose per image, '
        'geometry, viewpoints, composition and reflectance controls from MARBLE35/14; it is not a pure '
        'training-method or data-generation ablation. Training approval does not establish photographic '
        'identity, wood/cloth supervision or successful disentanglement.\n', encoding='utf8')
    state = {'status': 'planned', 'git': git, 'dataset': str(dataset), 'candidates': str(candidates),
        'source_mode': audit['source_mode'], 'training_images': 180, 'gpu': gpu, 'execute': execute,
        'resume': False, 'max_steps_per_method': 1000,
        'completed_steps': {method: 0 for method in training.METHODS},
        'checkpoint_steps': list(inference.STEPS), 'inference_steps': [], 'monitoring': False,
        'best_metric': None, 'best_checkpoint': None, 'stop_reason': None}
    baseline._report(output, state)
    try:
        if not execute:
            state.update(status='dry_run_verified', stop_reason='execution not requested')
            return state
        selections = []
        for job in jobs:
            if git_info() != git:
                raise ValueError('checkout changed during TRELLIS180 training/inference')
            state.update(status=job['phase'], active_job=job['name'])
            baseline._report(output, state)
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
                if (status.get('status') != 'training_verified' or status.get('completed_steps') != 1000
                        or index.get('git') != git or len(runs) != 1 or any(runs[0].get(key) != value
                        for key, value in {'status': 'verified', 'method': job['name'], 'branch': 'subject',
                                           'token': '<S*>', 'steps': 1000}.items())):
                    raise ValueError('training did not produce a completed Subject audit')
                selections.extend(runs)
                state['completed_steps'][job['name']] = 1000
                if len(selections) == 2:
                    training._write(output / 'training_index.json', {'git': git, 'runs': selections})
            else:
                step = int(job['name'][4:])
                provenance = training.read_json(output / job['name'] / 'provenance.json')
                if (provenance.get('status') != 'succeeded' or provenance.get('generated_images') != 20
                        or provenance.get('step') != step or provenance.get('sampling') != inference.SAMPLING
                        or provenance.get('training_images') != 180
                        or len(provenance.get('contact_sheets', [])) != 3):
                    raise ValueError('incomplete Subject material-transfer test')
                state['inference_steps'].append(step)
            baseline._report(output, state)
        training.verify_dataset(dataset)
        if git_info() != git:
            raise ValueError('checkout changed during TRELLIS180 training/inference')
        state.update(status='execution_verified', active_job=None, stop_reason='all requested jobs completed')
        return state
    except BaseException as error:
        state.update(status='failed', stop_reason=f'{type(error).__name__}: {error}')
        raise
    finally:
        baseline._report(output, state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpu', required=True, type=int)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    run(args.candidates, args.output, args.gpu, args.execute)
