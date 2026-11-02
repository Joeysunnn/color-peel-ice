"""Paired SD2.1 Mailbox Subject27 transfer using the nine historical probes."""

import argparse
import json
from pathlib import Path

from experiments.sd21_mailbox_subject27_v1 import contract
from experiments.sd21_wood_branches_v1 import inference as shared, snapshots
from experiments.sd21_wood_branches_v1.prepare_model import verify_model
from experiments.subject_trellis_dataset_v1.inference import blue_green_prompt_rows
from experiments.sd_native_comparison_v1.generate import write_json
from scripts.launch.colorpeel_run import git_info, resolve_run_dir

SEEDS = tuple(range(30, 41))
SAMPLING = {**shared.SAMPLING, 'seeds': list(SEEDS)}


def prompt_rows(method, step, *, experiment_contract=contract):
    contract = experiment_contract
    if method not in contract.METHODS or type(step) is not int or step not in contract.STEPS:
        raise ValueError('require declared Subject method and paired checkpoint step')
    templates = [row for row in blue_green_prompt_rows(500) if row['method'] == method and row['seed'] == 42]
    rows = []
    for template in templates:
        label = template.get('condition', template.get('material'))
        slug = label.replace(' ', '_')
        for seed in SEEDS:
            rows.append({**template, 'id': f'{method}__step{step}__{slug}__seed{seed}',
                'branch': 'subject', 'step': step, 'seed': seed, 'label': label, 'cfg': 3.5,
                'subject_adapter_loaded': True, 'image_path': f'images/{slug}__seed{seed}.png'})
    return rows


def run(method, train_run, output, device='cuda:0', execute=False, *, experiment_contract=contract):
    contract = experiment_contract
    prompt_rows(method, contract.STEPS[0], experiment_contract=contract)
    output = resolve_run_dir(Path(output))
    train_run = Path(train_run).resolve()
    if (output.exists() or output.is_relative_to(shared.ROOT) or output.is_relative_to(train_run)
            or train_run.is_relative_to(output) or output.is_relative_to(Path(contract.MODEL_DIR).resolve())):
        raise ValueError('require a fresh external inference directory separate from training/model')
    revision = git_info()
    model = verify_model(contract.MODEL_DIR)
    locks = snapshots.model_locks(contract.MODEL_DIR, model)
    output.mkdir(parents=True)
    state = {'status': 'running', 'git': revision, 'method': method, 'branch': 'subject',
        'base_model': model['model'], 'sampling': SAMPLING,
        'generated_images': 0, 'completed_steps': [], 'monitoring': False,
        'training_images': None, 'expected_images': 99 * len(contract.STEPS)}
    try:
        exported = snapshots.audit_export(train_run, 'subject', method, output / 'ckpts', False,
                                          experiment_contract=contract)
        state['training_images'] = exported['training_images']
        locks.update(exported['source_locks'])
        locks.update(shared.reference_review(output, exported['references']))
        for step, selection in exported['steps'].items():
            directory = output / f'step{step}'
            directory.mkdir()
            rows = prompt_rows(method, step, experiment_contract=contract)
            (directory / 'generation_manifest.jsonl').write_text(
                ''.join(json.dumps(row) + '\n' for row in rows), encoding='utf8')
            generated = []
            if execute:
                import torch
                snapshots.check_locks(locks)
                pipe = shared.load_pipeline([selection], device, model_dir=contract.MODEL_DIR,
                                            experiment_contract=contract)
                generated = shared._generate(pipe, rows, directory, device, '<S*>')
                del pipe
                torch.cuda.empty_cache()
                if len(generated) != len(rows):
                    raise ValueError('incomplete Subject transfer matrix')
                state['generated_images'] += len(generated)
            shared.review(directory, generated or rows, training=True)
            state['completed_steps'].append(step)
            write_json(output / 'status.json', state)
        snapshots.check_locks(locks)
        if git_info() != revision:
            raise ValueError('execution checkout changed')
        state['status'] = 'succeeded' if execute else 'dry_run_verified'
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        state['source_locks'] = locks
        write_json(output / 'status.json', state)
        write_json(output / 'provenance.json', state)
        links = ''.join(f'<li><a href="step{step}/index.html">step{step}</a></li>' for step in state['completed_steps'])
        (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>SD2.1 Mailbox Subject</h1>'
            + f'<p>{method}; {state["status"]}; {state["generated_images"]} images; PNDM100/CFG3.5/512px</p>'
            + f'<a href="training_references.html">Actual {state["training_images"]} training images and captions</a><ul>' + links + '</ul>', encoding='utf8')
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', required=True, choices=contract.METHODS)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    run(args.method, args.run, args.output, execute=args.execute)
