"""Paired SD2.1 Color-only transfer with the ten historical prompts and CFG6."""

import argparse
import json
from pathlib import Path

from experiments.mailbox_color_sd15_v1 import prompts
from experiments.sd21_color_gray9_v1 import contract
from experiments.sd21_wood_branches_v1 import inference as shared, snapshots
from experiments.sd21_wood_branches_v1.prepare_model import verify_model
from experiments.sd_native_comparison_v1.generate import write_json
from scripts.launch.colorpeel_run import git_info, resolve_run_dir

SEEDS = tuple(range(30, 41))
SAMPLING = {**prompts.SAMPLING, 'seeds': list(SEEDS)}


def prompt_rows(method, step, smoke=False):
    if method not in contract.METHODS or step not in ((2,) if smoke else contract.STEPS):
        raise ValueError('require declared Color method and paired step')
    conditions = [row for row in prompts.conditions() if row['arm'] == 'color_only']
    rows = [{'id': f'{method}__step{step}__{row["condition"]}__seed{seed}',
             'branch': 'color', 'method': method, 'step': step, 'seed': seed,
             **row, 'label': row['condition'], 'cfg': 6.0, 'color_adapter_loaded': True,
             'image_path': f'images/{row["condition"]}__seed{seed}.png'}
            for row in conditions for seed in ((30,) if smoke else SEEDS)]
    return rows[:1] if smoke else rows


def run(method, train_run, output, smoke=False, device='cuda:0', execute=False):
    output = resolve_run_dir(Path(output))
    train_run = Path(train_run).resolve()
    if (output.exists() or output.is_relative_to(shared.ROOT) or output.is_relative_to(train_run)
            or train_run.is_relative_to(output)):
        raise ValueError('require a fresh external inference directory separate from training')
    revision = git_info()
    model = verify_model()
    locks = snapshots.model_locks(contract.MODEL_DIR, model)
    locks[str(prompts.SOURCE)] = prompts.sha256(prompts.SOURCE)
    output.mkdir(parents=True)
    state = {'status': 'running', 'git': revision, 'method': method, 'branch': 'color',
             'base_model': model['model'], 'sampling': {**SAMPLING, 'seeds': [30] if smoke else list(SEEDS)},
             'smoke': smoke, 'generated_images': 0, 'completed_steps': [], 'monitoring': False,
             'training_images': 9, 'expected_images': 1 if smoke else 440}
    try:
        exported = snapshots.audit_export(train_run, 'color', method, output / 'ckpts', smoke,
                                          experiment_contract=contract)
        locks.update(exported['source_locks'])
        reference_hashes = shared.reference_review(output, exported['references'])
        locks.update(reference_hashes)
        for step, selection in exported['steps'].items():
            directory = output / f'step{step}'
            directory.mkdir()
            rows = prompt_rows(method, step, smoke)
            (directory / 'generation_manifest.jsonl').write_text(
                ''.join(json.dumps(row) + '\n' for row in rows), encoding='utf8')
            generated = []
            if execute:
                import torch
                snapshots.check_locks(locks)
                pipe = shared.load_pipeline([selection], device, experiment_contract=contract)
                generated = shared._generate(pipe, rows, directory, device, '<C*>')
                del pipe
                torch.cuda.empty_cache()
                if len(generated) != len(rows):
                    raise ValueError('incomplete Color transfer matrix')
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
        (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>SD2.1 Color gray9</h1>'
            + f'<p>{method}; {state["status"]}; {state["generated_images"]} images; PNDM100/CFG6/512px</p>'
            + '<a href="training_references.html">Actual nine training images and captions</a><ul>' + links + '</ul>', encoding='utf8')
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', required=True, choices=contract.METHODS)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    run(args.method, args.run, args.output, args.smoke, execute=args.execute)
