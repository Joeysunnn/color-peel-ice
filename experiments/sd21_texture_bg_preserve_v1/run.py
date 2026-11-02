"""Audit, probe, resume two LoRA arms to600, then evaluate48 paired transfers."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.sd21_background_diagnosis_v1 import run as diagnosis
from experiments.sd21_texture_bg_preserve_v1 import contract, resume_source
from experiments.subject_multicolor_multimaterial_b.training import _write

sct = diagnosis.sct
ROOT = Path(__file__).resolve().parents[2]


def prepare(previous_path):
    audit = diagnosis.prepare(Path(previous_path).resolve())
    audit['texture_sources'] = {}
    for step in (400, 600):
        item, locks = sct.st.source._pair(sct.st.source.TEXTURE_ROOT, 'texture', 'lora_r64', step,
            sct.st.source.TEXTURE_GIT, audit['sources']['texture']['lora_r64']['base_model'],
            recipe_contract=contract.original)
        audit['texture_sources'][str(step)] = item
        sct.st.source._merge(audit['source_locks'], locks)
    audit['new_training_git'] = sct.st.git_info()
    audit['original_texture_training_git'] = sct.st.source.TEXTURE_GIT
    return audit


def compare(original, resumed):
    import torch

    result = {}
    for name, old_name, new_name in (('adapter', original['weight'], resumed['weight']), ('token', '<T*>.bin', '<T*>.bin')):
        old_path, new_path = Path(original['checkpoint']) / old_name, Path(resumed['checkpoint']) / new_name
        old, new = torch.load(old_path, map_location='cpu'), torch.load(new_path, map_location='cpu')
        if set(old) != set(new):
            raise ValueError('old/new T600 tensor names differ')
        rows = {}
        for key in old:
            if old[key].shape != new[key].shape or not bool(torch.isfinite(new[key]).all()):
                raise ValueError('old/new T600 tensor shapes or finite values differ')
            delta = (new[key].double() - old[key].double())
            rows[key] = {'equal': torch.equal(old[key], new[key]), 'old_dtype': str(old[key].dtype),
                         'new_dtype': str(new[key].dtype), 'same_dtype': old[key].dtype == new[key].dtype,
                         'max_abs': float(delta.abs().max()),
                         'l2': float(delta.norm()), 'reference_l2': float(old[key].double().norm())}
        result[name] = {'all_tensors_equal': all(row['equal'] for row in rows.values()), 'tensors': rows,
                        'original_sha256': sct.sha256(old_path), 'resumed_sha256': sct.sha256(new_path)}
        result[name]['file_sha_equal'] = result[name]['original_sha256'] == result[name]['resumed_sha256']
    result['interpretation'] = 'Observed tensor and file comparisons; no assumption of exact continuation equivalence.'
    return result


def run(output, previous_path):
    output, previous_path = sct.st.resolve_run_dir(Path(output)), Path(previous_path).resolve()
    roots = (ROOT, sct.st.SUBJECT_PARENT, sct.st.source.TEXTURE_ROOT, sct.COLOR_ROOT,
             contract.MODEL_DIR, contract.original.NATURAL_DATASET, previous_path.parent)
    if output.exists() or any(output.is_relative_to(root) or root.is_relative_to(output) for root in roots):
        raise ValueError('require fresh external output separate from code, models, datasets and sources')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    sct.st.require_idle_gpu0()
    audit = prepare(previous_path)
    sct.st.require_idle_gpu0()
    output.mkdir(parents=True)
    from experiments.sd21_texture_bg_preserve_v1.inference import plan
    (output / 'generation_manifest.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in plan()), encoding='utf8')
    state = {'status': 'preparing', 'gpu': 0, 'arms': {}, 'max_steps': 600, 'resume_step': 400,
             'new_training_git': audit['new_training_git'], 'original_texture_training_git': audit['original_texture_training_git'],
             'inference_git': audit['inference_git'], 'expected_images': 48, 'generated_images': 0}
    sct.write(output / 'source_audit.json', audit)
    sct.write(output / 'status.json', state)
    environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': '0', 'PYTHONDONTWRITEBYTECODE': '1',
                   'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'DIFFUSERS_OFFLINE': '1'}

    def execute(command, label):
        state.update(status=label, command=command)
        sct.write(output / 'status.json', state)
        sct.st.shared.snapshots.check_locks(audit['source_locks'])
        if sct.st.git_info() != audit['new_training_git']:
            raise ValueError('checkout changed during paired continuation')
        with (output / f'{label}.log').open('x', encoding='utf8') as log:
            subprocess.run(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                           stdout=log, stderr=subprocess.STDOUT, check=True)

    try:
        derived = resume_source.derive(output / 'resume_parent', audit['texture_sources']['400'], sct.st.shared.verify_model())
        sct.st.source._merge(audit['source_locks'], derived['source_locks'])
        sct.st.source._merge(audit['source_locks'], derived['derived_locks'])
        audit['derived_resume'] = derived
        sct.write(output / 'source_audit.json', audit)
        execute([sys.executable, '-u', '-B', '-m', 'experiments.sd21_texture_bg_preserve_v1.probe',
                 '--audit', str(output / 'source_audit.json'), '--output', str(output / 'gradient_probe.json')], 'gradient_probe')
        if sct.st.read_json(output / 'gradient_probe.json').get('passed') is not True:
            raise ValueError('real T400 gradient probe did not pass')
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        exports = {}
        for arm in contract.ARMS:
            group = output / arm
            group.mkdir()
            config = contract.build_config(arm, derived['checkpoint'])
            config_path = group / 'config.json'
            _write(config_path, config)
            from scripts.launch.colorpeel_run import read_config
            if read_config(config_path) != config:
                raise ValueError('serialized training config differs')
            name = f"{stamp}__{contract.STUDY}__{config['run']['variant']}__{audit['new_training_git']['commit'][:7]}__42"
            train_run = group / 'training' / name
            state['arms'][arm] = {'status': 'training', 'weight': contract.ARMS[arm], 'run': str(train_run)}
            execute([sys.executable, '-u', '-B', str(ROOT / 'scripts/launch/colorpeel_run.py'),
                     '--config', str(config_path), '--run-dir', str(train_run)], f'{arm}_training')
            prefix = (Path(derived['checkpoint']).parent / 'training_metrics.jsonl').read_bytes()
            metrics = (train_run / 'checkpoints/training_metrics.jsonl').read_bytes()
            if b''.join(metrics.splitlines(keepends=True)[:400]) != prefix:
                raise ValueError('continued metrics do not preserve the exact original400 prefix')
            result = sct.st.shared.snapshots.audit_export(train_run, 'texture', 'lora_r64', group / 'ckpts',
                experiment_contract=contract.recipe(arm, derived['checkpoint']))
            exports[arm] = {str(step): item for step, item in result['steps'].items()}
            sct.st.source._merge(audit['source_locks'], result['source_locks'])
            for item in exports[arm].values():
                sct.st.source._merge(audit['source_locks'], item['hashes'])
            state['arms'][arm]['status'] = 'trained_and_exported'
            sct.write(output / 'status.json', state)
            sct.write(output / 'source_audit.json', audit)
            sct.write(output / 'exports.json', exports)
        sct.write(output / 'lambda0_vs_original_t600.json', compare(audit['texture_sources']['600'], exports['lambda0']['600']))
        execute([sys.executable, '-u', '-B', '-m', 'experiments.sd21_texture_bg_preserve_v1.inference',
                 '--experiment', str(output)], 'inference')
        inference_state = sct.st.read_json(output / 'inference/status.json')
        if inference_state['status'] != 'succeeded' or inference_state['generated_images'] != 48:
            raise ValueError('48-image evaluation did not complete')
        sct.st.shared.snapshots.check_locks(audit['source_locks'])
        state.update(status='succeeded', generated_images=48)
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        sct.write(output / 'status.json', state)
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--source-audit', required=True, type=Path)
    args = parser.parse_args()
    run(args.output, args.source_audit)
