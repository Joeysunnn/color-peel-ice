"""Train and export the two SD1.5 token-local Color recipes."""

import argparse
import copy
from pathlib import Path

from experiments.mailbox_color_sd15_v1 import data
from experiments.subject_multicolor_multimaterial_b import training as baseline
from experiments.sd15_token_local_kv_v1 import snapshots
from experiments.sd15_token_local_kv_v1.contract import read_json, sha256
from experiments.texture_material_disentangled_v1 import train as shared
from scripts.launch.colorpeel_run import read_config

ROOT, METHODS = baseline.ROOT, baseline.METHODS
STUDY = 'mailbox_color_sd15_v1'
STEPS = (400, 600, 800, 1000)
ARG_CHANGES = {'concepts_list', 'max_train_steps', 'checkpoint_steps', 'checkpointing_steps',
               'modifier_token', 'initializer_token', 'hflip', 'freeze_model'}


def recipe(method):
    if method not in METHODS:
        raise ValueError('unsupported Color method')
    return {'method': method, 'branch': 'color', 'concept_id': 'mailbox_original_orange', 'token': '<C*>'}


def build_config(method, dataset):
    recipe(method)
    dataset = Path(dataset).resolve()
    config = copy.deepcopy(read_json(baseline.BASELINES[method]))
    config.pop('sd15_migration')
    config.update(status='authorized_mailbox_color_sd15')
    config['run'] = {'study': STUDY, 'variant': f'mailbox_orange_{method}_steps1000', 'seed': 42}
    config['environment'].pop('COLORPEEL_MAILBOX_MATTE_ROOT', None)
    config['data_manifest'] = str(dataset / 'training_assets_manifest.jsonl')
    config['args'].update(concepts_list=str(dataset / 'concepts.json'), max_train_steps=1000,
        checkpointing_steps=200, checkpoint_steps=list(STEPS), modifier_token='<C*>',
        initializer_token='orange', hflip=True, freeze_model='crossattn_kv')
    return config


def build_configuration_bundle(dataset, output):
    return baseline.build_configuration_bundle(dataset, output, verify_dataset=data.verify_dataset,
        config_builder=build_config, allowed_arg_changes=ARG_CHANGES)


def run(dataset, output, method, gpu, execute_training=False):
    result = baseline.run(dataset, output, method, gpu, execute_training,
        verify_dataset=data.verify_dataset, config_builder=build_config,
        allowed_arg_changes=ARG_CHANGES, audit_recipe=recipe(method))
    if execute_training:
        index = Path(output) / 'training_index.json'
        baseline._write(index, {**read_json(index), 'dataset': str(Path(dataset).resolve())})
    return result


def audit_run(method, run, dataset, expected_git=None):
    data.verify_dataset(Path(dataset))
    return shared.audit_run(recipe(method), Path(run), build_config(method, dataset), expected_git=expected_git)


def _locks(locks):
    if not locks or any(not Path(p).is_absolute() or sha256(p) != digest for p, digest in locks.items()):
        raise ValueError('Color source artifact changed or missing locks')


def export_step(selection, step, output):
    """Extract C and its adapter from the same audited Accelerator snapshot."""
    import torch

    if (selection.get('token') != '<C*>' or selection.get('branch') != 'color'
            or selection.get('status') != 'verified' or selection.get('method') not in METHODS
            or selection.get('steps') != 1000 or type(step) is not int or step not in STEPS):
        raise ValueError('require an audited Color selection and declared step')
    _locks(selection['hashes'])
    config = read_config(Path(selection['run']) / 'config.yaml')
    if (config['args']['modifier_token'] != '<C*>' or config['args']['max_train_steps'] != 1000
            or snapshots.checkpoint_plan(config['args']) != list(STEPS)):
        raise ValueError('Color checkpoint plan differs')
    output = snapshots._new_output(output, Path(selection['run']).resolve())
    checkpoint = Path(selection['checkpoint'])
    directory = checkpoint / f'checkpoint-{step}'
    if any(not (directory / name).is_file() or not (directory / name).stat().st_size
           for name in snapshots.STATE_FILES):
        raise ValueError('Color snapshot state missing or empty')
    token_id, _ = snapshots._embedding_audit(checkpoint / 'embedding_update_audit.json', 1000, '<C*>')
    shapes, empty = snapshots.adapter_schema(selection['method'])
    adapter_path, encoder_path = directory / 'pytorch_model.bin', directory / 'pytorch_model_1.bin'
    source_hashes = {str(p.resolve()): sha256(p) for p in (adapter_path, encoder_path)}
    adapter = torch.load(adapter_path, map_location='cpu')
    snapshots._check_state(torch, adapter, shapes, {}, f'Color step{step}')
    encoder = torch.load(encoder_path, map_location='cpu')
    table = encoder.get(snapshots.EMBEDDING_KEY) if isinstance(encoder, dict) else None
    if (not isinstance(table, torch.Tensor) or table.ndim != 2 or table.shape[1] != 768
            or not 0 <= token_id < table.shape[0]):
        raise ValueError('Color embedding table or token ID differs')
    embedding = table[token_id].clone().contiguous()
    del table, encoder
    if not bool(torch.isfinite(embedding).all()):
        raise ValueError('Color embedding is nonfinite')
    if step == 1000:
        final = torch.load(checkpoint / selection['weight'], map_location='cpu')
        snapshots._check_state(torch, final, shapes, empty, 'final Color')
        final_token = torch.load(checkpoint / '<C*>.bin', map_location='cpu')
        if (not isinstance(final_token, dict) or set(final_token) != {'<C*>'}
                or not torch.equal(final_token['<C*>'], embedding)
                or any(not torch.equal(adapter[k], final[k]) for k in shapes)):
            raise ValueError('final Color snapshot differs from paired export')
    _locks({**selection['hashes'], **source_hashes})
    record = {'source_adapter_sha256': source_hashes[str(adapter_path.resolve())],
              'source_text_encoder_sha256': source_hashes[str(encoder_path.resolve())]}
    derived = snapshots._write_export(torch, adapter, embedding, empty, selection, step, record, output)
    derived.update(steps=step, training_steps=1000, source_snapshot_hashes=source_hashes,
                   paired_embedding_finite=True)
    derived['hashes'] = {**selection['hashes'], **source_hashes, **derived['hashes']}
    baseline._write(output / 'receipt.json', derived)
    return derived


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--execute-training', action='store_true')
    args = parser.parse_args()
    run(args.dataset, args.output, args.method, args.gpu, args.execute_training)
