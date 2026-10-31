"""Audit SD2.1 training and export matching adapter/token pairs at every saved step."""

from pathlib import Path

from experiments.sd15_token_local_kv_v1 import snapshots as shared
from experiments.sd15_token_local_kv_v1.contract import read_json, sha256
from experiments.sd_native_comparison_v1.generate import write_json
from experiments.sd21_wood_branches_v1 import contract
from scripts.launch.colorpeel_run import git_info, read_config

MODEL_DIR = contract.MODEL_DIR
WIDTH = 1024


def check_locks(locks):
    if not locks or any(not Path(path).is_absolute() or sha256(path) != digest for path, digest in locks.items()):
        raise ValueError('SD2.1 source bytes changed or locks are missing')


def model_locks(model_dir, receipt):
    model_dir = Path(model_dir).resolve()
    if Path(receipt['model']['root']).resolve() != model_dir:
        raise ValueError('SD2.1 receipt points to another model directory')
    return {str(model_dir / name): digest for name, digest in receipt['files'].items()} | {
        str(model_dir / 'model_source.json'): sha256(model_dir / 'model_source.json')}


def adapter_schema(method):
    """The pinned SD2.1-base topology has16 cross-attention blocks and1024 text channels."""
    processors = {}
    for block, hidden in ((0, 320), (1, 640), (2, 1280)):
        for attention in range(2):
            processors[f'down_blocks.{block}.attentions.{attention}.transformer_blocks.0'] = hidden
    processors['mid_block.attentions.0.transformer_blocks.0'] = 1280
    for block, hidden in ((1, 1280), (2, 640), (3, 320)):
        for attention in range(3):
            processors[f'up_blocks.{block}.attentions.{attention}.transformer_blocks.0'] = hidden
    shapes = {}
    for name, hidden in processors.items():
        for projection in ('k', 'v'):
            prefix = name + '.attn2.processor.'
            if method == 'fullrank':
                shapes[prefix + f'delta_{projection}.weight'] = (hidden, WIDTH)
            elif method == 'lora_r64':
                shapes[prefix + f'to_{projection}_lora.down.weight'] = (64, WIDTH)
                shapes[prefix + f'to_{projection}_lora.up.weight'] = (hidden, 64)
            else:
                raise ValueError('unknown SD2.1 adapter method')
    empty = {name + '.attn1.processor': {} for name in processors} if method == 'fullrank' else {}
    return shapes, empty


def audit_export(train_run, branch, method, output, smoke=False):
    import torch
    from experiments.sd21_wood_branches_v1.prepare_model import verify_model

    if branch not in contract.BRANCHES or method not in contract.METHODS or type(smoke) is not bool:
        raise ValueError('select a declared SD2.1 branch, method and smoke flag')
    run = Path(train_run).resolve()
    output = shared._new_output(output, run)
    if output.is_relative_to(Path(__file__).resolve().parents[2]) or run.is_relative_to(output):
        raise ValueError('SD2.1 exports must be fresh and outside code and training runs')
    model_dir = Path(MODEL_DIR).resolve()
    receipt = verify_model(model_dir)
    expected = contract.build_config(branch, method, model_dir=model_dir, smoke=smoke)
    config, manifest = read_config(run / 'config.yaml'), read_json(run / 'manifest.json')
    revision = git_info()
    if (config != expected or manifest.get('status') != 'succeeded' or manifest.get('returncode') != 0
            or manifest.get('stage') != 'train' or manifest.get('run') != config['run']
            or manifest.get('git') != revision):
        raise ValueError('SD2.1 training recipe, revision or completion differs')
    maximum = 2 if smoke else 1000
    steps = [2] if smoke else list(contract.STEPS)
    if (config['args'].get('max_train_steps') != maximum or shared.checkpoint_plan(config['args']) != steps
            or Path(config['args']['pretrained_model_name_or_path']).resolve() != model_dir):
        raise ValueError('SD2.1 base or checkpoint plan differs')
    data = contract.verify_dataset(branch)
    checkpoint, token = run / 'checkpoints', contract.TOKENS[branch]
    adaptation = read_json(checkpoint / 'adaptation_config.json')
    mode = 'token_local_kv' if method == 'fullrank' else ('lora_material_kv' if branch == 'material' else 'lora_subject_kv')
    if (adaptation.get('modifier_tokens') != [token] or adaptation.get('adaptation_mode') != mode
            or (method == 'lora_r64' and any(adaptation.get(key) != value for key, value in
                {'rank': 64, 'alpha': 64, 'mode': 'token_local_kv'}.items()))):
        raise ValueError('SD2.1 adaptation token, method or rank differs')
    if {path.name for path in checkpoint.glob('checkpoint-*') if path.is_dir()} != {f'checkpoint-{step}' for step in steps}:
        raise ValueError('saved SD2.1 snapshot directories differ from the declared plan')
    metrics = shared._metrics(checkpoint / 'training_metrics.jsonl', maximum, token)
    token_id, embedding_audit = shared._embedding_audit(checkpoint / 'embedding_update_audit.json', maximum, token)
    if (embedding_audit.get('non_modifier_embedding_drift') or {}).get('changed_rows') != 0:
        raise ValueError('SD2.1 ordinary token embeddings changed')
    shapes, empty = adapter_schema(method)
    weight = 'pytorch_token_local_kv_weights.bin' if method == 'fullrank' else 'pytorch_lora_kv_weights.bin'
    final_adapter = torch.load(checkpoint / weight, map_location='cpu')
    shared._check_state(torch, final_adapter, shapes, empty, 'SD2.1 final')
    final_token = torch.load(checkpoint / (token + '.bin'), map_location='cpu')
    if (not isinstance(final_token, dict) or set(final_token) != {token}
            or not isinstance(final_token[token], torch.Tensor) or tuple(final_token[token].shape) != (WIDTH,)
            or not bool(torch.isfinite(final_token[token]).all())):
        raise ValueError('SD2.1 final token must be a finite1024-vector')
    paths = [run / 'config.yaml', run / 'manifest.json'] + [checkpoint / name for name in
        (weight, token + '.bin', 'adaptation_config.json', 'training_metrics.jsonl', 'embedding_update_audit.json')]
    locks = {**model_locks(model_dir, receipt), **data['source_locks'], **{str(path): sha256(path) for path in paths}}
    for step in steps:
        for name in shared.STATE_FILES:
            path = checkpoint / f'checkpoint-{step}' / name
            if not path.is_file() or not path.stat().st_size:
                raise ValueError('SD2.1 paired snapshot has missing or empty training state')
            locks[str(path)] = sha256(path)
    check_locks(locks)
    parent = {'method': method, 'branch': branch, 'concept_id': 'wood_spoon' if branch == 'subject' else 'wood_spoon_' + branch,
        'token': token, 'status': 'verified', 'run': str(run), 'checkpoint': str(checkpoint), 'weight': weight,
        'steps': maximum, 'metrics': metrics, 'embedding_audit': embedding_audit, 'embedding_dim': WIDTH,
        'base_model': receipt['model'], 'hashes': locks, 'final_snapshot_matches_export': True}
    exports = {}
    for step in steps:
        directory = checkpoint / f'checkpoint-{step}'
        adapter = torch.load(directory / 'pytorch_model.bin', map_location='cpu')
        shared._check_state(torch, adapter, shapes, {}, f'SD2.1 step{step}')
        encoder = torch.load(directory / 'pytorch_model_1.bin', map_location='cpu')
        table = encoder.get(shared.EMBEDDING_KEY) if isinstance(encoder, dict) else None
        if (not isinstance(table, torch.Tensor) or table.ndim != 2 or table.shape[1] != WIDTH
                or not 0 <= token_id < table.shape[0]):
            raise ValueError('SD2.1 paired text encoder table or token ID differs')
        embedding = table[token_id].clone().contiguous()
        del table, encoder
        if not bool(torch.isfinite(embedding).all()):
            raise ValueError('SD2.1 paired token is nonfinite')
        if step == maximum and (not torch.equal(embedding, final_token[token])
                or any(not torch.equal(adapter[key], final_adapter[key]) for key in shapes)):
            raise ValueError('SD2.1 final adapter/token differs from its paired final snapshot')
        record = {'source_adapter_sha256': locks[str(directory / 'pytorch_model.bin')],
                  'source_text_encoder_sha256': locks[str(directory / 'pytorch_model_1.bin')]}
        derived = shared._write_export(torch, adapter, embedding, empty, parent, step, record, output / f'step{step}')
        derived.update(steps=step, training_steps=maximum, paired_embedding_finite=True,
            final_snapshot_matches_export=step == maximum,
            source_snapshot_hashes={str(directory / name): locks[str(directory / name)]
                                    for name in ('pytorch_model.bin', 'pytorch_model_1.bin')})
        derived['hashes'] = {**locks, **derived['hashes']}
        write_json(output / f'step{step}/receipt.json', derived)
        exports[step] = derived
    check_locks(locks)
    if git_info() != revision:
        raise ValueError('SD2.1 checkout changed during paired export')
    result = {'training': parent, 'steps': exports, 'source_locks': locks, 'references': data['references'],
              'training_images': data['rows'], 'git': revision, 'smoke': smoke}
    write_json(output / 'index.json', result)
    return result
