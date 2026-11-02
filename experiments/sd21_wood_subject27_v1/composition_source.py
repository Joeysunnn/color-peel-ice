"""Audit immutable S27@400/600 and T16@600 pairs despite paused generation jobs."""

from pathlib import Path

from experiments.sd15_token_local_kv_v1.contract import read_json, sha256
from experiments.sd21_wood_branches_v1 import composition as previous
from experiments.sd21_wood_subject27_v1 import contract
from scripts.launch.colorpeel_run import git_info, read_config

SUBJECT_ROOT = Path('/var/tmp/colorpeel-r12user5/runs/sd21_wood_subject27_v1/'
    '20261101-120109__sd21_wood_subject27_v1__both1000__675609d__42')
SUBJECT_GIT = {'commit': '675609d04c96d036793d8c32a547f390f4984573',
    'branch': 'codex/sd21-wood-subject27-training-v1'}
TEXTURE_ROOT, TEXTURE_GIT = previous.SOURCE, previous.SOURCE_GIT
SUBJECT_STEPS = (400, 600)
TEXTURE_STEP = 600


def _merge(target, extra):
    for path, digest in extra.items():
        if path in target and target[path] != digest:
            raise ValueError('Immutable composition source locks conflict')
        target[path] = digest


def _job(source, branch, method, expected_git):
    # Generation can be interrupted after training/export. This mutable file is
    # used only to resolve a pinned job description, never as a completion gate/lock.
    description = read_json(source / 'experiment.json')
    matches = [job for job in description.get('jobs', [])
               if job.get('branch') == branch and job.get('method') == method]
    if description.get('git') != expected_git or len(matches) != 1:
        raise ValueError('Pinned composition source Git/job description differs')
    group = source / 'methods' / method if branch == 'subject' else source / branch / method
    run = Path(matches[0].get('run', ''))
    if not run.is_absolute() or run.parent != group / 'training':
        raise ValueError('Pinned composition job training path differs')
    return group, run


def _pair(source, branch, method, step, expected_git, model, *, recipe_contract=None):
    group, run = _job(source, branch, method, expected_git)
    folder = group / 'inference'
    checkpoint = folder / f'ckpts/step{step}'
    receipt_path, index_path = checkpoint / 'receipt.json', folder / 'ckpts/index.json'
    item, index = read_json(receipt_path), read_json(index_path)
    if recipe_contract is None:
        recipe_contract = contract if branch == 'subject' else previous.contract
    expected = recipe_contract.build_config(branch, method)
    maximum = expected['args']['max_train_steps']
    manifest = read_json(run / 'manifest.json')
    token = recipe_contract.TOKENS[branch]
    weight = 'pytorch_token_local_kv_weights.bin' if method == 'fullrank' else 'pytorch_lora_kv_weights.bin'
    identity = {'branch': branch, 'method': method, 'step': step, 'steps': step,
        'training_steps': maximum, 'token': token, 'weight': weight, 'embedding_dim': 1024,
        'status': 'verified', 'paired_embedding_finite': True, 'final_snapshot_matches_export': step == maximum,
        'concept_id': getattr(recipe_contract, 'CONCEPT_IDS', {}).get(
            branch, 'wood_spoon' if branch == 'subject' else 'wood_spoon_texture')}
    if (not item.get('hashes') or item != index.get('steps', {}).get(str(step))
            or index.get('git') != expected_git or index.get('smoke') is not False
            or item.get('run') != str(run) or item.get('checkpoint') != str(checkpoint)
            or item.get('base_model') != model or item.get('source_checkpoint') != str(run / 'checkpoints')
            or any(item.get(key) != value for key, value in identity.items())
            or read_config(run / 'config.yaml') != expected
            or manifest.get('status') != 'succeeded' or manifest.get('returncode') != 0
            or manifest.get('stage') != 'train' or manifest.get('git') != expected_git
            or manifest.get('run') != expected['run']):
        raise ValueError('Selected immutable S27/T16 training or paired export differs')
    adaptation = read_json(checkpoint / 'adaptation_config.json')
    if (adaptation.get('modifier_tokens') != [token]
            or adaptation.get('adaptation_mode') != ('token_local_kv' if method == 'fullrank' else 'lora_subject_kv')
            or (method == 'lora_r64' and any(adaptation.get(key) != value
                for key, value in {'rank': 64, 'alpha': 64, 'mode': 'token_local_kv'}.items()))):
        raise ValueError('Selected composition adapter metadata differs')
    snapshot = run / 'checkpoints' / f'checkpoint-{step}'
    snapshot_paths = {str(snapshot / name) for name in ('pytorch_model.bin', 'pytorch_model_1.bin')}
    if set(item.get('source_snapshot_hashes', {})) != snapshot_paths:
        raise ValueError('Selected composition snapshot source paths differ')
    paths = [run / 'config.yaml', run / 'manifest.json', checkpoint / weight,
        checkpoint / (token + '.bin'), checkpoint / 'adaptation_config.json',
        run / 'checkpoints/embedding_update_audit.json', run / 'checkpoints/training_metrics.jsonl']
    paths += [Path(path) for path in snapshot_paths]
    hashes = item['hashes']
    if any(str(path) not in hashes for path in paths):
        raise ValueError('Selected immutable payload locks are incomplete')
    locks = {str(path): hashes[str(path)] for path in paths}
    if (any(item['source_snapshot_hashes'][path] != locks[path] for path in snapshot_paths)
            or item.get('source_adapter_sha256') != locks[str(snapshot / 'pytorch_model.bin')]
            or item.get('source_text_encoder_sha256') != locks[str(snapshot / 'pytorch_model_1.bin')]
            or item.get('derived_adapter_sha256') != locks[str(checkpoint / weight)]
            or item.get('derived_token_sha256') != locks[str(checkpoint / (token + '.bin'))]):
        raise ValueError('Selected composition payload and snapshot hashes disagree')
    _merge(locks, {str(path): sha256(path) for path in (receipt_path, index_path)})
    previous.snapshots.check_locks(locks)
    previous.check_pair(item, branch, method, step=step, training_steps=maximum)
    return {**item, 'hashes': locks}, locks


def prepare():
    revision = git_info()
    receipt = previous.verify_model()
    common_locks = previous.snapshots.model_locks(contract.MODEL_DIR, receipt)
    subject_data = contract.verify_dataset('subject')
    texture_data = previous.contract.verify_dataset('texture')
    if subject_data['rows'] != 27 or texture_data['rows'] != 16:
        raise ValueError('Composition requires the approved S27 and original T16 datasets')
    _merge(common_locks, subject_data['source_locks'])
    _merge(common_locks, texture_data['source_locks'])
    textures, texture_locks = {}, {}
    for method in contract.METHODS:
        textures[method], locks = _pair(TEXTURE_ROOT, 'texture', method, TEXTURE_STEP, TEXTURE_GIT, receipt['model'])
        _merge(texture_locks, locks)
    results = {}
    for step in SUBJECT_STEPS:
        sources, locks = {}, {**common_locks, **texture_locks}
        for method in contract.METHODS:
            sources[method], selected = _pair(SUBJECT_ROOT, 'subject', method, step, SUBJECT_GIT, receipt['model'])
            _merge(locks, selected)
        previous.snapshots.check_locks(locks)
        results[step] = {'inference_git': revision,
            'training_gits': {'subject': SUBJECT_GIT, 'texture': TEXTURE_GIT},
            'sources': {'subject': sources, 'texture': textures},
            'references': {'subject': subject_data['references'], 'texture': texture_data['references']},
            'source_locks': locks}
    if git_info() != revision:
        raise ValueError('Inference checkout changed during composition source validation')
    return results
