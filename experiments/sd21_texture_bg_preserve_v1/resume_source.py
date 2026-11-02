"""Derive a read-only-resume package without relabelling the old final audit."""

import json
import math
from pathlib import Path
import shutil

from experiments.sd21_mailbox_sct_v1 import run as sct
from src.train.resume_utils import METRIC_FIELDS, STATE_FILES


def validated_prefix(metrics_bytes, final_audit):
    lines = metrics_bytes.splitlines(keepends=True)
    rows = [json.loads(line) for line in lines]
    if len(rows) != 1000 or any(type(row.get('step')) is not int or row['step'] != index
                               for index, row in enumerate(rows, 1)):
        raise ValueError('require exactly 1000 continuous original metric lines')
    for row in rows:
        flags = row.get('losses_finite', {})
        if (row.get('present_modifier_tokens') != ['<T*>'] or not flags
                or any(value is not True for value in flags.values())
                or any(type(row.get(key)) not in (int, float) or not math.isfinite(row[key]) for key in METRIC_FIELDS)):
            raise ValueError('original metrics have invalid loss or Texture exposure')
    records = final_audit.get('modifier_tokens', [])
    if final_audit.get('observed_optimization_steps') != 1000 or len(records) != 1:
        raise ValueError('require the original completed1000 audit')
    record = records[0]
    if (record.get('token') != '<T*>' or type(record.get('token_id')) is not int
            or record.get('exposure_steps') != 1000 or record.get('nonzero_gradient_steps') != 1000
            or final_audit.get('non_modifier_embedding_drift', {}).get('changed_rows') != 0):
        raise ValueError('prefix counts cannot be inferred unless every original step exposed and updated Texture')
    return b''.join(lines[:400]), record['token_id']


def derive(output, source400, model_receipt):
    import torch
    from transformers import AutoTokenizer
    from src.train.training_audit import EmbeddingUpdateAudit

    output = Path(output)
    if output.exists():
        raise ValueError('derived resume package must be fresh')
    run = Path(source400['run'])
    parent = run / 'checkpoints'
    source_checkpoint = parent / 'checkpoint-400'
    audit_path, metrics_path = parent / 'embedding_update_audit.json', parent / 'training_metrics.jsonl'
    final_audit = sct.st.read_json(audit_path)
    prefix, token_id = validated_prefix(metrics_path.read_bytes(), final_audit)
    model = Path(model_receipt['model']['root'])
    base_path = model / 'text_encoder/model.safetensors'
    paths = [audit_path, metrics_path, base_path] + [source_checkpoint / name for name in STATE_FILES]
    locks = {str(path): sct.sha256(path) for path in paths}
    sct.st.shared.snapshots.check_locks(locks)
    key = sct.st.shared.snapshots.shared.EMBEDDING_KEY
    from safetensors import safe_open
    with safe_open(str(base_path), framework='pt', device='cpu') as handle:
        base = handle.get_tensor(key)
    current = torch.load(source_checkpoint / 'pytorch_model_1.bin', map_location='cpu')[key]
    tokenizer = AutoTokenizer.from_pretrained(str(model), subfolder='tokenizer', use_fast=False, local_files_only=True)
    initializer = tokenizer.encode('texture', add_special_tokens=False)
    if len(initializer) != 1 or token_id != base.shape[0] or current.shape != (token_id + 1, 1024):
        raise ValueError('original Texture token ID, initializer or encoder shape differs')
    initial = torch.cat((base, base[initializer[0]:initializer[0] + 1]), dim=0)
    observer = EmbeddingUpdateAudit({'<T*>': token_id}, initial)
    observer.observed_optimization_steps = 400
    observer.exposure_steps['<T*>'] = observer.nonzero_gradient_steps['<T*>'] = 400
    derived_audit = observer.finalize(current)
    delta = derived_audit['modifier_tokens'][0]['initial_final_l2_delta']
    if (derived_audit['non_modifier_embedding_drift']['changed_rows'] != 0
            or not math.isfinite(delta) or delta <= 0 or not bool(torch.isfinite(current).all())):
        raise ValueError('actual checkpoint400 embedding drift is invalid')
    del base, current, initial, observer
    optimizer = torch.load(source_checkpoint / 'optimizer.bin', map_location='cpu')
    candidates = [(identifier, state) for identifier, state in optimizer['state'].items()
                  if tuple(state.get('exp_avg', torch.empty(0)).shape) == (token_id + 1, 1024)]
    if len(candidates) != 1:
        raise ValueError('cannot uniquely identify the restored embedding AdamW state')
    identifier, embedding_state = candidates[0]
    groups = [group for group in optimizer['param_groups'] if identifier in group['params']]
    if len(groups) != 1 or groups[0]['weight_decay'] != 0 or float(embedding_state['step']) != 400:
        raise ValueError('embedding optimizer decay or step differs')
    moments = {}
    for name in ('exp_avg', 'exp_avg_sq'):
        value = embedding_state[name]
        if value.shape != (token_id + 1, 1024) or not bool(torch.isfinite(value).all()) or bool(torch.count_nonzero(value[:token_id])):
            raise ValueError('ordinary embedding AdamW moments must be finite and exactly zero')
        moments[name] = {'ordinary_nonzero_elements': 0, 'texture_l2': float(value[token_id].norm())}
    optimizer_evidence = {'embedding_parameter_id': identifier, 'step': 400, 'weight_decay': 0, 'moments': moments}
    del optimizer, candidates, embedding_state, groups, value
    destination = output / 'checkpoints/checkpoint-400'
    destination.mkdir(parents=True)
    copied = {}
    for name in STATE_FILES:
        target = destination / name
        shutil.copyfile(source_checkpoint / name, target)
        copied[str(target)] = sct.sha256(target)
        if copied[str(target)] != locks[str(source_checkpoint / name)]:
            raise ValueError('copied complete training state differs')
    (destination.parent / 'training_metrics.jsonl').write_bytes(prefix)
    derived_audit['derived_prefix_evidence'] = {
        'not_an_original_final400_audit': True,
        'count_derivation': 'All 1000 observed steps had exposure and nonzero gradient; therefore the first400 did too.',
        'embedding_drift': 'Recomputed from actual checkpoint400 text encoder and pinned base initializer.',
        'original_final_audit': str(audit_path), 'original_final_audit_sha256': locks[str(audit_path)]}
    sct.write(destination.parent / 'embedding_update_audit.json', derived_audit)
    for path in (destination.parent / 'training_metrics.jsonl', destination.parent / 'embedding_update_audit.json'):
        copied[str(path)] = sct.sha256(path)
    receipt = {'checkpoint': str(destination), 'source_run': str(run), 'source_step': 400,
        'source_training_steps': 1000, 'source_locks': locks, 'derived_locks': copied,
        'prefix_metrics_bytes': len(prefix), 'derivation': derived_audit['derived_prefix_evidence'],
        'embedding_optimizer': optimizer_evidence,
        'data_cursor': {'batches_per_epoch': 16, 'first_epoch': 25, 'skip_batches': 0},
        'rng_scope': 'Complete accelerator RNG restored; dataloader worker RNG is not checkpointed; equality is measured, not assumed.'}
    sct.write(output / 'derived_receipt.json', receipt)
    sct.st.shared.snapshots.check_locks(locks | copied)
    return receipt
