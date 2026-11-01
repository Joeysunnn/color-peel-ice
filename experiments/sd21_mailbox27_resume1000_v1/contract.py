"""Read-only600-step parents and unchanged Mailbox27 continuation recipe."""

from pathlib import Path

from experiments.sd21_mailbox_subject27_v1 import contract as parent
from experiments.sd21_mailbox_subject27_v1.contract import (
    BRANCHES, CONCEPT_IDS, INITIALIZERS, METHODS, MODEL_DIR, TOKENS, check_locks,
)
from experiments.sd15_token_local_kv_v1.contract import read_json, sha256
from experiments.subject_trellis_wood27_v1.prepare import verify_output
from scripts.launch.colorpeel_run import read_config

STUDY = 'sd21_mailbox27_resume1000_v1'
STEPS = (800, 1000)
PARENT = Path('/var/tmp/colorpeel-r12user5/runs/sd21_mailbox_subject27_v1/'
    '20261101-221114__both600__a513371__42')
DATASET = PARENT / 'experiment/dataset'
OLD_CHECKOUT = Path('/var/tmp/colorpeel-r12user5/checkouts/sd21-mailbox-subject27-600-v1')
OLD_COMMIT = 'a5133714dbcfa0cefea26aa705b33a5787b5e5bb'
STATE_NAMES = ('pytorch_model.bin', 'pytorch_model_1.bin', 'optimizer.bin',
               'scheduler.bin', 'random_states_0.pkl', 'custom_checkpoint_0.pkl')


def source_run(method):
    if method not in METHODS:
        raise ValueError('require original or rank64')
    return PARENT / 'experiment/methods' / method / 'training' / (
        f'20261101-221228__sd21_mailbox_subject27_v1__mailbox27_{method}_steps600__a513371__42')


def build_config(branch, method, model_dir=MODEL_DIR, smoke=False):
    if branch != 'subject' or smoke or Path(model_dir) != MODEL_DIR:
        raise ValueError('only the pinned SD2.1 Subject continuation is authorized')
    run = source_run(method)
    config = read_config(run / 'config.yaml')
    old = read_json(PARENT / 'experiment/methods' / method / 'config.json')
    manifest = read_json(run / 'manifest.json')
    if (config != old or manifest['status'] != 'succeeded' or manifest['returncode'] != 0
            or manifest['git']['commit'] != OLD_COMMIT or config['args']['max_train_steps'] != 600
            or config['args']['checkpoint_steps'] != [200, 400, 600]
            or config['args']['concepts_list'] != str(DATASET / 'concepts.json')
            or config['args']['pretrained_model_name_or_path'] != str(MODEL_DIR)):
        raise ValueError('parent600 training recipe or successful completion differs')
    config['run'].update(study=STUDY, variant=f'mailbox27_{method}_resume600_to1000')
    config['args'].update(max_train_steps=1000, checkpoint_steps=list(STEPS),
        resume_from_checkpoint=str(run / 'checkpoints/checkpoint-600'))
    config['data_lineage'].update(old_weights_reused=True, resume_parent=str(run), resume_step=600)
    return config


def verify_dataset(branch):
    if branch != 'subject':
        raise ValueError('only Subject is authorized')
    audit_path = DATASET / 'dataset_audit.json'
    audit = read_json(audit_path)
    if (audit['rows'], audit['synthetic_rows'], audit['photo_rows'], audit['subject_id']) != (27, 24, 3, 'mailbox'):
        raise ValueError('parent dataset differs')
    expected = {str(OLD_CHECKOUT / Path(path).relative_to(parent.ROOT))
        if Path(path).is_relative_to(parent.ROOT) else path: digest
        for path, digest in parent.SOURCE_HASHES.items()}
    if any(audit['source_locks'].get(path) != digest for path, digest in expected.items()):
        raise ValueError('parent source provenance differs')
    locks = {**audit['source_locks'], **audit['files'], str(audit_path): sha256(audit_path)}
    for method in METHODS:
        build_config(branch, method)
        run = source_run(method)
        paths = [run / 'config.yaml', run / 'manifest.json',
            PARENT / 'experiment/methods' / method / 'config.json']
        paths += [run / 'checkpoints' / name for name in ('training_metrics.jsonl', 'embedding_update_audit.json')]
        paths += [run / 'checkpoints/checkpoint-600' / name for name in STATE_NAMES]
        locks.update({str(path): sha256(path) for path in paths})
    check_locks(locks)
    references = verify_output(DATASET.parent)
    if any(row['subject_id'] != 'mailbox' for row in references):
        raise ValueError('wrong subject in parent dataset')
    return {'branch': branch, 'dataset': str(DATASET), 'rows': 27, 'references': references,
        'source_locks': locks, 'token': '<S*>', 'initializer': 'mailbox'}
