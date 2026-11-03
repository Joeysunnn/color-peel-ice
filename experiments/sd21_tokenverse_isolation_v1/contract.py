"""One paired SD2.1 S/C/T trial of TokenVerse-inspired concept isolation."""

import copy
import os
from pathlib import Path
from types import SimpleNamespace

from experiments.sd21_mailbox_center11_v1 import contract as subject
from experiments.sd21_color_rendered9_150_v1 import contract as color
from experiments.sd21_wood_branches_v1 import contract as texture
from experiments.sd21_mailbox_sct_v1 import run as sct

STUDY = 'sd21_tokenverse_isolation_v1'
BRANCHES, METHODS = ('subject', 'color', 'texture'), ('lora_r64',)
ARMS = {'control': 0.0, 'isolation': 1.0}
TOKENS = {'subject': '<S*>', 'color': '<C*>', 'texture': '<T*>'}
INITIALIZERS = {'subject': 'mailbox', 'color': 'orange', 'texture': 'texture'}
STEPS = {'subject': (500, 1000), 'color': (100, 150), 'texture': (500, 1000)}
GPUS = {'subject': 0, 'color': 1, 'texture': 2}
MODEL_DIR = texture.MODEL_DIR
RECIPES = {'subject': subject, 'color': color, 'texture': texture}
CONCEPT_IDS = {'subject': 'mailbox', 'color': color.CONCEPT_IDS['color'], 'texture': 'wood_spoon_texture'}


def configure_data():
    os.environ[subject.DATASET_ENV] = str(sct.st.SUBJECT_ROOT / 'dataset')
    os.environ[color.DATASET_ENV] = str(sct.COLOR_ROOT / 'dataset')


def verify_dataset(branch):
    if branch not in BRANCHES:
        raise ValueError('only the three branches in the current SCT composition are included')
    configure_data()
    return RECIPES[branch].verify_dataset(branch)


def build_config(branch, arm, manifest, method='lora_r64', model_dir=MODEL_DIR, smoke=False):
    if branch not in BRANCHES or arm not in ARMS or method != 'lora_r64' or smoke or Path(model_dir) != MODEL_DIR:
        raise ValueError('require the fixed SD2.1 LoRA S/C/T paired experiment')
    manifest = Path(manifest)
    if not manifest.is_absolute():
        raise ValueError('require an absolute reviewed bank manifest')
    configure_data()
    config = copy.deepcopy(RECIPES[branch].build_config(branch, method, model_dir=model_dir))
    config['status'] = 'tokenverse_inspired_isolation_paired'
    config['run'].update(study=STUDY, variant=f'{branch}_{arm}_fresh{STEPS[branch][-1]}')
    config['environment']['CUDA_VISIBLE_DEVICES'] = str(GPUS[branch])
    config['args'].update(max_train_steps=STEPS[branch][-1], checkpoint_steps=list(STEPS[branch]),
                          gradient_checkpointing=True, concept_isolation_weight=ARMS[arm])
    if ARMS[arm]:
        config['args']['concept_isolation_manifest'] = str(manifest)
    if config['args'].get('resume_from_checkpoint'):
        raise ValueError('this experiment trains both arms fresh from the same base')
    config['data_lineage'].update(old_weights_reused=False, concept_isolation_weight=ARMS[arm],
        concept_isolation_manifest=str(manifest),
        note='Original single-image objective plus optional joint-canvas right-half base-teacher consistency; not full TokenVerse.')
    return config


def recipe(branch, arm, manifest):
    return SimpleNamespace(BRANCHES=(branch,), METHODS=METHODS, STEPS=STEPS[branch], MODEL_DIR=MODEL_DIR,
        TOKENS=TOKENS, CONCEPT_IDS=CONCEPT_IDS, verify_dataset=verify_dataset,
        build_config=lambda selected, method, model_dir=MODEL_DIR, smoke=False:
            build_config(selected, arm, manifest, method, model_dir, smoke))
