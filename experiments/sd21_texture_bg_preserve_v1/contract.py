"""Only the authorized LoRA T400-to600 background-preservation comparison."""

import copy
from pathlib import Path
from types import SimpleNamespace

from experiments.sd21_wood_branches_v1 import contract as original

STUDY = 'sd21_texture_bg_preserve_v1'
ARMS = {'lambda0': 0.0, 'lambda1': 1.0}
STEPS, SEEDS = (500, 600), (31, 38)
MODEL_DIR = original.MODEL_DIR


def build_config(arm, resume_checkpoint, branch='texture', method='lora_r64', model_dir=MODEL_DIR, smoke=False):
    if arm not in ARMS or branch != 'texture' or method != 'lora_r64' or smoke or Path(model_dir) != MODEL_DIR:
        raise ValueError('only LoRA Texture400-to600, lambda0/lambda1, pinned SD2.1 is authorized')
    resume_checkpoint = Path(resume_checkpoint)
    if not resume_checkpoint.is_absolute() or resume_checkpoint.name != 'checkpoint-400':
        raise ValueError('require an absolute derived checkpoint-400')
    config = copy.deepcopy(original.build_config(branch, method))
    config['run'].update(study=STUDY, variant=f'wood_texture_lora_r64_resume400_to600_{arm}')
    config['args'].update(max_train_steps=600, checkpoint_steps=list(STEPS),
        resume_from_checkpoint=str(resume_checkpoint), texture_background_preservation_weight=ARMS[arm])
    config['data_lineage'].update(old_weights_reused=True, resume_step=400,
        resume_checkpoint=str(resume_checkpoint), background_preservation_weight=ARMS[arm],
        note='Exact original wood Texture16 assets; copied complete T400 state; only background loss differs between arms.')
    return config


def recipe(arm, resume_checkpoint):
    return SimpleNamespace(BRANCHES=('texture',), METHODS=('lora_r64',), STEPS=STEPS,
        MODEL_DIR=MODEL_DIR, TOKENS={'texture': '<T*>'}, CONCEPT_IDS={'texture': 'wood_spoon_texture'},
        verify_dataset=original.verify_dataset,
        build_config=lambda branch, method, model_dir=MODEL_DIR, smoke=False:
            build_config(arm, resume_checkpoint, branch, method, model_dir, smoke))
