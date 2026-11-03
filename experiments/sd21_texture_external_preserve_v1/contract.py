"""Paired T400-to600 continuation with ordinary-context preservation images."""

import copy
from pathlib import Path
from types import SimpleNamespace

from experiments.sd21_wood_branches_v1 import contract as original

STUDY = 'sd21_texture_external_preserve_v1'
ARMS = {'control': 0.0, 'preservation': 10.0}
STEPS, SEEDS = (500, 600), (31, 38)
MODEL_DIR = original.MODEL_DIR


def build_config(arm, resume_checkpoint, manifest, branch='texture', method='lora_r64', model_dir=MODEL_DIR, smoke=False):
    if arm not in ARMS or branch != 'texture' or method != 'lora_r64' or smoke or Path(model_dir) != MODEL_DIR:
        raise ValueError('require the fixed paired LoRA Texture preservation protocol')
    resume_checkpoint, manifest = Path(resume_checkpoint), Path(manifest)
    if not resume_checkpoint.is_absolute() or resume_checkpoint.name != 'checkpoint-400' or not manifest.is_absolute():
        raise ValueError('require absolute T400 checkpoint and preservation manifest')
    config = copy.deepcopy(original.build_config(branch, method))
    config['run'].update(study=STUDY, variant=f'wood_texture_lora_r64_resume400_to600_{arm}')
    config['args'].update(max_train_steps=600, checkpoint_steps=list(STEPS),
        resume_from_checkpoint=str(resume_checkpoint), texture_external_preservation_weight=ARMS[arm])
    if ARMS[arm]:
        config['args']['texture_external_preservation_manifest'] = str(manifest)
    config['data_lineage'].update(old_weights_reused=True, resume_step=400,
        resume_checkpoint=str(resume_checkpoint), preservation_manifest=str(manifest),
        external_preservation_weight=ARMS[arm],
        note='Original Texture16 foreground objective; optional ordinary-context full-token adapter teacher consistency.')
    return config


def recipe(arm, resume_checkpoint, manifest):
    return SimpleNamespace(BRANCHES=('texture',), METHODS=('lora_r64',), STEPS=STEPS,
        MODEL_DIR=MODEL_DIR, TOKENS={'texture': '<T*>'}, CONCEPT_IDS={'texture': 'wood_spoon_texture'},
        verify_dataset=original.verify_dataset,
        build_config=lambda branch, method, model_dir=MODEL_DIR, smoke=False:
            build_config(arm, resume_checkpoint, manifest, branch, method, model_dir, smoke))
