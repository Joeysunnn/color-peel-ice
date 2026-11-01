"""Authorized fresh200-update supplement to the rendered-background9 experiment."""

from pathlib import Path
import sys

from experiments.sd21_color_rendered9_v1 import contract as previous
from experiments.sd21_color_gray9_v1 import contract as staging

STUDY = 'sd21_color_rendered9_200_v1'
BRANCHES, METHODS = previous.BRANCHES, previous.METHODS
TOKENS, INITIALIZERS = previous.TOKENS, previous.INITIALIZERS
MODEL_DIR, DATASET_ENV = previous.MODEL_DIR, previous.DATASET_ENV
CONCEPT_IDS = previous.CONCEPT_IDS
SOURCE_ROOT, SOURCE_INDEX_SHA = previous.SOURCE_ROOT, previous.SOURCE_INDEX_SHA
STEPS = (200,)
MIN_FREE_GIB = 10
PARENT = Path('/home/r12user5/Documents/Jiawei/colorpeel-runs/sd21_color_rendered9_v1/'
              '20261101-183502__rendered9_both800__2fc8981__42/experiment')
SOURCE_NOTE = (previous.SOURCE_NOTE + ' User authorized independent200-update reruns because the '
               'original experiment saved only400/600/800. Do not resume later weights or claim '
               'recovery of the missing historical200 snapshot. Old experiment remains read-only.')
source_snapshot = previous.source_snapshot


def authorization():
    return {**previous.authorization(), 'max_train_steps': 200,
            'checkpoint_steps': list(STEPS), 'transfer_seeds': list(range(30, 46)),
            'parent_experiment': str(PARENT), 'fresh_training_from_base': True}


def stage_dataset(output):
    return staging.stage_dataset(output, recipe=sys.modules[__name__])


def verify_dataset(branch, *, dataset=None):
    return staging.verify_dataset(branch, dataset=dataset, recipe=sys.modules[__name__])


def build_config(branch, method, model_dir=MODEL_DIR, smoke=False, *, dataset=None):
    config = previous.build_config(branch, method, model_dir, smoke, dataset=dataset)
    config['status'] = 'authorized_sd21_color_rendered9_200'
    config['run'].update(study=STUDY, variant=f'rendered9_color_{method}_{"smoke2" if smoke else "steps200"}')
    config['args'].update(max_train_steps=2 if smoke else 200, checkpoint_steps=[2] if smoke else list(STEPS))
    config['data_lineage'].update(source_note=SOURCE_NOTE, parent_experiment=str(PARENT))
    return config
