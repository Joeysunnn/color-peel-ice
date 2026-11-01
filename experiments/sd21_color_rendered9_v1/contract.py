"""Accepted rendered-background9, unchanged Color recipe, 800 updates."""

from pathlib import Path
import sys

import numpy as np
from PIL import Image

from experiments.sd21_color_gray9_v1 import contract as previous
from experiments.sd15_token_local_kv_v1.contract import read_json, sha256

STUDY = 'sd21_color_rendered9_v1'
BRANCHES, METHODS = previous.BRANCHES, previous.METHODS
TOKENS, INITIALIZERS, MODEL_DIR = previous.TOKENS, previous.INITIALIZERS, previous.MODEL_DIR
DATASET_ENV = previous.DATASET_ENV
CONCEPT_IDS = {'color': 'mailbox_original_orange_rendered9'}
STEPS = (400, 600, 800)
SOURCE_ROOT = Path('/var/tmp/colorpeel-r12user5/runs/color_geometry_lightness_v1/'
                   '20261101-175834__rendered_background9__e44d9cb__42/result9')
SOURCE_INDEX_SHA = 'c712520a3b92c2b1c7f5990f4e595e0db692529a948758cf63af60a4dcdbc700'
SOURCE_NOTE = ('User authorized the nine continuous rendered gray-background images for SD2.1 Color. '
               'Background/shadows are render pixels, not constant RGB128. Existing fixed-a/b bounded-L '
               'foreground mapping is retained. Whole valid-image loss; no supplied object masks. '
               'Historical preview eligibility is unchanged. New weights start from the same SD2.1 base.')


def authorization():
    return {**previous.authorization(), 'dataset': 'rendered_background9', 'max_train_steps': 800,
            'checkpoint_steps': list(STEPS), 'transfer_seeds': list(range(30, 35))}


def source_snapshot():
    index = SOURCE_ROOT / 'files.json'
    if sha256(index) != SOURCE_INDEX_SHA:
        raise ValueError('Frozen rendered9 index changed')
    locks = {str(index.resolve()): SOURCE_INDEX_SHA}
    for name, digest in read_json(index).items():
        path = (SOURCE_ROOT / name).resolve()
        if not path.is_relative_to(SOURCE_ROOT.resolve()) or sha256(path) != digest:
            raise ValueError('Frozen rendered9 file changed')
        locks[str(path)] = digest
    audit = read_json(SOURCE_ROOT / 'audit.json')
    if (audit['count'] != 9 or audit['numeric_pass'] is not True
            or audit['background_changed_pixels_by_mapping'] != 0 or audit['old_masks_identical'] is not True):
        raise ValueError('Rendered background audit differs')
    rows = read_json(SOURCE_ROOT / 'manifest.json')
    expected = [(s, v) for s in ('cube', 'sphere', 'cylinder') for v in (0, 8, 16)]
    if len(rows) != 9 or {(r['shape'], r['view']) for r in rows} != set(expected):
        raise ValueError('Rendered9 coverage differs')
    rows.sort(key=lambda r: expected.index((r['shape'], r['view'])))
    for row in rows:
        if row['caption'] != f"a photo of {row['shape']} shape in <C*> color":
            raise ValueError('Color caption differs')
        image_path, mask_path = [(SOURCE_ROOT / row[key]).resolve() for key in ('image', 'mask')]
        if locks.get(str(image_path)) != row['image_sha256'] or str(mask_path) not in locks:
            raise ValueError('Rendered9 manifest disagrees with index')
        with Image.open(image_path) as image:
            if image.mode != 'RGB' or image.size != (512, 512):
                raise ValueError('Require512px RGB training images')
            rgb = np.asarray(image)
        mask = np.asarray(Image.open(mask_path).convert('L'))
        neutral = np.asarray(Image.open(SOURCE_ROOT / row['id'] / 'neutral.png').convert('RGB'))
        if mask.shape != (512, 512) or set(np.unique(mask)) != {0, 255} or not np.array_equal(rgb[mask == 0], neutral[mask == 0]):
            raise ValueError('Background is not preserved from the neutral render')
    return rows, locks


def stage_dataset(output):
    return previous.stage_dataset(output, recipe=sys.modules[__name__])


def verify_dataset(branch, *, dataset=None):
    return previous.verify_dataset(branch, dataset=dataset, recipe=sys.modules[__name__])


def build_config(branch, method, model_dir=MODEL_DIR, smoke=False, *, dataset=None):
    config = previous.build_config(branch, method, model_dir, smoke, dataset=dataset)
    config['status'] = 'authorized_sd21_color_rendered9'
    config['run'].update(study=STUDY, variant=f'rendered9_color_{method}_{"smoke2" if smoke else "steps800"}')
    config['args'].update(max_train_steps=2 if smoke else 800, checkpoint_steps=[2] if smoke else list(STEPS))
    config['data_lineage'].update(source=str(SOURCE_ROOT), source_note=SOURCE_NOTE)
    return config
