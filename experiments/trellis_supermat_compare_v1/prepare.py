"""Freeze one original orange Mailbox input for two estimators; never train."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from experiments.subject_trellis_counterfactual_v1.convert_dino import sha256
from scripts.launch.colorpeel_run import git_info, resolve_run_dir

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'experiments/natural_material_supermat_v1/assets/source_mailbox.jpg'
MASK = SOURCE.parent / 'object_mask.png'
ROI = SOURCE.parent / 'material_region_preliminary.png'
LOCKS = {SOURCE: 'ba9c6624d9998cdfcd16308b0cf8fd8f3778909d8655efe2ba9cebba370483d0',
         MASK: '1b226d6b66c95c49ff2dddb734ff2c9efac172cddf6d3231bd12eebbad5d7157',
         ROI: '5d9eedeffc14ff71c3afca6109bf3f3c9ad9eb9f21acfa6554f35ea432df6661'}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf8')


def prepare(output):
    git = git_info()
    output = resolve_run_dir(Path(output))
    if output.exists() or output.is_relative_to(ROOT):
        raise ValueError('require fresh external output')
    if any(sha256(path) != digest for path, digest in LOCKS.items()):
        raise ValueError('frozen orange source, foreground mask or reviewed ROI changed')
    rgb = np.array(Image.open(SOURCE).convert('RGB'))
    mask = np.array(Image.open(MASK).convert('L'))
    roi = np.array(Image.open(ROI).convert('L'))
    if rgb.shape != (512, 512, 3) or mask.shape != (512, 512) or roi.shape != mask.shape:
        raise ValueError('require original512 canvas for image, foreground and ROI')
    if set(np.unique(mask)) != {0, 255} or np.any((roi > 0) & (mask == 0)):
        raise ValueError('require binary full-object alpha containing the reviewed ROI')
    output.mkdir(parents=True)
    inputs = output / 'inputs'
    inputs.mkdir()
    rgba = np.dstack((rgb, mask))
    rgba[mask == 0, :3] = 0
    cutout = inputs / 'mailbox.png'
    Image.fromarray(rgba).save(cutout)
    shutil.copyfile(SOURCE, output / 'original_mailbox.jpg')
    shutil.copyfile(MASK, output / 'object_mask.png')
    shutil.copyfile(ROI, output / 'reviewed_roi.png')
    receipt = {name + suffix: value for name, path in
               (('source', SOURCE), ('mask', MASK), ('cutout', cutout))
               for suffix, value in (('_path', str(path)), ('_sha256', sha256(path)))}
    write(output / 'source_receipt.json', receipt)
    write(output / 'model_revisions.json', {
        'microsoft/TRELLIS.2-4B': 'af44b45f2e35a493886929c6d786e563ec68364d',
        'microsoft/TRELLIS-image-large': '25e0d31ffbebe4b5a97464dd851910efc3002d96'})
    write(output / 'preparation.json', {'git': git, 'training': False, 'seed': 42,
        'source_locks': {str(path): digest for path, digest in LOCKS.items()},
        'shared_input_sha256': sha256(cutout), 'shared_input': str(cutout),
        'foreground_pixels': int((mask > 0).sum()), 'reviewed_roi_pixels': int((roi > 0).sum()),
        'policy': 'same orange original and binary alpha; foreground RGB unchanged; hidden RGB zero',
        'preprocessing': {'supermat': 'official original512 canvas; alpha over gray0.5',
                          'trellis2': 'official alpha crop/resize/black composite; actual_condition.png retained'},
        'roi_provenance': 'same 2815px small_painted_lid mask used by historical Material8; local filename retains preliminary label',
        'comparison': 'estimated-camera visible overlap, not calibrated pixel/surface correspondence'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    prepare(parser.parse_args().output)
