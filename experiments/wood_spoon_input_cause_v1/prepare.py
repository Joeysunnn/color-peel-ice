"""One diagnostic input: harden the historical spoon alpha, preserving its RGB."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from experiments.subject_trellis_counterfactual_v1.run import check_source_receipt
from scripts.launch.colorpeel_run import git_info
from src.methods.colorpeel_ice.single_image_subject import check_locks, fresh, sha256, write_json

EXPECTED = {
    'source': 'b17fb667be977d2ed879f3c216011e6dde8324951a6f331136a7c52ab2bc4f9d',
    'cutout': 'c54251c91afc98ba3e9f626a71ac234f1350dc239bc36aedfd846295da168842',
    'mask': '9647fbc0ca9ad0c8155ca17ad1e8987573c5adc061af6dbd5062000b0e230383'}
NATIVE_SHA = 'a3d581d2bcf28f0ad7ef05ac15a74d24a22decac2c05b5d63fb3a06b3b856aa9'


def official_crop_box(alpha):
    """The pinned official preprocessing crop for an already512 RGBA input."""
    yy, xx = np.where(alpha > .8 * 255)
    if not len(xx):
        raise ValueError('historical alpha has no foreground above204')
    cx, cy = (int(xx.min()) + int(xx.max())) / 2, (int(yy.min()) + int(yy.max())) / 2
    size = max(int(xx.max()) - int(xx.min()), int(yy.max()) - int(yy.min()))
    return [cx - size // 2, cy - size // 2, cx + size // 2, cy + size // 2]


def harden_alpha(rgba):
    result = rgba.copy()
    result[..., 3] = np.where(rgba[..., 3] > 204, 255, 0)
    result[result[..., 3] == 0, :3] = 0
    if official_crop_box(result[..., 3]) != official_crop_box(rgba[..., 3]):
        raise ValueError('alpha intervention changed the official crop')
    return result


def prepare(old_receipt, output):
    old_receipt, output = Path(old_receipt).resolve(), Path(output).resolve()
    original = json.loads(old_receipt.read_text())
    if any(original[name + '_sha256'] != expected for name, expected in EXPECTED.items()):
        raise ValueError('require the exact historical wood spoon inputs')
    inputs = check_source_receipt(old_receipt)
    # Historical bundle layout: inputs/wood_spoon/ and wood_spoon/native_asset.pt.
    native = Path(original['cutout_path']).parents[2] / 'wood_spoon/native_asset.pt'
    locks = {**inputs['sha256'], str(native): NATIVE_SHA}
    check_locks(locks)
    if any(output.is_relative_to(Path(path).parent) for path in locks):
        raise ValueError('diagnostic output must be outside all source folders')
    revision = git_info()
    with Image.open(original['cutout_path']) as image:
        rgba = np.array(image)
    hardened = harden_alpha(rgba)
    output = fresh(output)
    cutout, mask = output / 'cutout512.png', output / 'mask512.png'
    reference = output / 'original_alpha512.png'
    Image.fromarray(hardened).save(cutout)
    Image.fromarray(hardened[..., 3]).save(mask)
    shutil.copyfile(original['mask_path'], reference)
    for path in (cutout, mask, reference):
        locks[str(path)] = sha256(path)
    receipt = {**original, 'cutout_path': str(cutout), 'cutout_sha256': locks[str(cutout)],
        'mask_path': str(mask), 'mask_sha256': locks[str(mask)], 'source_locks': locks,
        'soft_alpha': False, 'segmentation': 'diagnostic historical alpha>204; not a SAM mask',
        'diagnostic': {'variant': 'old_hard_alpha', 'git': revision,
            'parent_receipt_path': str(old_receipt), 'parent_receipt_sha256': locks[str(old_receipt)],
            'historical_native_path': str(native), 'historical_native_sha256': NATIVE_SHA,
            'original_alpha_path': str(reference), 'original_alpha_sha256': locks[str(reference)],
            'original_segmentation': original['segmentation'],
            'intervention': 'alpha=(old_alpha>204)*255; retained RGB byte-exact; transparent RGB zero',
            'official_crop_xyxy': official_crop_box(rgba[..., 3]),
            'original_support_above204': int((rgba[..., 3] > 204).sum()),
            'production_input_changed': False, 'training': False, 'quality_accepted_by_user': False}}
    check_locks(locks)
    write_json(output / 'source_receipt.json', receipt)
    check_source_receipt(output / 'source_receipt.json')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-receipt', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    prepare(args.old_receipt, args.output)
