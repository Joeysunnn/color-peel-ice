"""Recolor only the TRELLIS condition; retain the original photo and SAM mask."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.subject_trellis_counterfactual_v1.run import check_source_receipt
from src.methods.colorpeel_ice.natural_subject_recolor_border_repair import derive_repaired_alpha
from src.methods.colorpeel_ice.natural_subject_recolor_pilot import recolor
from src.methods.colorpeel_ice.single_image_subject import check_locks, fresh, sha256, write_json


def prepare(source_receipt, output):
    source_receipt, output = Path(source_receipt).resolve(), Path(output).resolve()
    parent_sha = sha256(source_receipt)
    original = json.loads(source_receipt.read_text())
    check_source_receipt(source_receipt)
    locks = {**original['source_locks'], str(source_receipt): parent_sha,
             original['image512_path']: original['image512_sha256']}
    check_locks(locks)
    if any(output.is_relative_to(Path(path).parent) for path in locks):
        raise ValueError('pink output must be outside all source folders')
    with Image.open(original['image512_path']) as image, Image.open(original['mask_path']) as mask_image:
        if image.mode != 'RGB' or image.size != (512, 512) or mask_image.mode != 'L':
            raise ValueError('require the original RGB512 photo and binary L-mode mask')
        rgb, mask = np.asarray(image), np.asarray(mask_image)
    if not set(np.unique(mask)) <= {0, 255} or not mask.any():
        raise ValueError('require an unchanged nonempty binary mask')
    # This helper derives alpha only. Never call the historical mask-repair function.
    soft_alpha, alpha_parameters = derive_repaired_alpha(mask > 0)
    pink, metrics, _ = recolor(rgb, mask, soft_alpha, 0.0)
    output = fresh(output)
    cutout = output / 'cutout512.png'
    Image.fromarray(np.dstack([np.where(mask[..., None] > 0, pink, 0), mask])).save(cutout)
    locks[str(cutout)] = sha256(cutout)
    receipt = {**original, 'cutout_path': str(cutout), 'cutout_sha256': locks[str(cutout)],
        'source_locks': locks, 'conditioning_recolor': {
            'scope': 'TRELLIS conditioning only; original photo anchors and C/M/T inputs unchanged',
            'method': 'historical deterministic Lab/LCh hue=0 with soft-alpha RGB blending',
            'hue_degrees': 0.0, 'mask_policy': 'unchanged supplied binary SAM mask; no mask repair',
            'parent_receipt_path': str(source_receipt), 'parent_receipt_sha256': parent_sha,
            'original_cutout_path': original['cutout_path'],
            'original_cutout_sha256': original['cutout_sha256'],
            'soft_alpha_parameters': alpha_parameters, 'recolor_metrics': metrics}}
    check_locks(locks)
    write_json(output / 'source_receipt.json', receipt)
    check_source_receipt(output / 'source_receipt.json')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-receipt', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    prepare(args.source_receipt, args.output)
