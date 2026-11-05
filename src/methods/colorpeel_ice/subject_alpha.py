"""Use SAM for instance identity and pinned BiRefNet soft alpha for TRELLIS."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

from experiments.subject_trellis_counterfactual_v1.run import check_source_receipt
from experiments.subject_trellis_spoons_v1.prepare import ASSETS, SPEC, square_cutout
from .single_image_subject import ROOT, check_locks, fresh, sha256, write_json

MIN_SAM_COVERAGE = 0.80  # Engineering association guard, not a geometry-quality score.
MIN_OWNER_OVERLAP = 0.01  # Intersection divided by the smaller component/SAM support.


def cached_bundle():
    """Verify the existing pinned files without downloading or replacing anything."""
    spec = json.loads(SPEC.read_text())
    segmentation = spec['segmentation']
    bundle = ASSETS / 'models/birefnet' / segmentation['revision']
    locks = {str(SPEC.resolve()): sha256(SPEC)}
    for name, expected in segmentation['files'].items():
        path = bundle / name
        if path.stat().st_size != expected['size']:
            raise ValueError(f'pinned BiRefNet file size mismatch: {name}')
        digest = sha256(path)
        if 'sha256' in expected:
            matched = digest == expected['sha256']
        else:
            data = path.read_bytes()
            blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            matched = blob == expected['git_blob_sha1']
        if not matched:
            raise ValueError(f'pinned BiRefNet file hash mismatch: {name}')
        locks[str(path.resolve())] = digest
    return bundle, spec, locks


def load_segmenter():
    bundle, spec, locks = cached_bundle()
    official = ASSETS / 'sources/TRELLIS.2'
    if (subprocess.check_output(['git', '-C', str(official), 'rev-parse', 'HEAD'], text=True).strip()
            != spec['official_commit']
            or subprocess.check_output(['git', '-C', str(official), 'status', '--porcelain'], text=True).strip()):
        raise ValueError('require the pinned clean official TRELLIS.2 checkout')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1',
        HF_MODULES_CACHE=str(ASSETS / 'cache/huggingface_modules'))
    sys.path.insert(0, str(official))
    from trellis2.pipelines.rembg import BiRefNet
    segmenter = BiRefNet(str(bundle))
    segmenter.cuda()
    metadata = {'repo': spec['segmentation']['repo'], 'revision': spec['segmentation']['revision'],
                'bundle': str(bundle), 'official_commit': spec['official_commit'],
                'model_files_sha256': locks, 'offline': True}
    return segmenter, metadata, locks


def associate_components(alpha, masks, object_id):
    """Retain complete 8-connected alpha>0 components belonging only to this SAM instance."""
    alpha = np.asarray(alpha)
    if alpha.ndim != 2 or alpha.dtype != np.uint8:
        raise ValueError('require native uint8 soft alpha')
    if object_id not in masks:
        raise ValueError('selected SAM instance is missing')
    masks = {name: np.asarray(mask, dtype=bool) for name, mask in masks.items()}
    if any(mask.shape != alpha.shape or not mask.any() for mask in masks.values()):
        raise ValueError('require nonempty native aligned SAM masks')
    from scipy.ndimage import label
    labels, count = label(alpha > 0, structure=np.ones((3, 3), dtype=np.uint8))
    component_sizes = np.bincount(labels.ravel())
    owners = {component: [] for component in range(1, count + 1)}
    overlaps = {component: [] for component in range(1, count + 1)}
    for name, mask in masks.items():
        sam_size = int(mask.sum())
        components, intersections = np.unique(labels[mask], return_counts=True)
        for component, intersection in zip(components, intersections):
            if component:
                component, intersection = int(component), int(intersection)
                support_size = int(component_sizes[component])
                ratio = intersection / min(support_size, sam_size)
                is_owner = ratio >= MIN_OWNER_OVERLAP
                overlaps[component].append({'object_id': name, 'intersection_pixels': intersection,
                    'component_support_pixels': support_size, 'sam_pixels': sam_size,
                    'overlap_ratio': ratio, 'is_owner': is_owner, 'ignored_contact': not is_owner})
                if is_owner:
                    owners[component].append(name)
    ambiguous = {component: names for component, names in owners.items()
                 if object_id in names and len(names) > 1}
    if ambiguous:
        contacts = {component: overlaps[component] for component in ambiguous}
        raise ValueError(f'BiRefNet components overlap multiple SAM instances for {object_id}: {ambiguous}; '
                         f'minimum ownership overlap {MIN_OWNER_OVERLAP}; contacts: {contacts}')
    selected = [component for component, names in owners.items() if names == [object_id]]
    kept = np.isin(labels, selected)
    coverage = float((kept & masks[object_id]).sum() / masks[object_id].sum())
    if not selected or coverage < MIN_SAM_COVERAGE:
        raise ValueError(f'BiRefNet association for {object_id} has SAM coverage {coverage:.6f}; '
                         f'require at least {MIN_SAM_COVERAGE}')
    result = np.where(kept, alpha, 0).astype(np.uint8)
    return result, {'object_id': object_id, 'connectivity': 8, 'support': 'alpha > 0',
        'component_count': int(count), 'component_owners': owners, 'selected_components': selected,
        'ownership_overlaps': overlaps, 'minimum_ownership_overlap': MIN_OWNER_OVERLAP,
        'ownership_overlap_definition': 'intersection / min(component alpha>0 pixels, SAM pixels)',
        'sam_coverage': coverage, 'minimum_sam_coverage': MIN_SAM_COVERAGE,
        'selected_native_alpha_values_unchanged': bool(np.array_equal(result[kept], alpha[kept])),
        'policy': 'Ignore sub-threshold contacts for ownership only; keep complete uniquely owned components; '
                  'remove other/unassigned components whole; '
                  'no hard-mask clipping, opacity normalization, hole filling or morphology.'}


def prepare(source_receipt, objects_json, object_id, output, segmenter=None):
    source_receipt, objects_json, output = [Path(path).resolve()
                                          for path in (source_receipt, objects_json, output)]
    original = json.loads(source_receipt.read_text())
    check_source_receipt(source_receipt)
    objects = json.loads(objects_json.read_text())
    source = Path(original['source_path']).resolve()
    if (objects['status'] not in ('succeeded', 'partial')
            or Path(objects['source']['canonical_image']).resolve() != source
            or objects['source']['canonical_image_sha256'] != original['source_sha256']):
        raise ValueError('SAM objects and semantic receipt describe different native photographs')
    locks = {**original['source_locks'], str(source_receipt): sha256(source_receipt),
             str(objects_json): sha256(objects_json)}
    with Image.open(source) as opened:
        native = opened.convert('RGB')
    masks = {}
    for row in objects['objects']:
        if row['id'] in masks or Path(row['source_image']).resolve() != source:
            raise ValueError('duplicate SAM id or differing instance source')
        path = Path(row['object_mask']).resolve()
        if sha256(path) != row['object_mask_sha256']:
            raise ValueError('SAM instance mask changed')
        locks[str(path)] = row['object_mask_sha256']
        with Image.open(path) as opened:
            mask = np.asarray(opened)
        if mask.shape != (native.height, native.width) or not set(np.unique(mask)) <= {0, 255}:
            raise ValueError('require native aligned binary SAM masks')
        masks[row['id']] = mask > 0
        if row['id'] == object_id and (path != Path(original['original_mask_path']).resolve()
                or row['object_mask_sha256'] != original['original_mask_sha256']):
            raise ValueError('selected SAM mask differs from semantic receipt')
    if object_id not in masks:
        raise ValueError('selected SAM instance is missing')
    check_locks(locks)
    protected = (ROOT, ASSETS, *(Path(path).parent for path in locks))
    if any(output.is_relative_to(path) for path in protected):
        raise ValueError('alpha output must be outside source, model and code folders')
    output = fresh(output)
    owned = segmenter is None
    metadata = {'backend': 'injected_for_test'}
    if owned:
        segmenter, metadata, model_locks = load_segmenter()
        locks.update(model_locks)
    try:
        segmented = segmenter(native.copy())
    finally:
        if owned:
            segmenter.cpu()
    if segmented.mode != 'RGBA' or segmented.size != native.size:
        raise ValueError('BiRefNet must preserve the full native RGBA canvas')
    rgba = np.asarray(segmented).copy()
    if not np.array_equal(rgba[..., :3], np.asarray(native)):
        raise ValueError('BiRefNet changed native source RGB')
    full_alpha_path = output / 'full_native_alpha.png'
    Image.fromarray(rgba[..., 3]).save(full_alpha_path)
    selected_alpha, association = associate_components(rgba[..., 3], masks, object_id)
    selected_alpha_path = output / 'selected_native_alpha.png'
    Image.fromarray(selected_alpha).save(selected_alpha_path)
    rgba[..., 3] = selected_alpha
    cutout, transform = square_cutout(Image.fromarray(rgba))
    cutout_path, alpha_path = output / 'cutout512.png', output / 'conditioning_alpha512.png'
    cutout.save(cutout_path)
    cutout.getchannel('A').save(alpha_path)
    write_json(output / 'association.json', association)
    for path in (full_alpha_path, selected_alpha_path, cutout_path, alpha_path, output / 'association.json'):
        locks[str(path)] = sha256(path)
    with Image.open(original['mask_path']) as semantic_mask:
        matches = cutout.getchannel('A').tobytes() == semantic_mask.convert('L').tobytes()
    receipt = {**original, 'cutout_path': str(cutout_path), 'cutout_sha256': locks[str(cutout_path)],
        'conditioning_alpha_path': str(alpha_path), 'conditioning_alpha_sha256': locks[str(alpha_path)],
        'semantic_receipt_path': str(source_receipt), 'semantic_receipt_sha256': locks[str(source_receipt)],
        'source_locks': locks, 'alpha_matches_mask': matches, 'transparent_rgb_zero': True,
        'conditioning_alpha': {'method': 'pinned_BiRefNet_full_native_RGBA_then_joint_LANCZOS/v1',
            'segmentation': metadata, 'association': association, 'transform': transform,
            'full_native_alpha_path': str(full_alpha_path), 'selected_native_alpha_path': str(selected_alpha_path),
            'semantic_mask_modified': False, 'opacity_normalized': False, 'quality_accepted_by_user': False}}
    check_locks(locks)
    write_json(output / 'source_receipt.json', receipt)
    check_source_receipt(output / 'source_receipt.json')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-receipt', required=True, type=Path)
    parser.add_argument('--objects-json', required=True, type=Path)
    parser.add_argument('--object-id', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    prepare(args.source_receipt, args.objects_json, args.object_id, args.output)
