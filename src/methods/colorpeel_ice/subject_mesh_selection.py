"""Compare original/pink native meshes at their saved cameras against one SAM mask.

This is a two-dimensional alignment heuristic, not a geometry-quality acceptance.
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
from PIL import Image

from src.methods.colorpeel_ice.single_image_subject import (
    BORDER_TOLERANCE, MIN_SOURCE_IOU, check_locks, check_source_framing, fresh,
    reviewed_source_camera, sha256,
)

CLOSE_IOU = .01
CLOSE_DISTANCE = .005
CONFIG_KEYS = ('official_commit', 'seed', 'pipeline_type', 'encoder_receipt_sha256',
               'flow_dtype', 'sampling', 'models', 'sparse_algorithm')
SEMANTIC_KEYS = tuple(key for name in ('source', 'original_mask', 'image512', 'mask')
                      for key in (name + '_path', name + '_sha256'))


def geometry_checks(vertices, faces, attrs):
    """Reject broken native arrays, without imposing watertightness or topology rules."""
    vertices, faces, attrs = map(np.asarray, (vertices, faces, attrs))
    result = {'passed': False, 'reason': None, 'vertices': int(len(vertices)) if vertices.ndim else 0,
              'faces': int(len(faces)) if faces.ndim else 0,
              'topology_policy': 'No watertightness/manifold requirement; isolated degenerate faces allowed.'}
    if (vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) < 3
            or vertices.dtype.kind not in 'fiu' or not np.isfinite(vertices).all()):
        result['reason'] = 'vertices require finite numeric Nx3 values'
    elif (faces.ndim != 2 or faces.shape[1:] != (3,) or not len(faces)
            or faces.dtype.kind not in 'iu'):
        result['reason'] = 'faces require a nonempty integer Mx3 array'
    elif faces.min() < 0 or faces.max() >= len(vertices):
        result['reason'] = 'face index outside native vertices'
    elif attrs.dtype.kind not in 'fiu' or not attrs.size or not np.isfinite(attrs).all():
        result['reason'] = 'native attributes must be nonempty and finite'
    else:
        points = vertices.astype(np.float64)
        extent = np.ptp(points, axis=0)
        if not np.isfinite(extent).all() or not np.any(extent > 0):
            result['reason'] = 'native geometry requires finite nonzero extent'
        else:
            triangles = points[faces]
            with np.errstate(over='ignore', invalid='ignore'):
                areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                               triangles[:, 2] - triangles[:, 0]), axis=1) * .5
                total = areas.sum()
            result.update(extent=extent.tolist(), degenerate_faces=int((areas == 0).sum()),
                          positive_area_faces=int((areas > 0).sum()))
            if not np.isfinite(areas).all() or not np.isfinite(total) or total <= 0:
                result['reason'] = 'native geometry requires finite positive total triangle area'
            else:
                result.update(passed=True, reason='native array checks passed', total_triangle_area=float(total))
    return result


def mask_metrics(rendered_mask, sam_mask):
    """Symmetric mean boundary distance: equal directional weights, holes included."""
    rendered, sam = np.asarray(rendered_mask, dtype=bool), np.asarray(sam_mask, dtype=bool)
    if rendered.shape != (512, 512) or sam.shape != (512, 512) or not rendered.any() or not sam.any():
        raise ValueError('metrics require nonempty aligned 512 masks')
    from scipy.ndimage import binary_erosion, distance_transform_edt

    boundary = lambda mask: mask & ~binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)
    rendered_boundary, sam_boundary = boundary(rendered), boundary(sam)
    forward = float(distance_transform_edt(~sam_boundary)[rendered_boundary].mean())
    reverse = float(distance_transform_edt(~rendered_boundary)[sam_boundary].mean())
    y, x = np.where(sam)
    width, height = int(x.max() - x.min() + 1), int(y.max() - y.min() + 1)
    diagonal = float(np.hypot(width, height))
    distance = .5 * (forward + reverse)
    iou = float((rendered & sam).sum() / (rendered | sam).sum())
    return {'mask_iou': iou, 'boundary_distance_pixels': distance,
            'render_to_sam_boundary_pixels': forward, 'sam_to_render_boundary_pixels': reverse,
            'normalized_boundary_distance': distance / diagonal, 'sam_bbox_diagonal_pixels': diagonal,
            'sam_bbox_xyxy_inclusive': [int(x.min()), int(y.min()), int(x.max()), int(y.max())],
            'render_boundary_pixels': int(rendered_boundary.sum()), 'sam_boundary_pixels': int(sam_boundary.sum()),
            'score': iou - distance / diagonal}


def choose(candidates, source_sha256, sam_sha256, object_id, seed=42):
    """Select eligible evidence without mutating it; close candidates use a recorded draw."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be an integer in [0, 2**32)')
    candidates = copy.deepcopy(candidates)
    if len(candidates) != 2 or {row['name'] for row in candidates} != {'original', 'pink'}:
        raise ValueError('require exactly original and pink candidates')
    salt = {'source_sha256': source_sha256, 'native_sam_sha256': sam_sha256,
            'object_id': object_id, 'declared_seed': seed}
    salt_hash = hashlib.sha256(json.dumps(salt, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    derived_seed = int(salt_hash[:16], 16)
    eligible = []
    for row in candidates:
        if row['eligible']:
            metrics = row['metrics']
            iou, distance = metrics['mask_iou'], metrics['normalized_boundary_distance']
            if not np.isfinite([iou, distance]).all() or not MIN_SOURCE_IOU <= iou <= 1 or distance < 0:
                raise ValueError('eligible candidate has invalid metrics')
            row['score'] = float(iou - distance)
            eligible.append(row)
    selected, close, differences = None, False, None
    reason = 'No candidate passed provenance, native geometry, saved review, and actual framing guards.'
    if len(eligible) == 1:
        selected, reason = eligible[0]['name'], 'Only one candidate passed all guards.'
    elif len(eligible) == 2:
        differences = {key: abs(eligible[0]['metrics'][key] - eligible[1]['metrics'][key])
                       for key in ('mask_iou', 'normalized_boundary_distance')}
        close = (differences['mask_iou'] <= CLOSE_IOU + 1e-12
                 and differences['normalized_boundary_distance'] <= CLOSE_DISTANCE + 1e-12)
        if close:
            selected = random.Random(derived_seed).choice(sorted(row['name'] for row in eligible))
            reason = 'Both metrics are close; selected by the recorded deterministic random draw.'
        else:
            selected = sorted(eligible, key=lambda row: (-row['score'], row['name']))[0]['name']
            reason = 'Selected highest IoU minus normalized boundary distance (name breaks exact score ties).'
    return {'status': 'selected' if selected is not None else 'blocked', 'selected': selected,
            'reason': reason, 'candidates': candidates, 'close': close, 'metric_differences': differences,
            'random': {'salt': salt, 'salt_sha256': salt_hash, 'derived_seed': derived_seed,
                       'draw_order': ['original', 'pink'], 'draw_used': close},
            'thresholds': {'minimum_source_iou': MIN_SOURCE_IOU, 'close_iou': CLOSE_IOU,
                           'close_normalized_boundary_distance': CLOSE_DISTANCE,
                           'comparison_roundoff_tolerance': 1e-12, 'border_tolerance_pixels': BORDER_TOLERANCE},
            'score_definition': 'SAM mask IoU minus symmetric mean boundary distance / SAM bbox diagonal',
            'boundary_definition': '8-neighbor erosion with zero padding; all components and holes; each direction weight 0.5',
            'quality_accepted_by_user': False,
            'limitations': ['Two-dimensional mesh plus saved-camera alignment heuristic only.',
                           'Does not assess faceting, depth, backside, or geometric fidelity.',
                           'Saved cameras retain their original weak-RGB fitting differences; no camera search.']}


def _read_candidate(row, semantic_path, semantic, locks, seed):
    from experiments.subject_trellis_counterfactual_v1.run import OFFICIAL, check_source_receipt
    run = Path(row['trellis_run'])
    state_path, review_path, asset = run / 'repro_outputs/status.json', run / 'review/review.json', run / 'native_asset.pt'
    # Lock all existing evidence even when generation stopped before completion.
    for path in (state_path, review_path, asset, run / 'source_cutout512.png', run / 'pipeline.json'):
        if path.is_file():
            locks[str(path)] = sha256(path)
    state = json.loads(state_path.read_text())
    row['generation_status'] = state.get('status')
    provenance = state.get('source_receipt', {})
    row['source_receipt'] = provenance.get('path')
    if state.get('status') != 'generated_pending_user_review' or state.get('generated_assets') != 1:
        raise ValueError('candidate generation did not complete')
    receipt_path = Path(provenance['path'])
    if receipt_path.is_file():
        locks[str(receipt_path)] = sha256(receipt_path)
    if not receipt_path.is_absolute() or sha256(receipt_path) != provenance['sha256']:
        raise ValueError('candidate source receipt path/hash mismatch')
    copied = Path(provenance['copied_path'])
    if not copied.is_absolute() or sha256(copied) != provenance['sha256']:
        raise ValueError('copied generation source receipt hash mismatch')
    locks[str(copied)] = sha256(copied)
    receipt = json.loads(receipt_path.read_text())
    locks.update(check_source_receipt(receipt_path)['sha256'])
    if any(receipt.get(key) != semantic[key] for key in SEMANTIC_KEYS):
        raise ValueError('candidate does not preserve the same semantic original photo and SAM mask')
    if (receipt.get('semantic_receipt_path') != str(semantic_path)
            or receipt.get('semantic_receipt_sha256') != sha256(semantic_path)):
        raise ValueError('candidate semantic receipt binding mismatch')
    if not receipt.get('conditioning_alpha_path') or not receipt.get('conditioning_alpha_sha256'):
        raise ValueError('candidate requires independent BiRef soft alpha')
    recolor = receipt.get('conditioning_recolor')
    if (row['name'] == 'original' and recolor is not None
            or row['name'] == 'pink' and (not recolor or recolor.get('hue_degrees') != 0)):
        raise ValueError('candidate original/pink conditioning policy mismatch')
    if sha256(run / 'source_cutout512.png') != receipt['cutout_sha256']:
        raise ValueError('generated source cutout differs from conditioning receipt')
    review = json.loads(review_path.read_text())
    if state.get('review') != review:
        raise ValueError('saved review differs from generation status')
    config = {key: state[key] for key in CONFIG_KEYS}
    pipeline_args = json.loads((run / 'pipeline.json').read_text())['args']
    config['normalization'] = {key: pipeline_args[key] for key in
                               ('shape_slat_normalization', 'tex_slat_normalization')}
    commit = state['git']['commit']
    if not isinstance(commit, str) or len(commit) != 40 or any(char not in '0123456789abcdef' for char in commit):
        raise ValueError('require full generation Git commit')
    config['generation_git_commit'] = commit
    if config['seed'] != seed or config['official_commit'] != OFFICIAL:
        raise ValueError('candidate generation seed or official commit differs from the declared protocol')
    row['generation_config'] = config
    row['conditioning_alpha_sha256'] = receipt['conditioning_alpha_sha256']
    row['native_asset'] = str(asset)
    row['saved_review'] = str(review_path)
    return review


def _load_native(asset):
    import torch
    return torch.load(asset, map_location='cpu', weights_only=False)


def _as_numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def _render_opaque(saved, camera, output):
    from experiments.subject_trellis_counterfactual_v1.run import ASSETS, OFFICIAL
    official = ASSETS / 'sources/TRELLIS.2'
    if (subprocess.check_output(['git', '-C', str(official), 'rev-parse', 'HEAD'], text=True).strip() != OFFICIAL
            or subprocess.check_output(['git', '-C', str(official), 'status', '--porcelain'], text=True).strip()):
        raise ValueError('require pinned clean TRELLIS renderer checkout')
    os.environ.update(TORCH_EXTENSIONS_DIR=str(ASSETS / 'cache/torch_extensions'),
                      TRITON_CACHE_DIR=str(output / 'triton_cache'))
    sys.path.insert(0, str(official))
    import torch
    from trellis2.representations.mesh import MeshWithPbrMaterial, PbrMaterial
    from trellis2.renderers import MeshRenderer

    vertices, faces = saved['vertices'].cuda(), saved['faces'].cuda()
    mesh = MeshWithPbrMaterial(vertices, faces, torch.zeros(len(faces), dtype=torch.long, device='cuda'),
                              torch.zeros((len(faces), 3, 2), device='cuda'), [PbrMaterial()])
    extr, intr = [vertices.new_tensor(camera[key]) for key in ('extrinsics', 'intrinsics')]
    settings = {'resolution': 512, 'near': .1, 'far': 100, 'ssaa': 1, 'antialias': False}
    with torch.inference_mode():
        mask = MeshRenderer(settings).render(mesh, extr, intr, return_types=['mask'])['mask'].cpu().numpy()
    if mask.shape != (512, 512) or not np.isfinite(mask).all():
        raise ValueError('opaque renderer returned invalid mask')
    return mask > .5, {'class': 'MeshRenderer', 'official_commit': OFFICIAL, **settings,
                       'return_types': ['mask'], 'learned_opacity_used': False,
                       'camera': 'unchanged saved source camera'}


def select_mesh(semantic_receipt, candidates, object_id, output, seed=42):
    """Write a reviewable selection, including blocked and failed-generation evidence."""
    from experiments.subject_trellis_counterfactual_v1.run import check_source_receipt
    if set(candidates) != {'original', 'pink'}:
        raise ValueError('require original and pink candidate run paths')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be an integer in [0, 2**32)')
    semantic_path = Path(semantic_receipt).resolve()
    semantic = json.loads(semantic_path.read_text())
    locks = check_source_receipt(semantic_path)['sha256']
    for name in ('source', 'original_mask', 'image512', 'mask'):
        path = semantic[name + '_path']
        if sha256(path) != semantic[name + '_sha256']:
            raise ValueError('semantic original input changed')
        locks[path] = semantic[name + '_sha256']
    with Image.open(semantic['mask_path']) as image:
        sam = np.asarray(image)
        if image.mode != 'L' or image.size != (512, 512) or not set(np.unique(sam)) <= {0, 255} or not sam.any():
            raise ValueError('semantic reference must be a nonempty binary SAM512 mask')
    output = fresh(output)
    rows, ready = [], []
    for name in ('original', 'pink'):
        row = {'name': name, 'trellis_run': str(Path(candidates[name]).resolve()),
               'source_receipt': None, 'eligible': False, 'reason': None, 'metrics': None}
        rows.append(row)
        try:
            review = _read_candidate(row, semantic_path, semantic, locks, seed)
            ready.append((row, review))
        except (KeyError, ValueError, OSError, TypeError) as exc:
            row['reason'] = f'{type(exc).__name__}: {exc}'
    comparable = (len(ready) < 2 or
                  (ready[0][0]['generation_config'] == ready[1][0]['generation_config']
                   and ready[0][0]['conditioning_alpha_sha256'] == ready[1][0]['conditioning_alpha_sha256']))
    for row, review in ready:
        if not comparable:
            row['reason'] = 'Completed candidates differ in generation configuration or conditioning alpha.'
            continue
        try:
            saved = _load_native(row['native_asset'])
            row['geometry'] = geometry_checks(*(_as_numpy(saved[key]) for key in ('vertices', 'faces', 'attrs')))
            if not row['geometry']['passed']:
                raise ValueError(row['geometry']['reason'])
            camera, row['saved_review_guard'] = reviewed_source_camera(review)
            opaque, row['renderer'] = _render_opaque(saved, camera, output)
            mask_path = output / (row['name'] + '_opaque_mask.png')
            Image.fromarray(opaque.astype(np.uint8) * 255).save(mask_path)
            locks[str(mask_path)] = sha256(mask_path)
            row.update(opaque_mask_path=str(mask_path), opaque_mask_sha256=locks[str(mask_path)])
            row['metrics'] = mask_metrics(opaque, sam)
            row['actual_framing_guard'] = check_source_framing(opaque, sam)
            row.update(eligible=True, reason='All engineering guards passed; not user quality acceptance.')
        except Exception as exc:
            # Preserve a broken candidate and still assess its independently generated sibling.
            row['reason'] = f'{type(exc).__name__}: {exc}'
    check_locks(locks)
    result = choose(rows, semantic['source_sha256'], semantic['original_mask_sha256'], object_id, seed)
    result.update(semantic_receipt=str(semantic_path), source_locks=locks,
                  completed_candidates_comparable=comparable)
    (output / 'selection.json').write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--semantic-receipt', type=Path, required=True)
    parser.add_argument('--candidate', action='append', required=True, metavar='NAME=TRELLIS_RUN')
    parser.add_argument('--object-id', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    pairs = [value.split('=', 1) for value in args.candidate]
    if len(pairs) != 2 or any(len(pair) != 2 or not pair[1] for pair in pairs):
        parser.error('supply exactly --candidate original=PATH and --candidate pink=PATH')
    select_mesh(args.semantic_receipt, dict(pairs), args.object_id, args.output, args.seed)


if __name__ == '__main__':
    main()
