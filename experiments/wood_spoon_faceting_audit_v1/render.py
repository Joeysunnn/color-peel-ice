"""Compare flat and vertex-normal shading of three unchanged saved spoon meshes."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

from experiments.subject_trellis_counterfactual_v1.run import ASSETS, OFFICIAL
from experiments.subject_trellis_dataset_v1.build import array_hash, write
from scripts.launch.colorpeel_run import git_info, resolve_run_dir
from src.methods.colorpeel_ice.single_image_subject import ROOT, check_locks, sha256

RESOLUTION = 1024
ROI512 = (200, 20, 304, 185)


def summary(values):
    values = np.asarray(values).reshape(-1)
    if not values.size:
        return {'count': 0}
    return {'count': int(values.size), 'min': float(values.min()), 'max': float(values.max()),
            'mean': float(values.mean()), 'p50_p90_p95': np.quantile(values, [.5, .9, .95]).tolist()}


def normal_jumps(normal, mask, roi):
    x0, y0, x1, y1 = roi
    normal, mask = normal[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    lengths = np.linalg.norm(normal, axis=-1)
    mask = mask & (lengths > 1e-6)
    normal = normal / np.maximum(lengths[..., None], 1e-6)
    angles = []
    for a, b, valid in ((normal[1:], normal[:-1], mask[1:] & mask[:-1]),
                         (normal[:, 1:], normal[:, :-1], mask[:, 1:] & mask[:, :-1])):
        angles.append(np.degrees(np.arccos(np.clip((a * b).sum(-1)[valid], -1, 1))))
    angles = np.concatenate(angles)
    return {**summary(angles), 'over_5_degrees': int((angles > 5).sum()),
            'over_15_degrees': int((angles > 15).sum()), 'units': 'degrees between adjacent visible pixels'}


def save_image(path, values, mask):
    if values.ndim == 2:
        values = np.repeat(values[..., None], 3, axis=-1)
    rgb = np.where(mask[..., None], np.rint(np.clip(values, 0, 1) * 255), 127).astype(np.uint8)
    Image.fromarray(rgb).save(path)


def run(assets, reviews, output, gpu=0):
    assets, reviews = [[Path(path).resolve() for path in group] for group in (assets, reviews)]
    if len(assets) != 3 or len(reviews) != 3:
        raise ValueError('require historical, current pink and current original asset/review pairs')
    output = resolve_run_dir(Path(output))
    if output.exists() or any(output.is_relative_to(path) or path.is_relative_to(output)
                             for path in (ROOT, ASSETS, *(path.parent for path in assets + reviews))):
        raise ValueError('require a fresh external directory separate from all inputs')
    revision = git_info()
    locks = {str(path): sha256(path) for path in assets + reviews}
    cameras = [json.loads(path.read_text())['camera'] for path in reviews]
    official = ASSETS / 'sources/TRELLIS.2'
    if (subprocess.check_output(['git', '-C', str(official), 'rev-parse', 'HEAD'], text=True).strip() != OFFICIAL
            or subprocess.check_output(['git', '-C', str(official), 'status', '--porcelain'], text=True).strip()):
        raise ValueError('require pinned clean official renderer source')
    from experiments.sd21_qwen_initializers_v1.run import require_idle
    if gpu < 0 or os.environ.get('CUDA_VISIBLE_DEVICES') not in (None, str(gpu)):
        raise ValueError('GPU selection disagrees with CUDA visibility')
    require_idle([gpu])
    output.mkdir(parents=True)
    roi = tuple(value * 2 for value in ROI512)
    state = {'status': 'running', 'git': revision, 'source_locks': locks, 'official_commit': OFFICIAL,
        'resolution': RESOLUTION, 'source512_bowl_roi_xyxy_exclusive': ROI512, 'render_roi_xyxy_exclusive': roi,
        'roi_policy': 'Fixed diagnostic rectangle, not a segmentation or training mask.',
        'normal_policy': 'Per-face cross products versus area-weighted incident-face vertex normals; vertices/faces unchanged.',
        'shading': 'Identical camera-space Lambert: .22 + .78*max(dot(n,normalize([.3,-.4,-1])),0); no PBR, SSAO or tone mapping.',
        'normal_encoding': '-camera_space_normal*.5+.5; flip both modes using the same visible-face orientation',
        'coordinate_policy': 'Official projection and raster orientation; no vertical image flip.',
        'limitations': ['Each asset uses its own saved camera; cross-asset ROI metrics are not pose-controlled quality scores.',
                       'Vertex-normal interpolation changes shading only and does not repair mesh geometry.'],
        'new_model_samples': 0, 'training_started': False, 'geometry_modified': False,
        'quality_accepted_by_user': False, 'expected_meshes': 3, 'completed_meshes': 0, 'meshes': []}
    write(output / 'status.json', state)
    try:
        os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),
            TORCH_EXTENSIONS_DIR=str(ASSETS / 'cache/torch_extensions'), TRITON_CACHE_DIR=str(output / 'triton_cache'))
        sys.path.insert(0, str(official))
        import torch
        import torch.nn.functional as F
        import nvdiffrast.torch as dr
        from trellis2.renderers.pbr_mesh_renderer import intrinsics_to_projection

        context = dr.RasterizeCudaContext(device='cuda')
        light = np.array([.3, -.4, -1.], dtype=np.float32)
        light /= np.linalg.norm(light)
        for label, asset, camera in zip(('historical', 'current_pink', 'current_original'), assets, cameras):
            folder = output / label
            folder.mkdir()
            with torch.serialization.safe_globals([slice]):
                saved = torch.load(asset, map_location='cuda', weights_only=True)
            vertices, faces = saved['vertices'].contiguous(), saved['faces'].contiguous()
            hashes = {name: array_hash(saved[name].cpu().numpy()) for name in ('vertices', 'faces')}
            if not torch.isfinite(vertices).all():
                raise ValueError('nonfinite native vertices')
            extr, intr = [vertices.new_tensor(camera[name]) for name in ('extrinsics', 'intrinsics')]
            if extr.shape != (4, 4) or intr.shape != (3, 3) or not torch.isfinite(extr).all() or not torch.isfinite(intr).all():
                raise ValueError('invalid saved camera')
            homogeneous = torch.cat([vertices, torch.ones_like(vertices[:, :1])], dim=1)
            camera_xyz = (homogeneous @ extr.T)[:, :3]
            clip = homogeneous @ (intrinsics_to_projection(intr, .1, 100) @ extr).T
            raster, _ = dr.rasterize(context, clip[None], faces, resolution=[RESOLUTION, RESOLUTION])
            triangle_ids = raster[0, ..., 3].long() - 1
            mask = (triangle_ids >= 0).cpu().numpy()
            if not mask.any():
                raise ValueError('empty fixed-camera raster')
            tri = vertices[faces.long()]
            cross = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1)
            face_normal = F.normalize(cross, dim=1)
            vertex_normal = torch.zeros_like(vertices)
            for corner in range(3):
                vertex_normal.index_add_(0, faces[:, corner].long(), cross)
            vertex_normal = F.normalize(vertex_normal, dim=1)
            normal_indices = torch.arange(len(faces), device='cuda', dtype=torch.int32)[:, None].expand(-1, 3).contiguous()
            flat = dr.interpolate(face_normal[None], raster, normal_indices)[0][0] @ extr[:3, :3].T
            smooth = F.normalize(dr.interpolate(vertex_normal[None], raster, faces)[0][0], dim=-1) @ extr[:3, :3].T
            position = dr.interpolate(camera_xyz[None].contiguous(), raster, faces)[0][0]
            sign = torch.where((flat * position).sum(-1, keepdim=True) > 0, -1., 1.)
            flat, smooth = [(normal * sign).cpu().numpy() for normal in (flat, smooth)]
            depth = position[..., 2].cpu().numpy()
            for mode, normal in (('flat', flat), ('vertex', smooth)):
                save_image(folder / (mode + '_normal.png'), -normal * .5 + .5, mask)
                save_image(folder / (mode + '_lambert.png'), .22 + .78 * np.maximum(normal @ light, 0), mask)
            low, high = float(depth[mask].min()), float(depth[mask].max())
            save_image(folder / 'depth.png', (depth - low) / max(high - low, 1e-12), mask)
            Image.fromarray(mask.astype(np.uint8) * 255).save(folder / 'mask.png')
            np.savez_compressed(folder / 'buffers.npz', flat_normal=flat, vertex_normal=smooth, depth=depth, mask=mask)
            x0, y0, x1, y1 = roi
            visible = torch.unique(triangle_ids[y0:y1, x0:x1])
            visible = visible[visible >= 0]
            screen = (clip[:, :2] / clip[:, 3:]) * (RESOLUTION / 2)
            projected = screen[faces[visible].long()]
            ab, ac = projected[:, 1] - projected[:, 0], projected[:, 2] - projected[:, 0]
            areas = (.5 * (ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]).abs()).cpu().numpy()
            state['meshes'].append({'id': label, 'asset': str(asset), 'camera': camera, 'vertices': len(vertices),
                'faces': len(faces), 'geometry_sha256': hashes, 'depth_range': [low, high],
                'roi_foreground_pixels': int(mask[y0:y1, x0:x1].sum()),
                'roi_flat_normal_jumps': normal_jumps(flat, mask, roi),
                'roi_vertex_normal_jumps': normal_jumps(smooth, mask, roi),
                'roi_visible_face_projected_area_pixels2': summary(areas),
                'area_policy': 'Full projected triangle area for each face visible inside the fixed ROI; not ROI-clipped area.',
                'artifact_sha256': {path.name: sha256(path) for path in folder.iterdir() if path.is_file()}})
            if any(array_hash(saved[name].cpu().numpy()) != digest for name, digest in hashes.items()):
                raise ValueError('native geometry changed during diagnostic')
            state['completed_meshes'] += 1
            write(output / 'status.json', state)
            print('RENDERED', label, flush=True)
        check_locks(locks)
        if git_info() != revision:
            raise ValueError('checkout changed during diagnostic')
        state.update(status='completed_pending_visual_review', torch=torch.__version__, gpu=torch.cuda.get_device_name())
    except Exception as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write(output / 'status.json', state)
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, nargs=3, required=True, help='Trusted historical, current pink, current original assets')
    parser.add_argument('--reviews', type=Path, nargs=3, required=True, help='Corresponding saved camera reviews')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    run(args.assets, args.reviews, args.output, args.gpu)
