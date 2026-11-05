"""Render one or more unchanged native meshes with one reference and three shared cameras."""

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from experiments.subject_trellis_counterfactual_v1.run import ASSETS, OFFICIAL
from experiments.subject_trellis_dataset_v1.build import array_hash, write
from experiments.wood_spoon_faceting_audit_v1.render import save_image
from scripts.launch.colorpeel_run import git_info, resolve_run_dir
from src.methods.colorpeel_ice.single_image_subject import ROOT, check_locks, sha256

RESOLUTION = 512
POSES = (('yaw0', 0.), ('yaw90', math.pi / 2), ('yaw_minus90', -math.pi / 2))


def run(assets, labels, reference_review, output, gpu=0):
    assets = [Path(path).resolve() for path in assets]
    reference_review, output = Path(reference_review).resolve(), resolve_run_dir(Path(output))
    if (len(assets) < 1 or len(assets) != len(labels) or len(set(labels)) != len(labels)
            or any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', name) for name in labels)):
        raise ValueError('require at least one asset and matching unique simple labels')
    if output.exists() or any(output.is_relative_to(path) or path.is_relative_to(output)
            for path in (ROOT, ASSETS, reference_review.parent, *(path.parent for path in assets))):
        raise ValueError('require a fresh external output separate from inputs and code')
    revision = git_info()
    locks = {str(path): sha256(path) for path in [*assets, reference_review]}
    reference = json.loads(reference_review.read_text())['camera']
    extr, intr = [np.asarray(reference[key], dtype=np.float64) for key in ('extrinsics', 'intrinsics')]
    if (extr.shape != (4, 4) or intr.shape != (3, 3) or not np.isfinite(extr).all()
            or not np.isfinite(intr).all() or not np.allclose(extr[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(extr[:3, :3] @ extr[:3, :3].T, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(extr[:3, :3]), 1, atol=1e-5)
            or min(intr[0, 0], intr[1, 1]) <= 0):
        raise ValueError('invalid reference camera matrices')
    official = ASSETS / 'sources/TRELLIS.2'
    if (subprocess.check_output(['git', '-C', str(official), 'rev-parse', 'HEAD'], text=True).strip() != OFFICIAL
            or subprocess.check_output(['git', '-C', str(official), 'status', '--porcelain'], text=True).strip()):
        raise ValueError('require pinned clean official renderer checkout')
    from experiments.sd21_qwen_initializers_v1.run import require_idle
    if gpu < 0 or os.environ.get('CUDA_VISIBLE_DEVICES') not in (None, str(gpu)):
        raise ValueError('GPU selection disagrees with CUDA visibility')
    require_idle([gpu])
    output.mkdir(parents=True)
    state = {'status': 'running', 'git': revision, 'official_commit': OFFICIAL, 'source_locks': locks,
        'reference_review': str(reference_review), 'physical_gpu': gpu, 'render_seed': 42,
        'renderer': {'class': 'nvdiffrast.RasterizeCudaContext', 'resolution': RESOLUTION,
                     'near': .1, 'far': 100, 'antialias': False, 'learned_opacity_used': False},
        'light': {'camera_space_direction': [.3, -.4, -1.], 'normalized_before_use': True,
                  'lambert_ambient': .22, 'lambert_diffuse': .78},
        'normal_policy': 'Native per-face cross products; display face normals toward camera; no vertex smoothing.',
        'normal_encoding': '-camera_space_normal*.5+.5; official raster orientation without vertical flip',
        'geometry_policy': 'Native vertices/faces unchanged; no recentering, scaling, alignment, fitting or remeshing.',
        'new_model_samples': 0, 'training_started': False, 'quality_accepted_by_user': False,
        'expected_renders': 4 * len(assets), 'completed_renders': 0, 'meshes': [], 'views': [],
        'limitations': ['Shared canonical cameras target the native world origin, not a separately fitted center.',
                       'A source-camera crop is retained and reported; yaw labels are not semantic front/back.']}
    write(output / 'status.json', state)
    try:
        os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),
            TORCH_EXTENSIONS_DIR=str(ASSETS / 'cache/torch_extensions'), TRITON_CACHE_DIR=str(output / 'triton_cache'))
        sys.path.insert(0, str(official))
        import torch
        import torch.nn.functional as F
        import nvdiffrast.torch as dr
        from trellis2.renderers.pbr_mesh_renderer import intrinsics_to_projection
        from trellis2.utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics

        torch.manual_seed(42)
        cameras = [('reference', torch.tensor(extr, dtype=torch.float32, device='cuda'),
                    torch.tensor(intr, dtype=torch.float32, device='cuda'))]
        cameras += [(name, *yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, .4, 2.8, 40)) for name, yaw in POSES]
        state['cameras'] = {name: {'extrinsics': view.cpu().tolist(), 'intrinsics': projection.cpu().tolist()}
                            for name, view, projection in cameras}
        state['canonical_camera_parameters'] = {'yaw_radians': dict(POSES), 'pitch_radians': .4,
            'radius': 2.8, 'fov_degrees': 40, 'principal_point': [.5, .5], 'target_world_xyz': [0., 0., 0.]}
        context = dr.RasterizeCudaContext(device='cuda')
        light = np.array([.3, -.4, -1.], dtype=np.float32)
        light /= np.linalg.norm(light)
        sheets = {kind: Image.new('RGB', (4 * RESOLUTION, len(assets) * (RESOLUTION + 32)), 'white')
                  for kind in ('clay', 'normal', 'mask')}
        try:
            font = ImageFont.truetype('DejaVuSans.ttf', 18)
        except OSError:
            font = ImageFont.load_default()
        for row, (label, asset) in enumerate(zip(labels, assets)):
            with torch.serialization.safe_globals([slice]):
                saved = torch.load(asset, map_location='cuda', weights_only=True)
            vertices, faces = saved['vertices'].contiguous(), saved['faces'].contiguous()
            hashes = {name: array_hash(saved[name].cpu().numpy()) for name in ('vertices', 'faces')}
            if (vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3
                    or not len(vertices) or not len(faces) or not torch.isfinite(vertices).all()
                    or faces.min() < 0 or faces.max() >= len(vertices)):
                raise ValueError('invalid native mesh geometry')
            xyz = vertices.cpu().numpy()
            state['meshes'].append({'label': label, 'asset': str(asset), 'vertices': len(vertices), 'faces': len(faces),
                'geometry_sha256': hashes, 'minimum_xyz': xyz.min(0).tolist(), 'maximum_xyz': xyz.max(0).tolist()})
            homogeneous = torch.cat([vertices, torch.ones_like(vertices[:, :1])], dim=1)
            tri = vertices[faces.long()]
            face_normal = F.normalize(torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1), dim=1)
            normal_indices = torch.arange(len(faces), dtype=torch.int32, device='cuda')[:, None].expand(-1, 3).contiguous()
            folder = output / label
            folder.mkdir()
            for column, (name, view, projection) in enumerate(cameras):
                camera_xyz = (homogeneous @ view.T)[:, :3]
                clip = homogeneous @ (intrinsics_to_projection(projection, .1, 100) @ view).T
                raster, _ = dr.rasterize(context, clip[None], faces, resolution=[RESOLUTION, RESOLUTION])
                mask = (raster[0, ..., 3] > 0).cpu().numpy()
                normal = dr.interpolate(face_normal[None], raster, normal_indices)[0][0] @ view[:3, :3].T
                position = dr.interpolate(camera_xyz[None].contiguous(), raster, faces)[0][0]
                normal *= torch.where((normal * position).sum(-1, keepdim=True) > 0, -1., 1.)
                normal = normal.cpu().numpy()
                if not mask.any() or not np.isfinite(normal[mask]).all():
                    raise ValueError(f'empty or nonfinite fixed-camera render: {label}/{name}')
                paths = {kind: folder / f'{name}_{kind}.png' for kind in sheets}
                save_image(paths['clay'], .22 + .78 * np.maximum(normal @ light, 0), mask)
                save_image(paths['normal'], -normal * .5 + .5, mask)
                Image.fromarray(mask.astype(np.uint8) * 255).save(paths['mask'])
                yy, xx = np.where(mask)
                state['views'].append({'label': label, 'camera': name, 'foreground_pixels': int(mask.sum()),
                    'bbox_xyxy_exclusive': [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1],
                    'touches_border': bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()),
                    'artifacts': {kind: {'path': str(path), 'sha256': sha256(path)} for kind, path in paths.items()}})
                for kind, sheet in sheets.items():
                    x, y = column * RESOLUTION, row * (RESOLUTION + 32)
                    with Image.open(paths[kind]) as image:
                        sheet.paste(image.convert('RGB'), (x, y + 32))
                    ImageDraw.Draw(sheet).text((x + 8, y + 6), f'{label} / {name} / {kind}', fill='black', font=font)
                state['completed_renders'] += 1
                write(output / 'status.json', state)
            if any(array_hash(tensor.cpu().numpy()) != hashes[name]
                   for name, tensor in (('vertices', vertices), ('faces', faces))):
                raise ValueError('native geometry changed during rendering')
            del saved, vertices, faces, homogeneous, tri, face_normal, normal_indices, raster, position
            print('RENDERED', label, flush=True)
        state['contact_sheets'] = {}
        for kind, sheet in sheets.items():
            path = output / f'comparison_{kind}.png'
            sheet.save(path)
            state['contact_sheets'][kind] = {'path': str(path), 'sha256': sha256(path)}
        check_locks(locks)
        if git_info() != revision or state['completed_renders'] != state['expected_renders']:
            raise ValueError('checkout changed or comparison incomplete')
        state.update(status='completed_pending_visual_review', geometry_modified=False,
                     torch=torch.__version__, gpu=torch.cuda.get_device_name())
    except Exception as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write(output / 'status.json', state)
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, nargs='+', required=True)
    parser.add_argument('--labels', nargs='+', required=True)
    parser.add_argument('--reference-review', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    run(args.assets, args.labels, args.reference_review, args.output, args.gpu)
