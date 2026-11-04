"""Bounded same-mesh clay views; no model sampling, fitting or training."""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

from experiments.subject_trellis_counterfactual_v1.run import ASSETS, OFFICIAL, check_source_receipt
from experiments.subject_trellis_dataset_v1.build import array_hash, camera, sha256, write
from scripts.launch.colorpeel_run import git_info, resolve_run_dir
from src.methods.colorpeel_ice.single_image_subject import RENDERER, ROOT, STUDIO_SHA, check_locks


def pose_grid():
    return [(f'yaw{index * 45:+d}deg_pitch{pitch:g}rad',
             {'yaw_radians': index * math.pi / 4, 'pitch_radians': pitch, 'roll_radians': 0.0})
            for pitch in (0.0, 0.4) for index in range(-4, 4)]


def checked_inputs(asset, expected_sha, source_receipt, subject_manifest):
    subject = json.loads(subject_manifest.read_text())
    review_path = asset.parent / 'review/review.json'
    paths = (asset, source_receipt, review_path)
    if (sha256(asset) != expected_sha or Path(subject['native_asset']).resolve() != asset
            or Path(subject['source_receipt']).resolve() != source_receipt
            or any(subject['source_locks'].get(str(path)) != sha256(path) for path in paths)):
        raise ValueError('asset, source receipt or camera review differs from the Subject dataset')
    source = check_source_receipt(source_receipt)
    locks = {**source['sha256'], **{str(path): sha256(path) for path in paths},
             str(subject_manifest): sha256(subject_manifest)}
    return subject, json.loads(review_path.read_text()), source, locks


def run(asset, expected_sha, source_receipt, subject_manifest, output, gpu):
    asset, source_receipt, subject_manifest = [Path(path).resolve()
                                             for path in (asset, source_receipt, subject_manifest)]
    output = resolve_run_dir(Path(output))
    if output.exists() or any(output.is_relative_to(path) or path.is_relative_to(output)
                             for path in (ROOT, ASSETS, asset.parent, source_receipt.parent)):
        raise ValueError('require a fresh external output separate from code and source assets')
    revision = git_info()
    subject, review, source, locks = checked_inputs(asset, expected_sha, source_receipt, subject_manifest)
    official = ASSETS / 'sources/TRELLIS.2'
    if (subprocess.check_output(['git', '-C', str(official), 'rev-parse', 'HEAD'], text=True).strip() != OFFICIAL
            or subprocess.check_output(['git', '-C', str(official), 'status', '--porcelain'], text=True).strip()):
        raise ValueError('require pinned clean official renderer checkout')
    light_path = official / 'assets/hdri/studio.exr'
    locks[str(light_path)] = STUDIO_SHA
    check_locks(locks)
    from experiments.sd21_qwen_initializers_v1.run import require_idle
    if gpu < 0 or os.environ.get('CUDA_VISIBLE_DEVICES') not in (None, str(gpu)):
        raise ValueError('GPU selection disagrees with CUDA visibility')
    require_idle([gpu])
    output.mkdir(parents=True)
    state = {'status': 'running', 'git': revision, 'source_locks': locks, 'physical_gpu': gpu,
             'new_model_samples': 0, 'training_started': False, 'camera_optimization': False,
             'expected_renders': 18, 'completed_renders': 0, 'quality_accepted_by_user': False,
             'geometry_modified': False, 'formal_source_receipt': str(source_receipt), 'views': []}
    write(output / 'status.json', state)
    try:
        os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), OPENCV_IO_ENABLE_OPENEXR='1',
            TORCH_EXTENSIONS_DIR=str(ASSETS / 'cache/torch_extensions'),
            TRITON_CACHE_DIR=str(output / 'triton_cache'))
        sys.path.insert(0, str(official))
        import cv2
        import torch
        from trellis2.representations.mesh import MeshWithPbrMaterial, PbrMaterial
        from trellis2.renderers import EnvMap, PbrMeshRenderer

        saved = torch.load(asset, map_location='cuda', weights_only=False)
        vertices, faces = saved['vertices'], saved['faces']
        geometry = {key: array_hash(saved[key].cpu().numpy()) for key in ('vertices', 'faces')}
        if geometry != subject['geometry_sha256']:
            raise ValueError('native geometry differs from the Subject dataset')
        xyz = vertices.cpu().numpy().astype(np.float64)
        centered = xyz - xyz.mean(0)
        eigenvalues, axes = np.linalg.eigh(centered.T @ centered / len(xyz))
        principal = centered @ axes[:, ::-1]
        state['geometry'] = {'hashes': geometry, 'vertices': len(vertices), 'faces': len(faces),
            'minimum_xyz': xyz.min(0).tolist(), 'maximum_xyz': xyz.max(0).tolist(),
            'extent_xyz': np.ptp(xyz, axis=0).tolist(),
            'pca_variances_descending': eigenvalues[::-1].tolist(),
            'pca_extents_in_variance_order': np.ptp(principal, axis=0).tolist(),
            'note': 'Diagnostic bounds only; thin geometry is not automatically judged incorrect.'}
        material = PbrMaterial()
        material.base_color_factor = vertices.new_tensor([.5, .5, .5])
        material.metallic_factor, material.roughness_factor = 0.0, 0.5
        mesh = MeshWithPbrMaterial(vertices, faces, torch.zeros(len(faces), dtype=torch.long, device='cuda'),
                                  torch.zeros((len(faces), 3, 2), device='cuda'), [material])
        poses = []
        for name, value in (('current_subject_camera', subject['camera']),
                            ('current_source_fit_camera', review['camera'])):
            poses.append((name, vertices.new_tensor(value['extrinsics']),
                          vertices.new_tensor(value['intrinsics']), value))
        for name, parameters in pose_grid():
            extr, intr, record = camera(vertices, parameters, 0, {'renderer': RENDERER})
            poses.append((name, extr, intr, record))
        env = cv2.imread(str(light_path), cv2.IMREAD_UNCHANGED)
        if env is None or not np.isfinite(env).all():
            raise ValueError('invalid studio environment')
        light = EnvMap(torch.tensor(cv2.cvtColor(env, cv2.COLOR_BGR2RGB), device='cuda'))
        _ = light._backend
        renderer = PbrMeshRenderer({'resolution': 512, **{
            key: value for key, value in RENDERER.items() if key != 'fov_degrees'}})
        with Image.open(source['cutout_path']) as opened:
            rgba = np.array(opened)
        source_mask = rgba[..., 3] > 127
        reference = Image.fromarray(np.where(source_mask[..., None], rgba[..., :3], 127).astype(np.uint8))
        reference.save(output / 'source_orange.png')
        state['source_display_sha256'] = sha256(output / 'source_orange.png')
        sheet = Image.new('RGB', (4 * 512, 5 * 542), 'white')
        draw = ImageDraw.Draw(sheet)
        sheet.paste(reference, (0, 30))
        draw.text((8, 8), 'Original source foreground (reference)', fill='black')
        for index, (name, extr, intr, record) in enumerate(poses, start=1):
            torch.manual_seed(42)
            with torch.inference_mode():
                rendered = renderer.render(mesh, extr, intr, {'studio': light}, use_envmap_bg=False)
            mask = rendered['mask'].cpu().numpy() > .5
            rgb = rendered['shaded_studio'].cpu().numpy().transpose(1, 2, 0)
            if not mask.any() or not np.isfinite(rgb).all():
                raise ValueError(f'empty or nonfinite clay view: {name}')
            image = Image.fromarray(np.where(mask[..., None], np.rint(np.clip(rgb, 0, 1) * 255), 127).astype(np.uint8))
            image.save(output / (name + '.png'))
            Image.fromarray(mask.astype(np.uint8) * 255).save(output / (name + '_mask.png'))
            ys, xs = np.nonzero(mask)
            state['views'].append({'id': name, 'camera': record, 'image': name + '.png',
                'image_sha256': sha256(output / (name + '.png')),
                'mask_sha256': sha256(output / (name + '_mask.png')),
                'bbox_xyxy_exclusive': [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                'foreground_pixels': int(mask.sum()),
                'touches_border': bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())})
            x, y = index % 4 * 512, index // 4 * 542
            draw.text((x + 8, y + 8), name, fill='black')
            sheet.paste(image, (x, y + 30))
            state['completed_renders'] += 1
            write(output / 'status.json', state)
            print('RENDERED', name, flush=True)
        sheet.save(output / 'contact_sheet.png')
        if any(array_hash(getattr(mesh, key).cpu().numpy()) != digest for key, digest in geometry.items()):
            raise ValueError('native geometry changed during probe')
        check_locks(locks)
        if git_info() != revision or state['completed_renders'] != state['expected_renders']:
            raise ValueError('inference checkout changed or probe incomplete')
        state.update(status='completed_pending_visual_review', contact_sheet='contact_sheet.png',
                     contact_sheet_sha256=sha256(output / 'contact_sheet.png'))
    except Exception as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write(output / 'status.json', state)
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset', required=True, type=Path)
    parser.add_argument('--asset-sha256', required=True)
    parser.add_argument('--source-receipt', required=True, type=Path)
    parser.add_argument('--subject-manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    run(args.asset, args.asset_sha256, args.source_receipt, args.subject_manifest, args.output, args.gpu)
