"""Four shared opaque-clay cameras for five saved meshes; no fitting or sampling."""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

from experiments.subject_trellis_counterfactual_v1.run import ASSETS, OFFICIAL, check_source_receipt
from experiments.subject_trellis_dataset_v1.build import array_hash, write
from scripts.launch.colorpeel_run import git_info, resolve_run_dir
from src.methods.colorpeel_ice.single_image_subject import ROOT, STUDIO_SHA, check_locks, sha256

LABELS = ('pink', 'green', 'cyan', 'blue', 'purple')
HUES = dict(zip(LABELS, (0., 120., 180., 240., 300.)))
POSES = (('yaw_0', 0.), ('yaw_90', math.pi / 2), ('yaw_180', math.pi), ('yaw_minus90', -math.pi / 2))
RENDERER = {'resolution': 512, 'near': .1, 'far': 100, 'ssaa': 1, 'peel_layers': 1}


def checked_runs(runs):
    if len(runs) != 5 or {path.parent.name for path in runs} != set(LABELS):
        raise ValueError('require five runs with parent names pink/green/cyan/blue/purple')
    records, locks, baseline = [], {}, None
    for path in runs:
        status_path, asset = path / 'repro_outputs/status.json', path / 'native_asset.pt'
        state = json.loads(status_path.read_text())
        if state.get('status') != 'generated_pending_user_review' or state.get('generated_assets') != 1:
            raise ValueError('require one completed native asset per color')
        receipt_path = Path(state['source_receipt']['path'])
        source = check_source_receipt(receipt_path)
        receipt = json.loads(receipt_path.read_text())
        if receipt.get('conditioning_recolor', {}).get('hue_degrees') != HUES[path.parent.name]:
            raise ValueError('color label differs from declared conditioning hue')
        if (sha256(receipt_path) != state['source_receipt']['sha256']
                or sha256(path / 'source_cutout512.png') != receipt['cutout_sha256']):
            raise ValueError('generated asset source receipt or cutout changed')
        comparable = {key: state[key] for key in ('git', 'official_commit', 'seed', 'models',
            'encoder_receipt_sha256', 'sampling', 'flow_dtype', 'sparse_algorithm', 'pipeline_type')}
        comparable.update(source_sha256=receipt['source_sha256'], mask_sha256=receipt['mask_sha256'])
        if state['seed'] != 42 or state['official_commit'] != OFFICIAL or (baseline is not None and comparable != baseline):
            raise ValueError('five runs must share seed42, model configuration, original photo and mask')
        baseline = comparable
        locks.update(source['sha256'])
        locks.update({str(item): sha256(item) for item in (asset, status_path,
            path / 'pipeline.json', path / 'source_cutout512.png', path / 'actual_condition.png',
            path / 'review/review.json')})
        records.append({'label': path.parent.name, 'run': str(path), 'asset': str(asset),
                        'generation_git': state['git'], 'source_receipt': str(receipt_path)})
    return records, locks, baseline


def run(runs, output, gpu=0):
    runs = [Path(path).resolve() for path in runs]
    output = resolve_run_dir(Path(output))
    if output.exists() or any(output.is_relative_to(path) or path.is_relative_to(output)
                             for path in (ROOT, ASSETS, *(root.parent for root in runs))):
        raise ValueError('require fresh external output separate from code, assets and input runs')
    revision = git_info()
    records, locks, comparable = checked_runs(runs)
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
        'shared_generation_configuration': comparable, 'runs': records, 'views': [],
        'renderer': RENDERER, 'light': 'studio', 'render_seed': 42, 'background_rgb': [127]*3,
        'clay': {'base_color': [.5]*3, 'metallic': 0., 'roughness': .5, 'alpha': 1.},
        'expected_renders': 20, 'completed_renders': 0, 'new_model_samples': 0, 'training_started': False,
        'camera_optimization': False, 'geometry_modified': False, 'quality_accepted_by_user': False,
        'limitations': ['Native axes and world origin are retained; yaw labels do not identify semantic front/back.',
                       'Opaque clay shows full mesh geometry independently of generated appearance and alpha.',
                       'These fixed views are diagnostics, not geometry-quality acceptance.']}
    write(output / 'status.json', state)
    try:
        os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), OPENCV_IO_ENABLE_OPENEXR='1',
            TORCH_EXTENSIONS_DIR=str(ASSETS / 'cache/torch_extensions'), TRITON_CACHE_DIR=str(output / 'triton_cache'))
        sys.path.insert(0, str(official))
        import cv2
        import torch
        from trellis2.representations.mesh import MeshWithPbrMaterial, PbrMaterial
        from trellis2.renderers import EnvMap, PbrMeshRenderer
        from trellis2.utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics

        cameras = [(name, *yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, .4, 2.8, 40)) for name, yaw in POSES]
        state['cameras'] = {name: {'yaw_radians': yaw, 'pitch_radians': .4, 'roll_radians': 0.,
            'radius': 2.8, 'fov_degrees': 40, 'cx': .5, 'cy': .5,
            'extrinsics': extr.cpu().tolist(), 'intrinsics': intr.cpu().tolist()}
            for (name, yaw), (_, extr, intr) in zip(POSES, cameras)}
        write(output / 'status.json', state)
        environment = cv2.imread(str(light_path), cv2.IMREAD_UNCHANGED)
        if environment is None or not np.isfinite(environment).all():
            raise ValueError('invalid studio environment')
        light = EnvMap(torch.tensor(cv2.cvtColor(environment, cv2.COLOR_BGR2RGB), device='cuda'))
        _ = light._backend
        renderer = PbrMeshRenderer(RENDERER)
        for record in records:
            saved = torch.load(record['asset'], map_location='cuda', weights_only=False)
            vertices, faces = saved['vertices'], saved['faces']
            xyz = vertices.cpu().numpy()
            if not np.isfinite(xyz).all():
                raise ValueError('nonfinite native geometry')
            geometry = {key: array_hash(saved[key].cpu().numpy()) for key in ('vertices', 'faces')}
            record['geometry'] = {'sha256': geometry, 'vertices': len(vertices), 'faces': len(faces),
                'minimum_xyz': xyz.min(0).tolist(), 'maximum_xyz': xyz.max(0).tolist(),
                'extent_xyz': np.ptp(xyz, axis=0).tolist()}
            material = PbrMaterial()
            material.base_color_factor = vertices.new_tensor([.5, .5, .5])
            material.metallic_factor, material.roughness_factor = 0., .5
            mesh = MeshWithPbrMaterial(vertices, faces, torch.zeros(len(faces), dtype=torch.long, device='cuda'),
                                      torch.zeros((len(faces), 3, 2), device='cuda'), [material])
            folder = output / record['label']
            folder.mkdir()
            for name, extr, intr in cameras:
                torch.manual_seed(42)
                with torch.inference_mode():
                    rendered = renderer.render(mesh, extr, intr, {'studio': light}, use_envmap_bg=False)
                mask = rendered['mask'].cpu().numpy() > .5
                rgb = rendered['shaded_studio'].cpu().numpy().transpose(1, 2, 0)
                if (not mask.any() or not np.isfinite(rgb).all() or mask[0].any() or mask[-1].any()
                        or mask[:, 0].any() or mask[:, -1].any()):
                    raise ValueError(f'empty, nonfinite or clipped fixed view: {record["label"]}/{name}')
                image_path, mask_path = folder / (name + '.png'), folder / (name + '_mask.png')
                Image.fromarray(np.where(mask[..., None], np.rint(np.clip(rgb, 0, 1) * 255), 127).astype(np.uint8)).save(image_path)
                Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
                ys, xs = np.nonzero(mask)
                state['views'].append({'label': record['label'], 'camera': name,
                    'image': image_path.relative_to(output).as_posix(), 'image_sha256': sha256(image_path),
                    'mask': mask_path.relative_to(output).as_posix(), 'mask_sha256': sha256(mask_path),
                    'bbox_xyxy_exclusive': [int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1],
                    'foreground_pixels': int(mask.sum()), 'touches_border': False})
                state['completed_renders'] += 1
                write(output / 'status.json', state)
                print('RENDERED', record['label'], name, flush=True)
            if any(array_hash(getattr(mesh, key).cpu().numpy()) != digest for key, digest in geometry.items()):
                raise ValueError('native geometry changed during rendering')
            del saved, mesh, vertices, faces, rendered
        check_locks(locks)
        if git_info() != revision or state['completed_renders'] != 20:
            raise ValueError('checkout changed or fixed-view comparison incomplete')
        state.update(status='completed_pending_visual_review', torch=torch.__version__, gpu=torch.cuda.get_device_name())
    except Exception as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write(output / 'status.json', state)
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs=5, required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    run(args.runs, args.output, args.gpu)
