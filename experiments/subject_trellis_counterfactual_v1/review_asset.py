"""Review one native TRELLIS asset at an estimated original-canvas camera."""

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def review(mesh, source_cutout_path, output, trellis_root):
    """Fit a display camera only; never change the supplied mesh or its attributes."""
    import cv2
    import torch
    from scipy.optimize import minimize
    from trellis2.renderers import EnvMap, MeshRenderer, PbrMeshRenderer
    from trellis2.utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics

    source_cutout_path, output = Path(source_cutout_path), Path(output).resolve()
    if output.exists() or output.is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError('review output must be new and outside the repository')
    with Image.open(source_cutout_path) as image:
        if image.mode != 'RGBA' or image.size != (512, 512):
            raise ValueError('require the original-canvas RGBA512 foreground cutout')
        source = image.copy()
    rgba = np.asarray(source, dtype=np.float32) / 255
    source_mask = rgba[..., 3] > 0.5
    if not source_mask.any():
        raise ValueError('source foreground is empty')
    target = np.asarray(source.getchannel('A').resize((128, 128), Image.Resampling.BILINEAR)) / 255
    target_rgb = np.asarray(source.convert('RGB').resize((128, 128), Image.Resampling.BILINEAR)) / 255
    target_y, target_x = np.indices(target.shape) / 128
    principal = [float((target_x * target).sum() / target.sum()),
                 float(1 - (target_y * target).sum() / target.sum())]
    raster = MeshRenderer({'resolution': 128, 'near': 0.1, 'far': 100,
                           'ssaa': 1, 'antialias': False})
    evaluations = []

    def camera(parameters):
        yaw, pitch, roll, radius, cx, cy = map(float, parameters)
        extr, intr = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, radius, 40)
        rotation = extr.new_tensor([[math.cos(roll), -math.sin(roll), 0],
                                   [math.sin(roll), math.cos(roll), 0], [0, 0, 1]])
        extr[:3] = rotation @ extr[:3]
        intr[0, 2], intr[1, 2] = cx, cy
        return extr, intr

    def objective(parameters):
        with torch.no_grad():
            extr, intr = camera(parameters)
            rendered = raster.render(mesh, extr, intr, return_types=['mask', 'attr'])
            mask = rendered['mask'].detach().cpu().numpy()
            color = rendered['base_color'].detach().cpu().numpy().transpose(1, 2, 0)
        intersection = np.minimum(mask, target).sum()
        union = np.maximum(mask, target).sum()
        overlap = (mask > 0.9) & (target > 0.9)
        mae = float(np.abs(color[overlap] - target_rgb[overlap]).mean()) if overlap.any() else 1.0
        loss = float(1 - intersection / max(float(union), 1) + 0.05 * mae)
        if not np.isfinite(loss):
            raise ValueError('nonfinite camera-fit output')
        evaluations.append((loss, list(map(float, parameters))))
        return loss

    # These are camera candidates for the same generated asset, never new assets.
    for yaw in np.linspace(-np.pi, np.pi, 8, endpoint=False):
        for pitch in (-0.3, 0.2, 0.7):
            for roll in (-.35, 0, .35):
                objective([yaw, pitch, roll, 2.8, *principal])
    initial = min(evaluations)[1]
    fit = minimize(objective, initial, method='Powell',
                   bounds=[(-2 * np.pi, 2 * np.pi), (-0.8, 1.1), (-.8, .8), (1.2, 5.5),
                           (0.05, 0.95), (0.05, 0.95)],
                   options={'maxfev': 350, 'maxiter': 6, 'xtol': 0.005, 'ftol': 0.001})
    loss, parameters = min(evaluations)
    extr, intr = camera(parameters)
    environment = Path(trellis_root) / 'assets/hdri/forest.exr'
    env_rgb = cv2.imread(str(environment), cv2.IMREAD_UNCHANGED)
    if env_rgb is None or not np.isfinite(env_rgb).all():
        raise ValueError('official forest.exr must be readable and finite')
    envmap = EnvMap(torch.tensor(cv2.cvtColor(env_rgb, cv2.COLOR_BGR2RGB),
                                dtype=torch.float32, device=mesh.device))
    renderer = PbrMeshRenderer({'resolution': 512, 'near': 0.1, 'far': 100,
                               'ssaa': 1, 'peel_layers': 8})
    with torch.random.fork_rng(devices=[mesh.device]), torch.no_grad():
        torch.manual_seed(42)  # Official SSAO is stochastic; fix its review-only randomness.
        native = renderer.render(mesh, extr, intr, envmap, use_envmap_bg=False)
        clay_mesh = copy.copy(mesh)
        clay_mesh.attrs = mesh.attrs.clone()
        for name, value in [('base_color', 0.5), ('metallic', 0), ('roughness', 0.5)]:
            clay_mesh.attrs[..., mesh.layout[name]] = value
        torch.manual_seed(42)
        clay = renderer.render(clay_mesh, extr, intr, envmap, use_envmap_bg=False)
    arrays = {key: value.detach().cpu().numpy() for key, value in native.items()}
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError('nonfinite native PBR review output')
    fitted_mask = arrays['mask'] > 0.5
    union = np.logical_or(source_mask, fitted_mask).sum()
    iou = float(np.logical_and(source_mask, fitted_mask).sum() / max(int(union), 1))
    output.mkdir(parents=True)
    panels = {'source_rgb': rgba[..., :3] * rgba[..., 3:] + (127 / 255) * (1 - rgba[..., 3:]),
              'pbr': arrays['shaded'], 'clay': clay['shaded'].detach().cpu().numpy(),
              'base_color': arrays['base_color'], 'metallic': arrays['metallic'],
              'roughness': arrays['roughness'], 'normal': arrays['normal']}
    artifacts = {}
    sheet = Image.new('RGB', (2048, 1080), 'white')
    draw = ImageDraw.Draw(sheet)
    for index, (name, value) in enumerate(panels.items()):
        if name != 'source_rgb':
            value = value.transpose(1, 2, 0) if value.ndim == 3 else np.repeat(value[..., None], 3, axis=-1)
            # Native shaded output already includes alpha compositing, ACES and gamma.
            # Replace geometry-exterior display pixels only; never multiply AA/alpha twice.
            value = np.where(fitted_mask[..., None], value, 127 / 255)
        if not np.isfinite(value).all():
            raise ValueError('nonfinite review panel')
        image = Image.fromarray(np.rint(np.clip(value, 0, 1) * 255).astype(np.uint8))
        path = output / f'{name}.png'
        image.save(path)
        artifacts[name] = str(path)
        x, y = (index % 4) * 512, (index // 4) * 540
        draw.text((x + 10, y + 8), name + (' (estimated view)' if name != 'source_rgb' else ' (original canvas)'), fill='black')
        sheet.paste(image, (x, y + 28))
    Image.fromarray(fitted_mask.astype(np.uint8) * 255).save(output / 'fitted_mask.png')
    artifacts['fitted_mask'] = str(output / 'fitted_mask.png')
    draw.multiline_text((1546, 555), f'Estimated camera; original512 canvas\nMask IoU: {iou:.4f}\nNative PBR attributes; no texture edits\nClay is a separate attribute clone\nNo calibrated-camera or Albedo claim', fill='black', spacing=8)
    sheet.save(output / 'contact_sheet.png')
    artifacts['contact_sheet'] = str(output / 'contact_sheet.png')
    record = {'status': 'pending_user_review', 'camera': {
        'status': 'estimated_from_original512_mask_and_weak_RGB_correspondence',
        'parameters': dict(zip(['yaw_radians', 'pitch_radians', 'roll_radians', 'radius', 'cx', 'cy'], parameters)),
        'fov_degrees': 40, 'extrinsics': extr.detach().cpu().tolist(),
        'intrinsics': intr.detach().cpu().tolist(), 'fit_resolution': 128,
        'output_resolution': 512, 'objective': '1-soft_mask_IoU + 0.05*RGB_BaseColor_MAE',
        'evaluations': len(evaluations), 'loss': loss,
        'refinement_success': bool(fit.success), 'refinement_message': str(fit.message)},
        'metrics': {'original512_mask_iou': iou}, 'artifacts': artifacts,
        'environment': {'path': str(environment), 'sha256': hashlib.sha256(environment.read_bytes()).hexdigest()},
        'rendering': {'ssaa': 1, 'peel_layers': 8, 'near': 0.1, 'far': 100,
                      'use_envmap_bg': False, 'display_exterior_rgb': [127, 127, 127],
                      'shaded_conversion': 'official ACES and gamma only; no additional conversion',
                      'attribute_conversion': 'native renderer values clipped to display range only',
                      'ssao_seed': 42, 'clay': 'cloned attrs: BC=.5 M=0 R=.5; unchanged alpha and geometry'},
        'limitations': ['Camera fit is approximate and may have pose/scale ambiguities.',
                       'Source RGB versus generated Base Color is only a weak alignment cue, not an Albedo metric.',
                       'No geometry, shading, hidden surfaces or source identity accuracy is implied.',
                       'Native transparency is rendered over black; exterior display pixels alone become neutral.',
                       'One fixed generated asset; no material-counterfactual dataset or training.']}
    (output / 'review.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf8')
    return record


if __name__ == '__main__':
    import argparse
    import os
    import sys
    from scripts.launch.colorpeel_run import git_info, resolve_run_dir
    parser = argparse.ArgumentParser(description='Re-render a saved native asset; no model sampling.')
    parser.add_argument('--asset', required=True, type=Path)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    git = git_info()
    output = resolve_run_dir(args.output)
    assets = Path('/var/tmp/colorpeel-r12user5/assets/trellis2_v1')
    os.environ.update(OPENCV_IO_ENABLE_OPENEXR='1',
        TORCH_EXTENSIONS_DIR=str(assets / 'cache/torch_extensions'),
        TRITON_CACHE_DIR=str(output.parent / (output.name + '_triton_cache')))
    sys.path.insert(0, str(assets / 'sources/TRELLIS.2'))
    import torch
    from trellis2.representations import MeshWithVoxel
    saved = torch.load(args.asset, map_location='cuda', weights_only=False)
    saved['origin'] = saved['origin'].tolist()
    mesh = MeshWithVoxel(**saved)
    record = review(mesh, args.source, output, assets / 'sources/TRELLIS.2')
    (output / 'saved_asset_source.json').write_text(json.dumps({'git': git,
        'asset': str(args.asset), 'asset_sha256': hashlib.sha256(args.asset.read_bytes()).hexdigest(),
        'source': str(args.source), 'source_sha256': hashlib.sha256(args.source.read_bytes()).hexdigest(),
        'new_model_samples': 0, 'metrics': record['metrics']}, indent=2) + '\n')
    print(json.dumps(record['metrics']))
