"""Compare centered and source-cropped cameras on the unchanged Mailbox mesh."""

import argparse
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

from experiments.subject_trellis_dataset_v1.build import (
    ASSETS, PROTOCOL, ROOT, array_hash, camera, sha256, write,
)
from scripts.launch.colorpeel_run import git_info, resolve_run_dir

SOURCE_SHA = 'e263e3f1dd319426baa6cb8f72ff56495230962eea40659497d1cf49b665c7c1'


def sheet(output, name, cases):
    canvas = Image.new('RGB', (1536, len(cases) * 552), 'white')
    draw = ImageDraw.Draw(canvas)
    source = Image.open(output / 'source/source_rgb.png').convert('RGB')
    for row, case in enumerate(cases):
        for col, (label, image) in enumerate([
            ('Source: actual pink recolored input', source),
            ('Centered: existing dataset camera', Image.open(output / f'{case}/centered.png')),
            ('Source framing: estimated camera', Image.open(output / f'{case}/source_framing.png')),
        ]):
            draw.text((col * 512 + 8, row * 552 + 5), label, fill='black')
            draw.text((col * 512 + 8, row * 552 + 20), case, fill='black')
            canvas.paste(image, (col * 512, row * 552 + 40))
    canvas.save(output / name)


def run(output, stage):
    output = resolve_run_dir(output)
    git = git_info()
    config = json.loads(PROTOCOL.read_text())
    official = ASSETS / 'sources/TRELLIS.2'
    source = Path(config['asset']).with_name('source_cutout512.png')
    environment = official / 'assets/hdri/studio.exr'
    locks = {config['asset']: config['asset_sha256'],
             config['camera_source']: config['camera_source_sha256'],
             str(source): SOURCE_SHA, str(environment): config['lights']['studio']}

    def check_inputs():
        for path, digest in locks.items():
            if sha256(path) != digest:
                raise ValueError(f'input hash changed: {path}')
        if subprocess.check_output(['git', '-C', str(official), 'rev-parse', 'HEAD'], text=True).strip() != config['official_commit']:
            raise ValueError('official revision changed')
        if subprocess.check_output(['git', '-C', str(official), 'status', '--porcelain'], text=True).strip() or git_info() != git:
            raise ValueError('source tree changed')

    if output.is_relative_to(ROOT) or output.is_relative_to(ASSETS):
        raise ValueError('require external experiment directory')
    check_inputs()
    if stage == 'clay':
        output.mkdir(parents=True, exist_ok=False)
        (output / 'source').mkdir()
        shutil.copyfile(source, output / 'source/source_cutout512.png')
        shutil.copyfile(config['camera_source'], output / 'source/estimated_camera_review.json')
        shutil.copyfile(PROTOCOL, output / 'source/dataset_protocol.json')
        write(output / 'experiment.json', {'git': git, 'input_hashes': locks,
            'scope': 'One source-near view, two framings; clay followed by eight appearance pairs.',
            'training_started': False, 'new_model_samples': 0,
            'note': 'Actual TRELLIS input is the pink recolored Mailbox, not the orange photograph. No geometry repair or camera refit.'})
    else:
        receipt = json.loads((output / 'experiment.json').read_text())
        if receipt['git'] != git or receipt['input_hashes'] != locks:
            raise ValueError('experiment provenance mismatch')
        previous = json.loads((output / 'clay_status.json').read_text())
        if previous['status'] != 'completed_pending_visual_review':
            raise ValueError('clay stage must complete before appearance')
        for path, digest in previous['artifact_hashes'].items():
            if sha256(output / path) != digest:
                raise ValueError('clay output changed')
    status_path = output / f'{stage}_status.json'
    if status_path.exists():
        raise ValueError('stage already attempted; preserve it and use a new experiment')
    state = {'status': 'started', 'git': git, 'stage': stage, 'render_count': 0,
             'training_started': False, 'new_model_samples': 0}
    write(status_path, state)
    try:
        os.environ.update(OPENCV_IO_ENABLE_OPENEXR='1',
            TORCH_EXTENSIONS_DIR=str(ASSETS / 'cache/torch_extensions'),
            TRITON_CACHE_DIR=str(output / 'triton_cache'))
        sys.path.insert(0, str(official))
        import cv2
        import torch
        from trellis2.representations.mesh import MeshWithPbrMaterial, PbrMaterial
        from trellis2.renderers import EnvMap, PbrMeshRenderer

        saved = torch.load(config['asset'], map_location='cuda', weights_only=False)
        vertices, faces = saved['vertices'], saved['faces']
        geometry_hashes = {key: array_hash(saved[key].cpu().numpy()) for key in ('vertices', 'faces')}
        material = PbrMaterial()
        mesh = MeshWithPbrMaterial(vertices, faces, torch.zeros(len(faces), dtype=torch.long, device='cuda'),
                                  torch.zeros((len(faces), 3, 2), device='cuda'), [material])
        original = json.loads(Path(config['camera_source']).read_text())['camera']
        centered_extr, centered_intr, centered = camera(vertices, original['parameters'], 0, config)
        cameras = {'centered': (centered_extr, centered_intr),
                   'source_framing': (vertices.new_tensor(original['extrinsics']), vertices.new_tensor(original['intrinsics']))}
        camera_record = {'centered': centered, 'source_framing': original,
                         'note': 'Reuse saved source-camera matrices verbatim. Approximate fit, no geometry completion.'}
        if stage == 'clay':
            write(output / 'cameras.json', camera_record)
        elif json.loads((output / 'cameras.json').read_text()) != camera_record:
            raise ValueError('camera changed between stages')
        env_rgb = cv2.imread(str(environment), cv2.IMREAD_UNCHANGED)
        if env_rgb is None or not np.isfinite(env_rgb).all():
            raise ValueError('invalid HDRI')
        light = EnvMap(torch.tensor(cv2.cvtColor(env_rgb, cv2.COLOR_BGR2RGB), device='cuda'))
        _ = light._backend
        renderer = PbrMeshRenderer({'resolution': 512, **{key: config['renderer'][key]
            for key in ('near', 'far', 'ssaa', 'peel_layers')}})
        rgba = np.asarray(Image.open(source).convert('RGBA'))
        source_mask = rgba[..., 3] > 127
        if stage == 'clay':
            Image.fromarray(np.where(source_mask[..., None], rgba[..., :3], 127).astype(np.uint8)).save(output / 'source/source_rgb.png')
            Image.fromarray(source_mask.astype(np.uint8) * 255).save(output / 'source/mask.png')
        cases = [('clay', [.5, .5, .5], 0., .5)] if stage == 'clay' else [
            (f'{color}_M{metal}_R{int(round(rough * 100)):02d}', config['colors'][color], float(metal), rough)
            for color, metal, rough in itertools.product(('blue', 'green'), (0, 1), (.15, .85))]
        records = []
        baseline = {}
        for case, color, metallic, roughness in cases:
            folder = output / case
            folder.mkdir(exist_ok=False)
            material.base_color_factor = vertices.new_tensor(color)
            material.metallic_factor, material.roughness_factor = metallic, roughness
            for framing, (extr, intr) in cameras.items():
                torch.manual_seed(42)
                with torch.inference_mode():
                    rendered = renderer.render(mesh, extr, intr, {'studio': light}, use_envmap_bg=False)
                arrays = {key: value.cpu().numpy() for key, value in rendered.items()}
                if any(not np.isfinite(value).all() for value in arrays.values()):
                    raise ValueError('nonfinite rendering')
                mask = arrays['mask'] > .5
                if not mask.any():
                    raise ValueError('empty foreground')
                signatures = {key: array_hash(value) for key, value in {'mask': mask, 'normal': arrays['normal']}.items()}
                if framing not in baseline:
                    baseline[framing] = signatures
                if baseline[framing] != signatures:
                    raise ValueError('geometry changed with material')
                if stage == 'appearance':
                    if signatures != previous['geometry_buffers'][framing]:
                        raise ValueError('geometry differs from clay')
                for key, target in [('base_color', np.array(color)[:, None]), ('metallic', metallic), ('roughness', roughness)]:
                    values = arrays[key][:, mask] if key == 'base_color' else arrays[key][mask]
                    if not np.allclose(values, target, atol=1e-6, rtol=0):
                        raise ValueError(f'incorrect material buffer: {key}')
                raw = np.rint(np.clip(arrays['shaded_studio'].transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
                Image.fromarray(raw).save(folder / f'{framing}_raw.png')
                Image.fromarray(np.where(mask[..., None], raw, 127).astype(np.uint8)).save(folder / f'{framing}.png')
                Image.fromarray(mask.astype(np.uint8) * 255).save(folder / f'{framing}_mask.png')
                overlay = np.full((512, 512, 3), 127, dtype=np.uint8)
                overlay[source_mask & ~mask] = [255, 100, 60]
                overlay[mask & ~source_mask] = [40, 190, 255]
                overlay[mask & source_mask] = [240, 240, 240]
                if stage == 'clay':
                    Image.fromarray(overlay).save(folder / f'{framing}_mask_overlay.png')
                ys, xs = np.where(mask)
                records.append({'case': case, 'framing': framing, 'base_color': color,
                    'metallic': metallic, 'roughness': roughness,
                    'bbox_xyxy': [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                    'bottom_border_pixels': int(mask[-1].sum()),
                    'source_mask_iou': float((mask & source_mask).sum() / (mask | source_mask).sum()),
                    'image': f'{case}/{framing}.png', 'image_sha256': sha256(folder / f'{framing}.png')})
                state['render_count'] += 1
                print('RENDERED', case, framing, flush=True)
        sheet(output, f'{stage}_comparison.png', [case[0] for case in cases])
        write(output / f'{stage}_records.json', records)
        (output / 'index.html').write_text(
            '<!doctype html><meta charset="utf-8"><title>Mailbox framing review</title>'
            '<style>body{font-family:sans-serif;margin:24px}img{max-width:100%}</style>'
            '<h1>Mailbox framing review</h1><p>Source / centered / estimated source framing. '
            'Same native geometry; no completion, training or camera refit. Studio light.</p>'
            '<h2>Clay</h2><a href="clay_comparison.png"><img src="clay_comparison.png"></a>'
            '<p>Mask overlays: white=overlap; orange=source only; cyan=render only.</p>'
            '<a href="clay/source_framing_mask_overlay.png">Source-framing mask overlay</a>'
            + ('<h2>Blue/green, M=0/1, R=.15/.85</h2>'
               '<a href="appearance_comparison.png"><img src="appearance_comparison.png"></a>'
               if stage == 'appearance' else ''), encoding='utf8')
        check_inputs()
        if any(array_hash(getattr(mesh, key).cpu().numpy()) != value for key, value in geometry_hashes.items()):
            raise ValueError('native geometry changed')
        state.update(status='completed_pending_visual_review', geometry_sha256=geometry_hashes,
            geometry_buffers=baseline, renderer=config['renderer'], light='studio', seed=42,
            torch=torch.__version__, gpu=torch.cuda.get_device_name(), native_inputs_unchanged=True,
            artifact_hashes={p.relative_to(output).as_posix(): sha256(p)
                for p in output.rglob('*') if p.is_file() and p.suffix in ('.png', '.json') and p != status_path})
    except BaseException as error:
        state.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write(status_path, state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=('clay', 'appearance'), required=True)
    args = parser.parse_args()
    run(args.output, args.stage)
