"""Render a balanced material-counterfactual dataset from the saved native asset."""

import argparse
import csv
import hashlib
import html
import itertools
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

from experiments.subject_trellis_counterfactual_v1.convert_dino import sha256
from scripts.launch.colorpeel_run import git_info, resolve_run_dir

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = Path(__file__).with_name('protocol.json')
ASSETS = Path('/var/tmp/colorpeel-r12user5/assets/trellis2_v1')


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf8')


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def material_id(metallic, roughness):
    return f'M{int(metallic)}_R{int(round(100 * roughness)):02d}'


def rows(config):
    result = []
    for view, color, metallic, roughness, light in itertools.product(
            config['views_yaw_offset_radians'], config['colors'], config['metallic'],
            config['roughness'], config['lights']):
        material = material_id(metallic, roughness)
        adjective = {0.15: 'glossy', 0.5: 'moderately rough', 0.85: 'rough'}[roughness]
        label = 'metallic' if metallic else 'non-metallic'
        result.append({'id': f'{view}__{color}__{material}__{light}',
            'kind': 'synthetic_material_counterfactual', 'view': view, 'color': color,
            'light': light, 'material': material, 'pair_group': f'{view}__{color}__{light}',
            'controls': {'metallic': metallic, 'roughness': roughness,
                         'base_color': config['colors'][color], 'alpha': 1.0},
            'caption': f'a photo of <S*> mailbox in {color} color with a {adjective} {label} surface',
            'caption_status': 'parameter-derived proposed label; rendered appearance pending review',
            'training_eligible': False, 'review_status': 'pending_user_review'})
    return result


def camera(vertices, source, yaw_offset, config):
    """Fit all vertices once; image pose and geometry remain fixed for each view."""
    import torch
    from trellis2.utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics
    yaw = source['yaw_radians'] + yaw_offset
    pitch, roll = source['pitch_radians'], source['roll_radians']
    extr, intr = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, 1, config['renderer']['fov_degrees'])
    rotation = extr.new_tensor([[math.cos(roll), -math.sin(roll), 0],
                               [math.sin(roll), math.cos(roll), 0], [0, 0, 1]])
    extr[:3] = rotation @ extr[:3]
    center = (vertices.amin(0) + vertices.amax(0)) / 2
    extr[:3, 3] -= extr[:3, :3] @ center
    intr[0, 2] = intr[1, 2] = 0.5
    transformed = vertices @ extr[:3, :3].T + extr[:3, 3]
    z_offset = transformed[:, 2] - 1
    required = torch.maximum(transformed[:, 0].abs() * intr[0, 0],
                             transformed[:, 1].abs() * intr[1, 1]) / 0.38 - z_offset
    radius = max(float(required.max()), config['renderer']['near'] - float(z_offset.min())) + 0.02
    extr[2, 3] += radius - 1
    final = vertices @ extr[:3, :3].T + extr[:3, 3]
    if float(final[:, 2].min()) <= config['renderer']['near']:
        raise ValueError('geometry crosses the near clipping plane')
    return extr, intr, {'yaw_radians': yaw, 'pitch_radians': pitch, 'roll_radians': roll,
                        'radius': radius, 'geometry_center': center.cpu().tolist(),
                        'minimum_vertex_depth': float(final[:, 2].min()),
                        'extrinsics': extr.cpu().tolist(), 'intrinsics': intr.cpu().tolist()}


def package(output, row, shaded, mask):
    """Use actual rendered pixels; only replace exterior with the common background."""
    raw = np.rint(np.clip(shaded.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
    Image.fromarray(raw).save(output / 'raw' / (row['id'] + '.png'))
    image = np.where(mask[..., None], raw, 127).astype(np.uint8)
    image_path = output / 'dataset/images' / row['id'] / 'image.png'
    mask_path = output / 'dataset/masks' / row['id'] / 'image.png'
    image_path.parent.mkdir(parents=True)
    mask_path.parent.mkdir(parents=True)
    Image.fromarray(image).save(image_path)
    Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
    cutout = np.concatenate([np.where(mask[..., None], raw, 0), mask[..., None].astype(np.uint8) * 255], axis=-1)
    Image.fromarray(cutout).save(output / 'cutouts' / (row['id'] + '.png'))
    return {**row, 'image': image_path.relative_to(output / 'dataset').as_posix(),
        'image_sha256': sha256(image_path), 'mask': mask_path.relative_to(output / 'dataset').as_posix(),
        'mask_sha256': sha256(mask_path), 'raw_image': f"raw/{row['id']}.png",
        'raw_sha256': sha256(output / 'raw' / (row['id'] + '.png')),
        'foreground_rgba': f"cutouts/{row['id']}.png",
        'display_qc': {'foreground_mean_rgb': raw[mask].mean(0).tolist(),
                       'black_channel_fraction': float((raw[mask] == 0).mean()),
                       'clipped_channel_fraction': float((raw[mask] == 255).mean())}}


def review(output, records, config):
    folder = output / 'review'
    folder.mkdir()
    links = []
    for view, light in itertools.product(config['views_yaw_offset_radians'], config['lights']):
        sheet = Image.new('RGB', (1536, 1440), 'white')
        draw = ImageDraw.Draw(sheet)
        for i, color in enumerate(config['colors']):
            for j, (metallic, roughness) in enumerate(itertools.product(config['metallic'], config['roughness'])):
                row = next(r for r in records if (r['view'], r['light'], r['color'], r['material']) ==
                           (view, light, color, material_id(metallic, roughness)))
                with Image.open(output / 'dataset' / row['image']) as image:
                    sheet.paste(image.resize((256, 256), Image.Resampling.LANCZOS), (j * 256, i * 288 + 32))
                draw.text((j * 256 + 5, i * 288 + 5), f'{color} | M={metallic:g} R={roughness:g}\n{view} / {light}', fill='black')
        name = f'{view}__{light}.png'
        sheet.save(folder / name)
        links.append(f'<h2>{html.escape(view)} / {html.escape(light)}</h2><a href="{name}"><img width="1200" src="{name}"></a>')
    with (folder / 'captions.csv').open('w', newline='', encoding='utf8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['id', 'pair_group', 'caption', 'image'])
        for row in records:
            writer.writerow([row[key] for key in ('id', 'pair_group', 'caption', 'image')])
    cards = [f'<figure><img width="256" src="../dataset/{html.escape(r["image"])}"><figcaption>{html.escape(r["caption"])}</figcaption></figure>' for r in records]
    (folder / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Mailbox material counterfactuals</title>'
        '<h1>180 synthetic candidates; pending review</h1><p>Fixed native geometry. Uniform colors and opaque M/R interventions. No training.</p>'
        + ''.join(links) + ''.join(cards), encoding='utf8')


def report(output, state):
    folder = output / 'repro_outputs'
    folder.mkdir(exist_ok=True)
    write(folder / 'status.json', state)
    documents = {
        'SUMMARY.md': f"Status: {state['status']}; images: {state['images_completed']}/180. Synthetic candidates, pending review. No training.\n",
        'COMMANDS.md': shlex.join([sys.executable, '-B', '-m', 'experiments.subject_trellis_dataset_v1.build', '--output', str(output)]) + '\n',
        'LOG.md': 'See protocol.json, cameras.json, runtime.json, dataset/manifest.json, dataset/pairs.json, verification.json and captured sibling log.\n',
        'SCIENTIFIC_CHANGELOG.md': 'Same complete native vertices/faces in official MeshWithPbrMaterial, one texture-free opaque material. Uniform display-domain BC replaces all predicted appearance; exact scalar M/R. Three source-near camera orientations, centered/full-frame once each. Two official HDRIs, full factorial. SSAO seed42 reset each render. One opaque depth layer, official ACES/gamma once.\n',
        'COMPARABILITY_REPORT.md': 'Synthetic data from one generated geometry, not exact photographic counterfactuals. Appearance-only patterns discarded. Same mesh identity across samples does not establish match to real source or successful learned disentanglement. M/R approximated by official IBL renderer, no wood/cloth labels. Not a controlled comparison to MARBLE35: size, geometry, composition, materials and light differ.\n',
        'PATCHES.md': f"New independent experiment at {state['git']['commit']}. Source changes committed locally, pushed, pulled on server. No official source patch, existing-result edits or new model calls.\n"}
    for name, value in documents.items():
        (folder / name).write_text(value, encoding='utf8')


def run(output):
    output = resolve_run_dir(Path(output))
    config, git = json.loads(PROTOCOL.read_text()), git_info()
    official = ASSETS / 'sources/TRELLIS.2'
    if (output.exists() or output.is_relative_to(ROOT) or output.is_relative_to(ASSETS)
            or output.is_relative_to(Path(config['asset']).parent)):
        raise ValueError('require new external output, separate from old artifacts and assets')
    locks = {config['asset']: config['asset_sha256'], config['camera_source']: config['camera_source_sha256'],
             str(PROTOCOL): sha256(PROTOCOL)}
    locks.update({str(official / f'assets/hdri/{name}.exr'): digest for name, digest in config['lights'].items()})

    def check():
        if any(sha256(path) != digest for path, digest in locks.items()):
            raise ValueError('immutable input changed')
        if subprocess.check_output(['git', '-C', str(official), 'rev-parse', 'HEAD'], text=True).strip() != config['official_commit']:
            raise ValueError('official revision mismatch')
        if subprocess.check_output(['git', '-C', str(official), 'status', '--porcelain'], text=True).strip() or git_info() != git:
            raise ValueError('project or official source changed')

    check()  # Verify the trusted own checkpoint before unpickling.
    output.mkdir(parents=True)
    for folder in ('raw', 'cutouts', 'dataset', 'geometry', 'maps'):
        (output / folder).mkdir()
    shutil.copyfile(PROTOCOL, output / 'protocol.json')
    state = {'status': 'initializing', 'git': git, 'input_hashes': locks, 'images_completed': 0,
             'expected_images': 180, 'new_model_samples': 0, 'training_started': False, 'monitoring': False}
    report(output, state)
    try:
        os.environ.update(OPENCV_IO_ENABLE_OPENEXR='1', PYTHONDONTWRITEBYTECODE='1',
            TORCH_EXTENSIONS_DIR=str(ASSETS / 'cache/torch_extensions'), TRITON_CACHE_DIR=str(output / 'triton_cache'))
        sys.path.insert(0, str(official))
        import cv2
        import torch
        from trellis2.representations.mesh import MeshWithPbrMaterial, PbrMaterial
        from trellis2.renderers import EnvMap, PbrMeshRenderer
        saved = torch.load(config['asset'], map_location='cuda', weights_only=False)
        vertices, faces = saved['vertices'], saved['faces']
        geometry_hashes = {name: array_hash(saved[name].cpu().numpy()) for name in ('vertices', 'faces')}
        material = PbrMaterial()
        mesh = MeshWithPbrMaterial(vertices, faces, torch.zeros(len(faces), dtype=torch.long, device='cuda'),
                                  torch.zeros((len(faces), 3, 2), device='cuda'), [material])
        lights = {}
        for name in config['lights']:
            image = cv2.imread(str(official / f'assets/hdri/{name}.exr'), cv2.IMREAD_UNCHANGED)
            if image is None or not np.isfinite(image).all():
                raise ValueError('invalid official HDRI')
            lights[name] = EnvMap(torch.tensor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), device='cuda'))
        # Finish lazy light initialization before resetting the paired SSAO RNG.
        for light in lights.values():
            _ = light._backend
        renderer = PbrMeshRenderer({'resolution': config['resolution'], **{
            key: config['renderer'][key] for key in ('near', 'far', 'ssaa', 'peel_layers')}})
        source_camera = json.loads(Path(config['camera_source']).read_text())['camera']['parameters']
        cameras, records, baseline = {}, [], {}
        planned = rows(config)
        write(output / 'runtime.json', {'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
            'geometry': {'vertices': len(vertices), 'faces': len(faces), 'sha256': geometry_hashes},
            'official_commit': config['official_commit'], 'renderer': config['renderer']})
        started = time.monotonic()
        for view, offset in config['views_yaw_offset_radians'].items():
            extr, intr, cameras[view] = camera(vertices, source_camera, offset, config)
            write(output / 'cameras.json', cameras)
            for color, metallic, roughness in itertools.product(config['colors'], config['metallic'], config['roughness']):
                material.base_color_factor = vertices.new_tensor(config['colors'][color])
                material.metallic_factor, material.roughness_factor = metallic, roughness
                torch.manual_seed(config['seed'])
                with torch.inference_mode():
                    rendered = renderer.render(mesh, extr, intr, lights, use_envmap_bg=False)
                arrays = {key: value.cpu().numpy() for key, value in rendered.items()}
                if any(not np.isfinite(value).all() for value in arrays.values()):
                    raise ValueError('non-finite rendered values')
                mask = arrays['mask'] > .5
                y, x = np.where(mask)
                if not len(x) or x.min() < 20 or y.min() < 20 or x.max() > 491 or y.max() > 491:
                    raise ValueError('empty or clipped framing')
                signatures = {'mask': array_hash(mask), 'normal': array_hash(arrays['normal'])}
                if view not in baseline:
                    baseline[view] = signatures
                    Image.fromarray(mask.astype(np.uint8) * 255).save(output / 'geometry' / (view + '_mask.png'))
                    np.save(output / 'geometry' / (view + '_normal.npy'), arrays['normal'])
                if signatures != baseline[view]:
                    raise ValueError('geometry buffers differ within a view')
                targets = {'base_color': np.array(config['colors'][color])[:, None],
                           'metallic': metallic, 'roughness': roughness, 'alpha': 1}
                for name, target in targets.items():
                    values = arrays[name][:, mask] if name == 'base_color' else arrays[name][mask]
                    if not np.allclose(values, target, atol=1e-6, rtol=0):
                        raise ValueError(f'rendered {name} differs from intervention')
                map_name = f'{view}__{color}__{material_id(metallic, roughness)}.npz'
                np.savez_compressed(output / 'maps' / map_name, **{key: arrays[key] for key in targets})
                for light in config['lights']:
                    row = next(r for r in planned if (r['view'], r['color'], r['material'], r['light']) ==
                               (view, color, material_id(metallic, roughness), light))
                    record = package(output, row, arrays['shaded_' + light], mask)
                    record.update(geometry_buffer_sha256=signatures, maps='maps/' + map_name,
                                  maps_sha256=sha256(output / 'maps' / map_name))
                    records.append(record)
                state.update(status='rendering', images_completed=len(records), seconds=time.monotonic() - started)
                report(output, state)
                print('COUNTERFACTUAL', view, color, material_id(metallic, roughness), len(records), flush=True)
        write(output / 'dataset/manifest.json', {'schema': config['schema'], 'records': records,
            'git': git, 'source_hashes': locks, 'training_started': False,
            'training_eligible_images': 0, 'geometry_sha256': geometry_hashes,
            'pair_group_definition': 'same native geometry, view, color, light; six M/R states',
            'interpretation': config['limitations']})
        concepts = [{'instance_prompt': [r['caption']],
            'instance_data_dir': str(output / 'dataset' / r['image']).rsplit('/', 1)[0],
            'instance_mask_dir': str(output / 'dataset' / r['mask']).rsplit('/', 1)[0]} for r in records]
        write(output / 'dataset/concepts.json', concepts)
        review(output, records, config)
        from experiments.subject_trellis_dataset_v1.verify import verify
        state['verification'] = verify(output)
        check()
        if any(array_hash(getattr(mesh, name).cpu().numpy()) != geometry_hashes[name] for name in geometry_hashes):
            raise ValueError('native geometry changed')
        state.update(status='dataset180_constructed_pending_user_review', seconds=time.monotonic() - started,
                     native_asset_unchanged=True, dataset_manifest_sha256=sha256(output / 'dataset/manifest.json'))
    except BaseException as error:
        state.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report(output, state)
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    print(run(parser.parse_args().output)['status'])
