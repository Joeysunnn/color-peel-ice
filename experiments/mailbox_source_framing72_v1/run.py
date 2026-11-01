"""Preview three cropped views, then build 72 Mailbox counterfactuals."""

import argparse
import html
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

from experiments.mailbox_framing_preview_v1.run import SOURCE_SHA
from experiments.subject_trellis_dataset_v1.build import (
    ASSETS, PROTOCOL, ROOT, array_hash, package, rows, sha256, write,
)
from experiments.subject_trellis_dataset_v1.verify import one_factor_pairs, verify_record, SPLIT_POLICY
from scripts.launch.colorpeel_run import git_info, resolve_run_dir


def source_camera(source, offset, vertices):
    """Keep source distance, principal point, pitch and roll; vary only yaw."""
    from trellis2.utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics
    p = source['parameters']
    if offset == 0:
        return vertices.new_tensor(source['extrinsics']), vertices.new_tensor(source['intrinsics'])
    extr, intr = yaw_pitch_r_fov_to_extrinsics_intrinsics(
        p['yaw_radians'] + offset, p['pitch_radians'], p['radius'], source['fov_degrees'])
    roll = p['roll_radians']
    rotation = extr.new_tensor([[math.cos(roll), -math.sin(roll), 0],
                               [math.sin(roll), math.cos(roll), 0], [0, 0, 1]])
    extr[:3] = rotation @ extr[:3]
    intr[0, 2], intr[1, 2] = p['cx'], p['cy']
    return extr, intr


def review(output, records, config):
    links = []
    for view, light in itertools.product(config['views_yaw_offset_radians'], config['lights']):
        canvas = Image.new('RGB', (1536, 576), 'white')
        draw = ImageDraw.Draw(canvas)
        for i, color in enumerate(config['colors']):
            for j, (metal, rough) in enumerate(itertools.product(config['metallic'], config['roughness'])):
                record = next(r for r in records if (r['view'], r['light'], r['color'],
                    r['controls']['metallic'], r['controls']['roughness']) == (view, light, color, metal, rough))
                image = Image.open(output / 'dataset' / record['image'])
                canvas.paste(image.resize((256, 256), Image.Resampling.LANCZOS), (j * 256, i * 288 + 32))
                draw.text((j * 256 + 5, i * 288 + 4), f'{color} M={metal:g} R={rough:g}\n{view} / {light}', fill='black')
        name = f'{view}__{light}.png'
        canvas.save(output / 'review' / name)
        links.append(f'<h2>{view} / {light}</h2><a href="{name}"><img src="{name}"></a>')
    cards = ''.join(f'<figure><a href="../dataset/{r["image"]}"><img width="256" '
        f'src="../dataset/{r["image"]}"></a><figcaption>{html.escape(r["id"])}<br>'
        f'{html.escape(r["caption"])}</figcaption></figure>' for r in records)
    (output / 'review/index.html').write_text('<!doctype html><meta charset="utf-8">'
        '<title>Mailbox source-framing72</title><style>body{font-family:sans-serif;margin:24px}'
        'img{max-width:100%}figure{display:inline-block;width:256px;vertical-align:top}</style>'
        '<h1>Mailbox: 72 source-framed counterfactuals</h1>'
        '<p>Same native geometry, cropped source composition; no geometry completion or training.</p>'
        '<h2>Source / left / center / right clay</h2><img src="../preview/clay_three_views.png">'
        + ''.join(links) + '<h2>Actual RGB candidates and captions</h2>' + cards, encoding='utf8')


def run(output, stage, preview_reviewed=False):
    output = resolve_run_dir(output)
    git = git_info()
    config = json.loads(PROTOCOL.read_text())
    config.update(schema='mailbox_source_framing72/v1', colors={c: config['colors'][c] for c in ('blue', 'green')},
        framing='Saved source camera: fixed radius/cx/cy/pitch/roll, yaw offsets -.25/0/+.25. No centering or fit-to-frame.',
        scope='Three clay previews followed by 72 candidates; no new mesh sampling or training.')
    official = ASSETS / 'sources/TRELLIS.2'
    source = Path(config['asset']).with_name('source_cutout512.png')
    locks = {config['asset']: config['asset_sha256'], config['camera_source']: config['camera_source_sha256'], str(source): SOURCE_SHA}
    locks.update({str(official / f'assets/hdri/{name}.exr'): digest for name, digest in config['lights'].items()})

    def check_inputs():
        if any(sha256(path) != digest for path, digest in locks.items()):
            raise ValueError('immutable input hash changed')
        if subprocess.check_output(['git', '-C', str(official), 'rev-parse', 'HEAD'], text=True).strip() != config['official_commit']:
            raise ValueError('official revision changed')
        if subprocess.check_output(['git', '-C', str(official), 'status', '--porcelain'], text=True).strip() or git_info() != git:
            raise ValueError('source checkout changed')

    check_inputs()
    if output.is_relative_to(ROOT) or output.is_relative_to(ASSETS):
        raise ValueError('output must be external')
    if stage == 'views':
        output.mkdir(parents=True, exist_ok=False)
        for folder in ('preview', 'source', 'geometry'):
            (output / folder).mkdir()
        shutil.copyfile(source, output / 'source/source_cutout512.png')
        write(output / 'protocol.json', config)
        write(output / 'experiment.json', {'git': git, 'input_hashes': locks, 'training_started': False, 'new_model_samples': 0})
    else:
        if not preview_reviewed:
            raise ValueError('inspect preview and explicitly pass --preview-reviewed')
        previous = json.loads((output / 'views_status.json').read_text())
        provenance = json.loads((output / 'experiment.json').read_text())
        if (previous['status'] != 'completed_pending_review' or provenance['git'] != git
                or provenance['input_hashes'] != locks or json.loads((output / 'protocol.json').read_text()) != config):
            raise ValueError('preview provenance mismatch')
        for name, digest in previous['artifact_hashes'].items():
            if sha256(output / name) != digest:
                raise ValueError('preview artifact changed')
    status_path = output / f'{stage}_status.json'
    if status_path.exists():
        raise ValueError('stage already attempted; never overwrite it')
    state = {'status': 'started', 'git': git, 'stage': stage, 'images_completed': 0,
             'preview_visually_checked': preview_reviewed, 'training_started': False, 'new_model_samples': 0}
    write(status_path, state)
    try:
        os.environ.update(OPENCV_IO_ENABLE_OPENEXR='1', TORCH_EXTENSIONS_DIR=str(ASSETS / 'cache/torch_extensions'),
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
        source_params = json.loads(Path(config['camera_source']).read_text())['camera']
        cameras = {name: source_camera(source_params, offset, vertices)
                   for name, offset in config['views_yaw_offset_radians'].items()}
        camera_record = {name: {'extrinsics': extr.cpu().tolist(), 'intrinsics': intr.cpu().tolist()}
                         for name, (extr, intr) in cameras.items()}
        if stage == 'views':
            write(output / 'cameras.json', camera_record)
        elif json.loads((output / 'cameras.json').read_text()) != camera_record:
            raise ValueError('camera differs from inspected preview')
        lights = {}
        for name in (['studio'] if stage == 'views' else config['lights']):
            rgb = cv2.imread(str(official / f'assets/hdri/{name}.exr'), cv2.IMREAD_UNCHANGED)
            if rgb is None or not np.isfinite(rgb).all():
                raise ValueError('invalid HDRI')
            lights[name] = EnvMap(torch.tensor(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), device='cuda'))
            _ = lights[name]._backend
        renderer = PbrMeshRenderer({'resolution': 512, **{key: config['renderer'][key]
            for key in ('near', 'far', 'ssaa', 'peel_layers')}})
        planned, records, geometry, framing = rows(config), [], {}, {}
        if stage == 'dataset':
            for folder in ('raw', 'cutouts', 'dataset', 'maps', 'review'):
                (output / folder).mkdir(exist_ok=False)
        for view, (extr, intr) in cameras.items():
            depth = vertices @ extr[:3, :3].T + extr[:3, 3]
            if float(depth[:, 2].min()) <= config['renderer']['near']:
                raise ValueError('near-plane clipping')
            controls = [([.5, .5, .5], 0., .5, None)] if stage == 'views' else [
                (config['colors'][color], metallic, roughness, color) for color, metallic, roughness in
                itertools.product(config['colors'], config['metallic'], config['roughness'])]
            for color, metallic, roughness, color_name in controls:
                material.base_color_factor = vertices.new_tensor(color)
                material.metallic_factor, material.roughness_factor = metallic, roughness
                torch.manual_seed(42)
                with torch.inference_mode():
                    rendered = renderer.render(mesh, extr, intr, lights, use_envmap_bg=False)
                arrays = {key: value.cpu().numpy() for key, value in rendered.items()}
                if any(not np.isfinite(value).all() for value in arrays.values()):
                    raise ValueError('nonfinite render')
                mask = arrays['mask'] > .5
                if not mask.any() or not mask[-1].any() or mask[0].any() or mask[:, 0].any() or mask[:, -1].any():
                    raise ValueError('require bottom crop only; inspect framing')
                signatures = {'mask': array_hash(mask), 'normal': array_hash(arrays['normal'])}
                if view not in geometry:
                    geometry[view] = {'mask': mask, 'normal': arrays['normal']}
                if signatures != {key: array_hash(value) for key, value in geometry[view].items()}:
                    raise ValueError('geometry differs across appearances')
                if stage == 'views':
                    ys, xs = np.where(mask)
                    framing[view] = {'bbox_xyxy': [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                                     'bottom_border_pixels': int(mask[-1].sum())}
                    Image.fromarray(mask.astype(np.uint8) * 255).save(output / f'geometry/{view}_mask.png')
                    np.save(output / f'geometry/{view}_normal.npy', arrays['normal'])
                    raw = np.rint(np.clip(arrays['shaded_studio'].transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
                    Image.fromarray(raw).save(output / f'preview/{view}_raw.png')
                    Image.fromarray(np.where(mask[..., None], raw, 127).astype(np.uint8)).save(output / f'preview/{view}.png')
                    state['images_completed'] += 1
                else:
                    if signatures != previous['geometry_buffers'][view]:
                        raise ValueError('geometry differs from reviewed clay')
                    targets = ('base_color', 'metallic', 'roughness', 'alpha')
                    map_name = f'{view}__{color_name}__M{int(metallic)}_R{int(round(roughness * 100))}.npz'
                    np.savez_compressed(output / 'maps' / map_name, **{key: arrays[key] for key in targets})
                    for light in lights:
                        row = next(r for r in planned if (r['view'], r['color'], r['light'], r['controls']['metallic'],
                            r['controls']['roughness']) == (view, color_name, light, metallic, roughness))
                        record = package(output, row, arrays['shaded_' + light], mask)
                        record.update(geometry_buffer_sha256=signatures, maps='maps/' + map_name,
                                      maps_sha256=sha256(output / 'maps' / map_name))
                        verify_record(output, record, config, geometry[view])
                        records.append(record)
                    state['images_completed'] = len(records)
                print('RENDERED', stage, view, color_name, metallic, roughness, state['images_completed'], flush=True)
        if stage == 'views':
            rgba = np.asarray(Image.open(source).convert('RGBA'))
            source_rgb = Image.fromarray(np.where(rgba[..., 3:] > 127, rgba[..., :3], 127).astype(np.uint8))
            canvas = Image.new('RGB', (2048, 544), 'white')
            draw = ImageDraw.Draw(canvas)
            for i, (name, image) in enumerate([('Actual pink source', source_rgb)] + [
                    (name, Image.open(output / f'preview/{name}.png')) for name in cameras]):
                draw.text((i * 512 + 8, 8), name, fill='black')
                canvas.paste(image, (i * 512, 32))
            canvas.save(output / 'preview/clay_three_views.png')
            state['framing'] = framing
        else:
            if len(records) != 72 or {r['id'] for r in records} != {r['id'] for r in planned}:
                raise ValueError('incomplete 72-image factorial')
            groups = {}
            for record in records:
                groups.setdefault(record['pair_group'], []).append(record)
            pairs = [pair for group in groups.values() for pair in one_factor_pairs(group)]
            if len(groups) != 12 or len(pairs) != 84:
                raise ValueError('incorrect matched material pairs')
            by_id = {r['id']: r for r in records}
            for pair in pairs:
                first, second = (by_id[pair[key]] for key in ('from', 'to'))
                a, b = [np.asarray(Image.open(output / 'dataset' / r['image'])).astype(np.int16) for r in (first, second)]
                difference = np.abs(a - b)[geometry[first['view']]['mask']]
                if not difference.any():
                    raise ValueError('material intervention produced identical foreground')
                pair['foreground_rgb_mae_0_255'] = float(difference.mean())
            write(output / 'dataset/manifest.json', {'schema': config['schema'], 'records': records, 'git': git,
                'source_hashes': locks, 'geometry_sha256': geometry_hashes, 'training_started': False,
                'training_eligible_images': 0, 'interpretation': config['limitations']})
            write(output / 'dataset/concepts.json', [{'instance_prompt': [r['caption']],
                'instance_data_dir': str((output / 'dataset' / r['image']).parent),
                'instance_mask_dir': str((output / 'dataset' / r['mask']).parent)} for r in records])
            write(output / 'dataset/pairs.json', {'groups': groups, 'pairs': pairs, 'split_policy': SPLIT_POLICY})
            review(output, records, config)
            state.update(pair_groups=12, single_factor_pairs=84)
        check_inputs()
        if any(array_hash(getattr(mesh, key).cpu().numpy()) != digest for key, digest in geometry_hashes.items()):
            raise ValueError('native geometry changed')
        state.update(status='completed_pending_review', native_inputs_unchanged=True, geometry_sha256=geometry_hashes,
            geometry_buffers={view: {key: array_hash(value) for key, value in buffers.items()} for view, buffers in geometry.items()},
            gpu=torch.cuda.get_device_name(), torch=torch.__version__,
            artifact_hashes={p.relative_to(output).as_posix(): sha256(p) for p in output.rglob('*')
                if p.is_file() and p.suffix in ('.png', '.json', '.npy', '.npz', '.html') and p != status_path})
    except BaseException as error:
        state.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write(status_path, state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=('views', 'dataset'), required=True)
    parser.add_argument('--preview-reviewed', action='store_true')
    args = parser.parse_args()
    run(args.output, args.stage, args.preview_reviewed)
