"""Fresh per-instance C9/M8/T16 data; no historical Mailbox derived assets."""

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BRANCH_COUNTS = {'color': 9, 'material': 8, 'texture': 16}
ASSET_ROOT = Path('/home/r12user5/Documents/Jiawei/papers/CLEVER/image_generation')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf8')


def check_locks(locks):
    if any(sha(path) != digest for path, digest in locks.items()):
        raise ValueError('CMT input or derived evidence changed')


def fresh(path):
    path = Path(path).resolve()
    if path.exists() or path.is_relative_to(ROOT):
        raise ValueError('require a fresh output outside the source checkout')
    path.mkdir(parents=True)
    return path


def eroded_region(mask, radius=5):
    from PIL import Image, ImageFilter, ImageOps
    image = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255)
    padded = ImageOps.expand(image, border=radius, fill=0)
    return np.asarray(padded.filter(ImageFilter.MinFilter(radius * 2 + 1)).crop(
        (radius, radius, radius + image.width, radius + image.height))) == 255


def largest_rectangle(valid):
    """Same all-valid rectangle policy as natural_texture_material_v1.extract."""
    heights, best, box = np.zeros(valid.shape[1], dtype=int), None, None
    for y, row in enumerate(valid):
        heights = np.where(row, heights + 1, 0)
        stack = []
        for x in range(len(heights) + 1):
            height, start = int(heights[x]) if x < len(heights) else 0, x
            while stack and stack[-1][1] > height:
                left, previous = stack.pop()
                top, bottom, right = y - previous + 1, y + 1, x
                key = (-previous * (right - left), top, left, bottom, right)
                if best is None or key < best:
                    best, box = key, (left, top, right, bottom)
                start = left
            if height and (not stack or stack[-1][1] < height):
                stack.append((start, height))
    if box is None:
        raise ValueError('empty texture region')
    return box


def prepare_inputs(prepared, output):
    from PIL import Image
    prepared, output = Path(prepared).resolve(), fresh(output)
    source_path = prepared / 'source_receipt.json'
    source = read(source_path)
    locks = dict(source['source_locks'])
    for key in ('image512', 'mask', 'cutout'):
        path = source[key + '_path']
        if sha(path) != source[key + '_sha256']:
            raise ValueError('shared prepared source changed')
        locks[path] = sha(path)
    locks[str(source_path)] = sha(source_path)
    check_locks(locks)
    with Image.open(source['mask_path']) as opened:
        mask = np.asarray(opened.convert('L')) == 255
    region = eroded_region(mask)
    if region.sum() < 2:
        raise ValueError('automatic eroded region has fewer than two valid pixels')
    Image.fromarray(region.astype(np.uint8) * 255).save(output / 'material_region.png')
    with Image.open(source['image512_path']) as opened:
        preview = np.array(opened.convert('RGB'))
    preview[region] = np.rint(.65 * preview[region] + .35 * np.array([255, 0, 0])).astype(np.uint8)
    Image.fromarray(preview).save(output / 'roi_overlay.png')
    inputs = output / 'supermat_input'
    inputs.mkdir()
    shutil.copyfile(source['cutout_path'], inputs / 'input.png')
    locks[str(output / 'material_region.png')] = sha(output / 'material_region.png')
    locks[str(output / 'roi_overlay.png')] = sha(output / 'roi_overlay.png')
    locks[str(inputs / 'input.png')] = sha(inputs / 'input.png')
    receipt = {'source': source, 'source_locks': locks, 'region': str(output / 'material_region.png'),
        'supermat_input': str(inputs / 'input.png'), 'region_pixels': int(region.sum()),
        'roi_overlay': str(output / 'roi_overlay.png'), 'overlay_use': 'preview_only_never_model_input',
        'region_policy': 'full object eroded 5px on shared512 canvas; no semantic/manual ROI',
        'region_reviewed_by_user': False, 'mixed_surfaces_possible': True}
    write(output / 'input_receipt.json', receipt)
    return output / 'input_receipt.json'


def color_target(rgb, valid):
    from . import neutral_lch_injection as color
    lab = color.linear_rgb_to_lab(color._srgb_to_linear(rgb[valid]))
    target = np.median(lab, axis=0)
    lightness = np.linspace(0, 100, 1001)
    values = np.column_stack([lightness, np.broadcast_to(target[1:], (len(lightness), 2))])
    linear = color.lab_to_linear_rgb(values)
    feasible = lightness[((linear >= 0) & (linear <= 1)).all(axis=1)]
    if len(feasible) < 11:
        raise ValueError('pixel chromaticity has insufficient in-gamut lightness range')
    low, high = float(feasible[0] + .1), float(feasible[-1] - .1)
    center = float(np.clip(50, low + .4, high - .4))
    amplitude = min(8., center - low, high - center)
    return {'lab_median': target.tolist(), 'a': float(target[1]), 'b': float(target[2]),
        'estimator': 'componentwise median Lab in automatic eroded object mask',
        'pixel_count': int(valid.sum()), 'L_center': center, 'L_amplitude': amplitude,
        'feasible_L_range': [low, high], 'chroma_reduction': False, 'rgb_clipping': False}


def remap_color(rgb, mask, target):
    from . import neutral_lch_injection as color
    valid = np.asarray(mask, dtype=bool)
    interior = eroded_region(valid)
    source_lab = color.linear_rgb_to_lab(color._srgb_to_linear(rgb.reshape(-1, 3))).reshape(*valid.shape, 3)
    q05, center, q95 = np.quantile(source_lab[interior, 0], [.05, .5, .95])
    scale = (q95 - q05) / 2
    if scale < .5:
        raise ValueError('neutral render has insufficient lightness contrast')
    lightness = target['L_center'] + target['L_amplitude'] * np.tanh((source_lab[valid, 0] - center) / scale)
    lab = np.column_stack([lightness, np.full(len(lightness), target['a']), np.full(len(lightness), target['b'])])
    result = rgb.copy()
    result[valid] = color._linear_to_srgb_uint8(color.lab_to_linear_rgb(lab))
    return result, {'source_L_q05_median_q95': [float(q05), float(center), float(q95)],
                    'outside_mask_changed_pixels': int(np.any(result[~valid] != rgb[~valid], axis=1).sum())}


def extract_gray(albedo, region):
    from PIL import Image
    from experiments.texture_material_disentangled_v1.data import extract_texture
    rgb = np.asarray(albedo.convert('RGB'), dtype=np.float64) / 255
    linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    luminance = linear @ np.array([.2126, .7152, .0722])
    valid = np.asarray(region) >= 128
    lower, upper = np.quantile(luminance[valid], [.02, .98])
    if upper - lower <= .0001:
        return Image.new('L', albedo.size, 128), {
            'method': 'uniform_texture_midpoint_for_degenerate_robust_normalization',
            'lower_luminance': float(lower), 'upper_luminance': float(upper),
            'valid_pixels': int(valid.sum()), 'constant_epsilon': .0001,
            'normalization_degenerate': True, 'spatial_pattern_invented': False,
            'interpretation': 'valid nearly uniform albedo; learn uniform, do not amplify noise'}
    return extract_texture(albedo, region)


def build_plan(prepared, maps, output, branches=('color', 'material', 'texture')):
    from PIL import Image, ImageDraw
    from scripts.launch.colorpeel_run import git_info
    from experiments.natural_material_supermat_v1.material_pipeline import map_values
    branches = list(dict.fromkeys(branches))
    if not branches or not set(branches) <= set(BRANCH_COUNTS):
        raise ValueError('require selected Color/Material/Texture branches')
    prepared, output = Path(prepared).resolve(), fresh(output)
    receipt = read(prepared / 'input_receipt.json')
    locks = dict(receipt['source_locks'])
    locks[str(prepared / 'input_receipt.json')] = sha(prepared / 'input_receipt.json')
    check_locks(locks)
    protocol_path = ROOT / 'experiments/texture_material_disentangled_v1/protocol.json'
    old = read(protocol_path)
    protocol = {key: old[key] for key in ('seed', 'dataset', 'renderer', 'texture_extraction')}
    protocol['materials'] = {name: spec for name, spec in old['materials'].items() if spec['role'] == 'nuisance'}
    locks[str(protocol_path)] = sha(protocol_path)
    with Image.open(receipt['region']) as opened:
        region = opened.convert('L')
    valid = np.asarray(region) == 255
    result = {'git': git_info(), 'requested_branches': branches, 'branches': [], 'failures': {},
              'source': receipt['source'], 'source_locks': locks, 'records': [], 'protocol': protocol,
              'automatic_roi_reviewed': False, 'training_started': False}
    if set(branches) & {'material', 'texture'}:
        maps = Path(maps).resolve()
        inference_path = maps.parent / 'inference_manifest.json'
        inference = read(inference_path)
        expected = [row for row in inference['outputs'] if row['input'] == receipt['supermat_input']]
        if (inference.get('status') != 'complete' or len(expected) != 1
                or expected[0]['input_sha256'] != sha(receipt['supermat_input'])):
            raise ValueError('SuperMat outputs do not match this instance input')
        locks[str(inference_path)] = sha(inference_path)
        for channel in ('albedo', 'roughness', 'metallic'):
            path = maps / (channel + '.png')
            if str(path) != expected[0]['maps'][channel]['path'] or sha(path) != expected[0]['maps'][channel]['sha256']:
                raise ValueError('SuperMat map receipt differs')
            locks[str(path)] = sha(path)
    for branch in branches:
        try:
            if branch == 'color':
                with Image.open(receipt['source']['image512_path']) as opened:
                    result['color_target'] = color_target(np.asarray(opened.convert('RGB')), valid)
                rows = [{'id': f'color_{shape}_{view:02d}', 'branch': branch, 'shape': shape,
                         'view': view, 'seed': 42, 'caption': f'a photo of {shape} shape in <C*> color'}
                        for shape, view in itertools.product(('cube', 'sphere', 'cylinder'), (0, 8, 16))]
            else:
                if branch == 'material':
                    distributions = {channel: map_values(maps / (channel + '.png'), region)
                                     for channel in ('roughness', 'metallic')}
                    target = {channel: dist['median'] for channel, dist in distributions.items()}
                    result['material_target'] = target
                    result['material_distributions'] = distributions
                    protocol['materials']['target'] = target
                else:
                    with Image.open(maps / 'albedo.png') as opened:
                        gray, statistics = extract_gray(opened, region)
                    box = largest_rectangle(valid)
                    gray.crop(box).save(output / 'texture.png')
                    with Image.open(receipt['roi_overlay']) as opened:
                        preview = opened.convert('RGB')
                    ImageDraw.Draw(preview).rectangle((box[0], box[1], box[2]-1, box[3]-1), outline='blue', width=3)
                    preview.save(output / 'texture_roi_overlay.png')
                    locks[str(output / 'texture_roi_overlay.png')] = sha(output / 'texture_roi_overlay.png')
                    locks[str(output / 'texture.png')] = sha(output / 'texture.png')
                    result['texture_extraction'] = {'statistics': statistics, 'crop_xyxy': list(box),
                        'retained_region_fraction': (box[2]-box[0])*(box[3]-box[1])/int(valid.sum()),
                        'no_inpainting': True, 'physical_scale_known': False}
                rows = []
                for i, (shape, color, light, material) in enumerate(itertools.product(
                        ('sphere', 'cube'), ('red', 'blue'), ('soft_front', 'side_directional'),
                        ('smooth_dielectric', 'rough_metal') if branch == 'texture' else ('target',))):
                    phrase = protocol['dataset']['light_phrases'][light]
                    caption = f'a photo of a {color} {shape}'
                    caption += (f" with <T*> texture made of {protocol['materials'][material]['literal']} under {phrase}"
                                if branch == 'texture' else f' with a <M*> surface under {phrase}')
                    row = {'id': f'{branch}_{i:02d}', 'branch': branch, 'shape': shape, 'color': color,
                           'light': light, 'material_id': material, 'seed': 42+i, 'caption': caption}
                    if branch == 'texture':
                        row.update(texture_path=str(output / 'texture.png'), texture_sha256=sha(output / 'texture.png'))
                    rows.append(row)
            result['records'].extend(rows)
            result['branches'].append(branch)
        except ValueError as error:
            result['failures'][branch] = str(error)
    from scripts.methods.colorpeel_ice.run_d1_gamut_aware_lch_injection import ASSETS
    for relative in ASSETS.values():
        path = ASSET_ROOT / relative
        locks[str(path)] = sha(path)
    profile_path = ROOT / protocol['renderer']['profile']
    locks[str(profile_path)] = sha(profile_path)
    check_locks(locks)
    write(output / 'plan.json', result)
    return output / 'plan.json'


def render_color(row, output, bpy):
    from scripts.methods.colorpeel_ice import render_d1_color_calibration_preflight as shared
    from scripts.methods.colorpeel_ice import render_d1_color_calibration_reduced_direct as direct
    from scripts.methods.colorpeel_ice import run_d1_gamut_aware_lch_injection as original
    from experiments.color_geometry_lightness_v1.render_background import rebuild_ground
    from src.methods.colorpeel_ice import neutral_lch_injection as color
    assets = {key: ASSET_ROOT / value for key, value in original.ASSETS.items()}
    bpy.ops.wm.open_mainfile(filepath=str(assets['base_scene']))
    scene = bpy.context.scene
    devices = shared._configure_cycles(scene, row['seed'])
    shared.configure_color_management(scene)
    for obj in list(scene.objects):
        if obj.type == 'MESH' and obj.name != 'Ground':
            bpy.data.objects.remove(obj, do_unlink=True)
    shared._configure_background(scene)
    rebuild_ground(scene)
    obj = original._append_shape(row['shape'], assets[original.SHAPE_DETAILS[row['shape']][0]])
    linear = color.lab_to_linear_rgb(np.array([[50., 0., 0.]]))[0].tolist()
    material = direct._configure_rubber(obj, {'material': 'Rubber', 'linear_rgb': linear,
        'socket_rgba': linear + [1.]}, assets['material_asset'])
    camera, lights, base = direct._camera_and_lights(obj, row['view'])
    folder = output / row['id']
    folder.mkdir()
    shared._configure_mask_output(folder, obj)
    scene.render.filepath = str(folder / 'image.png')
    bpy.ops.render.render(write_still=True)
    mask = shared._finalize_mask(folder)
    record = {**row, 'image': str(folder / 'image.png'), 'mask': str(mask), 'camera': camera,
              'lights': lights, 'material': material, 'cuda_devices': devices,
              'cycles_samples': scene.cycles.samples, 'background': 'continuous gray studio'}
    write(folder / 'metadata.json', record)
    return {**record, 'metadata': str(folder / 'metadata.json')}


def render(plan_path, output, gpu):
    from scripts.launch.colorpeel_run import git_info
    import bpy
    from experiments.natural_material_supermat_v1 import render_calibration as base
    from experiments.natural_material_threeway_v1.render_grid import append_shape
    from experiments.texture_material_disentangled_v1.render import render_one as texture_render
    from experiments.material_uniform_albedo_v1.render import render_one as material_render
    from experiments.natural_texture_material_v1.render_dataset import shader_audit
    plan = read(plan_path)
    if (plan['git'] != git_info() or os.environ.get('CUDA_VISIBLE_DEVICES') != str(gpu)
            or tuple(bpy.app.version) != (4, 2, 11)):
        raise ValueError('plan Git, selected GPU, or Blender version differs')
    check_locks(plan['source_locks'])
    output = fresh(output)
    profile = read(ROOT / plan['protocol']['renderer']['profile'])
    profile['blender'].update(cuda_visible_devices=str(gpu), cycles_samples=64, resolution=[512, 512])
    result = {'status': 'running', 'git': plan['git'], 'plan_path': str(Path(plan_path).resolve()),
              'plan_sha256': sha(plan_path), 'records': []}
    try:
        for row in plan['records']:
            if row['branch'] == 'color':
                record = render_color(row, output, bpy)
            else:
                function = texture_render if row['branch'] == 'texture' else material_render
                record = function(row, output, plan['protocol'], profile, bpy, base, append_shape)
                record.update(shader_audit(bpy, row['branch']))
            for key in ('image', 'mask', 'metadata'):
                record[key + '_sha256'] = sha(record[key])
            result['records'].append(record)
            write(output / 'manifest.json', result)
            print('CMT_RENDERED', row['id'], flush=True)
        check_locks(plan['source_locks'])
        result['status'] = 'complete'
    except BaseException as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write(output / 'manifest.json', result)
    return output / 'manifest.json'


def finalize(plan_path, renders, output):
    from PIL import Image
    from scripts.launch.colorpeel_run import git_info
    plan = read(plan_path)
    renders = Path(renders).resolve()
    rendered = read(renders / 'manifest.json')
    if (rendered['status'] != 'complete' or rendered['plan_sha256'] != sha(plan_path)
            or rendered['git'] != plan['git'] or git_info() != plan['git']):
        raise ValueError('complete rendering and preparation must use the same plan and Git')
    output = fresh(output)
    locks = {**plan['source_locks'], str(Path(plan_path).resolve()): sha(plan_path),
             str(renders / 'manifest.json'): sha(renders / 'manifest.json')}
    check_locks(locks)
    datasets = {}
    for branch in plan['branches']:
        rows = []
        for original in (row for row in rendered['records'] if row['branch'] == branch):
            row = dict(original)
            for key in ('image', 'mask', 'metadata'):
                if sha(row[key]) != row[key + '_sha256']:
                    raise ValueError('render artifact hash changed')
                locks[row[key]] = row[key + '_sha256']
            for key in ('image', 'mask'):
                row['source_' + key] = row[key]
                row['source_' + key + '_sha256'] = row[key + '_sha256']
            folder = output / branch / row['id']
            folder.mkdir(parents=True)
            with Image.open(row['image']) as opened:
                rgb = np.asarray(opened.convert('RGB'))
            with Image.open(row['mask']) as opened:
                mask = np.asarray(opened.convert('L')) >= 128
            if rgb.shape != (512, 512, 3) or mask.shape != (512, 512) or not mask.any():
                raise ValueError('expected RGB512 and nonempty aligned mask')
            if branch == 'color':
                rgb, row['color_metrics'] = remap_color(rgb, mask, plan['color_target'])
            Image.fromarray(rgb).save(folder / 'image.png')
            Image.fromarray(mask.astype(np.uint8) * 255).save(folder / 'mask.png')
            row.update(image=str(folder / 'image.png'), image_sha256=sha(folder / 'image.png'),
                       mask=None if branch == 'color' else str(folder / 'mask.png'))
            row['mask_sha256'] = None if row['mask'] is None else sha(row['mask'])
            rows.append(row)
        if len(rows) != BRANCH_COUNTS[branch]:
            raise ValueError(f'incorrect {branch} image count')
        datasets[branch] = {'rows': rows, 'source_locks': locks,
            'source_image_sha256': plan['source']['source_sha256'],
            'source_mask_sha256': plan['source']['original_mask_sha256']}
        if branch == 'material':
            datasets[branch]['material_target'] = plan['material_target']
    for path in output.rglob('*.png'):
        locks[str(path)] = sha(path)
    check_locks(locks)
    write(output / 'datasets.json', {'branches': datasets, 'failures': plan['failures'],
          'automatic_roi_reviewed': False, 'git': plan['git']})
    return datasets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('inputs', 'plan', 'render', 'finalize'), required=True)
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--maps', type=Path)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--renders', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--branches', nargs='+', default=list(BRANCH_COUNTS), choices=list(BRANCH_COUNTS))
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else None)
    if args.stage == 'inputs':
        prepare_inputs(args.prepared, args.output)
    elif args.stage == 'plan':
        build_plan(args.prepared, args.maps, args.output, args.branches)
    elif args.stage == 'render':
        render(args.plan, args.output, args.gpu)
    else:
        finalize(args.plan, args.renders, args.output)


if __name__ == '__main__':
    main()
