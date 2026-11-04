"""Fresh per-instance inputs and Subject11: original, two recolors, eight renders."""

import argparse
from itertools import product
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageOps

from src.methods.colorpeel_ice.object_segmentation import sha256, write_json

ROOT = Path(__file__).resolve().parents[3]
COLORS = {'blue': [.2, .35, .8], 'green': [.25, .65, .3]}
RENDERER = {'near': .1, 'far': 100, 'ssaa': 1, 'peel_layers': 1, 'fov_degrees': 40}
STUDIO_SHA = 'c36c7fd390b2cdf8693360cf3662cb1d710c95fa799d9f822f42997167839b56'


def check_locks(locks):
    if any(sha256(path) != digest for path, digest in locks.items()):
        raise ValueError('instance source or generated asset changed')


def fresh(output):
    output = Path(output).resolve()
    if output.exists() or output.is_relative_to(ROOT):
        raise ValueError('require a fresh external output directory')
    output.mkdir(parents=True)
    return output


def square_pair(image, mask):
    """Map the full native canvas together, never crop or stretch an instance."""
    mask = np.asarray(mask)
    if mask.shape != (image.height, image.width) or not set(np.unique(mask)) <= {0, 255}:
        raise ValueError('require native aligned binary mask')
    if not mask.any():
        raise ValueError('empty subject mask')
    scale = min(512 / image.width, 512 / image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    offset = ((512 - size[0]) // 2, (512 - size[1]) // 2)
    rgb = Image.new('RGB', (512, 512), (127, 127, 127))
    alpha = Image.new('L', (512, 512), 0)
    rgb.paste(image.convert('RGB').resize(size, Image.Resampling.LANCZOS), offset)
    alpha.paste(Image.fromarray(mask.astype(np.uint8)).resize(size, Image.Resampling.NEAREST), offset)
    if not np.asarray(alpha).any():
        raise ValueError('subject disappeared when mapping to the 512 canvas')
    return rgb, alpha, {'native_size': list(image.size), 'resized_size': list(size),
        'scale': scale, 'scale_xy': [size[0] / image.width, size[1] / image.height],
        'pad_left_top': list(offset), 'output_size': [512, 512], 'crop': False,
        'rgb_resampling': 'LANCZOS', 'mask_resampling': 'NEAREST', 'padding_rgb': [127]*3}


def prepare_inputs(image_path, mask_path, output):
    image_path, mask_path = Path(image_path).resolve(), Path(mask_path).resolve()
    locks = {str(path): sha256(path) for path in (image_path, mask_path)}
    with Image.open(image_path) as opened, Image.open(mask_path) as opened_mask:
        image = ImageOps.exif_transpose(opened).convert('RGB')
        rgb, mask, transform = square_pair(image, np.asarray(opened_mask.convert('L')))
    output = fresh(output)
    rgb.save(output / 'image512.png')
    mask.save(output / 'mask512.png')
    pixels, alpha = np.asarray(rgb), np.asarray(mask)
    Image.fromarray(np.dstack([np.where(alpha[..., None] > 0, pixels, 0), alpha])).save(output / 'cutout512.png')
    for name in ('image512.png', 'mask512.png', 'cutout512.png'):
        locks[str(output / name)] = sha256(output / name)
    receipt = {'source_path': str(image_path), 'source_sha256': locks[str(image_path)],
        'original_mask_path': str(mask_path), 'original_mask_sha256': locks[str(mask_path)],
        'image512_path': str(output / 'image512.png'), 'image512_sha256': locks[str(output / 'image512.png')],
        'mask_path': str(output / 'mask512.png'), 'mask_sha256': locks[str(output / 'mask512.png')],
        'cutout_path': str(output / 'cutout512.png'), 'cutout_sha256': locks[str(output / 'cutout512.png')],
        'transform': transform, 'source_locks': locks, 'alpha_matches_mask': True,
        'transparent_rgb_zero': True, 'segmentation': 'supplied per-instance Scheme A binary SAM mask'}
    check_locks(locks)
    write_json(output / 'source_receipt.json', receipt)
    return receipt


def appearance_rows(category):
    if not isinstance(category, str) or not category.strip() or any(x in category for x in '<>\n\r'):
        raise ValueError('require a plain per-object category')
    return [{'id': f'center__{color}__M{metal}_R{int(rough*100):02d}__studio',
             'kind': 'trellis_counterfactual', 'color': color, 'metallic': metal, 'roughness': rough,
             'caption': f'a photo of <S*> {category} in {color} color with a '
                        f'{"glossy" if rough == .15 else "rough"} '
                        f'{"metallic" if metal else "non-metallic"} surface'}
            for color, metal, rough in product(COLORS, (0, 1), (.15, .85))]


def save_row(output, row, rgb, mask):
    rgb, mask = np.asarray(rgb), np.asarray(mask)
    if (rgb.shape != (512, 512, 3) or rgb.dtype != np.uint8 or mask.shape != (512, 512)
            or not set(np.unique(mask)) <= {0, 255} or not mask.any()):
        raise ValueError('require RGB512 and nonempty binary mask512')
    result = dict(row)
    for key, array in (('image', rgb), ('mask', mask)):
        path = output / 'assets' / row['id'] / key / 'image.png'
        path.parent.mkdir(parents=True)
        Image.fromarray(array).save(path)
        result[key], result[key + '_sha256'] = str(path), sha256(path)
    return result


def photo_rows(receipt, output, category):
    from experiments.subject_trellis_wood27_v1.prepare import HUES, recolor_photo
    appearance_rows(category)  # Validate before constructing captions.
    with Image.open(receipt['image512_path']) as opened, Image.open(receipt['mask_path']) as mask_image:
        rgb, mask = np.array(opened), np.array(mask_image)
    base = {'id': 'photo_original', 'kind': 'original_photograph',
            'caption': f'a photo of <S*> {category}'}
    rows = [save_row(output, base, rgb, mask)]
    for color, hue in HUES.items():
        colored, metrics, _, _ = recolor_photo(rgb, mask, hue)
        rows.append(save_row(output, {'id': f'photo_lch_{color}', 'kind': 'lch_recolor',
            'caption': f'a photo of <S*> {category} in {color} color',
            'hue_degrees': hue, 'recolor_metrics': metrics}, colored, mask))
    return rows


def render_dataset(inputs, trellis_run, output, category):
    """Use the newly sampled instance mesh; never resolve historical Mailbox assets."""
    from experiments.subject_trellis_counterfactual_v1.run import ASSETS, OFFICIAL
    from experiments.subject_trellis_dataset_v1.build import camera, array_hash

    inputs, trellis_run = Path(inputs).resolve(), Path(trellis_run).resolve()
    receipt_path = inputs / 'source_receipt.json'
    receipt = json.loads(receipt_path.read_text())
    state_path = trellis_run / 'repro_outputs/status.json'
    state = json.loads(state_path.read_text())
    if (state.get('status') != 'generated_pending_user_review' or state.get('generated_assets') != 1
            or state.get('source_receipt', {}).get('sha256') != sha256(receipt_path)
            or sha256(trellis_run / 'source_cutout512.png') != receipt['cutout_sha256']):
        raise ValueError('TRELLIS asset is not a completed fresh sample for this instance receipt')
    official = ASSETS / 'sources/TRELLIS.2'
    if (subprocess.check_output(['git', '-C', str(official), 'rev-parse', 'HEAD'], text=True).strip() != OFFICIAL
            or subprocess.check_output(['git', '-C', str(official), 'status', '--porcelain'], text=True).strip()):
        raise ValueError('require pinned clean TRELLIS renderer checkout')
    asset, review_path, light_path = (trellis_run / 'native_asset.pt', trellis_run / 'review/review.json',
                                     official / 'assets/hdri/studio.exr')
    locks = {**receipt['source_locks'], str(receipt_path): sha256(receipt_path),
             str(asset): sha256(asset), str(review_path): sha256(review_path),
             str(state_path): sha256(state_path), str(light_path): STUDIO_SHA}
    check_locks(locks)
    output = fresh(output)
    os.environ.update(OPENCV_IO_ENABLE_OPENEXR='1',
        TORCH_EXTENSIONS_DIR=str(ASSETS / 'cache/torch_extensions'), TRITON_CACHE_DIR=str(output / 'triton_cache'))
    sys.path.insert(0, str(official))
    import cv2
    import torch
    from trellis2.representations.mesh import MeshWithPbrMaterial, PbrMaterial
    from trellis2.renderers import EnvMap, PbrMeshRenderer

    rows = photo_rows(receipt, output, category)
    saved = torch.load(asset, map_location='cuda', weights_only=False)
    vertices, faces = saved['vertices'], saved['faces']
    geometry_hashes = {name: array_hash(saved[name].cpu().numpy()) for name in ('vertices', 'faces')}
    material = PbrMaterial()
    mesh = MeshWithPbrMaterial(vertices, faces, torch.zeros(len(faces), dtype=torch.long, device='cuda'),
                              torch.zeros((len(faces), 3, 2), device='cuda'), [material])
    source_camera = json.loads(review_path.read_text())['camera']['parameters']
    extr, intr, camera_receipt = camera(vertices, source_camera, 0, {'renderer': RENDERER})
    env = cv2.imread(str(light_path), cv2.IMREAD_UNCHANGED)
    if env is None or not np.isfinite(env).all():
        raise ValueError('invalid studio environment')
    light = EnvMap(torch.tensor(cv2.cvtColor(env, cv2.COLOR_BGR2RGB), device='cuda'))
    _ = light._backend
    renderer = PbrMeshRenderer({'resolution': 512, **{key: value for key, value in RENDERER.items() if key != 'fov_degrees'}})
    baseline = None
    for row in appearance_rows(category):
        material.base_color_factor = vertices.new_tensor(COLORS[row['color']])
        material.metallic_factor, material.roughness_factor = row['metallic'], row['roughness']
        torch.manual_seed(42)
        with torch.inference_mode():
            rendered = renderer.render(mesh, extr, intr, {'studio': light}, use_envmap_bg=False)
        arrays = {key: value.cpu().numpy() for key, value in rendered.items()}
        if any(not np.isfinite(value).all() for value in arrays.values()):
            raise ValueError('nonfinite Subject render')
        inside = arrays['mask'] > .5
        ys, xs = np.nonzero(inside)
        if not len(xs) or min(xs.min(), ys.min()) < 20 or max(xs.max(), ys.max()) > 491:
            raise ValueError('empty or clipped centered Subject render')
        signatures = {'mask': array_hash(inside), 'normal': array_hash(arrays['normal'])}
        if baseline is not None and signatures != baseline:
            raise ValueError('geometry changes across appearance interventions')
        baseline = signatures
        for key, expected in (('base_color', np.array(COLORS[row['color']])[:, None]),
                              ('metallic', row['metallic']), ('roughness', row['roughness'])):
            values = arrays[key][:, inside] if key == 'base_color' else arrays[key][inside]
            if not np.allclose(values, expected, atol=1e-6, rtol=0):
                raise ValueError(f'{key} buffer differs from the requested intervention')
        raw = np.rint(np.clip(arrays['shaded_studio'].transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
        rgb = np.where(inside[..., None], raw, 127).astype(np.uint8)
        rows.append(save_row(output, row, rgb, inside.astype(np.uint8) * 255))
    for row in rows:
        locks.update({row[key]: row[key + '_sha256'] for key in ('image', 'mask')})
    check_locks(locks)
    result = {'rows': rows, 'source_locks': locks, 'source_image_sha256': receipt['source_sha256'],
        'source_mask_sha256': receipt['original_mask_sha256'], 'counts': {'original': 1, 'recolor': 2, 'trellis': 8},
        'camera': camera_receipt, 'geometry_sha256': geometry_hashes, 'renderer': RENDERER,
        'source_receipt': str(receipt_path), 'native_asset': str(asset), 'quality_accepted_by_user': False,
        'limitations': ['New geometry is a generated estimate, not a geometry-quality acceptance.',
                       'Full-canvas photographs retain other objects; each loss uses only this instance mask.',
                       'Synthetic renders use centered geometry; this does not repair reconstruction errors.']}
    write_json(output / 'subject.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('inputs', 'render'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--image', type=Path)
    parser.add_argument('--mask', type=Path)
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--trellis-run', type=Path)
    parser.add_argument('--category')
    args = parser.parse_args()
    if args.stage == 'inputs':
        if args.image is None or args.mask is None:
            parser.error('inputs requires --image and --mask')
        prepare_inputs(args.image, args.mask, args.output)
    else:
        if args.inputs is None or args.trellis_run is None or args.category is None:
            parser.error('render requires --inputs, --trellis-run and --category')
        render_dataset(args.inputs, args.trellis_run, args.output, args.category)


if __name__ == '__main__':
    main()
