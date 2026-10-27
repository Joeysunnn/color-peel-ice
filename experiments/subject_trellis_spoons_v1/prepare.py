"""Segment both complete original spoon photos with one pinned official BiRefNet."""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from scripts.launch.colorpeel_run import git_info, resolve_run_dir

ROOT = Path(__file__).resolve().parents[2]
SPEC = Path(__file__).with_name('inputs.json')
ASSETS = Path('/var/tmp/colorpeel-r12user5/assets/trellis2_v1')


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf8')


def download_bundle(segmentation):
    """GET only pinned files; never overwrite an existing complete or partial file."""
    import requests
    bundle = ASSETS / 'models/birefnet' / segmentation['revision']
    bundle.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, expected in segmentation['files'].items():
        path = bundle / name
        if not path.exists():
            url = f"{segmentation['endpoint']}/{segmentation['repo']}/resolve/{segmentation['revision']}/{name}"
            with requests.get(url, stream=True, timeout=(15, 120)) as response:
                response.raise_for_status()
                with path.open('xb') as stream:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            stream.write(chunk)
        if path.stat().st_size != expected['size']:
            raise ValueError(f'pinned model file size mismatch: {name}')
        digest = sha256(path)
        if 'sha256' in expected:
            matched = digest == expected['sha256']
        else:
            data = path.read_bytes()  # Git blob metadata is used only for small code/config files.
            blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            matched = blob == expected['git_blob_sha1']
        if not matched:
            raise ValueError(f'pinned model file hash mismatch: {name}')
        hashes[name] = digest
    return bundle, hashes


def square_cutout(image):
    """Resize the entire canvas without a subject crop; preserve straight RGBA."""
    import numpy as np
    from PIL import Image
    width, height = image.size
    scale = min(512 / width, 512 / height)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    offset = ((512 - size[0]) // 2, (512 - size[1]) // 2)
    resized = image.resize(size, Image.Resampling.LANCZOS)
    canvas = Image.new('RGBA', (512, 512), (0, 0, 0, 0))
    canvas.paste(resized, offset)
    rgba = np.array(canvas)
    rgba[rgba[..., 3] == 0, :3] = 0
    return Image.fromarray(rgba), {'original_size': [width, height],
        'resize_size': list(size), 'scale': scale, 'pad_left_top': list(offset),
        'pad_right_bottom': [512 - size[0] - offset[0], 512 - size[1] - offset[1]],
        'output_size': [512, 512], 'resampling': 'PIL LANCZOS RGBA', 'cropped': False}


def review_panel(cutout):
    from PIL import Image, ImageDraw
    panel = Image.new('RGB', (1536, 536), 'white')
    for index, (name, background) in enumerate((('white', 255), ('gray127', 127), ('soft alpha', None))):
        if background is None:
            view = cutout.getchannel('A').convert('RGB')
        else:
            view = Image.new('RGBA', cutout.size, (background,) * 3 + (255,))
            view = Image.alpha_composite(view, cutout).convert('RGB')
        panel.paste(view, (512 * index, 24))
        ImageDraw.Draw(panel).text((512 * index + 8, 5), name, fill='black')
    return panel


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0', choices=['cuda:0'])
    args = parser.parse_args(argv)
    spec = json.loads(SPEC.read_text(encoding='utf8'))
    sources = [(item, (ROOT / item['source']).resolve()) for item in spec['subjects']]
    if any(not path.is_relative_to(ROOT) or sha256(path) != item['source_sha256'] for item, path in sources):
        raise ValueError('complete original source hash mismatch')
    official = ASSETS / 'sources/TRELLIS.2'
    def official_git(*command):
        return subprocess.check_output(['git', '-C', str(official), *command], text=True).strip()
    if official_git('rev-parse', 'HEAD') != spec['official_commit'] or official_git('status', '--porcelain'):
        raise ValueError('require the frozen clean official TRELLIS.2 checkout')
    git = git_info()
    output = resolve_run_dir(args.output)
    if output.is_relative_to(ROOT) or output.is_relative_to(ASSETS):
        raise ValueError('require a new external run outside source/assets')
    output.mkdir(parents=True, exist_ok=False)
    segmentation = spec['segmentation']
    cache = ASSETS / 'cache'
    environment = {'HF_ENDPOINT': segmentation['endpoint'], 'HF_HOME': str(cache / 'huggingface'),
        'HF_HUB_CACHE': str(cache / 'huggingface'), 'HF_MODULES_CACHE': str(cache / 'huggingface_modules'),
        'TORCH_HOME': str(cache / 'torch'), 'TORCH_EXTENSIONS_DIR': str(cache / 'torch_extensions'),
        'TRITON_CACHE_DIR': str(output / 'triton_cache'), 'TMPDIR': '/var/tmp',
        'TMP': '/var/tmp', 'TEMP': '/var/tmp', 'PYTHONDONTWRITEBYTECODE': '1'}
    os.environ.update(environment)
    state = {'status': 'started', 'stage': 'download_segmentation', 'git': git,
        'official_commit': spec['official_commit'], 'inputs_sha256': sha256(SPEC),
        'segmentation': segmentation, 'environment': environment, 'subjects': [],
        'policy': 'Full original canvas; soft alpha only; no crop, threshold, morphology or hole filling. No 3D sampling or training.'}
    segmenter = None
    try:
        write(output / 'prepare_receipt.json', state)
        snapshot, state['model_files_sha256'] = download_bundle(segmentation)
        state['model_bundle_path'] = str(snapshot)
        state['stage'] = 'segment_complete_originals'
        write(output / 'prepare_receipt.json', state)
        sys.path.insert(0, str(official))
        import torch
        from PIL import Image
        from trellis2.pipelines.rembg import BiRefNet
        torch.cuda.set_device(args.device)
        segmenter = BiRefNet(str(snapshot))
        segmenter.cuda()
        panels = []
        for item, source in sources:
            directory = output / item['subject_id']
            directory.mkdir()
            copied = directory / source.name
            shutil.copyfile(source, copied)
            with Image.open(source) as opened:
                orientation = opened.getexif().get(274)
                if orientation not in (None, 1):
                    raise ValueError('unexpected source EXIF orientation')
                original = opened.convert('RGB')
            segmented = segmenter(original.copy())
            if segmented.mode != 'RGBA' or segmented.size != original.size:
                raise ValueError('official segmenter must preserve full original canvas')
            full_mask = directory / 'full_resolution_mask.png'
            segmented.getchannel('A').save(full_mask)
            cutout, transform = square_cutout(segmented)
            cutout_path, mask_path = directory / 'cutout512.png', directory / 'mask512.png'
            cutout.save(cutout_path)
            cutout.getchannel('A').save(mask_path)
            receipt = {'subject_id': item['subject_id'], 'source_path': str(source),
                'source_sha256': sha256(source), 'copied_source_path': str(copied),
                'copied_source_sha256': sha256(copied), 'cutout_path': str(cutout_path),
                'cutout_sha256': sha256(cutout_path), 'mask_path': str(mask_path),
                'mask_sha256': sha256(mask_path), 'full_resolution_mask_path': str(full_mask),
                'full_resolution_mask_sha256': sha256(full_mask), 'transform': transform,
                'exif_orientation': orientation, 'official_commit': spec['official_commit'],
                'segmentation': segmentation, 'alpha_matches_mask': True,
                'transparent_rgb_zero': True, 'soft_alpha': True, 'hole_filling': False}
            write(directory / 'source_receipt.json', receipt)
            panel = review_panel(cutout)
            panel.save(directory / 'review.png')
            panels.append(panel)
            state['subjects'].append(receipt)
            write(output / 'prepare_receipt.json', state)
        combined = Image.new('RGB', (1536, 536 * len(panels)), 'white')
        for index, panel in enumerate(panels):
            combined.paste(panel, (0, index * 536))
        combined.save(output / 'review.png')
        if git_info() != git or any(sha256(path) != item['source_sha256'] for item, path in sources):
            raise ValueError('source checkout changed during preparation')
        state.update(status='prepared_pending_visual_review', stage='completed')
    except BaseException as error:
        state.update(status='failed', error_type=type(error).__name__)
        raise
    finally:
        if segmenter is not None:
            segmenter.cpu()
            del segmenter
            gc.collect()
            torch.cuda.empty_cache()
        write(output / 'prepare_receipt.json', state)
    print(json.dumps({'status': state['status'], 'output': str(output)}))


if __name__ == '__main__':
    main()
