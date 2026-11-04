"""Audit saved TRELLIS attributes and renderer channels without model sampling."""

import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

from experiments.subject_trellis_counterfactual_v1.run import ASSETS, OFFICIAL
from experiments.trellis_supermat_compare_v1.compare_maps import display, sha256, write_json
from scripts.launch.colorpeel_run import git_info

KEYS = ('base_color', 'roughness', 'metallic', 'mask', 'alpha')
ATOL = 1e-4


def layout_record(layout):
    if set(layout) != {'base_color', 'metallic', 'roughness', 'alpha'}:
        raise ValueError('Require the four named PBR attributes')
    if any(not isinstance(value, slice) for value in layout.values()):
        raise ValueError('PBR attribute layout must use slices')
    return {name: [value.start, value.stop, value.step] for name, value in layout.items()}


def official_layout(official):
    """Read literal slice definitions, without constructing a pipeline/model."""
    definitions = []
    for path in sorted((Path(official) / 'trellis2/pipelines').glob('*.py')):
        for node in ast.walk(ast.parse(path.read_text(encoding='utf8'))):
            if not isinstance(node, ast.Assign) or not any(
                    isinstance(target, ast.Attribute) and target.attr == 'pbr_attr_layout' for target in node.targets):
                continue
            if not isinstance(node.value, ast.Dict):
                raise ValueError('Expected literal official pbr_attr_layout dictionary')
            layout = {}
            for key, value in zip(node.value.keys, node.value.values):
                if (not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name)
                        or value.func.id != 'slice' or value.keywords):
                    raise ValueError('Expected literal slice in official PBR layout')
                layout[ast.literal_eval(key)] = slice(*(ast.literal_eval(arg) for arg in value.args))
            definitions.append({'file': str(path), 'line': node.lineno, 'sha256': sha256(path),
                                'layout': layout_record(layout)})
    if not definitions or any(item['layout'] != definitions[0]['layout'] for item in definitions):
        raise ValueError('Missing or inconsistent official PBR attribute layouts')
    return definitions[0]['layout'], definitions


def statistics(array):
    array = np.asarray(array, dtype=np.float64).reshape(-1)
    valid = np.isfinite(array)
    values = array[valid]
    return {'count': int(array.size), 'finite_count': int(valid.sum()), 'nonfinite_count': int((~valid).sum()),
            'min': float(values.min()) if values.size else None, 'max': float(values.max()) if values.size else None,
            'mean': float(values.mean()) if values.size else None, 'std': float(values.std()) if values.size else None,
            'quantiles': dict(zip(('q00', 'q05', 'q25', 'q50', 'q75', 'q95', 'q100'),
                map(float, np.quantile(values, (0, .05, .25, .5, .75, .95, 1))))) if values.size else {},
            'below_zero': int((values < 0).sum()), 'above_one': int((values > 1).sum()),
            'near_zero_atol1e_minus4': int((np.abs(values) <= ATOL).sum()),
            'near_one_atol1e_minus4': int((np.abs(values - 1) <= ATOL).sum())}


def compare_values(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        raise ValueError(f'Comparison shape mismatch: {actual.shape} != {expected.shape}')
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return {'array_equal': bool(np.array_equal(actual, expected)), 'different_elements': int(np.count_nonzero(difference)),
            'max_abs_diff': float(difference.max()) if difference.size else None,
            'mean_abs_diff': float(difference.mean()) if difference.size else None,
            'atol': ATOL, 'rtol': 0,
            'passed': bool(difference.size and np.isfinite(difference).all() and np.all(difference <= ATOL))}


def audit(source_run, roi, output):
    source_run, roi, output = [Path(value).resolve() for value in (source_run, roi, output)]
    if output.exists() or output.is_relative_to(source_run) or source_run.is_relative_to(output):
        raise ValueError('Require a fresh diagnostic directory outside the source asset run')
    git = git_info()
    output.mkdir(parents=True)
    state = {'status': 'running', 'stage': 'preflight', 'git': git, 'source_run': str(source_run),
             'training': False, 'generated_assets': 0, 'atol': ATOL, 'rtol': 0, 'script_sha256': sha256(__file__)}
    try:
        official = ASSETS / 'sources/TRELLIS.2'
        revision = subprocess.check_output(['git', '-C', str(official), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', str(official), 'status', '--porcelain'], text=True).strip()
        if revision != OFFICIAL or dirty:
            raise ValueError('Require the unchanged pinned official TRELLIS checkout')
        expected_layout, definitions = official_layout(official)
        asset, review_path = source_run / 'native_asset.pt', source_run / 'review/review.json'
        old_npz, receipt_path = source_run / 'review/attributes.npz', source_run / 'review/raw_attributes.json'
        receipt, review = [json.loads(path.read_text()) for path in (receipt_path, review_path)]
        locks = {str(path): sha256(path) for path in (asset, review_path, old_npz, receipt_path, roi)}
        if (locks[str(asset)] != receipt['asset_sha256'] or locks[str(review_path)] != receipt['review_sha256']
                or locks[str(old_npz)] != receipt['npz_sha256'] or receipt['official_commit'] != OFFICIAL):
            raise ValueError('Saved native asset or old export differs from its receipt')
        comparison = json.loads((roi.parent / 'report.json').read_text())
        if (comparison['common_roi_sha256'] != locks[str(roi)]
                or comparison['source_locks'].get(str(old_npz)) != locks[str(old_npz)]):
            raise ValueError('ROI must belong to the original visible-attribute comparison')
        with Image.open(roi) as opened:
            if opened.size != (512, 512):
                raise ValueError('Require the original512 common ROI')
            region = np.asarray(opened.convert('L')) >= 128
        if not region.any():
            raise ValueError('Common ROI is empty')
        environment = Path(review['environment']['path'])
        locks[str(environment)] = sha256(environment)
        if locks[str(environment)] != review['environment']['sha256']:
            raise ValueError('Original HDRI changed')
        state.update(official_commit=OFFICIAL, official_layout_definitions=definitions, source_locks=locks,
                     roi_pixels=int(region.sum()), stage='native_attributes')
        os.environ.update(OPENCV_IO_ENABLE_OPENEXR='1', TORCH_EXTENSIONS_DIR=str(ASSETS / 'cache/torch_extensions'),
                          TRITON_CACHE_DIR=str(output / 'triton_cache'))
        sys.path.insert(0, str(official))
        import cv2
        import torch
        from trellis2.representations import MeshWithVoxel
        from trellis2.renderers import EnvMap, PbrMeshRenderer

        if torch.cuda.device_count() != 1:
            raise ValueError('Expose exactly one authorized CUDA GPU')
        saved = torch.load(asset, map_location='cpu', weights_only=False)
        state['native_layout'] = layout_record(saved['layout'])
        if state['native_layout'] != expected_layout or saved['attrs'].ndim != 2 or saved['attrs'].shape[-1] != 6:
            raise ValueError('Native six-channel layout disagrees with the official pipeline')
        original_attrs = saved['attrs'].clone()
        values = original_attrs.double().numpy()
        state['native_attributes'] = {'shape': list(values.shape), 'dtype': str(original_attrs.dtype),
            'channels': {str(index): statistics(values[:, index]) for index in range(6)},
            'named_attributes': {name: statistics(values[:, selection]) for name, selection in saved['layout'].items()}}
        write_json(output / 'native_channels.json', {'layout': state['native_layout'], **state['native_attributes']})
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite native attributes')
        arguments = {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in saved.items()}
        arguments['origin'] = saved['origin'].tolist()
        mesh = MeshWithVoxel(**arguments)
        extr, intr = [torch.tensor(review['camera'][key], device='cuda', dtype=torch.float32)
                      for key in ('extrinsics', 'intrinsics')]
        rgb = cv2.cvtColor(cv2.imread(str(environment), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        envmap = EnvMap(torch.tensor(rgb, device='cuda'))
        renderer = PbrMeshRenderer({'resolution': 512, 'near': .1, 'far': 100, 'ssaa': 1, 'peel_layers': 8})
        state.update(gpu=torch.cuda.get_device_name(0), camera=review['camera'], stage='baseline')

        def render(candidate, name):
            torch.manual_seed(42)
            with torch.no_grad():
                result = renderer.render(candidate, extr, intr, envmap, use_envmap_bg=False)
            arrays = {key: result[key].detach().cpu().numpy() for key in KEYS}
            if any(not np.isfinite(value).all() for value in arrays.values()):
                raise ValueError(f'{name}: nonfinite renderer output')
            np.savez_compressed(output / (name + '.npz'), **arrays)
            return arrays

        baseline = render(mesh, 'baseline')
        with np.load(old_npz, allow_pickle=False) as previous:
            state['baseline_comparison'] = {key: compare_values(baseline[key], previous[key]) for key in KEYS}
        state['baseline_roi'] = {key: statistics(baseline[key][region]) for key in ('roughness', 'metallic', 'alpha')}
        cases = [('baseline', baseline)]
        sentinels = ({'base_color': [.11, .22, .33], 'metallic': .7, 'roughness': .2, 'alpha': 1.0},
                     {'base_color': [.73, .41, .19], 'metallic': .15, 'roughness': .8, 'alpha': 1.0})
        state.update(stage='sentinels', sentinels=[])
        write_json(output / 'status.json', state)
        for index, target in enumerate(sentinels, 1):
            candidate = MeshWithVoxel(**{**arguments, 'attrs': mesh.attrs.clone()})
            if candidate.attrs.data_ptr() == mesh.attrs.data_ptr():
                raise ValueError('Sentinel attributes unexpectedly share native storage')
            for name, value in target.items():
                candidate.attrs[..., saved['layout'][name]] = torch.tensor(value, device='cuda', dtype=candidate.attrs.dtype)
            name = f'sentinel{index}'
            arrays = render(candidate, name)
            reliable = region & (arrays['alpha'] >= .9999)
            ratio_region = region & (arrays['alpha'] > 1e-8)
            checks, direct, ratios = {}, {}, {}
            for key, value in target.items():
                field = arrays[key].transpose(1, 2, 0) if key == 'base_color' else arrays[key]
                actual = field[reliable]
                checks[key] = {**compare_values(actual, np.broadcast_to(value, actual.shape)), 'values': statistics(actual)}
                actual = field[region]
                direct[key] = {**compare_values(actual, np.broadcast_to(value, actual.shape)), 'values': statistics(actual)}
                if key != 'alpha':
                    denominator = arrays['alpha'][ratio_region]
                    actual = field[ratio_region] / (denominator[:, None] if key == 'base_color' else denominator)
                    ratios[key] = {**compare_values(actual, np.broadcast_to(value, actual.shape)), 'values': statistics(actual)}
            checks['mask_unchanged'] = compare_values(arrays['mask'], baseline['mask'])
            support = {'original_roi_pixels': int(region.sum()), 'alpha_threshold': .9999,
                       'reliable_pixels': int(reliable.sum()), 'retention_fraction': float(reliable.sum() / region.sum()),
                       'minimum_retention': .95, 'ratio_valid_pixels': int(ratio_region.sum()),
                       'passed': bool(reliable.sum() >= .95 * region.sum()), 'alpha': statistics(arrays['alpha'][region])}
            state['sentinels'].append({'name': name, 'target': target, 'checks': checks, 'support': support,
                                      'original_roi_direct': direct, 'original_roi_divided_by_alpha': ratios})
            cases.append((name, arrays))
            if not torch.equal(mesh.attrs.detach().cpu(), original_attrs):
                raise ValueError('Native in-memory attributes changed during diagnostics')
            write_json(output / 'status.json', state)
            print('CHANNEL_AUDIT', name, {key: value['passed'] for key, value in checks.items()}, flush=True)
        canvas = Image.new('RGB', (2048, 550 * len(cases)), 'white')
        draw = ImageDraw.Draw(canvas)
        for row, (name, arrays) in enumerate(cases):
            for column, key in enumerate(('base_color', 'roughness', 'metallic', 'alpha')):
                value = arrays[key].transpose(1, 2, 0) if key == 'base_color' else arrays[key]
                draw.text((column * 512 + 8, row * 550 + 8), f'{name} / {key}', fill='black')
                canvas.paste(display(value, arrays['mask'] > .5), (column * 512, row * 550 + 32))
        canvas.save(output / 'channels.png')
        if any(sha256(path) != digest for path, digest in locks.items()):
            raise ValueError('Original source asset or diagnostic input changed')
        state['native_asset_unchanged'] = True
        passed = (all(item['passed'] for item in state['baseline_comparison'].values())
                  and all(case['support']['passed'] for case in state['sentinels'])
                  and all(item['passed'] for case in state['sentinels'] for item in case['checks'].values()))
        state.update(status='passed' if passed else 'failed', stage='completed',
                     output_sha256={path.name: sha256(path) for path in output.glob('*') if path.suffix in ('.npz', '.png')})
        if not passed:
            raise ValueError('Baseline or sentinel numeric validation failed; all diagnostics retained')
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(output / 'status.json', state)
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--roi', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.source_run, args.roi, args.output)
