"""Compare saved reconstruction inputs/configuration and native tensors on CPU."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.launch.colorpeel_run import git_info, resolve_run_dir

ROOT = Path(__file__).resolve().parents[2]


def array_comparison(reference, candidate):
    same_shape = reference.shape == candidate.shape
    equal = same_shape and reference.dtype == candidate.dtype and np.array_equal(reference, candidate)
    result = {'reference_shape': list(reference.shape), 'candidate_shape': list(candidate.shape),
              'reference_dtype': str(reference.dtype), 'candidate_dtype': str(candidate.dtype),
              'reference_finite': bool(np.isfinite(reference).all()),
              'candidate_finite': bool(np.isfinite(candidate).all()), 'exactly_equal': bool(equal)}
    if same_shape:
        difference = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
        result.update(max_absolute_difference=float(difference.max()) if difference.size else 0.0,
                      mean_absolute_difference=float(difference.mean()) if difference.size else 0.0)
    return result


def plain(value):
    if isinstance(value, slice):
        return {'slice': [value.start, value.stop, value.step]}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    return value


def compare(reference, candidate, output):
    reference, candidate = Path(reference).resolve(), Path(candidate).resolve()
    output = resolve_run_dir(Path(output))
    if output.exists() or any(output.is_relative_to(root) for root in (ROOT, reference, candidate)):
        raise ValueError('comparison output must be fresh and outside code/input directories')
    revision = git_info()
    states = [json.loads((root / 'repro_outputs/status.json').read_text()) for root in (reference, candidate)]
    configs = [json.loads((root / 'pipeline.json').read_text())['args'] for root in (reference, candidate)]
    config_equal = {key: states[0][key] == states[1][key] for key in
        ('official_commit', 'seed', 'pipeline_type', 'encoder_receipt_sha256', 'sampling',
         'flow_dtype', 'sparse_algorithm', 'models')}
    config_equal.update({key: configs[0][key] == configs[1][key] for key in
                        ('shape_slat_normalization', 'tex_slat_normalization')})
    inputs = {}
    for name in ('source_cutout512.png', 'actual_condition.png'):
        with Image.open(reference / name) as a, Image.open(candidate / name) as b:
            inputs[name] = {'reference_size': list(a.size), 'candidate_size': list(b.size),
                'reference_mode': a.mode, 'candidate_mode': b.mode,
                'decoded_pixels_equal': a.mode == b.mode and a.size == b.size and a.tobytes() == b.tobytes()}
    import torch
    assets = [torch.load(root / 'native_asset.pt', map_location='cpu', weights_only=False)
              for root in (reference, candidate)]
    tensors = {name: array_comparison(assets[0][name].numpy(), assets[1][name].numpy())
               for name in ('vertices', 'faces', 'coords', 'attrs', 'origin')}
    metadata = {name: {'reference': plain(assets[0][name]), 'candidate': plain(assets[1][name]),
                      'equal': plain(assets[0][name]) == plain(assets[1][name])}
                for name in ('voxel_size', 'voxel_shape', 'layout')}
    inputs_match = all(config_equal.values()) and all(row['decoded_pixels_equal'] for row in inputs.values())
    mesh_match = inputs_match and all(tensors[key]['exactly_equal'] for key in ('vertices', 'faces', 'origin'))
    mesh_match = mesh_match and all(row['equal'] for row in metadata.values())
    native_match = mesh_match and all(row['exactly_equal'] for row in tensors.values())
    result = {'status': 'compared', 'comparison_git': revision, 'reference': str(reference),
        'candidate': str(candidate), 'reference_generation_git': states[0]['git'],
        'candidate_generation_git': states[1]['git'], 'configuration_equal': config_equal,
        'inputs': inputs, 'tensors': tensors, 'metadata': metadata,
        'input_and_configuration_reproduced': inputs_match, 'mesh_exactly_reproduced': mesh_match,
        'native_output_exactly_reproduced': native_match,
        'asset_file_sha256': {str(root): hashlib.sha256((root / 'native_asset.pt').read_bytes()).hexdigest()
                              for root in (reference, candidate)},
        'policy': 'Original tensor order, no alignment, tolerance relaxation or resampling; file hash is provenance only.',
        'training_started': False, 'quality_accepted_by_user': False}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: result[key] for key in
                     ('input_and_configuration_reproduced', 'mesh_exactly_reproduced', 'native_output_exactly_reproduced')}))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    arguments = parser.parse_args()
    compare(arguments.reference, arguments.candidate, arguments.output)
