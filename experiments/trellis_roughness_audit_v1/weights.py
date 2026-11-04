"""Audit existing TRELLIS weights on CPU; no downloads, inference or training."""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from experiments.subject_trellis_counterfactual_v1.convert_dino import sha256
from experiments.subject_trellis_counterfactual_v1.run import ASSETS, OFFICIAL

ROOT = Path(__file__).resolve().parents[2]
MODELS = {
    'sparse_structure_decoder', 'sparse_structure_flow_model', 'shape_slat_decoder',
    'shape_slat_flow_model_512', 'shape_slat_flow_model_1024',
    'tex_slat_decoder', 'tex_slat_flow_model_1024',
}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf8')


def file_record(path, expected):
    path = Path(path)
    digest = sha256(path)
    resolved = path.resolve()
    blob = resolved.name if resolved.parent.name == 'blobs' else None
    record = {'path': str(path), 'resolved_path': str(resolved), 'is_symlink': path.is_symlink(),
              'sha256': digest, 'recorded_sha256': expected, 'recorded_sha256_matches': digest == expected,
              'hf_cache_blob_name': blob, 'hf_cache_blob_digest_matches': None}
    if blob and re.fullmatch('[0-9a-f]{64}', blob):
        record.update(hf_cache_blob_digest_kind='lfs_sha256', hf_cache_blob_digest_matches=blob == digest)
    elif blob and re.fullmatch('[0-9a-f]{40}', blob):
        data = path.read_bytes()
        git_digest = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        record.update(hf_cache_blob_digest_kind='git_blob_sha1', hf_cache_blob_digest_matches=blob == git_digest)
    return record


def head_statistics(state):
    import torch
    weight, bias = state['output_layer.weight'], state['output_layer.bias']
    if weight.shape[0] != 6 or bias.shape != (6,):
        raise ValueError('Texture decoder output head must have six channels')
    result = []
    for index, name in enumerate(('red', 'green', 'blue', 'metallic', 'roughness', 'alpha')):
        row = weight[index].float()
        finite = bool(torch.isfinite(row).all() and torch.isfinite(bias[index]))
        result.append({'channel': index, 'name': name, 'finite': finite,
                       **({'weight_l2': row.norm().item(), 'weight_std': row.std(unbiased=False).item(),
                           'weight_min': row.min().item(), 'weight_max': row.max().item(),
                           'bias': bias[index].float().item()} if finite else {})})
    return result


def audit_model(name, stem, models):
    import torch
    record = {'name': name, 'checkpoint_stem': str(stem), 'status': 'running', 'load_calls': []}
    captured = {}
    original_load = torch.nn.Module.load_state_dict

    def capture(module, state, *args, **kwargs):
        captured.update(module=module, checkpoint=state)
        current = module.state_dict()
        call = {'strict': kwargs.get('strict', args[0] if args else True),
                'missing_keys': sorted(current.keys() - state.keys()),
                'unexpected_keys': sorted(state.keys() - current.keys()),
                'shape_mismatches': [key for key in state.keys() & current.keys()
                                     if state[key].shape != current[key].shape]}
        record['load_calls'].append(call)
        result = original_load(module, state, *args, **kwargs)
        call.update(missing_keys=list(result.missing_keys), unexpected_keys=list(result.unexpected_keys))
        return result

    model = None
    try:
        torch.nn.Module.load_state_dict = capture
        try:
            with torch.device('cpu'):
                model = models.from_pretrained(str(stem))
        finally:
            torch.nn.Module.load_state_dict = original_load
        if len(record['load_calls']) != 1:
            raise ValueError('Expected exactly one official checkpoint load')
        checkpoint, actual = captured['checkpoint'], model.state_dict()
        if any(value.device.type != 'cpu' for value in actual.values()):
            raise ValueError('Weight audit must stay on CPU')
        tensors = {}
        for key, value in checkpoint.items():
            loaded = actual.get(key)
            row = {'checkpoint_shape': list(value.shape), 'checkpoint_dtype': str(value.dtype),
                   'checkpoint_finite': bool(torch.isfinite(value).all()), 'present_in_model': loaded is not None}
            if loaded is not None:
                row.update(model_shape=list(loaded.shape), model_dtype=str(loaded.dtype),
                           model_finite=bool(torch.isfinite(loaded).all()),
                           exact_after_dtype_cast=torch.equal(loaded, value.to(loaded.dtype)))
            tensors[key] = row
        record['tensors'] = tensors
        record['model_extra_keys'] = sorted(actual.keys() - checkpoint.keys())
        if name == 'tex_slat_decoder':
            record['checkpoint_output_head'] = head_statistics(checkpoint)
            record['loaded_output_head'] = head_statistics(actual)
        if 'flow_model' in name:
            initial_dtypes = {key: value.dtype for key, value in actual.items()}
            model.convert_to(torch.float16)
            converted = model.state_dict()
            conversion = {}
            for key, value in converted.items():
                row = {'before_dtype': str(initial_dtypes[key]), 'after_dtype': str(value.dtype),
                       'bf16_converted_to_fp16': initial_dtypes[key] != torch.bfloat16 or value.dtype == torch.float16,
                       'nonfinite_count': int((~torch.isfinite(value)).sum())}
                if key in checkpoint:
                    before = checkpoint[key].to(initial_dtypes[key])
                    expected = before.to(value.dtype)
                    row.update(exact_after_dtype_cast=torch.equal(value, expected),
                               cast_overflow_count=int((torch.isfinite(before) & ~torch.isfinite(expected)).sum()))
                conversion[key] = row
            record['flow_conversion'] = {'model_dtype': str(model.dtype), 'tensors': conversion}
        keys_ok = all(not row['shape_mismatches'] and not row.get('missing_keys')
                      and not row.get('unexpected_keys') for row in record['load_calls'])
        tensors_ok = all(row['checkpoint_finite'] and row.get('model_finite', False)
                         and row.get('exact_after_dtype_cast', False) for row in tensors.values())
        casts_ok = all(row['nonfinite_count'] == 0 and row['bf16_converted_to_fp16']
                       and row.get('exact_after_dtype_cast', False)
                       and row.get('cast_overflow_count', 0) == 0
                       for row in record.get('flow_conversion', {}).get('tensors', {}).values())
        if 'flow_model' in name:
            casts_ok = casts_ok and model.dtype == torch.float16
        record['status'] = 'passed' if keys_ok and tensors_ok and casts_ok else 'failed'
    except Exception as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        torch.nn.Module.load_state_dict = original_load
        captured.clear()
        model = None
    return record


def audit(source_run, output):
    source_run, output = Path(source_run).resolve(), Path(output).resolve()
    if (output.exists() or output.is_relative_to(ROOT) or output.is_relative_to(source_run)
            or source_run.is_relative_to(output)):
        raise ValueError('Use a fresh output outside the checkout and source run')
    output.mkdir(parents=True)
    report = {'status': 'running', 'source_run': str(source_run), 'training': False,
              'inference': False, 'downloads': False, 'files': [], 'models': [],
              'blob_check_scope': 'Existing local HF cache blob names, not independent remote LFS metadata'}
    status_path = output / 'status.json'
    write(status_path, report)
    try:
        source_status_path = source_run / 'repro_outputs/status.json'
        pipeline_path = source_run / 'pipeline.json'
        source_status = json.loads(source_status_path.read_text(encoding='utf8'))
        args = json.loads(pipeline_path.read_text(encoding='utf8'))['args']
        report['source_files'] = {str(path): sha256(path) for path in (source_status_path, pipeline_path)}
        report['model_revisions'] = source_status['models']['revisions']
        if source_status['official_commit'] != OFFICIAL or set(args['models']) != MODELS:
            raise ValueError('Require the recorded official seven-model1024 cascade')
        official = ASSETS / 'sources/TRELLIS.2'
        commit = subprocess.check_output(['git', '-C', str(official), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', str(official), 'status', '--porcelain'], text=True).strip()
        if commit != OFFICIAL or dirty:
            raise ValueError('Require clean pinned official TRELLIS source')
        report['official_commit'] = commit
        for path, expected in source_status['models']['sha256'].items():
            result = file_record(path, expected)
            report['files'].append(result)
            write(status_path, report)
            print(f"HASH {len(report['files'])} {Path(path).name}", flush=True)
        if any(not row['recorded_sha256_matches'] or row['hf_cache_blob_digest_matches'] is False
               for row in report['files']):
            raise ValueError('Recorded model hash or HF cache blob digest differs')
        locked = source_status['models']['sha256']
        for stem in args['models'].values():
            if not Path(stem).is_absolute() or any(str(stem) + suffix not in locked
                                                  for suffix in ('.json', '.safetensors')):
                raise ValueError('Model stem must reference two recorded absolute local files')
        os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                          ATTN_BACKEND='xformers', SPARSE_ATTN_BACKEND='xformers')
        sys.path.insert(0, str(official))
        import torch
        from trellis2 import models
        report['torch_version'] = torch.__version__
        torch.set_num_threads(4)
        for name, stem in args['models'].items():
            report['current_model'] = name
            write(status_path, report)
            print(f'MODEL_START {name}', flush=True)
            record = audit_model(name, stem, models)
            model_path = output / (name + '.json')
            write(model_path, record)
            report['models'].append({'name': name, 'status': record['status'], 'report': str(model_path),
                                     'report_sha256': sha256(model_path)})
            report.pop('current_model', None)
            write(status_path, report)
            print(f"MODEL_DONE {name} {record['status']}", flush=True)
            del record
            gc.collect()
        if any(sha256(path) != digest for path, digest in report['source_files'].items()):
            raise ValueError('Source run metadata changed during audit')
        report['status'] = 'passed' if all(row['status'] == 'passed' for row in report['models']) else 'failed'
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        write(status_path, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    result = audit(options.source_run, options.output)
    print(json.dumps({'status': result['status'], 'models_checked': len(result['models'])}), flush=True)
    raise SystemExit(0 if result['status'] == 'passed' else 1)
