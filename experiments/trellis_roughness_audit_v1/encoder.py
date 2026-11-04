"""Check the saved orange conditioning image through the actual reloaded DINO extractor."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.subject_trellis_counterfactual_v1.convert_dino import (
    ASSETS, ATOL, REPO, REVISION, RTOL, sha256, tensor_sha,
)
from experiments.subject_trellis_counterfactual_v1.run import OFFICIAL, write
from scripts.launch.colorpeel_run import git_info


def verify_inputs(source_run):
    """Bind the saved model and conditioning image to the original run receipts."""
    status_path = source_run / 'repro_outputs/status.json'
    status = json.loads(status_path.read_text())
    if status['official_commit'] != OFFICIAL or status['generated_assets'] != 1:
        raise ValueError('require one existing asset from the pinned official code')
    encoder = Path(status['encoder']).resolve()
    if encoder != (ASSETS / 'dino_timm_v2/model').resolve():
        raise ValueError('require the reviewed dino_timm_v2 model directory')
    receipt_path = encoder.parent / 'conversion_receipt.json'
    if sha256(receipt_path) != status['encoder_receipt_sha256']:
        raise ValueError('conversion receipt differs from the generation run')
    receipt = json.loads(receipt_path.read_text())
    if (receipt['status'] != 'converted_parity_verified_not_original_hf_byte_verified'
            or receipt['source_repo'] != REPO or receipt['source_revision'] != REVISION):
        raise ValueError('encoder conversion receipt is not the reviewed public checkpoint')
    expected = receipt['model_files_sha256']
    if set(expected) != {path.name for path in encoder.iterdir() if path.is_file()}:
        raise ValueError('converted model file set changed')
    locks = {str(status_path): sha256(status_path), str(receipt_path): sha256(receipt_path)}
    for name, digest in expected.items():
        if Path(name).name != name or sha256(encoder / name) != digest:
            raise ValueError('saved converted model hash mismatch')
        locks[str(encoder / name)] = digest
    maps_path = source_run.parent / 'maps/report.json'
    maps = json.loads(maps_path.read_text())
    condition = source_run / 'actual_condition.png'
    condition_hash = maps['source_locks'].get(str(condition))
    if condition_hash is None or sha256(condition) != condition_hash:
        raise ValueError('actual conditioning image differs from the reviewed map comparison')
    locks.update({str(condition): condition_hash, str(maps_path): sha256(maps_path)})
    return encoder, condition, receipt, locks


def compare(a, b):
    import torch
    a, b = a.detach().float().cpu(), b.detach().float().cpu()
    if a.shape != b.shape:
        raise ValueError('feature shapes differ')
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    difference = (a - b).abs()
    return {'shape': list(a.shape), 'finite': finite,
            'allclose': finite and bool(torch.allclose(a, b, atol=ATOL, rtol=RTOL)),
            'max_error': difference.max().item() if finite else None,
            'mean_error': difference.mean().item() if finite else None}


def check_resolution(extractor, reference, image, resolution, device):
    import torch
    import torch.nn.functional as functional
    actual_pixels, actual_prenorm, reference_prenorm = [], [], []
    hooks = [
        extractor.model.embeddings.register_forward_pre_hook(
            lambda module, args: actual_pixels.append(args[0].detach().clone())),
        extractor.model.layer[-1].register_forward_hook(
            lambda module, args, result: actual_prenorm.append(result.detach())),
        reference.norm.register_forward_pre_hook(
            lambda module, args: reference_prenorm.append(args[0].detach())),
    ]
    extractor.image_size = resolution
    try:
        with torch.inference_mode():
            # This executes the official PIL resize, RGB conversion, normalization,
            # embeddings, rotary positions, layer loop and affine-free layer norm.
            actual = extractor([image])
            if len(actual_pixels) != 1 or len(actual_prenorm) != 1:
                raise ValueError('unexpected extractor hook invocation count')
            pixels = actual_pixels[0]
            if pixels.shape != (1, 3, resolution, resolution) or pixels.dtype != torch.float32:
                raise ValueError('unexpected actual extractor pixel shape or dtype')
            reference.forward_features(pixels.to(device))
            if len(reference_prenorm) != 1:
                raise ValueError('unexpected timm norm hook invocation count')
            original = reference_prenorm[0]
            normalized = functional.layer_norm(original, original.shape[-1:])
            expected_shape = (1, (resolution // 16) ** 2 + 5, 1024)
            if tuple(actual.shape) != expected_shape:
                raise ValueError('unexpected CLS/register/patch token layout')
            return {
                'resolution': resolution, 'dtype': str(actual.dtype),
                'pixels_sha256': tensor_sha(pixels),
                'pixels_finite': bool(torch.isfinite(pixels).all()),
                'pixel_source': 'captured at actual extractor embeddings; identical tensor passed to timm',
                'comparisons': {
                    'trellis_features': compare(actual, normalized),
                    'prenorm_diagnostic': compare(actual_prenorm[0], original),
                },
            }
    finally:
        for hook in hooks:
            hook.remove()


def run(source_run, output, device):
    source_run, output = Path(source_run).resolve(), Path(output).resolve()
    if output.is_relative_to(source_run) or output.is_relative_to(ASSETS.resolve()):
        raise ValueError('diagnostic output must be outside immutable source and asset directories')
    output.mkdir(parents=True, exist_ok=False)
    state = {'schema': 'trellis_encoder_audit/v1', 'status': 'started', 'git': git_info(),
             'source_run': str(source_run), 'device': device, 'official_commit': OFFICIAL,
             'source_repo': REPO, 'source_revision': REVISION, 'atol': ATOL, 'rtol': RTOL,
             'training': False, 'generation': False, 'downloads': False,
             'original_gated_hf_weights_compared': False,
             'acceptance': 'Both 512/1024 actual affine-free extractor outputs allclose; all captured tensors finite. Prenorm allclose is diagnostic only.'}
    write(output / 'report.json', state)
    try:
        encoder, condition, receipt, locks = verify_inputs(source_run)
        state.update(stage='inputs_verified', encoder=str(encoder), input_sha256=locks)
        state['versions'] = {name: importlib.metadata.version(name)
            for name in ('torch', 'timm', 'transformers', 'safetensors')}
        if state['versions']['transformers'] != '4.57.3' or state['versions']['timm'] != '1.0.26':
            raise ValueError('require the reviewed Transformers/timm versions')
        official = ASSETS / 'sources/TRELLIS.2'
        if subprocess.check_output(['git', '-C', str(official), 'rev-parse', 'HEAD'], text=True).strip() != OFFICIAL:
            raise ValueError('official checkout revision mismatch')
        if subprocess.check_output(['git', '-C', str(official), 'status', '--porcelain'], text=True).strip():
            raise ValueError('official checkout is dirty')
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                          HF_HUB_DISABLE_PROGRESS_BARS='1', PYTHONDONTWRITEBYTECODE='1',
                          ATTN_BACKEND='xformers', SPARSE_ATTN_BACKEND='xformers')
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(official))
        import timm
        import torch
        from huggingface_hub import hf_hub_download
        from PIL import Image
        from safetensors.torch import load_file
        from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor
        selected = torch.device(device)
        if selected.type != 'cuda' or not torch.cuda.is_available():
            raise ValueError('the actual official extractor requires CUDA')
        torch.cuda.set_device(selected)  # Official __call__ uses .cuda() without a device argument.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        torch.manual_seed(42)
        checkpoint = Path(hf_hub_download(REPO, 'model.safetensors', revision=REVISION,
            cache_dir=str(ASSETS / 'dino_timm_v2/cache'), local_files_only=True, token=False))
        if sha256(checkpoint) != receipt['source_file_sha256']:
            raise ValueError('cached public timm checkpoint differs from conversion receipt')
        locks[str(checkpoint)] = receipt['source_file_sha256']
        state.update(stage='reload_models', reference_checkpoint=str(checkpoint))
        write(output / 'report.json', state)
        # This reloads the saved converted directory through the production class,
        # rather than comparing only the in-memory model used before save_pretrained.
        extractor = DinoV3FeatureExtractor(str(encoder))
        extractor.to(selected)
        if next(extractor.model.parameters()).dtype != torch.float32:
            raise ValueError('converted extractor must retain the verified FP32 dtype')
        reference = timm.create_model('vit_large_patch16_dinov3', pretrained=False).eval()
        source = load_file(str(checkpoint), device='cpu')
        reference.load_state_dict(source, strict=True)
        del source
        reference.to(selected)
        with Image.open(condition) as opened:
            image = opened.copy()
        if image.mode != 'RGB':
            raise ValueError('saved official conditioning image must already be RGB')
        state.update(stage='compare_actual_extractor', condition_size=list(image.size), comparisons={})
        for resolution in (512, 1024):
            state['comparisons'][str(resolution)] = check_resolution(
                extractor, reference, image, resolution, selected)
            write(output / 'report.json', state)
        if any(sha256(path) != digest for path, digest in locks.items()):
            raise ValueError('a locked source file changed during the audit')
        passed = all(row['pixels_finite'] and row['comparisons']['trellis_features']['allclose']
            and all(result['finite'] for result in row['comparisons'].values())
            for row in state['comparisons'].values())
        state.update(status='passed' if passed else 'failed', stage='completed')
    except Exception as error:
        state.update(status='failed', error_type=type(error).__name__, error=str(error))
    write(output / 'report.json', state)
    print(json.dumps({'status': state['status'], 'output': str(output)}))
    return 0 if state['status'] == 'passed' else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True,
                        help='Existing TRELLIS directory containing actual_condition.png and repro_outputs/')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv)
    return run(args.source_run, args.output, args.device)


if __name__ == '__main__':
    raise SystemExit(main())
