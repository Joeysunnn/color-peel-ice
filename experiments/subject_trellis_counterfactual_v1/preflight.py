"""Read-only TRELLIS.2 environment/input probes; no installation or generation."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shlex
import sys
from pathlib import Path

from scripts.launch.colorpeel_run import git_info, resolve_run_dir

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path('/var/tmp/colorpeel-r12user5/runs/subject_marble_foreground_v1/'
              '20261026-131331__foreground30_plus5__7bc1a86__42')
MASK_SHA = '550c1e13bee1435e1145f06e44bdef02d75b8b8fdfa7462d3b39f456dfc9a978'
PINK_SHA = 'e263e3f1dd319426baa6cb8f72ff56495230962eea40659497d1cf49b665c7c1'
MANIFEST_SHA = '3c695a13444587cf85e9e57ff2eedb78c44d235e61fe523b847e90dc505a2353'
RECEIPT_SHA = '0c35d4043f5878d8449f65025ed018a9b33bc2f484dac376bd155a81c63b0874'
MODULES = ('torch', 'triton', 'flex_gemm', 'cumesh', 'nvdiffrast', 'nvdiffrec_render',
           'o_voxel', 'utils3d', 'transformers', 'huggingface_hub', 'PIL')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def failure(error):
    # Never serialize exception text, URLs, HTTP headers, or credential values.
    cause = error
    while getattr(cause, '__cause__', None) is not None:
        cause = cause.__cause__
    response = getattr(cause, 'response', None)
    return {'status': 'failed', 'exception_type': type(error).__name__,
            'cause_type': type(cause).__name__,
            'http_status': getattr(response, 'status_code', None)}


def check_inputs():
    from PIL import Image
    image_path, mask_path = SOURCE / 'source/pink/cutout512.png', SOURCE / 'source/object_mask512.png'
    manifest_path = SOURCE / 'dataset/manifest.json'
    expected = {image_path: PINK_SHA, mask_path: MASK_SHA, manifest_path: MANIFEST_SHA,
                SOURCE / 'prepare_receipt.json': RECEIPT_SHA}
    hashes = {str(path): sha256(path) for path in expected}
    if any(hashes[str(path)] != value for path, value in expected.items()):
        raise ValueError('frozen input hash mismatch')
    manifest = json.loads(manifest_path.read_text())
    originals = [row for row in manifest['records'] if row['id'] == 'pink__original']
    if len(originals) != 1 or len(manifest['records']) != 35:
        raise ValueError('unexpected foreground manifest')
    row = originals[0]
    for kind in ('image', 'mask'):
        path = (SOURCE / 'dataset' / row[kind]).resolve()
        if not path.is_relative_to((SOURCE / 'dataset').resolve()):
            raise ValueError('manifest path outside frozen dataset')
        if sha256(path) != row[kind + '_sha256']:
            raise ValueError('manifest asset hash mismatch')
    with Image.open(image_path) as image, Image.open(mask_path) as mask:
        if image.mode != 'RGBA' or image.size != (512, 512) or mask.size != image.size:
            raise ValueError('require aligned RGBA512 and mask')
        alpha = image.getchannel('A')
        if alpha.tobytes() != mask.convert('L').tobytes():
            raise ValueError('alpha does not match frozen mask')
        if any(a == 0 and (r or g or b) for r, g, b, a in image.getdata()):
            raise ValueError('transparent RGB must be zero')
    return {'status': 'passed', 'sha256': hashes, 'alpha_matches_mask': True,
            'transparent_rgb_zero': True, 'manifest_lock': 'independent frozen SHA256; pink asset rows verified'}


def _matmul_kernel(a, b, c):
    r = tl.arange(0, 32)
    x = tl.load(a + r[:, None] * 32 + r[None, :])
    y = tl.load(b + r[:, None] * 32 + r[None, :])
    tl.store(c + r[:, None] * 32 + r[None, :], tl.dot(x, y))


def check_gpu(device):
    import torch
    import torch.nn.functional as functional
    selected = torch.device(device)
    if selected.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('CUDA device required')
    with torch.cuda.device(selected):
        prop = torch.cuda.get_device_properties(selected)
        result = {'status': 'partial', 'device': str(selected), 'name': prop.name,
                  'memory_bytes': prop.total_memory, 'capability': [prop.major, prop.minor],
                  'torch': torch.__version__, 'torch_cuda': torch.version.cuda}
        try:
            result['native_bfloat16'] = torch.cuda.is_bf16_supported(including_emulation=False)
        except Exception as error:
            result['native_bfloat16'] = failure(error)
        try:
            torch.manual_seed(42)
            q, k, v = [torch.randn(1, 2, 16, 32, device=selected, dtype=torch.float16) for _ in range(3)]
            actual = functional.scaled_dot_product_attention(q, k, v)
            expected = (q.float() @ k.float().transpose(-1, -2) / (32 ** .5)).softmax(-1) @ v.float()
            torch.cuda.synchronize(selected)
            result['fp16_sdpa'] = {'status': 'passed' if torch.allclose(actual.float(), expected, atol=.01, rtol=.01) else 'failed',
                                   'finite': bool(torch.isfinite(actual).all())}
        except Exception as error:
            result['fp16_sdpa'] = failure(error)
        try:
            import triton
            import triton.language as language
            globals()['tl'] = language
            a, b = [torch.randn(32, 32, device=selected, dtype=torch.float16) for _ in range(2)]
            c = torch.empty_like(a)
            triton.jit(_matmul_kernel)[(1,)](a, b, c)
            torch.cuda.synchronize(selected)
            expected = a.float() @ b.float()
            result['fp16_triton_matmul'] = {'status': 'passed' if torch.allclose(c.float(), expected, atol=.02, rtol=.02) else 'failed',
                                            'finite': bool(torch.isfinite(c).all())}
        except Exception as error:
            result['fp16_triton_matmul'] = failure(error)
        if all(result[name]['status'] == 'passed' for name in ('fp16_sdpa', 'fp16_triton_matmul')):
            result['status'] = 'passed'
        return result


def check_access(output):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download('facebook/dinov3-vitl16-pretrain-lvd1689m', 'config.json',
                           cache_dir=str(output / 'access_cache'), force_download=True)
    return {'status': 'passed', 'config_sha256': sha256(path), 'weights_downloaded': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--check-access', action='store_true')
    args = parser.parse_args(argv)
    output = resolve_run_dir(args.output)
    if output.is_relative_to(ROOT) or output.is_relative_to(SOURCE) or SOURCE.is_relative_to(output):
        raise ValueError('require new external output outside frozen inputs')
    git = git_info()
    output.mkdir(parents=True, exist_ok=False)
    os.environ['TRITON_CACHE_DIR'] = str(output / 'triton_cache')
    state = {'git': git, 'python': sys.version, 'platform': platform.platform(),
             'device': args.device, 'modules': {}, 'generation_verified': False}
    distribution_map = importlib.metadata.packages_distributions()
    for name in MODULES:
        present = importlib.util.find_spec(name) is not None
        distributions = distribution_map.get(name, []) if present else []
        state['modules'][name] = {'present': present, 'versions': {
            distribution: importlib.metadata.version(distribution) for distribution in distributions}}
    state['backend_optional'] = {'xformers': {'present': importlib.util.find_spec('xformers') is not None}}
    for name, probe in (('inputs', check_inputs), ('gpu', lambda: check_gpu(args.device))):
        try:
            state[name] = probe()
        except Exception as error:
            state[name] = failure(error)
    state['access'] = {'status': 'not_checked', 'reason': 'requires --check-access'}
    if args.check_access:
        try:
            state['access'] = check_access(output)
        except Exception as error:
            state['access'] = {**failure(error), 'reason': 'official DINO config access unavailable; no credentials logged'}
    incomplete = [name for name, value in state['modules'].items() if not value['present']]
    state['runtime_adaptation_required'] = (['default BF16 flow precision requires explicit validated adaptation']
        if state['gpu'].get('native_bfloat16') is False else [])
    blocked = state['inputs']['status'] != 'passed' or state['access']['status'] == 'failed'
    partial = bool(incomplete or state['runtime_adaptation_required']) or state['gpu']['status'] != 'passed' or state['access']['status'] != 'passed'
    status = 'blocked' if blocked else 'partial' if partial else 'preflight_passed_generation_unverified'
    state.update(status=status, missing_modules=incomplete)
    report = output / 'repro_outputs'
    report.mkdir()
    (output / 'environment.json').write_text(json.dumps(state, indent=2) + '\n', encoding='utf8')
    (report / 'status.json').write_text(json.dumps({'status': status, 'generation_verified': False}, indent=2) + '\n', encoding='utf8')
    documents = {
        'SUMMARY.md': f"Status: {status}. Missing modules: {incomplete}. Runtime adaptation required: {state['runtime_adaptation_required']}. Module presence is not execution compatibility. No installation or generation.\n",
        'COMMANDS.md': shlex.join([sys.executable, '-B', '-m', 'experiments.subject_trellis_counterfactual_v1.preflight', *sys.argv[1:]]) + '\n',
        'LOG.md': 'See environment.json for per-probe results and sanitized exception types/status codes. Failed runs are retained.\n',
        'SCIENTIFIC_CHANGELOG.md': 'Only FP16 SDPA and 32x32 Triton matmul diagnostics. No model loaded, dtype adapted, image generated, training, or old input changed.\n',
        'COMPARABILITY_REPORT.md': 'Environment completeness and basic GPU probes do not establish TRELLIS.2 compatibility, FlexGEMM/CuMesh execution, model stability, identity preservation, or PBR accuracy. Default BF16 requires separate V100 runtime adaptation verification.\n',
    }
    for name, value in documents.items():
        (report / name).write_text(value, encoding='utf8')
    print(json.dumps({'status': status, 'output': str(output)}))
    return 2 if blocked else 1 if partial else 0


if __name__ == '__main__':
    raise SystemExit(main())
