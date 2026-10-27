"""Convert a pinned public timm DINOv3 checkpoint; require FP32 feature parity."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

from experiments.subject_trellis_counterfactual_v1.preflight import SOURCE, check_inputs, failure
from scripts.launch.colorpeel_run import git_info

REPO = 'timm/vit_large_patch16_dinov3.lvd1689m'
REVISION = '30c1109559f65dea34316b0d4842d35c5771fe11'
ASSETS = Path('/var/tmp/colorpeel-r12user5/assets/trellis2_v1')
ATOL, RTOL = 1e-3, 1e-3


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(-1).view(__import__('torch').uint8).numpy().tobytes()).hexdigest()


def map_state_dict(source, model):
    import torch
    converted, receipt = {}, {}
    direct = {'cls_token': 'embeddings.cls_token', 'reg_token': 'embeddings.register_tokens',
              'patch_embed.proj.weight': 'embeddings.patch_embeddings.weight',
              'patch_embed.proj.bias': 'embeddings.patch_embeddings.bias',
              'norm.weight': 'norm.weight', 'norm.bias': 'norm.bias'}
    for key, value in source.items():
        if key in direct:
            names, pieces = [direct[key]], [value]
        else:
            parts = key.split('.')
            if len(parts) < 3 or parts[0] != 'blocks' or not parts[1].isdigit():
                raise ValueError('unknown source tensor')
            prefix, suffix = f'layer.{parts[1]}.', '.'.join(parts[2:])
            if suffix == 'attn.qkv.weight':
                names = [prefix + f'attention.{name}_proj.weight' for name in ('q', 'k', 'v')]
                pieces = list(value.chunk(3, dim=0))
            else:
                renamed = suffix.replace('attn.proj.', 'attention.o_proj.')
                renamed = renamed.replace('mlp.fc1.', 'mlp.up_proj.').replace('mlp.fc2.', 'mlp.down_proj.')
                renamed = {'gamma_1': 'layer_scale1.lambda1', 'gamma_2': 'layer_scale2.lambda1'}.get(renamed, renamed)
                if renamed == suffix and not suffix.startswith(('norm1.', 'norm2.')):
                    raise ValueError('unknown block tensor')
                names, pieces = [prefix + renamed], [value]
        reconstructed = torch.cat(pieces, dim=0) if len(pieces) > 1 else pieces[0]
        if tensor_sha(reconstructed) != tensor_sha(value):
            raise ValueError('non-bitwise tensor mapping')
        for name, piece in zip(names, pieces):
            if name in converted:
                raise ValueError('duplicate destination tensor')
            converted[name] = piece
        receipt[key] = {'sha256': tensor_sha(value), 'dtype': str(value.dtype),
                        'shape': list(value.shape), 'destinations': names, 'bitwise_reassembly': True}
    expected = model.state_dict()
    generated = ['embeddings.mask_token'] + [f'layer.{i}.attention.{name}_proj.bias'
        for i in range(24) for name in ('q', 'v')]
    for name in generated:
        if name in converted:
            raise ValueError('unexpected source bias or mask token')
        converted[name] = torch.zeros_like(expected[name])
    model.load_state_dict(converted, strict=True)
    if any(tensor_sha(model.state_dict()[name]) != tensor_sha(value) for name, value in converted.items()):
        raise ValueError('load changed tensor bytes or dtype')
    return receipt, {name: tensor_sha(converted[name]) for name in generated}


def pink_input(output, resolution):
    """Pinned TRELLIS.2 RGBA branch: alpha bbox, square crop, black composite."""
    import numpy as np
    import torch
    from PIL import Image
    with Image.open(SOURCE / 'source/pink/cutout512.png') as image:
        positions = np.argwhere(np.array(image)[:, :, 3] > .8 * 255)
        left, top = positions[:, 1].min(), positions[:, 0].min()
        right, bottom = positions[:, 1].max(), positions[:, 0].max()
        cx, cy, size = (left + right) / 2, (top + bottom) / 2, int(max(right - left, bottom - top))
        crop = image.crop((cx - size // 2, cy - size // 2, cx + size // 2, cy + size // 2))
        rgba = np.array(crop).astype(np.float32) / 255
    rgb = Image.fromarray((rgba[:, :, :3] * rgba[:, :, 3:4] * 255).astype(np.uint8))
    rgb = rgb.resize((resolution, resolution), Image.Resampling.LANCZOS)
    rgb.save(output / f'parity_input{resolution}.png')
    pixels = torch.from_numpy(np.array(rgb).astype(np.float32) / 255).permute(2, 0, 1)[None]
    return (pixels - torch.tensor([.485, .456, .406])[None, :, None, None]) / torch.tensor([.229, .224, .225])[None, :, None, None]


def feature_check(source, model, pixels, device):
    import timm
    import torch
    import torch.nn.functional as functional
    reference = timm.create_model('vit_large_patch16_dinov3', pretrained=False).eval()
    reference.load_state_dict(source, strict=True)
    reference, model = reference.to(device), model.to(device).eval()
    captured = []
    hook = reference.norm.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    with torch.inference_mode():
        pixels = pixels.to(device)
        reference.forward_features(pixels)
        hook.remove()
        original = captured[0]
        hidden = model.embeddings(pixels, bool_masked_pos=None)
        position = model.rope_embeddings(pixels)
        for layer in model.layer:
            hidden = layer(hidden, position_embeddings=position)
        pairs = {'prenorm': (original, hidden), 'trellis_features': (
            functional.layer_norm(original, original.shape[-1:]), functional.layer_norm(hidden, hidden.shape[-1:]))}
        results = {}
        for name, (a, b) in pairs.items():
            difference = (a - b).abs()
            results[name] = {'max_error': difference.max().item(), 'mean_error': difference.mean().item(),
                'allclose': bool(torch.allclose(a, b, atol=ATOL, rtol=RTOL)),
                'finite': bool(torch.isfinite(a).all() and torch.isfinite(b).all()), 'shape': list(a.shape)}
    return {'atol': ATOL, 'rtol': RTOL, 'dtype': 'float32', 'resolution': pixels.shape[-1],
            'rope': 'both regenerated in float32; no BF16 period truncation', 'comparisons': results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if not output.is_relative_to(ASSETS.resolve()) or output == ASSETS.resolve():
        raise ValueError('output must be a new directory below the external TRELLIS assets root')
    git = git_info()
    output.mkdir(parents=True, exist_ok=False)
    attempt = output / 'attempt'
    attempt.mkdir()
    state = {'status': 'started', 'git': git, 'source_repo': REPO, 'source_revision': REVISION,
             'original_gated_hf_weights_compared': False, 'generation_verified': False}
    (output / 'conversion_receipt.json').write_text(json.dumps(state, indent=2) + '\n', encoding='utf8')
    try:
        state['stage'] = 'dependencies_and_inputs'
        state['versions'] = {name: importlib.metadata.version(name) for name in ('torch', 'timm', 'transformers', 'safetensors')}
        if state['versions']['transformers'] != '4.57.3' or state['versions']['timm'] != '1.0.26':
            raise ValueError('require reviewed Transformers/timm versions')
        state['input'] = check_inputs()
        import torch
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import DINOv3ViTConfig, DINOv3ViTModel
        os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        state['stage'] = 'public_checkpoint_download'
        path = hf_hub_download(REPO, 'model.safetensors', revision=REVISION,
                               cache_dir=str(output / 'cache'), token=False)
        state['source_file_sha256'] = sha256(path)
        source = load_file(path, device='cpu')
        config = DINOv3ViTConfig(patch_size=16, hidden_size=1024, intermediate_size=4096,
            num_hidden_layers=24, num_attention_heads=16, num_register_tokens=4,
            use_gated_mlp=False, hidden_act='gelu', layer_norm_eps=1e-5, rope_theta=100.,
            query_bias=True, key_bias=False, value_bias=True, proj_bias=True, mlp_bias=True)
        config._attn_implementation = 'sdpa'
        model = DINOv3ViTModel(config).eval()
        state['stage'] = 'strict_mapping'
        state['tensor_mapping'], state['synthesized_zero_tensors'] = map_state_dict(source, model)
        state['stage'] = 'fp32_feature_parity'
        state['parity'] = {str(resolution): feature_check(source, model,
            pink_input(attempt, resolution), args.device) for resolution in (512, 1024)}
        state['parity_input_sha256'] = {str(resolution): sha256(attempt / f'parity_input{resolution}.png')
            for resolution in (512, 1024)}
        state['parity_acceptance'] = 'TRELLIS consumes affine-free normalized features; prenorm is a finite-only diagnostic'
        if not all(result['comparisons']['trellis_features']['allclose']
                   and all(row['finite'] for row in result['comparisons'].values())
                   for result in state['parity'].values()):
            raise ValueError('feature parity threshold failed')
        state['stage'] = 'save_verified_conversion'
        candidate = attempt / 'model'
        model.cpu().save_pretrained(candidate, safe_serialization=True)
        state['model_files_sha256'] = {path.name: sha256(path) for path in candidate.iterdir() if path.is_file()}
        candidate.rename(output / 'model')
        state['status'] = 'converted_parity_verified_not_original_hf_byte_verified'
    except Exception as error:
        state.update(status='blocked', error=failure(error))
    (output / 'conversion_receipt.json').write_text(json.dumps(state, indent=2) + '\n', encoding='utf8')
    print(json.dumps({'status': state['status'], 'output': str(output)}))
    return 0 if state['status'].startswith('converted_') else 2


if __name__ == '__main__':
    raise SystemExit(main())
