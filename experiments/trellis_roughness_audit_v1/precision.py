"""One fixed-seed precision diagnostic, with exact-latent decoder comparison; no training."""

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.subject_trellis_counterfactual_v1.run import ASSETS, OFFICIAL, write
from experiments.subject_trellis_counterfactual_v1.convert_dino import sha256
from scripts.launch.colorpeel_run import git_info, resolve_run_dir


def statistics(value):
    import torch
    value = value.detach().float().cpu()
    if not torch.isfinite(value).all():
        raise ValueError('nonfinite intermediate tensor')
    return {'shape': list(value.shape), 'min': float(value.min()), 'max': float(value.max()),
            'mean': float(value.mean()), 'std': float(value.std()),
            'median': float(value.median())}


def channels(value):
    return {str(i): statistics(value[:, i]) for i in range(value.shape[1])}


def run(source_run, output):
    source_run, output = Path(source_run).resolve(), resolve_run_dir(Path(output))
    if output.exists() or output.is_relative_to(source_run) or output.is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError('require fresh external diagnostic output')
    official = ASSETS / 'sources/TRELLIS.2'
    if subprocess.check_output(['git', '-C', str(official), 'rev-parse', 'HEAD'], text=True).strip() != OFFICIAL:
        raise ValueError('official source revision mismatch')
    if subprocess.check_output(['git', '-C', str(official), 'status', '--porcelain'], text=True).strip():
        raise ValueError('official source is dirty')
    source_status = json.loads((source_run / 'repro_outputs/status.json').read_text())
    args = json.loads((source_run / 'pipeline.json').read_text())['args']
    condition_path = source_run / 'actual_condition.png'
    output.mkdir(parents=True)
    state = {'status': 'running', 'git': git_info(), 'training': False, 'source_run': str(source_run),
             'official_commit': OFFICIAL, 'condition_sha256': sha256(condition_path),
             'source_asset_sha256': sha256(source_run / 'native_asset.pt'),
             'flow_dtype': 'float32', 'seed': 42, 'pipeline_type': '1024_cascade',
             'policy': 'fixed original condition, samplers, weights and seed; change flow compute dtype only; then compare decoder dtypes on identical latents'}
    write(output / 'status.json', state)
    os.environ.update(ATTN_BACKEND='xformers', SPARSE_ATTN_BACKEND='xformers',
        TORCH_EXTENSIONS_DIR=str(ASSETS / 'cache/torch_extensions'),
        TRITON_CACHE_DIR=str(output / 'triton_cache'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    sys.path.insert(0, str(official))
    import torch
    from PIL import Image
    from trellis2.pipelines.base import Pipeline
    from trellis2.pipelines import Trellis2ImageTo3DPipeline, samplers
    from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor
    from trellis2.modules.sparse.conv import config as conv_config
    conv_config.FLEX_GEMM_ALGO = 'explicit_gemm'
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        state['stage'] = 'load_same_models'
        write(output / 'status.json', state)
        base = Pipeline.from_pretrained(str(source_run))
        for name, model in base.models.items():
            if 'flow_model' in name:
                model.convert_to(torch.float32)
                if model.dtype != torch.float32:
                    raise ValueError('flow precision conversion failed')
        kwargs = {}
        for name in ('sparse_structure', 'shape_slat', 'tex_slat'):
            spec = args[name + '_sampler']
            kwargs[name + '_sampler'] = getattr(samplers, spec['name'])(**spec['args'])
            kwargs[name + '_sampler_params'] = spec['params']
        pipe = Trellis2ImageTo3DPipeline(models=base.models, **kwargs,
            shape_slat_normalization=args['shape_slat_normalization'],
            tex_slat_normalization=args['tex_slat_normalization'],
            image_cond_model=DinoV3FeatureExtractor(source_status['encoder']), rembg_model=None,
            low_vram=True, default_pipeline_type='1024_cascade')
        pipe.cuda()
        captured = {}
        original_decode = pipe.decode_tex_slat

        def trace_decode(slat, subs):
            captured.update(slat=slat, subs=subs)
            state['texture_latent_denormalized'] = statistics(slat.feats)
            std = slat.feats.new_tensor(args['tex_slat_normalization']['std'])
            mean = slat.feats.new_tensor(args['tex_slat_normalization']['mean'])
            normalized = (slat.feats - mean) / std
            state['texture_latent_normalized'] = statistics(normalized)
            state['normalization_roundtrip_max_error'] = float((normalized * std + mean - slat.feats).abs().max())
            state['tex_normalization'] = args['tex_slat_normalization']
            result = original_decode(slat, subs)
            captured['decoded_fp16'] = result.feats.detach().float().cpu()
            captured['decoded_coords'] = result.coords.detach().cpu()
            state['decoder_fp16_physical_attributes'] = channels(result.feats)
            write(output / 'status.json', state)
            return result

        pipe.decode_tex_slat = trace_decode
        state['stage'] = 'fp32_flow_same_seed_generation'
        write(output / 'status.json', state)
        with Image.open(condition_path) as image:
            with torch.inference_mode():
                mesh = pipe.run(image.copy(), seed=42, num_samples=1, preprocess_image=False,
                                pipeline_type='1024_cascade')[0]
        state['generated_attributes'] = channels(mesh.attrs)
        torch.save({key: getattr(mesh, key).detach().cpu() for key in
                    ('vertices', 'faces', 'coords', 'attrs', 'origin')} | {
                    'voxel_size': mesh.voxel_size, 'voxel_shape': mesh.voxel_shape, 'layout': mesh.layout},
                   output / 'fp32_flow_native_asset.pt')
        baseline = torch.load(source_run / 'native_asset.pt', map_location='cpu', weights_only=False)
        state['baseline_attributes'] = channels(baseline['attrs'])
        del baseline, mesh
        gc.collect()
        torch.cuda.empty_cache()
        state['stage'] = 'same_latents_fp32_decoder'
        write(output / 'status.json', state)
        decoder = pipe.models['tex_slat_decoder']
        decoder.convert_to_fp32()
        decoder.dtype = torch.float32  # Official helper converts parameters but does not update this field.
        decoder.cuda()
        with torch.inference_mode():
            decoded = decoder(captured['slat'], guide_subs=captured['subs'])
        physical = decoded.feats.detach().float().cpu() * .5 + .5
        if not torch.equal(decoded.coords.cpu(), captured['decoded_coords']):
            raise ValueError('decoder comparison coordinates changed')
        state['decoder_fp32_physical_attributes'] = channels(physical)
        state['decoder_dtype_absolute_difference'] = channels((physical - captured['decoded_fp16']).abs())
        state['coordinates_equal'] = True
        if sha256(source_run / 'native_asset.pt') != state['source_asset_sha256']:
            raise ValueError('original asset changed')
        state.update(status='complete', stage='completed')
    except BaseException as error:
        state.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write(output / 'status.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    run(options.source_run, options.output)
