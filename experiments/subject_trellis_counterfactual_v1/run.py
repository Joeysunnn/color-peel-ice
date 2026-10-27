"""Generate exactly one native pink Mailbox asset for review, never train."""

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time

from experiments.subject_trellis_counterfactual_v1.preflight import SOURCE, check_inputs
from experiments.subject_trellis_counterfactual_v1.convert_dino import sha256
from scripts.launch.colorpeel_run import git_info, resolve_run_dir

ASSETS = Path('/var/tmp/colorpeel-r12user5/assets/trellis2_v1')
OFFICIAL = '75fbf0183001ed9876c8dbb35de6b68552ee08bd'


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf8')


def model_bundle(output):
    from huggingface_hub import HfApi, hf_hub_download
    api, revisions = HfApi(token=False), {}
    repo = 'microsoft/TRELLIS.2-4B'
    revisions[repo] = api.model_info(repo).sha
    cache = ASSETS / 'cache/huggingface'
    path = hf_hub_download(repo, 'pipeline.json', revision=revisions[repo], cache_dir=cache, token=False)
    args = json.loads(Path(path).read_text())['args']
    # The official1024 cascade does not use the512 texture flow.
    args['models'].pop('tex_slat_flow_model_512')
    hashes = {str(path): sha256(path)}
    for key, value in list(args['models'].items()):
        if value.startswith('microsoft/'):
            owner, name, *parts = value.split('/')
            source, stem = owner + '/' + name, '/'.join(parts)
        else:
            source, stem = repo, value
        if source not in revisions:
            revisions[source] = api.model_info(source).sha
        for suffix in ('.json', '.safetensors'):
            file = hf_hub_download(source, stem + suffix, revision=revisions[source], cache_dir=cache, token=False)
            hashes[str(file)] = sha256(file)
            if suffix == '.json':
                args['models'][key] = str(Path(file).with_suffix(''))
    # Public API assembly below supplies the verified local encoder and no rembg.
    write(output / 'pipeline.json', {'args': args})
    return args, {'revisions': revisions, 'sha256': hashes}


def run(output, encoder):
    output = resolve_run_dir(output)
    if output.is_relative_to(Path(__file__).resolve().parents[2]) or output.is_relative_to(SOURCE):
        raise ValueError('require fresh external output')
    git, inputs = git_info(), check_inputs()
    receipt_path = encoder.parent / 'conversion_receipt.json'
    receipt = json.loads(receipt_path.read_text())
    if receipt['status'] != 'converted_parity_verified_not_original_hf_byte_verified':
        raise ValueError('encoder conversion and feature parity must pass first')
    for name, digest in receipt['model_files_sha256'].items():
        if sha256(encoder / name) != digest:
            raise ValueError('verified encoder changed')
    output.mkdir(parents=True, exist_ok=False)
    report = output / 'repro_outputs'
    report.mkdir()
    state = {'status': 'started', 'git': git, 'official_commit': OFFICIAL, 'seed': 42,
        'pipeline_type': '1024_cascade', 'input': inputs, 'encoder': str(encoder),
        'encoder_receipt_sha256': sha256(receipt_path), 'training': False,
        'counterfactual_expansion': False, 'generated_assets': 0}
    os.environ.update(ATTN_BACKEND='xformers', SPARSE_ATTN_BACKEND='xformers',
        PYTHONDONTWRITEBYTECODE='1', TORCH_EXTENSIONS_DIR=str(ASSETS / 'cache/torch_extensions'),
        TRITON_CACHE_DIR=str(output / 'triton_cache'), OPENCV_IO_ENABLE_OPENEXR='1')
    official = ASSETS / 'sources/TRELLIS.2'
    import subprocess
    if subprocess.check_output(['git', '-C', str(official), 'rev-parse', 'HEAD'], text=True).strip() != OFFICIAL:
        raise ValueError('official checkout revision mismatch')
    sys.path.insert(0, str(official))
    write(report / 'status.json', state)
    try:
        import torch
        from PIL import Image
        from trellis2.pipelines.base import Pipeline
        from trellis2.pipelines import Trellis2ImageTo3DPipeline
        from trellis2.pipelines import samplers
        from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor
        state['stage'] = 'prepare_pinned_model_files'
        write(report / 'status.json', state)
        args, state['models'] = model_bundle(output)
        base = Pipeline.from_pretrained(str(output))
        flows = {}
        for name, model in base.models.items():
            if 'flow_model' in name:
                model.convert_to(torch.float16)
                flows[name] = str(model.dtype)
        state['flow_dtype'] = flows
        kwargs = {}
        for name in ('sparse_structure', 'shape_slat', 'tex_slat'):
            spec = args[name + '_sampler']
            kwargs[name + '_sampler'] = getattr(samplers, spec['name'])(**spec['args'])
            kwargs[name + '_sampler_params'] = spec['params']
        pipe = Trellis2ImageTo3DPipeline(models=base.models, **kwargs,
            shape_slat_normalization=args['shape_slat_normalization'],
            tex_slat_normalization=args['tex_slat_normalization'],
            image_cond_model=DinoV3FeatureExtractor(str(encoder)), rembg_model=None,
            low_vram=True, default_pipeline_type='1024_cascade')
        pipe.cuda()
        shutil.copyfile(SOURCE / 'source/pink/cutout512.png', output / 'source_cutout512.png')
        with Image.open(output / 'source_cutout512.png') as source:
            condition = pipe.preprocess_image(source)
        condition.save(output / 'actual_condition.png')
        state.update(stage='single_asset_generation', sampling={name: args[name + '_sampler']['params']
            for name in ('sparse_structure', 'shape_slat', 'tex_slat')})
        write(report / 'status.json', state)
        started = time.time()
        meshes = pipe.run(condition, seed=42, num_samples=1, preprocess_image=False,
                          pipeline_type='1024_cascade')
        if len(meshes) != 1:
            raise ValueError('expected exactly one asset')
        mesh = meshes[0]
        if not all(torch.isfinite(value).all() for value in (mesh.vertices, mesh.attrs)):
            raise ValueError('non-finite generated geometry or PBR attributes')
        torch.save({key: getattr(mesh, key).detach().cpu() for key in
            ('vertices', 'faces', 'coords', 'attrs', 'origin')} | {
            'voxel_size': mesh.voxel_size, 'voxel_shape': mesh.voxel_shape, 'layout': mesh.layout}, output / 'native_asset.pt')
        state.update(generated_assets=1, generation_seconds=time.time() - started,
                     vertices=len(mesh.vertices), faces=len(mesh.faces), stage='review_rendering')
        write(report / 'status.json', state)
        del pipe, base, meshes
        gc.collect()
        torch.cuda.empty_cache()
        from experiments.subject_trellis_counterfactual_v1.review_asset import review
        state['review'] = review(mesh, output / 'source_cutout512.png', output / 'review', official)
        import o_voxel
        state['stage'] = 'glb_export'
        write(report / 'status.json', state)
        glb = o_voxel.postprocess.to_glb(vertices=mesh.vertices, faces=mesh.faces,
            attr_volume=mesh.attrs, coords=mesh.coords, attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size, aabb=[[-.5, -.5, -.5], [.5, .5, .5]],
            decimation_target=1000000, texture_size=2048, remesh=False, verbose=True)
        glb.export(output / 'mailbox.glb')
        state['export'] = {'path': str(output / 'mailbox.glb'),
            'sha256': sha256(output / 'mailbox.glb'),
            'policy': 'official cleanup/simplification and UV baking; native_asset.pt and review retain pre-export native mesh'}
        state.update(status='generated_pending_user_review', stage='completed')
        if git_info() != git or check_inputs() != inputs:
            raise ValueError('code or immutable source changed')
    except BaseException as error:
        state.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write(report / 'status.json', state)
        (report / 'SUMMARY.md').write_text(f"Status: {state['status']}; native assets: {state['generated_assets']}. Pending identity/material review.\n")
        (report / 'COMMANDS.md').write_text(' '.join(sys.argv) + '\n')
        (report / 'LOG.md').write_text('See status.json and the captured run log. Failed outputs are retained; no automatic retry.\n')
        (report / 'SCIENTIFIC_CHANGELOG.md').write_text('Public same-model timm weights converted with feature parity, official flow torso converted BF16 to FP16 through convert_to, xformers attention, unchanged official sampler settings. Existing alpha supplies segmentation; unused rembg is not loaded. No Albedo substitution, data expansion or training.\n')
        (report / 'COMPARABILITY_REPORT.md').write_text('This is adapted V100 inference, not an exact default BF16 reproduction. Camera is estimated from one view. Native Base Color/R/M are generated estimates, not intrinsic or BRDF truth. No gated-original checkpoint byte comparison was possible.\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--encoder', required=True, type=Path)
    args = parser.parse_args()
    run(args.output, args.encoder.resolve())
