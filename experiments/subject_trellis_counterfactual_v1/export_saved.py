"""Export a smaller preview LOD of the existing pink asset; no model sampling."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.subject_trellis_counterfactual_v1.convert_dino import sha256
from scripts.launch.colorpeel_run import git_info, resolve_run_dir

ASSETS = Path('/var/tmp/colorpeel-r12user5/assets/trellis2_v1')
OFFICIAL = '75fbf0183001ed9876c8dbb35de6b68552ee08bd'
TRUSTED = Path('/var/tmp/colorpeel-r12user5/runs/subject_trellis_counterfactual_v1/'
               '20261027-110302__pink_asset__e0ca373__42/native_asset.pt')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    asset, output = args.asset.resolve(), resolve_run_dir(args.output)
    if asset != TRUSTED.resolve():
        raise ValueError('only the trusted task-generated native asset may be unpickled')
    if output.is_relative_to(Path(__file__).resolve().parents[2]) or output.is_relative_to(asset.parent):
        raise ValueError('require a fresh external output outside the native result')
    git = git_info()
    output.mkdir(parents=True, exist_ok=False)
    state = {'status': 'started', 'stage': 'load_native_asset', 'git': git,
        'asset': str(asset), 'asset_sha256': sha256(asset), 'new_model_samples': 0,
        'export_settings': {'decimation_target': 100000, 'texture_size': 2048, 'remesh': False},
        'interpretation': 'Preview LOD simplifies geometry and bakes textures. Native asset is unchanged; native review used pre-export geometry.'}
    receipt = output / 'export_receipt.json'

    def record():
        receipt.write_text(json.dumps(state, indent=2, allow_nan=False) + '\n', encoding='utf8')

    record()
    try:
        official = ASSETS / 'sources/TRELLIS.2'
        if subprocess.check_output(['git', '-C', str(official), 'rev-parse', 'HEAD'], text=True).strip() != OFFICIAL:
            raise ValueError('official source revision mismatch')
        os.environ.update(TORCH_EXTENSIONS_DIR=str(ASSETS / 'cache/torch_extensions'),
                          TRITON_CACHE_DIR=str(output / 'triton_cache'))
        sys.path.insert(0, str(official))
        import torch
        import o_voxel
        from trellis2.representations import MeshWithVoxel
        saved = torch.load(asset, map_location='cuda', weights_only=False)
        saved['origin'] = saved['origin'].tolist()
        mesh = MeshWithVoxel(**saved)
        state['native'] = {'vertices': len(mesh.vertices), 'faces': len(mesh.faces),
            'attributes': {name: {'finite': bool(torch.isfinite(mesh.attrs[..., mesh.layout[name]]).all()),
                'min': mesh.attrs[..., mesh.layout[name]].min().item(),
                'max': mesh.attrs[..., mesh.layout[name]].max().item()} for name in ('metallic', 'roughness')}}
        state['stage'] = 'official_glb_export'
        record()
        started = time.time()
        glb = o_voxel.postprocess.to_glb(vertices=mesh.vertices, faces=mesh.faces,
            attr_volume=mesh.attrs, coords=mesh.coords, attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size, aabb=[[-.5, -.5, -.5], [.5, .5, .5]],
            decimation_target=100000, texture_size=2048, remesh=False, verbose=True)
        state['stage'] = 'save_glb'
        record()
        path = output / 'mailbox_preview_lod.glb'
        glb.export(path)
        material = glb.visual.material
        state['preview'] = {'path': str(path), 'sha256': sha256(path),
            'vertices': len(glb.vertices), 'faces': len(glb.faces), 'alpha_mode': material.alphaMode,
            'textures': {name: {'size': list(getattr(material, name).size),
                'mode': getattr(material, name).mode} for name in ('baseColorTexture', 'metallicRoughnessTexture')},
            'export_seconds': time.time() - started}
        if sha256(asset) != state['asset_sha256'] or git_info() != git:
            raise ValueError('native input or project worktree changed')
        state.update(status='preview_lod_exported_pending_review', stage='completed', native_asset_unchanged=True)
    except BaseException as error:
        state.update(status='failed', error_type=type(error).__name__)
        raise
    finally:
        record()
    print(json.dumps({'status': state['status'], 'output': str(output)}))


if __name__ == '__main__':
    main()
