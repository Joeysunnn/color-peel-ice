"""Save unquantized visible TRELLIS A/R/M at the already fitted camera."""

import argparse
import json
import os
from pathlib import Path
import sys

from experiments.subject_trellis_counterfactual_v1.run import ASSETS, OFFICIAL, write
from experiments.subject_trellis_counterfactual_v1.convert_dino import sha256
from scripts.launch.colorpeel_run import git_info


def export(run):
    import subprocess
    run = Path(run).resolve()
    output = run / 'review/attributes.npz'
    if output.exists():
        raise FileExistsError(output)
    git = git_info()
    official = ASSETS / 'sources/TRELLIS.2'
    if subprocess.check_output(['git', '-C', str(official), 'rev-parse', 'HEAD'], text=True).strip() != OFFICIAL:
        raise ValueError('official revision changed')
    os.environ.update(OPENCV_IO_ENABLE_OPENEXR='1',
        TORCH_EXTENSIONS_DIR=str(ASSETS / 'cache/torch_extensions'),
        TRITON_CACHE_DIR=str(run / 'triton_cache'))
    sys.path.insert(0, str(official))
    import cv2
    import numpy as np
    import torch
    from trellis2.representations import MeshWithVoxel
    from trellis2.renderers import EnvMap, PbrMeshRenderer
    asset = run / 'native_asset.pt'
    saved = torch.load(asset, map_location='cuda', weights_only=False)
    saved['origin'] = saved['origin'].tolist()
    mesh = MeshWithVoxel(**saved)
    review_path = run / 'review/review.json'
    review = json.loads(review_path.read_text())
    extr, intr = [torch.tensor(review['camera'][key], device='cuda', dtype=torch.float32)
                  for key in ('extrinsics', 'intrinsics')]
    environment = Path(review['environment']['path'])
    if sha256(environment) != review['environment']['sha256']:
        raise ValueError('review illumination changed')
    env = cv2.cvtColor(cv2.imread(str(environment), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
    renderer = PbrMeshRenderer({'resolution': 512, 'near': .1, 'far': 100, 'ssaa': 1, 'peel_layers': 8})
    torch.manual_seed(42)
    with torch.no_grad():
        result = renderer.render(mesh, extr, intr, EnvMap(torch.tensor(env, device='cuda')), use_envmap_bg=False)
    arrays = {key: result[key].detach().cpu().numpy() for key in ('base_color', 'roughness', 'metallic', 'mask', 'alpha')}
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError('nonfinite visible attributes')
    np.savez_compressed(output, **arrays)
    write(run / 'review/raw_attributes.json', {'git': git, 'asset_sha256': sha256(asset),
        'review_sha256': sha256(review_path), 'npz_sha256': sha256(output),
        'official_commit': OFFICIAL, 'quantization': 'none',
        'base_color': 'native display-domain output before renderer pow2.2; no extra gamma',
        'roughness_metallic': 'raw linear parameters; retain out-of-range diagnostics',
        'camera': 'unchanged estimated original-canvas review camera', 'training': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    export(parser.parse_args().run)
