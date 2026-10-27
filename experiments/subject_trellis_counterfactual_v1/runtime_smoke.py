"""Exercise the actual V100 attention, sparse convolution and raster backends."""

import argparse
import json
import os
from pathlib import Path
import sys

from scripts.launch.colorpeel_run import git_info, resolve_run_dir


def main(output):
    output = resolve_run_dir(output)
    git = git_info()
    output.mkdir(parents=True, exist_ok=False)
    assets = Path('/var/tmp/colorpeel-r12user5/assets/trellis2_v1')
    os.environ.update(ATTN_BACKEND='xformers', SPARSE_ATTN_BACKEND='xformers',
        TRITON_CACHE_DIR=str(output / 'triton_cache'),
        TORCH_EXTENSIONS_DIR=str(assets / 'cache/torch_extensions'))
    sys.path.insert(0, str(assets / 'sources/TRELLIS.2'))
    state = {'status': 'started', 'git': git, 'generation_verified': False}
    try:
        import torch
        import torch.nn.functional as functional
        import xformers.ops
        from trellis2.modules.sparse import SparseTensor, SparseConv3d
        import nvdiffrast.torch as dr
        import cumesh
        torch.manual_seed(42)
        with torch.inference_mode():
            q, k, v = [torch.randn(1, 64, 12, 128, device='cuda', dtype=torch.float16) for _ in range(3)]
            actual = xformers.ops.memory_efficient_attention(q, k, v)
            expected = functional.scaled_dot_product_attention(*[x.transpose(1, 2).float() for x in (q, k, v)]).transpose(1, 2)
            state['xformers'] = bool(torch.allclose(actual.float(), expected, atol=.01, rtol=.01))
            coords = torch.tensor([[0, x, y, z] for x in range(2) for y in range(2) for z in range(2)],
                                  dtype=torch.int32, device='cuda')
            sparse = SparseTensor(feats=torch.full((8, 32), .125, device='cuda', dtype=torch.float16), coords=coords)
            conv = SparseConv3d(32, 32, 3).cuda().half()
            conv.weight.fill_(.25)
            conv.bias.zero_()
            values = conv(sparse).feats
            state['sparse_convolution'] = bool(torch.isfinite(values).all() and
                torch.allclose(values.float(), torch.full_like(values.float(), 8), atol=.01, rtol=.01))
            vertices = torch.tensor([[-.5, -.5, 0, 1], [.5, -.5, 0, 1], [0, .5, 0, 1]], device='cuda')
            faces = torch.tensor([[0, 1, 2]], device='cuda', dtype=torch.int32)
            raster, _ = dr.rasterize(dr.RasterizeCudaContext(), vertices[None], faces, (32, 32))
            state['rasterizer'] = bool(torch.isfinite(raster).all() and (raster[..., 3] > 0).any())
            mesh = cumesh.CuMesh()
            mesh.init(vertices[:, :3].contiguous(), faces)
            mesh.compute_vertex_normals()
            state['cumesh'] = bool(torch.isfinite(mesh.read_vertex_normals()).all())
            torch.cuda.synchronize()
        state['status'] = 'runtime_kernels_verified_generation_unverified' if all(
            state[name] for name in ('xformers', 'sparse_convolution', 'rasterizer', 'cumesh')) else 'failed'
    except Exception as error:
        state.update(status='failed', error_type=type(error).__name__, error=str(error))
    (output / 'runtime_smoke.json').write_text(json.dumps(state, indent=2) + '\n')
    print(json.dumps(state, indent=2))
    return 0 if state['status'].startswith('runtime_kernels_verified') else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    raise SystemExit(main(parser.parse_args().output))
