# TRELLIS.2: one foreground pink Mailbox asset

This study first checks the research server, then targets one seed42 asset
from the existing pink foreground original. The requested review consists
of an estimated source-view PBR render, clay render, Base Color, Metallic,
and Roughness. No counterfactual expansion, Subject training, parameter
sweep, or monitoring is authorized by this entrypoint.

The single asset and native review completed on the V100 adaptation. A separate
100k-face-target GLB export also completed. See [RESULTS.md](RESULTS.md) for
receipts, the interrupted first export, and unresolved identity/material issues.

## Frozen sources and official implementation

- Official repository: https://github.com/microsoft/TRELLIS.2
- Reviewed commit: `75fbf0183001ed9876c8dbb35de6b68552ee08bd`.
- Official weights: `microsoft/TRELLIS.2-4B`.
- Required image encoder: `facebook/dinov3-vitl16-pretrain-lvd1689m`.
- Existing foreground run:
  `/var/tmp/colorpeel-r12user5/runs/subject_marble_foreground_v1/20261026-131331__foreground30_plus5__7bc1a86__42`.
- Input: `source/pink/cutout512.png`, SHA256
  `e263e3f1dd319426baa6cb8f72ff56495230962eea40659497d1cf49b665c7c1`.
- Mask: `source/object_mask512.png`, SHA256
  `550c1e13bee1435e1145f06e44bdef02d75b8b8fdfa7462d3b39f456dfc9a978`.

The input is the foreground **original**, not a MARBLE-generated candidate.
It retains the original512 canvas with zero hidden RGB in transparent pixels.
Official preprocessing subsequently crops and resizes it; the source image
does not supply calibrated camera parameters. A future source-view render
must report camera fitting, not claim exact recovered camera or geometry.

## Environment/access preflight

Develop and check locally, commit and push, then pull the same clean commit
in an independent server checkout. The following command writes a new
external audit directory and performs only small GPU/access probes:

```bash
export PYTHONDONTWRITEBYTECODE=1 TMPDIR=/var/tmp
export COLORPEEL_OUTPUT_ROOT=/var/tmp/colorpeel-r12user5/runs
export CUDA_VISIBLE_DEVICES=1
PY=/home/r12user5/miniforge3/envs/univlr/bin/python
$PY -B -m experiments.subject_trellis_counterfactual_v1.preflight \
  --output "$NEW_AUDIT_RUN" --device cuda:0 --check-access
```

Select the physical GPU after checking current occupancy. Existing
environments are inspected without package installation or modification.
Only the small official DINOv3 config is requested with existing Hugging Face
credentials; credentials and request headers are never written to receipts.
The check does not request model access or accept license terms on the user's
behalf. A refused request is a blocker, not evidence that TRELLIS.2 ran.

Output contains `environment.json` and a `repro_outputs/` bundle distinguishing
hardware observations, missing dependencies, access failures, and unverified
end-to-end compatibility. An existing output directory is rejected. A failed
preflight is preserved; reruns require a new directory.

## Compatibility boundary

Official setup uses Torch2.6.0/cu124 and defaults to BF16 flow models.
The README documents xformers for V100, but V100 has no native BF16 support;
changing the attention backend does not automatically change flow precision.
The official flow `convert_to(torch.float16)` API is a possible runtime
adaptation, not a verified V100 result. It must be recorded and checked for
finite outputs before any asset-quality claim. Full-model `.half()` is not
an equivalent substitute for this API.

The official renderer requires CUDA extensions including FlexGEMM, CuMesh,
O-Voxel, nvdiffrast, and nvdiffrec render utilities. A successful dense
attention or Triton probe does not establish sparse-kernel, decoder,
exporter, memory, or complete pipeline compatibility.

New environments/models must go below
`/var/tmp/colorpeel-r12user5/assets/trellis2_v1`; new runs go below
`$COLORPEEL_OUTPUT_ROOT/subject_trellis_counterfactual_v1`.
Historical inputs/results remain read-only. Code deployment uses Git;
SCP is reserved for retrieving result artifacts.

## Current execution boundary

The original Facebook DINOv3 endpoint returned403 with existing server
credentials. The public Hugging Face/timm distribution of the same
ViT-L/16 LVD-1689M model is accessible without repository approval:
`timm/vit_large_patch16_dinov3.lvd1689m`, revision
`30c1109559f65dea34316b0d4842d35c5771fe11`.
Its weight layout differs; conversion and image-feature comparison are
required before using it with the original TRELLIS feature extractor.
This does not substitute DINOv2 or claim byte-level comparison with the
inaccessible original Facebook checkpoint.

`setup_runtime.sh` creates a new venv below the asset root, reusing only the
existing Torch2.6/cu124 and Triton3.2 installation through system site
packages. New dependencies are installed into that venv; the base environment
is unchanged. CUDA extensions target sm70. Official source checkouts remain
clean; wheel builds use separate disposable copies. No weights or images are
generated by setup. Failed setup artifacts are retained.

Generation requires verified isolated runtime and converted encoder first.
All raw native PBR output must be retained. No Albedo replacement, material
grid, or training is part of this first asset review.

The default masked-implicit Triton sparse GEMM aborted on this V100 with
`LLVM ERROR: Unsupported layout`. The documented `explicit_gemm` variant
passed the same sparse-convolution numeric check, xformers attention,
nvdiffrast rasterization and CuMesh normals. Generation selects it through
the official runtime configuration. It uses the same weights/operator with
CUDA neighbor maps and PyTorch im2col/GEMM, at greater memory cost; full
model compatibility is still established only by actual execution.

Install the memory-efficient attention backend into the new venv only:

```bash
/var/tmp/colorpeel-r12user5/assets/trellis2_v1/env/bin/python -B -m pip install \
  --no-deps xformers==0.0.29.post3 --index-url https://download.pytorch.org/whl/cu124
```

The first setup completed all CUDA wheel builds but its final import found
missing `trimesh` in the inherited environment. Preserve that setup log and
install the documented basic packages into the new venv before kernel checks:

```bash
/var/tmp/colorpeel-r12user5/assets/trellis2_v1/env/bin/python -B -m pip install \
  trimesh scipy imageio einops plyfile opencv-python-headless==4.11.0.86
```

`convert_dino` takes a fresh `--output` below the asset root. It maps the
pinned public timm tensors without changing their values, synthesizes only
zero Q/V biases and an unused zero mask token, and compares FP32 pre-norm and
TRELLIS-style normalized features on the actual pink condition at512/1024.
Acceptance uses the actual affine-free normalized features consumed by
TRELLIS at unchanged atol/rtol1e-3; pre-norm values remain finite-only
diagnostics because their scale is removed at the condition interface. Failed
parity leaves its attempt and receipt; it does not publish a converted model.
The reference is the public timm implementation, not an inaccessible original
Facebook checkpoint. RoPE follows Transformers4.57.3's FP32 reconstruction.

After parity passes, `run` takes a fresh `--output` below the run root and
`--encoder` pointing to that verified conversion's `model/`. It runs exactly
one official1024-cascade sample with the official three12-step samplers,
seed42 and FP16 flow-torso adaptation. It retains `native_asset.pt` and native
PBR channels; a separate clay clone changes only display material. A bounded
camera fit estimates source-canvas pose/scale/position. Contact-sheet labels
must say estimated view; mask IoU does not establish identity or Albedo truth.
`mailbox.glb` is a secondary official postprocessed export; native review
images refer to the retained pre-export mesh, not a claim of identical GLB
topology. No automatic scientific acceptance is made.

If the secondary export is interrupted, preserve its output and status rather
than rerunning generation. `export_saved --asset <native_asset.pt> --output
<new_external_run>` exports only the fixed task asset as a100k-face-target,
2048-texture preview LOD. It uses the official exporter with remesh disabled;
the native asset and native review images remain the primary evidence.
