# TRELLIS.2: one foreground pink Mailbox asset

This study first checks the research server, then targets one seed42 asset
from the existing pink foreground original. The requested review consists
of an estimated source-view PBR render, clay render, Base Color, Metallic,
and Roughness. No counterfactual expansion, Subject training, parameter
sweep, or monitoring is authorized by this entrypoint.

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

Only the environment/access preflight is implemented here. Generation is
not started while a required official model is inaccessible. When access
is available, resolve and lock model revisions, verify the isolated runtime,
then add and validate the single-asset runner locally before deployment.
All raw native PBR output must be retained. No Albedo replacement, material
grid, or training is part of this first asset review.
