# Historical pink reproduction, then current-SAM pink conditioning

Completed outcomes and evidence paths are recorded in [RESULTS.md](RESULTS.md).

The user authorized this bounded experiment on 2026-11-04 after the read-only
provenance audit. It does not change the production pipeline's default input,
train any branch, run transfer, alter masks or resume older paused studies.

1. Generate one seed42 asset from the exact frozen historical pink RGBA input,
   using the existing verified DINOv3 conversion, pinned TRELLIS weights and
   unchanged 1024-cascade samplers. Skip only the secondary GLB export. Compare
   actual conditioning pixels/configuration and native tensors against the
   October27 asset. Whole `.pt` serialization hashes alone are insufficient.
   Compare review against the later accepted six-parameter camera review, not
   the first five-parameter/no-roll review. Keep differences and inspect images
   if exact numerical reproduction does not hold.
2. Only after historical reproduction succeeds, create one new conditioning
   receipt from the current original-image/SAM receipt. Recolor the original
   foreground with historical deterministic Lab/LCh hue0 and soft-alpha rules,
   but do not apply either historical manual mask repair. Keep the binary SAM
   alpha, source photo, mask and downstream photo anchors unchanged. Generate
   exactly one seed42 mesh with the same model/sampling configuration.
3. Inspect the new review. The existing 0.80 source-mask guard and source-edge
   checks still apply before constructing a candidate S11. Passing them is not
   user quality acceptance. No extra seeds, automatic camera continuation or
   lower threshold are part of this experiment.

Pink is an explicit conditioning-only transformation derived from the original
photo. Color/Material/Texture sources and the original/recolored S11 photographic
anchors remain rooted in the original photograph. No old mesh is a fallback.

Preparation, from a clean committed checkout, writes only a fresh external dir:

```bash
python -B -m experiments.subject_pink_input_repro_v1.prepare \
  --source-receipt /absolute/path/original/input/source_receipt.json \
  --output /fresh/external/run/pink_inputs
```

The explicit new receipt binds its parent receipt, original cutout, original RGB,
unchanged mask and the recolor operation. It can be passed as `--source-receipt`
to the existing TRELLIS run command and as the inputs directory to the existing
Subject renderer. `image512_path` remains the original photo; only `cutout_path`
is the declared pink geometry condition. Prepare and any later training must
use one complete Git revision; no training is included in the present scope.
