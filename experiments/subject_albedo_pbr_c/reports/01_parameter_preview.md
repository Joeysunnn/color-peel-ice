# Experiment C first parameter preview

Confirmed execution, 2026-10-20, code revision
`ac92f46a328a21df13b6253272634c11ede1cc39`, clean branch
`codex/subject-experiment-c`.

Immutable server result:

```text
/var/tmp/colorpeel-r12user5/runs/subject_albedo_pbr_c/20261020-153651__parameter_preview__ac92f46__42
```

Execution used the independent server checkout
`/var/tmp/colorpeel-r12user5/checkouts/subject-experiment-c`; the Experiment B
checkout and branch were not switched or updated. All code was edited and
tested locally, committed and pushed, then cloned from that remote revision
on the server. Only CPU map editing ran. The selected SuperMat extraction
was reused with its existing verified receipt, not rerun.

```bash
export PYTHONDONTWRITEBYTECODE=1
export COLORPEEL_RUN_ROOT=/home/r12user5/Documents/Jiawei/colorpeel-runs
export COLORPEEL_OUTPUT_ROOT=/var/tmp/colorpeel-r12user5/runs
PY=/home/r12user5/miniforge3/envs/colorpeel017/bin/python
# Choose a NEW output; the result above is immutable.
$PY -B -m experiments.subject_albedo_pbr_c.prepare --output "$NEW_C_PREVIEW"
```

## Confirmed checks

- Local and server C tests:21 passed on each platform.
- Existing original photograph and original object mask exactly match the
  decoded RGB and alpha of the frozen SuperMat input. The mask contains42269
  foreground pixels. No repaired Subject mask or small measurement region
  was substituted.
- Original source copies are byte-identical. All source locks and all31
  generated artifact hashes were checked after execution. The downloaded
  review bundle also passed its file-hash verification locally.
- All30=5colorsx2metallicx3roughness parameter combinations passed numeric
  editing checks. Each factor's map hash is independent of the other two
  factors. M=0/1 and R=0.15/0.50/0.85 are exact float32 interventions inside
  the original mask; source parameter values outside it remain unchanged.
- Outside-mask Albedo changed pixels:0; reported RGB clipping pixels:0.
  Across colors, the largest foreground lightness p95 error is
  **0.1660173 Lab units**, and largest chromatic hue p95 error is
  **1.206416 degrees**. Source:`albedo_qc.json`.
- Status:`parameter_preview_verified_renderer_missing`.
  Actual counterfactual RGB renders:0. Training-eligible images:0.
  Training started:false. All manifest records explicitly remain unrendered.

## Observed visual limits

The three downloaded grids were visually inspected. Their labels and values
are readable, and they distinguish source RGB, extracted maps, edited Albedo
and parameter previews. Repeated Albedo thumbnails across M/R columns are
intentional: these panels do not show a material response or relighting.

The estimated Albedo visibly retains dark contours and brightness variation.
It is not established that SuperMat has completely removed illumination or
specular effects. Preserving this map's lightness therefore preserves any
such residuals as well as actual texture.

Gamut compression is substantial for some directions. For blue,99.67% of
foreground pixels require chroma reduction and the median reduction is65.0%;
cyan is similar (median62.1%). This is the unchanged legacy gamut-aware
algorithm, not clipping. The nominal B color labels/angles should not be
interpreted as calibrated perceptual colors for this new Albedo source.
Source:`albedo_qc.json` and `albedo_grid.png`.

## Pending before a training dataset exists

Reliable original-subject PBR recomposition remains unavailable. The missing
mesh or validated same-view geometry/normal, surface correspondences, camera
and lighting are listed in `renderer_readiness.json`. Generic CLEVR geometry
cannot establish mailbox identity preservation.

Obtain/fit and validate those inputs, reuse the existing Cycles PBR material
and renderer utilities, define the background/shadow policy, then check real
RGB silhouette, internal structure and material response. Only accepted RGB
can enter new Subject concepts. Identity preservation, color/material
controllability and glossy-plastic/ceramic/wood generalization are pending;
no scores or training results are inferred from numeric map edits.

The reproduction command, proposed captions and full A/M/R paths/hashes are
in the run's protocol, receipts and `parameter_manifest.jsonl`. The planned
Subject training and fixed-seed evaluation retain the B recipes described in
the study README; neither was started here.
