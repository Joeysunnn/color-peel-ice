# Five photographic hues with the current SAM mask

Completed metrics and review artifacts are recorded in [RESULTS.md](RESULTS.md).

User-authorized on 2026-11-05: compare one new TRELLIS mesh per historical
photographic recolor, using the current Mailbox Scheme A object_003 SAM mask.
This experiment generates five seed42 native assets and diagnostic images only.
No training, dataset selection, automatic retry or production-default change.

| Display name | Historical image label | Lab/LCh hue |
|---|---|---:|
| pink | red | 0 |
| green | green | 120 |
| cyan | cyan | 180 |
| blue | blue | 240 |
| purple | magenta | 300 |

The palette follows `d1_subject_recolor_mailbox_border_repair_protocol_v1.json`
under `experiments/natural_image_subject_color_pilot/configs/`; its yellow=60
candidate is explicitly excluded. Display-name corrections come from
`experiments/subject_marble_color30_v1/protocol.json`. Only their recolor rules
and labels are reused, not their historical images or manual mask repairs.

Each condition starts independently from the same original RGB, preserves Lab L,
reduces chroma only for gamut, and blends with the original using the same derived
soft RGB alpha. The RGBA segmentation alpha stays exactly the current binary SAM
mask. Original photo paths and C/M/T sources remain unchanged. The preparer adds
explicit hue selection while retaining the previous hue0 default.

All five runs use the same pinned weights, converted encoder, 1024 cascade,
samplers, seed42 and existing camera-fit budget. Pink is generated anew so all
five use the same complete code revision. GLB export is skipped; `native_asset.pt`
retains the full native mesh and attributes. There is no extra seed search.

From a fresh clean GitHub checkout, with both COLORPEEL_RUN_ROOT and
COLORPEEL_OUTPUT_ROOT pointing to `/home/r12user5/Documents/Jiawei/colorpeel-runs`:

```bash
bash experiments/subject_five_color_mesh_v1/run.sh \
  /absolute/path/current/original/source_receipt.json \
  /fresh/external/run_root 0
```

The script checks GPU idleness before every generation and stops on failure,
preserving partial evidence. Each color has `inputs/`, `trellis/` and `trellis.log`.
After generation, run the separate `compare_views.py` module on the five trellis
directories to render four exactly common cameras with opaque gray material.

The comparison separates independently fitted source views (native PBR and clay,
forest illumination) from common fixed views (opaque clay, studio illumination).
Fixed views do not rotate, center, scale, simplify or refit each mesh. Their yaw
labels are numeric, not presumed semantic front/back labels. Source-view IoU and
optimizer convergence are reported, but do not select a best mesh or imply user
acceptance. Existing 0.80 Subject guards remain unchanged; no S11 is built here.
