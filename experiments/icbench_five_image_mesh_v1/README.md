# Five randomly sampled ICBench images: eight meshes per Scheme A instance

User-requested2026-11-05. Read-only workbook selection yields36 eligible object
rows and29 distinct image IDs. Sort numeric image IDs, then make one
`random.Random(42).sample(pool, 5)` draw:53,3,0,74,10. Store the workbook SHA,
selection cells, full29-image pool, selected source paths/hashes and sampling
rule in an external selection JSON. Do not resample based on model results.

Use the original ICBench model input `img.jpg` (512x512) from each sample's
stage-one directory, not GT masks, manual object labels or training outputs.
The higher-resolution companion `original.jpg` is a distinct source version.
Verify the server JPEG hashes against the local D1 archive before execution.

## Scope and input semantics

1. Existing Scheme A: Qwen whole-object category discovery, Grounding DINO
   instances, SAM binary masks and existing mask deduplication. Include every
   retained instance; the workbook's36 selected objects do not filter them.
2. For each instance, prepare semantic input512 and associated BiRefNet soft
   alpha using existing unmodified component/coverage guards. Preserve failed
   images, rejected detections and blocked instances with their evidence.
3. Freeze the semantic receipt's original `image512.png` RGB once for all8
   cells. This is explicitly the `source_rgb=True` factorial policy; the old
   Mailbox/spoon joint-RGBA RGB policy stays the default for those studies.
   Source RGB prevents black holes in SAM pixels outside BiRef support.
4. Generate original/pink x recolor support SAM/BiRef x geometry alpha
   SAM-binary/BiRef-soft. Scope only controls pink recoloring. All8 are fresh
   calls, including two original duplicate pairs. Final RGBA zeros RGB only
   where the selected geometry alpha is0. Use the official automatic crop
   from each cell's alpha>204, retaining soft compositing and fractional boxes.
5. Compare all8 using one reference camera fixed in advance to this object's
   new `original__biref__biref` review, plus three shared canonical views. Do not
   choose a reference by score or appearance or borrow Mailbox/spoon cameras.

Seed42 and pinned TRELLIS settings remain unchanged. No initializer selection,
S11 generation, SuperMat, branch training, dual-candidate selector, extra seed
sweep, automatic retries or visual-quality acceptance belongs to this run.
An association failure remains blocked rather than silently using a GT mask,
hardening soft alpha, changing an ownership threshold or replacing the image.

## Execution

Use local changes/tests/review, push the authorized fork, then clone a new clean
server checkout and verify its full SHA. Set both COLORPEEL_RUN_ROOT and
COLORPEEL_OUTPUT_ROOT to `/home/r12user5/Documents/Jiawei/colorpeel-runs`.
Check GPU processes/owners before execution; existing execute() also checks
idle status before each child process. Runtime/model paths are inherited from
single_image_pipeline.RUNTIME and use cached pinned files without downloads.

```sh
python -B -m experiments.icbench_five_image_mesh_v1.run --stage intake \
  --selection EXTERNAL_SELECTION_JSON --output NEW_RUN/intake --gpus 0 1 2 3
python -B -m experiments.icbench_five_image_mesh_v1.run --stage meshes \
  --intake NEW_RUN/intake --output NEW_RUN/meshes --gpus 0 1 2 3
```

Both stages require the same Git. Each retained instance has8 planned jobs,
including blocked entries for failed alpha preparation. Generation failures
do not stop independent jobs; incomplete objects retain partial results and
block their full8 shared-camera comparison. Unknown objects in failed
discovery images cannot be counted as successfully processed.
