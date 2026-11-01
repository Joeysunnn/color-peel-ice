# Color geometry lightness preview

User-authorized nine-image preview only; no training or transfer generation.
Reuse the frozen Blender neutral cube/sphere/cylinder x views0/8/16 and the
historical nine Emission references. Check every source hash before and after.
Only final foreground Lab is synthesized: fixed original Mailbox a/b, with
L=50+8*tanh((source_L-interior_median)/(0.5*(interior_q95-interior_q05))).
Do not reduce chroma, clip RGB, apply another view transform or overwrite inputs.
Reject nearly uniform source lightness rather than amplify noise.

The smooth monotonic mapping preserves shading order but changes absolute
illumination and contrast. This is not physical PBR. Binary source masks remain
unchanged; no unmeasured subpixel coverage is invented. Background and cast
shadows are byte-identical to neutral renders. Emission reference backgrounds
have different colored spill, so this is not a strict shading-only ablation.
Review full-size RGBs and nearest-neighbor edge close-ups before accepting data.

Numerical checks use the historical eroded-interior definition: dominant a/b
error <=3, maximum interior pixel a/b error <=1, q95-q05 L contrast >=6,
no decoded brightness-order reversal >0.5 L, zero background changes. These
checks do not establish visual quality or trainability. No training concepts
are exported; all candidates retain training_eligible=false pending review.

Deploy through local commit/push and a fresh server checkout, then run:

```sh
PYTHONDONTWRITEBYTECODE=1 /home/r12user5/miniforge3/envs/colorpeel017/bin/python -B \
  -m experiments.color_geometry_lightness_v1.prepare --output "$NEW_EXTERNAL_RUN"
```

No GPU or fresh Blender rendering is required. index.html compares Emission,
neutral and candidate for all nine rows; comparison.png is a compact overview.
manifest.json retains exact captions and source records; source_locks.json
and files.json record source and output hashes.

## Uniform gray background derivative

After the user accepted geometry/color, `gray_background` creates a separate
nine-image version with every mask-exterior pixel set to sRGB uint8(128,128,128).
All foreground RGB pixels and mask files are unchanged; external cast shadows
are intentionally removed. No feathering, rerendering or training is performed.
The exact accepted source file index is hash-pinned and every source file is
verified before and after. Each saved PNG is checked for foreground identity
and exactly uniform background. Use the same deployment workflow, then:

```sh
PYTHONDONTWRITEBYTECODE=1 /home/r12user5/miniforge3/envs/colorpeel017/bin/python -B \
  -m experiments.color_geometry_lightness_v1.gray_background --output "$NEW_EXTERNAL_RUN"
```

The new index shows before/after; contact_sheet.png shows all nine new RGBs.
Source geometry acceptance does not authorize training of the background variant.

## Rendered continuous background

`render_background.py` replaces only the finite Ground mesh with a 360-degree
gray studio: radius-12 floor, radius-8 tangent cove, radius-20 wall to height32.
The original gray material, World strength0, lights, camera views0/8/16,
foreground geometry/material, Cycles512 samples and seed42 are retained.
All 512x512 camera rays plus frame corners must hit the new background.
The old camera/light/object/material metadata must match exactly. The source
asset contract is hash pinned; source files are verified before and after.

Deploy via local commit/push and server Git checkout. In a fresh experiment
parent, render `--shapes sphere --output "$RUN/sphere3"` first, using Blender
`--background --python-exit-code 1 --python experiments/color_geometry_lightness_v1/render_background.py --`
and `--asset-root /home/r12user5/Documents/Jiawei/papers/CLEVER/image_generation`.
Set CUDA_VISIBLE_DEVICES to a checked available GPU. Run ordinary Python
`-m experiments.color_geometry_lightness_v1.review_background --inputs "$RUN/sphere3" --output "$RUN/sphere_review"`.
Inspect all three sphere images before rendering `--shapes cube cylinder`
to a fresh `$RUN/remaining6`; then review both inputs into `$RUN/result9`.
The three approved sphere renders are reused without rerendering.

Review calls the existing `prepare.remap` unchanged and requires identical old
object-mask pixels. Background and cast shadows remain byte-identical to the
fresh neutral render, with no gray filling. Scene geometry can change indirect
illumination; foreground final RGB is not promised identical. Saved .blend
files and camera-ray audits retain scene evidence. Preview only; no training.
