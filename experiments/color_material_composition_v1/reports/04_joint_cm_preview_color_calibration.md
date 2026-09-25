# Joint C/M preview: orange socket calibration

The first 12-image soft-front preview was rendered under commit `abd4174` at
`$COLORPEEL_RUN_ROOT/color_material_composition_v1/joint_cm_caa_ablation_v1/preview`.
Its renderer status was `partial_smoke` with 12/12 rows. All 48 image, mask,
background-mask, and scene hashes matched the realization manifest. The four
color/material cells per shape used the same geometry and view, and the sphere
kept the selected ground-reflection split.

Visual review **failed for color**: both orange metal and orange rubber read as
deep red across cube, sphere, and cylinder. Metal and rubber remained visually
distinct, and blue remained blue. The orange socket used the linearized D1GT
emission reference `[0.738731741987261, 0.036888636218879646,
0.009661907129827573, 1.0]`; this source value did not transfer to a
perceptually orange native material under the selected renderer.

The next preview changes only the green channel of the orange material socket
to `0.2`. The D1GT target and nominal RGB remain provenance references, while
the material socket is now an empirical calibration input. The new preview
requires its own review record. No full 72-image rendering or joint training
is approved on the rejected first preview.
