# CLEVR subject + color + material (3 x 3 x 2)

This versioned stage extends the accepted subject+color experiment with one
material axis. It does not overwrite renderer v1/v2, their requests, renders,
staging, checkpoints, or evaluations.

The only renderer-side scientific change from `multiview_render_v2` is that
the request selects CLEVR's native `MyMetal` or `Rubber` material. Paired
metal/rubber views share the same render seed, orbit camera, lights, and Cycles
noise. All ColorPeel training math remains unchanged.

Execution stops after the 360-view realization and human contact-sheet review.
Training remains blocked until that review is explicitly accepted.

Accepted v2 metal equivalence is checked on decoded pixels because a same-profile
Blender control rerun showed one-level JPEG rounding differences. RGB requires
`max_abs_difference <= 1` and `mean_abs_difference <= 0.001`; decoded object and
background masks must match exactly. Raw SHA-256 values are retained in the
equivalence audit and are not used as cross-run equality gates.
