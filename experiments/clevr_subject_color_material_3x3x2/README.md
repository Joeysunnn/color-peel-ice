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
