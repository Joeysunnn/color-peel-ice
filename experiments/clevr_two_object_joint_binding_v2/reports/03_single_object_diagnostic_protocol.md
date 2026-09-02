# Single-object complete-bundle checkpoint diagnostic

This smoke isolates learned-token quality from multi-object instance binding.
It compares the accepted single-object full-grid checkpoint with the new
joint-binding checkpoint using identical prompts and sampling:

- 18 complete subject/color/material bundles;
- one object per prompt;
- generation seed 42;
- 100 inference steps, CFG 6.0, FP16 on GPU 3;
- SafetyChecker disabled under the existing acknowledged false-positive policy.

The primary gate is the complete three-token bundle. Individual subject-only,
color-only, or material-only prompts are intentionally excluded because they
are outside the training distribution and cannot diagnose multi-object binding.

Both checkpoints produce 18 images. The bundler validates all hashes and
provenance, then creates one 36-image checkpoint comparison sheet and a
randomized review CSV. The process stops for human review. The two 360-image
runs remain blocked until this smoke establishes whether the joint checkpoint's
single-object shape, color, and material semantics remain intact.
