# Grounded-SAM pinned-API diagnosis

- Failed run:
  `/home/r12user5/Documents/Jiawei/colorpeel-runs/clevr_subject_color_3x3/20260822-150219__clevr_subject_color_3x3__segment_grounded_sam__c8c874d__42`.
- Symptom: all 600 items recorded
  `TypeError: post_process_grounded_object_detection() got an unexpected
  keyword argument 'threshold'` and no masks were written.
- Direct evidence: the installed Transformers 4.48.1 signature accepts
  `box_threshold=0.25`, `text_threshold=0.25`, and `target_sizes`.
- Root cause: adapter keyword from a newer API, not model weights, image data,
  detection threshold, or the ColorPeel training result.
- Minimal fix: rename only `threshold` to `box_threshold`; keep its value 0.25.
- Fix commit: `b059bd5e92cf1994581d8600111d3ed5830dc7d5`.
- Regression evidence: 11 related tests passed locally. The clean server
  checkout was updated only with `git pull --ff-only`.
- Rerun:
  `/home/r12user5/Documents/Jiawei/colorpeel-runs/clevr_subject_color_3x3/20260822-150928__clevr_subject_color_3x3__segment_grounded_sam__b059bd5__42`.
  It produced 588 accepted masks and 12 explicit semantic/ratio failures.

Scientific semantics were unchanged: model IDs, numeric thresholds, prompts,
mask bounds, images, and downstream metric formulas are identical.
