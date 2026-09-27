# Caption-aligned mailbox S+M: owner visual review (2026-09-27)

## Frozen comparison

- Protocol: `protocols/mailbox_caption_aligned_subject_inference_v2.json`, code commit `fe06311`.
- Server output: `/home/r12user5/Documents/Jiawei/colorpeel-runs/subject_material_composition_v1/20260927-212006__subject_material_composition_v1__mailbox_caption_aligned_inference_252__fe06311__42/output`.
- The generation ledger completed 252/252 rows: baseline 80 visible and 4 safety-filtered; caption-aligned matte-only 84 visible; caption-aligned balanced 84 visible. The four filtered rows belong to baseline. These counts establish completion, not attribute quality.
- Matched grid: five reviewed caption colors (pink, green, cyan, blue, purple), held-out literal red, and city-street context; S-only, S plus literal matte plastic, S+selected M, and S plus literal metal; seeds 42–44, 100 steps, CFG 3.5.

## Project-owner observations

- Both caption-aligned S arms follow the requested color in many, but not all, images. The owner's visual estimate is roughly 70–80% color agreement. This is **not** a counted per-image accuracy or a scored metric.
- In the balanced arm, many S+`<M*>` samples appear metallic, while samples prompted with literal matte plastic appear matte. The owner judges its material response better than the matte-only arm.
- Visible baseline outputs appear metallic across conditions; visible matte-only outputs appear matte across conditions. The four safety-filtered baseline images have no inspectable surface appearance.

## Evidence boundary and decision

The balanced arm is the stronger **qualitative candidate for material controllability** in this matched grid. It is not selected as a validated disentangled S checkpoint: color control is imperfect, and the review does not provide per-color or per-condition counts, a mailbox-identity score, or a scene-adherence verdict. The paired training arms also differ in row count and material captions, so this contrast does not isolate one causal change.

Preserve the earlier 108-image prompt-alignment failure in `mailbox_matte_subject_prompt_alignment_20260926.md`; the caption-aligned comparison does not retroactively change its result. No new training or large generation is authorized by this review record.
