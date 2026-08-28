# Material 360-view realization: decoded-pixel gate v2

Verified code commit: `774241ad80cace8bfc1547daeafa6262e1fe0ec4`

The existing Blender run was reused without rerendering. It contains 360/360
successful records with `resume=false`, `limit=null`, and zero renderer
failures. `decoded_pixel_equivalence_v2` passed all 180 metal comparisons:

- maximum mean absolute RGB difference: `0.0009307861328125`
- maximum changed-channel fraction: `0.00083160400390625`
- maximum single-channel difference, record-only: `5`
- raw RGB SHA-256 matches: `0/180`
- object/background mask decoded-pixel matches: `180/180` each

Realization produced 360 rows, 288 image-only full-grid training JPEGs, and
192 image-only JPEGs for each Fold A/B/C. The review packet contains a 360-row
CSV, one 18-cell contact sheet, and nine paired metal/rubber sheets. The gate
remains `pending_human_review` with `training_authorized=false`; no smoke or
training command was run.

Server run root:
`/home/r12user5/Documents/Jiawei/colorpeel-runs/clevr_subject_color_material_3x3x2/20260826-104500__clevr_subject_color_material_3x3x2__multiview-render-v3-material__d1f4282__420000`

Review output:
`<run-root>/prepared_human_review_v2_gate`

18-cell contact-sheet SHA-256:
`88fe67a9c748392c6d3d404e8bc761baa2edf42f43e655eb0f7ef4a6ac4d3c75`

## Human decision

On 2026-08-28 the user accepted the rendered material review packet and
authorized the next protocol stage. This releases only the locked sequential
2-step and 18-step eight-token training smokes before the full-grid baseline.
It does not release Fold training or any claim about material disentanglement.
