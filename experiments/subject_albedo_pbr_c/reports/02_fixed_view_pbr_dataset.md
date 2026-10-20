# Experiment C: fixed-view PBR dataset, 2026-10-20

Thirty actual RGB training assets were generated, reviewed and staged using
clean revision `8c639fe4a4808fba2c6b0d8c9bdf455ecfe29d0d` on
`codex/subject-c-pbr-scene`. No training was started. Code was edited/tested
locally, committed/pushed, then pulled into the independent server checkout
`/var/tmp/colorpeel-r12user5/checkouts/subject-c-pbr-scene`.

## Immutable outputs

Accepted first fixed-view dataset:

```text
/var/tmp/colorpeel-r12user5/runs/subject_albedo_pbr_c/20261020-210555__pbr30_dataset__8c639fe__42
```

Its `assets/<id>/images/image.png` files are the actual training RGBs;
corresponding masks, `concepts.json`, `training_assets_manifest.jsonl`,
`dataset_audit.json`, `visual_review.json`, `source_locks.json`,
`contact_sheet.png` and `index.html` identify every image/caption and receipt.
The sheet labels actual RGBs separately from original/clay/normal/UV references.

Raw renders, packed `.blend`, EXRs, projection/geometry audits and frozen plan:

```text
/var/tmp/colorpeel-r12user5/runs/subject_albedo_pbr_c/20261020-210233__pbr30_bottom_complete__8c639fe__42
```

The five Albedos remain byte-identical to the earlier parameter preview:

```text
/var/tmp/colorpeel-r12user5/runs/subject_albedo_pbr_c/20261020-153651__parameter_preview__ac92f46__42
```

## Controlled factors and actual renderer

The full factorial is five fixed Albedos (pink, green, cyan, blue, purple),
Metallic=0/1, Roughness=0.15/0.50/0.85. Every color uses one identical Albedo
hash for all six M/R cases. Numeric scalar M/R inputs cover the entire rendered
surface; the earlier float32 map edits remain source provenance. Captions
condition the same `<S*>` mailbox on the corresponding color/material terms.

Blender4.2.11 Cycles uses one fixed manually fitted cylinder/cap/backplate
surface, UV mapping, orthographic camera, three rectangular area lights and
environment, Principled BSDF,256 samples, seed42, no denoising or adaptive
sampling. The exact poses, powers, shader inputs and checks are frozen in the
candidate `plan.json`. Only Albedo selection and M/R change across the grid.
An emission render verifies UV correspondence only; all training RGBs use
the Principled BSDF. There is no RGB material filter, imagegen, per-image
exposure tuning or per-image normalization.

The original42269-pixel mask loses the bottom crop of the mailbox. Final
rendering preserves all those pixels and adds682 pixels using the explicit
polygon `[[143,506],[375,506],[369,511],[142,511]]`. The resulting42951-pixel
mask has SHA256 `550c1e13bee1435e1145f06e44bdef02d75b8b8fdfa7462d3b39f456dfc9a978`.
Only added faces use nearest-valid-original-pixel UV continuation (1.588% of
visible faces; maximum extension5.099px). Original UVs and Albedo files stay
unchanged. A2px offscreen bottom strip avoids false frame-cut transparency.
One fixed TELEA-radius3 background plate inside the mask handles antialiasing;
every pixel outside the final mask is copied exactly from the original photo.

## Verification and visual review

- Local C suite:112 passed. Server affected geometry/staging suite:62 passed.
- All30 downloaded RGB SHA256 values matched the server candidate manifest.
- Final staging reverified30 RGBs,30 masks,30 captions and every source lock;
  source/candidate assets are unchanged and all copies match byte-for-byte.
- Silhouette mask IoU=1.0; minimum eroded-core coverage=1.0.
- UV scene-linear absolute error p95=0.0041453981 (gate0.01).
- Maximum fixed-display clipped-pixel fraction=0.0087541617, or0.8754%
  (gate1%). No per-image exposure compensation was used.
- Changed original-background pixels outside the final mask:0 for every RGB.
- Entire30-image contact sheet was inspected, with native512px edge/highlight
  spot checks. No large new deformation, disconnected foreground islands or
  obvious color spill remained at that review resolution. M=1/R=0.15 has
  concentrated colored reflections; increasing R broadens/softens highlights.
- A descriptive comparison of raw linear EXRs fitted the best single global
  brightness gain from M=0 to M=1 at each color/R. Relative residuals remained
  0.10086–0.90517, supporting that the M response is not solely brightness
  scaling. This is not a BRDF calibration or disentanglement metric. Exact
  values are in the sibling `.material_response_audit.json`.

## Rejected candidates and corrections

All prior attempts remain untouched. The full grid
`20261020-202754__pbr30_candidates__4065ba2__42` exposed >30% clipping for the
bright purple metal; lights were globally lowered before the next full run.
`20261020-203209__pbr30_calibrated__2d1f40c__42` passed numeric checks but left
orange pixels at the image bottom. The subsequent
`20261020-204457__pbr30_repaired_edges__744532a__42` reused B's larger mask;
full-size inspection found false background-post pixels and fragmented
foreground additions. It was not staged. Final8c639fe uses the original mask
plus only the explicit bottom polygon, with a hash-locked derivation recomputed
at staging. Earlier straight-cap geometry was also superseded by the fixed
elliptical-cap transition; rejected previews are not training assets.

## Limits of acceptance

This is real foreground PBR under an explicit assumed fixed scene, not recovery
of the actual mailbox geometry, normals, light or measured BRDF. The manual
single-view2.5D surface does not support a claim of correct novel views.
Door seams, handle and small components remain mostly Albedo details rather
than modeled relief. SuperMat Albedo can retain source shading/dark details;
freezing it does not prove intrinsic extraction is perfect. The682-pixel
bottom completion and its UV continuation are declared manual extrapolations.
Original background shadows stay fixed; they are not rerendered. At native
512px, source JPEG/segmentation edge uncertainty and Monte Carlo variation in
sharp glossy highlights remain. The nominal hue names do not guarantee
perceptual color constancy under changed reflectance.

Acceptance means usable for this first fixed-view counterfactual data
experiment with these limitations recorded; it does not establish disentangled
training or photorealistic recovery of all components. Subject training,
identity/color/material-control evaluation and comparison with B remain
separate work and have not been run here.

## Reproduce

Use the documented render/stage commands in the README at the recorded clean
revision. Select fresh output paths; never overwrite any paths above. Inspect
the new full grid and save explicit per-image visual decisions before staging.
Rendering and staging must use the same clean code revision. The candidate
records the complete command, protocol, source hashes and saved Blender scene.
Experiment B and the original parameter-preview C checkout were left on their
existing branches/revisions with clean worktrees.
