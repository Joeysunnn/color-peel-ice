# Full-spoon counterfactual construction, 2026-10-28

Completed two independent 180-image datasets (360 total), with no training or
monitor. Asset generation used commit `2943bf6`; the frozen rendering protocols
and dataset construction used `1a5a6235b2e1da4e0fab5a9745fb8340a0e74192`.
All source edits were committed/pushed locally and pulled into the clean server
checkout `/var/tmp/colorpeel-r12user5/checkouts/trellis-spoons-counterfactual-v1`.

All successful outputs belong to this new experiment directory:

```text
/var/tmp/colorpeel-r12user5/runs/subject_trellis_spoons_v1/20261028-013740__assets__2943bf6__42
  inputs/                         full-original copies, soft masks, cutouts, receipts
  metal_spoon/                    native_asset.pt, native PBR/clay/BC/M/R reviews
  wood_spoon/                     separate native asset and source-view camera
  datasets/metal_spoon/           180 rendered candidates and verification
  datasets/wood_spoon/            180 rendered candidates and verification
  asset_generation.log
  dataset_generation.log
```

`metal_spoon.json` and `wood_spoon.json` freeze separate asset/camera hashes.
Both use the original pink/green/cyan/blue/purple RGB values, M={0,1},
R={.15,.50,.85}, three source-near views and studio/courtyard HDRIs: 180 unique
512px images each. Geometry, camera, color and light stay fixed within each
six-material block. Source material names never enter the captions as a fixed
material: both subjects use `spoon`, with actual rendered M/R descriptions.
The datasets remain separate; their shared `<S*>` spelling is not a joint
two-identity training configuration.

| Subject | Native vertices / faces | Source-view mask IoU | Images / matched groups / edges | Render-loop seconds |
|---|---:|---:|---:|---:|
| metal_spoon | 351822 / 706420 | 0.82205 | 180 / 30 / 210 | 27.668 |
| wood_spoon | 408127 / 818898 | 0.94781 | 180 / 30 / 210 | 28.362 |

Each source-view camera is estimated, not calibrated. These IoUs describe
silhouette overlap, not identity accuracy. CPU tests and preflight checks passed
(11 dataset/subject tests plus receipt, pinned-download and skip-export checks).
Both server verifiers passed completeness, artifact hashes, RGB/mask/raw/RGBA
pairing, fixed mask/normal per view, uniform float32 BC/M/R/alpha maps,
one-image/one-caption concepts and nonidentical single-factor RGB interventions.

| Subject | Metallic-flip foreground MAE min / median | Roughness-edge foreground MAE min / median |
|---|---:|---:|
| metal_spoon | 25.062 / 39.091 | 3.384 / 12.510 |
| wood_spoon | 21.285 / 41.990 | 4.531 / 17.060 |

MAE uses the 0-255 RGB scale and is descriptive, not a material correctness
threshold. Each subject has 90 metallic-flip edges and 120 adjacent-roughness
edges. Total: 60 matched groups, 420 edges, 360 RGB images and 360 aligned masks.
Captions, concepts, manifest, pair index, raw renders, RGBA cutouts, exact material
maps, normal buffers, camera matrices and six full-color review sheets are saved
per subject. `review/index.html` exposes all images and captions.

Visual inspection covered full-object segmentation and native/source-view
PBR/clay/BC/M/R, then the rendered material grids. Both handles remain complete;
the wooden-source geometry retains a bowl cavity and handle hole. Native metal
Base Color contains dark reflection-like texture artifacts. Dataset rendering
discards all native appearance and uses uniform BC/scalar M/R, so those patterns
are not pasted into the counterfactuals. Original wood grain and logo are also
discarded. This is synthetic counterfactual data from reconstructed geometry,
not exact photographic counterfactuals or evidence of learned disentanglement.
All candidates remain pending user review (`training_eligible_images=0`).

The complete artifact archive is adjacent to the experiment directory:

```text
20261028-013740__assets__2943bf6__42.complete.tar.gz
bytes: 70631926
SHA256: 24e9047f6e0224d958bfc1406b125ddafee8dc69034db2e28362cbb2c0aa73d6
```

It contains both datasets, native assets, input provenance, reviews and logs;
compilation caches are excluded. The downloaded copy passed archive SHA256,
manifest SHA256, independent subject IDs and exact 180 RGB/180 mask counts for
each subject. Existing experiments and jobs were left untouched.

Two preprocessing-only failures remain preserved: `20261028-013330__assets__234e256__42`
(mirror HEAD redirect incompatible with the download library), and
`20261028-013553__assets__fb53bd4__42` (a transcribed SHA256 contained one extra
character). No 3D sample was generated in either attempt. The final GET-based
bundle validates all four files against pinned official Git-blob/LFS metadata;
the corrected 64-character weight digest matches the downloaded file. No model,
weight revision, sampler setting or pre-existing output was changed to resolve
these download/preflight failures.
