# Shared-soft-alpha original/pink selection: Mailbox production-helper verification

2026-11-05. The requested workflow is integrated and the same production
`prepare_subject_geometry` helper completed on Mailbox `object_003` from frozen
Scheme A segmentation. It freshly prepared one BiRefNet soft alpha, sampled one
original and one pink mesh, evaluated them, and selected **pink**. No S11 or
branch training was run. Producer commit:
`83719bfab04e86159a9dfdbc2188ccf0d1566eac`.

## Actual decision

| Measurement | Original RGB | Pink RGB |
| --- | ---: | ---: |
| Generation | Completed | Completed |
| Native structural checks | Passed | Passed |
| Vertices / faces | 830457 / 1664818 | 1033002 / 2067646 |
| Saved review soft-alpha IoU | 0.6242640500 | 0.8288188416 |
| Opaque geometry IoU against SAM | Skipped after review guard failed | 0.8190070200 |
| Symmetric mean boundary distance | Not evaluated | 13.1581268854 pixels |
| Boundary distance / SAM bbox diagonal | Not evaluated | 0.0356975718 |
| IoU minus normalized distance | Not eligible | 0.7833094482 |
| Eligibility | Rejected by unchanged 0.80 review guard | Passed |

Both meshes have finite valid native arrays and zero degenerate faces, but
the original candidate fails source-view alignment. Pink is therefore the only
eligible candidate. The close-case random draw was **not used** in this run;
its reproducibility, source/instance salt and threshold boundaries are unit-tested.
Both complete native outputs and all failed eligibility evidence remain retained.

The candidate alpha channels are byte-identical. Photo/SAM sources, model and
encoder hashes, seed42, samplers, precision, normalization and full generation
Git all match. The new pink conditioning pixels and native mesh SHA256 also
exactly reproduce the pink-only preview generated first at `87efe3a`:
`133afbc08d3932a07fbf4681e9b4626a38a8df0cd1cbb9c571b0ec1136b65380`.

## Limits of the result

Pink still shows a backplate/tube relationship different from the photograph.
Original is more severely distorted in the rendered view. Automated selection
here demonstrates the specified engineering rule; it does not prove the chosen
mesh has correct 3D shape or that soft alpha always improves every photograph.
IoU/boundary metrics cannot assess internal faceting, depth or hidden surfaces.
Both saved camera fits retain their original budget and did not declare
convergence. The user has seen the pink preview; neither this candidate nor
downstream four-branch learning is recorded as accepted.

The preceding Mailbox association fix is documented in
[the pink-preview record](../mailbox_soft_alpha_v1/RESULTS.md). It ignores weak
component/SAM contacts only for ownership, preserving all selected soft values.

## Integration and validation

- Production now keeps SAM semantic masks separate from BiRefNet alpha and
  generates original/pink candidates per object. It uses the selected receipt,
  native mesh and saved camera together, preserving the other candidate.
- A failed candidate does not skip its sibling. Zero eligible candidates block
  Subject preparation; other available C/M/T branches continue independently.
- Selection decisions, attempts and their source locks propagate into Subject
  dataset provenance. Original-photo anchors, Qwen and C/M/T sources remain original.
- Local relevant suite: 77 tests, 65 passed and 12 skipped for missing SciPy.
  The fresh clean server checkout ran all 77 with no skips in 40.564s.
- The real helper exercised fresh matting, both generation paths, structural
  checks, opaque GPU rendering, SciPy boundary metrics and actual selection.
  Code/evidence reviews were performed independently. No model download,
  geometry smoothing, asset deletion, extra seed search or training occurred.

## Retained results

Server root:
`/home/r12user5/Documents/Jiawei/colorpeel-runs/subject_dual_color_selection_v1/20261105__mailbox_dual__83719bf__42/`.

- `trellis_candidates/{original,pink}/native_asset.pt`: both new meshes.
- `trellis_selection/selection.json`: guards, scores, seed, source locks and decision.
- `trellis_selection/pink_opaque_mask.png`: actual scored geometry silhouette.
- `trellis_candidates/attempts.json`, `status.json`, `verification.json`: execution and input checks.
- `trellis_alpha/`, `trellis_input/`, both `review/` and `repro_outputs/`: conditioning and render evidence.

The source checkout remains at
`/home/r12user5/Documents/Jiawei/colorpeel-checkouts/20261105__dual_selection__83719bf`.
Server test log: the result root's parent `tests_83719bf.log`.
Local gallery:
`C:/Users/lenovo/.codex/visualizations/2026/11/04/01a1a4e9-817a-7d01-ae16-5db6ed9b42b7/subject_dual_color_selection_20261105/index.html`.
Downloaded evidence archive SHA256:
`e24d90c9205509dea5f23d519e6d12718ce66c4cc3b02e0121089559785a269b`.
