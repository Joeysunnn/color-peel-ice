# Object-mask A/B results

## A — preserved 2026-11-04 baseline

- Code: `4b537fac7d78946a0ffcf72c1c0d38e201b71ed7`, branch
  `codex/sam-multi-object-intake-v1`.
- Original run status: **partial**; four masks, three complete per-instance
  initializer manifests, one initializer failure. No training started and no
  four-branch checkpoints were produced by this intake test.
- Original implementation receipt records 64 passing local tests and clean
  local/server checkouts at that time. Those are historical checks, not a new
  test run performed while labeling the scheme.

| Instance | Discovery category | S / C / M / T initialization | Outcome |
| --- | --- | --- | --- |
| object_001 | number plate | tiles / blue / material / decorative | Initializer manifest written |
| object_002 | door | door / brown / material / grain | Initializer manifest written |
| object_003 | mailbox | box / orange / material / rejected | `embossed` splits into SD2.1 BPE IDs 23065 and 1804 |
| object_004 | door | pole / beige / glossy / marked | Initializer manifest written |

The two narrow door-region candidates remain; the per-crop VLM names one `pole`.
The receipt explicitly leaves whole-instance completeness unverified and mask
quality unaccepted by the user. Initializer completion is not scientific quality
acceptance or proof that these masks represent independent whole objects.

The Mailbox mask has descriptive IoU **0.9604056682411781** against the historical
reference, read only after prediction. The reference-check receipt originally
measured first-attempt `object_004`. During this documentation task its PNG SHA256
was verified identical to final-run `object_003`:
`44c8a935359f52c97f3cc788168d0103fc8482295be1f9531efff537a2052d0c`.
Thus the reported metric applies to that identical final mask, not to all four
masks or to exhaustive instance discovery.

An earlier attempt at commit `ba71002bb0a9b695a99be8882cdb924cd86b94e3`
was interrupted after producing 13 candidates including background and
overlapping parts. Those were not 13 accepted independent objects. Its source,
outputs and stop receipt remain preserved.

### Evidence locations

Server final run:

```text
/home/r12user5/Documents/Jiawei/colorpeel-runs/sam_multi_object_intake_v1/20261104__mailbox_intake__4b537fa
```

Local evidence root:

```text
C:/Users/lenovo/.codex/visualizations/2026/11/04/01a1a4e9-817a-7d01-ae16-5db6ed9b42b7/sam_multi_object_intake_20261104
```

Receipts within that root: `IMPLEMENTATION_RESULT.json`,
`MAILBOX_MASK_REFERENCE_CHECK.json`, `FIRST_ATTEMPT_STOP.json`,
`server_run/status.json`, `server_run/segmentation/objects.json`, and
`server_run/initializers/object_003/status.json`.

## B — bounded Mailbox test completed 2026-11-04

- Executed source: `3ef2ea6e6b6d3bda934702e0f9710c35040f7f2d`, fresh clean
  checkout `object-mask-b-3ef2ea6`; upstream ConceptExpress
  `db1dc61665866d208ba3a2e2152a83e7c6829e2f`, also clean.
- Same source photograph and SHA as A; verified cached SD2.1 base and the same
  SAM ViT-base revision. Seed 20, float32, physical GPU 0; no model downloads.
- Execution **succeeded** with exit 0, two coarse masks and two SAM refinements,
  no per-concept failures. No initializer step or training was run.
- Local related suite: 19 passed, one torch-dependent test skipped; all 20 passed
  on the server. Review found no blocking implementation defect. The initial
  local test attempt hit Windows sandbox TEMP permissions and was rerun with a
  writable external TEMP directory; no code workaround changed the algorithm.
- GPUs had no compute processes before the run or after completion. Existing
  non-training processes were left alone; source checkout remained clean.

Pre-clustering counts were `1580, 554, 162, 49, 13, 6, 2`. The selected level had
13 regions; EOT filtering kept five, and constrained post-clustering produced
two concepts. Visual review identifies `concept_001` as the number-plate tiles
and `concept_002` as the cylindrical mailbox body. Unlike A, this run does not
produce the two narrow door-region candidates. Fewer masks alone do not prove
more accurate or complete object discovery.

The mailbox's upper decorative backplate is already absent from B's coarse
mask and remains absent after SAM. All three sampled positive points lie on the
mailbox body. SAM improves the outline but does not recover the missing part.
The plate mask also excludes the surrounding wooden mounting frame; there is
no reference annotation here to decide whether that grouping is desired.

Historical Mailbox reference, read only after prediction:

| Mask | IoU | Precision | Recall |
| --- | ---: | ---: | ---: |
| A mailbox | 0.960406 | 0.978324 | 0.981287 |
| B coarse mailbox | 0.618261 | 0.872963 | 0.679387 |
| B SAM mailbox | 0.657318 | 0.970037 | 0.670941 |

These metrics describe one reference object, not dataset-wide quality. B does
not improve mailbox completeness in this test and has not replaced A. No
threshold/seed search or automatic quality acceptance followed. User acceptance
remains pending; no four-branch checkpoint was produced by this mask test.

The sampler and SAM candidate selection are our documented choices, not
recovered official SAM settings. Upstream FINCH's optional `pynndescent` warning
is non-blocking for this 4096-position exact-distance path. The SAM processor
reported an ignored `point_pad_value` option; every prompt contains exactly
three positive points, so no point-padding behavior is used.

Server results and execution log:

```text
/home/r12user5/Documents/Jiawei/colorpeel-runs/object_mask_ab_v1/20261104__mailbox_B__3ef2ea6__20
/home/r12user5/Documents/Jiawei/colorpeel-runs/object_mask_ab_v1/20261104__mailbox_B__3ef2ea6__20.log
```

Local evidence root:

```text
C:/Users/lenovo/.codex/visualizations/2026/11/04/01a1a4e9-817a-7d01-ae16-5db6ed9b42b7/object_mask_ab_v1_20261104
```

Contains `server_run/`, `execution.log`, `local_tests.txt`, `COMPARISON.json`,
`AB_comparison.png`, and `repro_outputs/`. Nine downloaded artifact hashes
(eight per-concept artifacts and the source image) match server receipts.
