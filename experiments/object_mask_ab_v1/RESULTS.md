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

## B — pending execution

The candidate is ConceptExpress coarse-mask extraction followed by SAM point
refinement, under the fixed protocol in [README.md](README.md). No B mask-quality
result, comparison winner, or user acceptance is claimed here. Append the actual
source commit, command, assets, output locations, failures and observations after
the run; preserve A as the baseline.
