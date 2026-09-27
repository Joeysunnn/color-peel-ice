# Perfusion-style mailbox Subject pilot

This is an independent Subject adaptation experiment. Existing Subject token-local K/V checkpoints, the selected ground-reflection Material checkpoint, and all frozen results remain read-only.

## Mechanism

For each cross-attention layer, the P1 `<S*>` Key is the frozen base `to_k` projection of the contextual `mailbox` encoding from `a photo of a mailbox`. The reference is computed once before training, stored in `mailbox_key_reference.pt`, and reused at inference. No Subject Key parameter is learned. The `<S*>` Value is base `to_v(x)` plus `a (bᵀx)` with fixed alpha 1.0, one rank-1 pair per layer. `a` starts at zero and `b` at small random values so the initial edit is zero and optimization can begin. At ordinary positions, base K/V projections are used without an adapter update. Self-attention and base SD are frozen.

This is a **Perfusion-style variant**, not a reproduction of the paper's dynamic similarity gate, covariance-weighted edit, background loss, or all-position Value changes. The [Perfusion paper](https://arxiv.org/abs/2305.01644) motivates the Key-Locking and low-rank Value constraint; this pilot follows the stricter token-position-only mechanism requested here.

## Matched cohorts

| Cohort | Frozen B0 Subject | P1 training inputs | P1 config |
| --- | --- | --- | --- |
| `original` | 2026-09-21 exposure-matched token-local K/V | Same 25 asset rows, prompts, masks, seed 42, 5000 steps and optimizer settings | `configs/mailbox_keylocked_rank1_value_5000.yaml` |
| `balanced_aligned` | 2026-09-27 caption-aligned balanced token-local K/V | Same 10 color/material rows, prompts, masks, seed 42, 5000 steps and optimizer settings | `configs/mailbox_balanced_aligned_keylocked_rank1_value_5000.yaml` |

The launcher verifies each input file against its frozen asset manifest and rejects any training-setting change other than the adaptation mode. The original source has five color families and 25 exposure-matched rows. The balanced source has five color families and two material captions per color. These cohorts answer separate within-cohort mechanism questions; comparing their outputs directly does not isolate the Subject adaptation mechanism.

Before the full runs, use `smoke_forward.py` with the selected run root on the server to check one actual pipeline forward for S-only and S+M. It uses zero-initialized P1 Value edits and the frozen B0 Subject embedding solely to test integration; it is not a scientific comparison result.

## Evaluation

Each cohort generates 36 new images: B0/P1 × plain/red/blue × Subject-only/Subject+selected-M × seeds 42–44. Red and blue prompt strings, CFG 3.5, 100 sampling steps, and the selected `<M*>` checkpoint come from the frozen mailbox diagnostic. The plain group adds a no-color-word control to both arms. Safety-filtered outputs are recorded as unavailable for visual review. Review mailbox identity, ordinary red/blue prompt response, metal appearance in Subject-only, and whether adding `<M*>` changes material while retaining identity.

The two P1 training runs and two evaluation runs use separate immutable directories under `$COLORPEEL_RUN_ROOT/perfusion_subject_pilot/`. Run them through `scripts/launch/colorpeel_run.py` with new run IDs after committing code and verifying a clean checkout. Set `COLORPEEL_PERFUSION_SUBJECT_RUN` to the matching completed P1 training directory for each evaluation config. Never reuse a dry-run or completed run directory.

The original cohort's red source caption is known to mismatch the visual source color, while red is held out in the balanced aligned cohort. Treat that as an interpretation limit, not as evidence about the adapter alone. Key-Locking may also weaken Subject identity over long training; inspect identity and prompt response together at the matched 5000-step endpoints.
