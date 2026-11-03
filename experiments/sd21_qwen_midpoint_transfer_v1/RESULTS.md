# Completed midpoint transfer, 2026-11-03

The requested S/M/T500 and C100 inference finished: 242 images, seeds30–40, with 227 unfiltered outputs and 15 retained safety-filtered black PNGs. No training occurred. This is the current Mailbox/Qwen round (`mailbox`, `orange`, `painted`, `uniform`), not the earlier wood-spoon Texture experiment.

| Group | Total | Unfiltered | Filtered |
| --- | ---: | ---: | ---: |
| Subject500 | 55 | 50 | 5 |
| Material500 | 44 | 44 | 0 |
| Texture500 | 44 | 44 | 0 |
| Color100 | 55 | 55 | 0 |
| S500+C100+M500+T500 | 44 | 34 | 10 |

41 focused server tests passed without skips. All five pipelines and the parent finished successfully. Source hashes were rechecked after inference; all downloaded PNG hashes, 512px RGB format, exact prompts/checkpoints/seeds, actual learned-vector receipts and token gates passed verification. Same-seed first denoiser inputs match across all groups. The eight repeated SCMT probes at seeds31/38 are byte-identical to the prior midpoint outputs. GPUs were released after completion.

## Visual observations

All242 positions were reviewed, including retained filtered outputs. These are visual observations, not quantitative scores or a causal ablation.

- Subject responds strongly to red and often to snow/flowers. Street33 has a clear street backdrop; most other street outputs remain simple backgrounds. Repeated mailbox parts, cropping and shape defects persist. White often becomes pale blue/purple.
- Material generally preserves object categories, with exceptions such as handleless mug30/33. Blue/purple color and painted/chipped styles are common. Backgrounds are not all gray studios: tables and colored/brick walls occur, and painted effects sometimes extend into the environment.
- Texture preserves many vases/mugs, but hair and full-sofa transfer remain unreliable. Hair30/35 become circular objects,31 a textured plane; sofa often becomes upholstery closeups. No consistent independently identifiable `uniform` texture is established.
- Color produces generally orange vases, sofas, parrots and roses with recognizable categories. Bowl is a notable failure:10/11 contain orange fruit; several bowls remain gray/black/white, and34 lacks a clear bowl. The cause of fruit intrusion was not isolated.
- SCMT generally retains orange mailbox appearance but not reliable scene control. None of the9 unfiltered street outputs clearly depicts a street. Snow30/37 show snowy greenery, while others often show dot patterns; flowers vary between floral/color-blob backdrops and plain backgrounds. Duplicated/cropped parts remain.

## Evidence locations

- Inference commit: `445ea2a3dc8ac83b48d2734b8c82aa4430f7c428`, branch `codex/sd21-qwen-midpoint-transfer-v1`.
- Source training commit: `cb169c6f1b4b8631c5dfae373145e217abf26ae3`.
- Server run: `/home/r12user5/Documents/Jiawei/colorpeel-runs/sd21_qwen_midpoint_transfer_v1/20261103-191800__mailbox_midpoints__445ea2a__30`.
- Server gallery: that run's `experiment/index.html`, with five group galleries and all original PNGs.
- Local review: `C:/Users/lenovo/.codex/visualizations/2026/11/02/01a19a4e-9ff6-77d1-8cab-1cf360089268/sd21_qwen_midpoint_transfer_20261103/review.html`.
- Local evidence: `verification.json`, `qualitative_review.json`, `RUN_LOCATIONS.json`,27 contact sheets and complete image/metadata copy under `server_run/experiment`.

PNDM100,512px,FP16, empty negative prompt, checker enabled; Color-only CFG6 and others CFG3.5. No rerolls or filtered-output removal. Original checkpoints and prior experiments were preserved.
