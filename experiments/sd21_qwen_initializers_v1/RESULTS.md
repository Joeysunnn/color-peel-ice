# Completed Mailbox integration trial, 2026-11-03

**The Qwen-to-training-to-inference integration passed. Independent attribute transfer and background behavior did not pass visual review.** These are separate outcomes.

Training/inference revision: `cb169c6f1b4b8631c5dfae373145e217abf26ae3`, branch `codex/sd21-qwen-initializers-v1`. No TokenVerse, external preservation or other auxiliary objective was enabled; all jobs started from the pinned SD2.1 base.

## Initialization and implementation evidence

Qwen3-VL-8B-Instruct, pinned revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`, read the original512px orange Mailbox photo. FP16/eager on V100 used a measured peak of16.69GiB (PyTorch allocated memory). Existing VLM/training environments were retained.

| Branch | Initializer | SD2.1 ordinary token ID | Qualitative VLM confidence |
| --- | --- | --- | --- |
| Subject | mailbox | 31482 | high |
| Color | orange | 4287 | high |
| Material | painted | 6433 | medium |
| Texture | uniform | 11075 | high |

The first response recognized category `mailbox` but listed `box` as its Subject candidate. The generic selector was corrected to check the VLM's own single-word category first; the original candidates remain unchanged. A second full VLM call with the **same prompt produced byte-identical raw text**. The first attempt did not train. The raw Subject description still included orange; it is preserved as evidence but is not used as the Subject initializer or pasted into captions. Confidence is not a calibrated probability.

Other repairs: opt-in manifest replaces the historical word whitelist; single-BPE/special-token checks; raw response/manifest/tokenizer binding; actual copied1024-vector audit; adjective-compatible Material captions; matching RGB/mask stems; same-step adapter/embedding exports; and a four-residual SCMT processor with disjoint text-position gates. See [the whole-method audit and protocol](README.md).

## Technical verification

- All50 server tests passed, none skipped, including real tensor/attention tests.
- Same-source S11/C9/M8/T16 data:44 unchanged RGB images; only declared captions changed. Texture is the Mailbox albedo patch, not historical wood-spoon texture.
- Four fresh2-step startup runs and four fresh formal runs completed. Formal steps: S1000/C150/M1000/T1000; saved S/M/T500+1000 and C100+150.
- All3150 formal and8 startup loss records were finite. Each learned token had a nonzero gradient on every update. Ordinary embedding rows had exactly0 measured drift.
- All actual initializer copies equaled their frozen base vectors. Eight formal paired exports passed source/schema/finiteness/step checks.
-112/112 inference PNGs completed;105 unfiltered and7 safety-filtered. No unflagged all-black image was found. All112 downloaded PNG hashes matched the server ledger.
-14 fresh pipeline groups, prompt gates and loaded embedding vectors were checked. Every same-seed initial denoiser input matched across conditions. Both top-level and inference status are `succeeded`; this is not a visual-quality claim.

The seven filtered cases are M native/literal vase seed31, M learned sofa seed31, T native/literal vase seed31, T learned sofa seed31, and midpoint SCMT flowers seed31. Their black PNGs and flags remain visible; they are not counted as observed visual failures or compared as usable controls.

## Visual findings, seeds31/38

| Group | Observation | Limit on the conclusion |
| --- | --- | --- |
| Subject1000 | Recognizable arched backplate/curved lid, but repeated or malformed bodies and cropping. Flowers appear for both seeds; street and snow environments are weak or absent. | Initializer is still the historical `mailbox`; this does not establish a Qwen-induced change or prove overfitting. |
| Color150 | All8 learned cases retain recognizable chair/mug/vase/sofa categories and acquire vivid red-orange. | No new quantitative Lab target-matching metric was run. |
| Material1000 | All7 visible learned cases use gray studio backgrounds;6/7 are blue. Chairs deform, mugs lose handles, vases simplify to cylinders; sofa38 remains a sofa. | Natural `painted` controls often retain class/background, so the word alone is not an established cause. |
| Texture1000 | All7 visible learned cases strongly favor blue and gray studio framing. Some classes survive, but sofa38 becomes a cube; no independent transfer of a distinctive Mailbox spatial pattern is established. | The original has little obvious pattern; derived albedo contains weak bands. `uniform texture` literal controls are themselves imperfect ground truth. |
| SCMT midpoint/final | Street remains planar/studio-like; snow is mainly dots on gray. Mid flowers38 has blurred flower-like shapes, while final flower cases lose that response. Final flowers38 gives a purple/gray mailbox on orange background, indicating foreground/background color misbinding. | Midpoint already fails; all four checkpoint steps change together, so this cannot isolate a branch or attribute the problem solely to1000 steps. |

Keep the initializer functionality, but do not adopt these M/T weights as a successful disentanglement baseline. Remaining priorities are data/target quality (especially a nearly plain Texture source), shape/background leakage, and spatial preservation. Token-local text gating does not enforce spatial locality. Do not interpret this round as solving background transfer or as an initializer-only causal ablation: Material caption grammar and the Texture source also differ from the earlier wood-texture SCT experiments.

## Records

Server: `/home/r12user5/Documents/Jiawei/colorpeel-runs/sd21_qwen_initializers_v1/20261103-174000__mailbox_four__cb169c6__42/experiment`.

Local review root: `C:/Users/lenovo/.codex/visualizations/2026/11/02/01a19a4e-9ff6-77d1-8cab-1cf360089268/sd21_qwen_initializers_20261103`.

- `method_audit.html`: Chinese method audit and all result links.
- `server_run/experiment/dataset/index.html`: all44 actual training RGB images/captions.
- `server_run/experiment/inference/index.html`: all112 planned/generated comparisons.
- `verification.json`: local metadata, vector-receipt, prompt-gate, latent and PNG verification.
- `subject_results.png`, `color_results.png`, `material_results.png`, `texture_results.png`, `scmt_seed31.png`, `scmt_seed38.png`: complete labelled contact sheets.

The173000/174000 directory labels are run identifiers. Actual launcher timestamps are recorded in the run manifests. The initial prepare command lacked `COLORPEEL_RUN_ROOT`; it failed before creating an experiment and was retried with the existing launcher's required environment variable. Both logs are retained. No training-environment upgrade, source mutation on the server, historical artifact deletion or prior-job interruption was performed.

The single-image recognition entry accepts new images. Fully automatic new-image target masks, TRELLIS/SuperMat processing and derived-data approval are **not** claimed: this first data resolver explicitly rejects images other than the audited Mailbox source.
