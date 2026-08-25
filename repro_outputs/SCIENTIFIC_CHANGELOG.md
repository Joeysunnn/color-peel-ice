# Scientific changelog

`confirmed` below refers only to this CLEVR adaptation's immutable server
artifacts. It is not an exact-paper-score reproduction or a claim that
entanglement is solved.

| Change | Status | Scientific/comparability effect |
|---|---|---|
| Replace official examples with nine CLEVR metal images | confirmed by 48-sample audit and nine-link staging | Changes domain and prevents exact paper-score comparison |
| Change token roles from 2 subject + 4 color to 3 + 3 | implemented | Keeps six tokens but changes concept allocation |
| Keep official prompt form for all nine cells | implemented/tested | Preserves conditioning form |
| Fix last modifier-token gradient boundary | implemented/tested | Makes token six trainable; highest verified patch risk |
| Literal official AdamW decay on ordinary vocabulary rows | confirmed in 2-, 9-, and 1500-step runs | Drift is observed/reported, never restored, and not a failure by itself |
| Preserve SD 1.4, Custom Diffusion K/V, CAA weight 0.2 | confirmed through 1500 steps | Core ColorPeel anchor |
| Keep GT masks out of training | implemented/tested in staging | Avoids ICE mask supervision |
| Deterministic 900-item sampling protocol | confirmed: 900/900 valid | Evaluation adaptation only |
| CLEVR scorer and axis contingency tables | confirmed | CLEVR diagnostic absent from paper |
| Color metrics using external masks | confirmed: 588 scored, 12 failures reported | Grounded-SAM is an external evaluation adaptation |
| Qwen3-VL deterministic predictions | confirmed: 300/300 valid JSON | Independent external-model evaluation stage |
| Server compatibility pins | environment created, frozen, and verified | Operational only |
| Accelerate 0.20.3 logging API compatibility | fixed and rerun at `c8c874d` | Logging configuration only; no scientific semantics changed |
| GitHub-only server deployment | confirmed through `c8c874d` | Workflow/provenance change only |
| Two real training smokes | confirmed | Two-step covers first two samples with no unseen-token requirement; nine-step covers all cells with exposure 3 per token |
| Grounding DINO 4.48.1 keyword compatibility | fixed at `b059bd5`; rerun confirmed | Operational API compatibility only; no threshold/model change |
| Independent Grounded-SAM and Qwen3-VL stages | confirmed | Keeps external model provenance and failure ledgers separate from generation/scoring |

Explicit exclusions: no ICE Stage Two, ICE loss, ICE token method, ICE baseline,
material token, alternative checkpoint, GT-mask training loss, or self-created
success threshold.

## Diagnosis-first follow-up changes

| Change | Status | Scientific/comparability effect |
|---|---|---|
| Freeze report-01 checkpoint and outputs | confirmed as workflow decision; existing run provenance retained | Creates a non-overwritable comparison anchor; no model-semantic change |
| Make human review primary for image semantics | approved; current review is qualitative only | Automated evaluators become secondary; no rate may be inferred without a per-item ledger |
| Diagnose cyan across trained/vanilla models, two prompt families, and learned/literal candidates | completed; 540/540 images `ok`; qualitative folder review completed | Separates base-model, K/V, token, initializer-candidate, and prompt-template hypotheses without retraining |
| Validate initializers as exactly one tokenizer token | implemented; server tokenizer IDs verified | Corrects silent use of a piece from a multi-piece word; changes initializer semantics for future runs |
| Change only `<c2*>` initializer to locked `turquoise` | dual smoke, 1500-step training, 900-image inference, and qualitative human gate completed | Scientific ablation against the frozen baseline; CAA, AdamW, mask, prompt, data, seed, steps, and other initializers stayed fixed |
| Three-stage black-image/SafetyChecker diagnosis | completed after Stage 2: checker-on FP16 19/19 flagged and black; checker-off FP16 19/19 finite and nonblack; FP32 not required | Confirms the exact-black outputs were checker filtering; diagnostic outputs do not replace baseline images |
| Disable SafetyChecker | executed only in isolated diagnostics with explicit acknowledgement | Safety-relevant and generation-semantic change; never applied to the frozen baseline artifacts |
| Disable confirmed false-positive SafetyChecker for matched turquoise evaluation | completed; 900/900 rows recorded paired acknowledgement, 0 black outputs | Prevents post-generation all-zero replacement; changes generation semantics but not training or the frozen baseline |
| Multiview held-out protocol and fold preparation | 180/180 rendered and validated; staging complete; human review pending; no training | Changes the data/view distribution and evaluation split; not comparable to the single-view baseline as a setting-only rerun |
| Add versioned `multiview_render_v2` orbit camera | implemented, unit-tested, real-smoke verified, and realized 180/180; human review passed; all nine Fold runs completed; v1 untouched | Sole scientific change is camera viewpoint sampling (±18° azimuth, ±10° elevation, ±5% distance) around the object center; all training and non-camera render semantics remain locked |
| Complete-bundle Fold held-out evaluation | protocol and auditable generation/bundle/Qwen/scoring adapters implemented; generation pending | Evaluation-only change: 3 folds × 3 training seeds × 9 complete prompts × 20 paired seeds; reports seen/held-out and both intervention directions without a single entanglement score |
| Factor-aware loss | conditional; no approved config or run | Would change the optimization objective and require a separate method-level ablation |
| Natural multi-object evaluation | conditional; no approved config or run | Extends the task domain; cannot retroactively validate the CLEVR baseline |

Verified tokenizer evidence from the server `colorpeel017` SD 1.4 cache:
`cube [11353]`, `sphere [6987]`, `cylinder [22092]`, `red [736]`,
`cyan [1470, 550]`, `gray [7048]`, `aqua [18613]`, `teal [22821]`, and
`turquoise [19899]`. The check used `AutoTokenizer` with
`local_files_only=True`.

## Single-object material extension

| Change | Status | Scientific/comparability effect |
|---|---|---|
| Add independent `clevr_subject_color_material_3x3x2` stage | implemented locally; runtime pending renderer gate | Does not overwrite two-axis artifacts |
| Add `<m1*>` metal and `<m2*>` rubber | implemented; tokenizer runtime preflight pending | Extends the learned bundle from six to eight tokens |
| Add request-controlled native CLEVR material | implemented as renderer v3 | Sole renderer scientific change from v2; camera, lights, background, scale, pose and Cycles settings remain v2-identical |
| Pair metal/rubber render seeds by shape-color-view | implemented/tested | Holds camera, lights and Cycles noise fixed across material interventions |
| Extend unchanged CAA to three prompt tokens | code unchanged; formula regression documented | CAA 0.2 now weights the mean of subject-color, subject-material and color-material pairs |
| Preserve literal official AdamW decay | unchanged | Ordinary vocabulary drift remains record-only |
| Add three-axis held-out evaluation | implemented/tested; not run | Reports shape/color/material/joint and three intervention directions, with no composite entanglement score |
| Add two human gates | implemented | Rendering review blocks all training; full-grid generation review blocks Fold training |

The checkpoint-1000 accelerator state remains evidence only: the upstream
training entry does not load resume state, so training recovery is not claimed.
