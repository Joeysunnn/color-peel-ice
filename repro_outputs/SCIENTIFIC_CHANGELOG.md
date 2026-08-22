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
