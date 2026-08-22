# Scientific changelog

No experiment has run. `implemented` below means code plus local tests, not a
paper-result reproduction.

| Change | Status | Scientific/comparability effect |
|---|---|---|
| Replace official examples with nine CLEVR metal images | implemented; real staging pending | Changes domain and prevents exact paper-score comparison |
| Change token roles from 2 subject + 4 color to 3 + 3 | implemented | Keeps six tokens but changes concept allocation |
| Keep official prompt form for all nine cells | implemented/tested | Preserves conditioning form |
| Fix last modifier-token gradient boundary | implemented/tested | Makes token six trainable; highest verified patch risk |
| Literal official AdamW decay on ordinary vocabulary rows | locked; not run | Drift is observed/reported, never restored, and not a failure by itself |
| Preserve SD 1.4, Custom Diffusion K/V, CAA weight 0.2 | configured; not run | Core ColorPeel anchor |
| Keep GT masks out of training | implemented/tested in staging | Avoids ICE mask supervision |
| Deterministic 900-item sampling protocol | implemented/tested; not generated | Evaluation adaptation only |
| CLEVR scorer and axis contingency tables | implemented/tested; no predictions | CLEVR diagnostic absent from paper |
| Color metrics using external masks | implemented/tested; no generated masks | Grounded-SAM remains an external runtime prerequisite |
| Qwen3-VL deterministic predictions | implemented/configured; not run | Independent external-model evaluation stage remains pending |
| Server compatibility pins | implemented in source; environment removed/not run | Operational only |
| GitHub-only server deployment | completed for `41d752a` | Workflow/provenance change only |
| Two real training smokes | configured; not run | Two-step covers first two samples with no unseen-token requirement; nine-step covers all cells with exposure 3 per token |
| Independent Grounded-SAM and Qwen3-VL stages | implemented/configured; not run | Keeps external model provenance and failure ledgers separate from generation/scoring |

Explicit exclusions: no ICE Stage Two, ICE loss, ICE token method, ICE baseline,
material token, alternative checkpoint, GT-mask training loss, or self-created
success threshold.
