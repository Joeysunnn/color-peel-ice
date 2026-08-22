# Official ColorPeel parameters and CLEVR reproduction locks

This table distinguishes values directly present in the official public
code/launcher from adaptations chosen in the approved CLEVR plan. The frozen
baseline run verified the listed CLEVR settings; diagnosis-first follow-up
variants remain pending unless stated otherwise.

| Item | Official public code / launcher | CLEVR 3×3 reproduction |
|---|---|---|
| Baseline commit | `021f5c74cee6c231a03b8b49bb96750cadfc4e06` | pinned |
| Backbone | `CompVis/stable-diffusion-v1-4` | unchanged |
| Diffusers | README states 0.17.0 | official source tag `v0.17.0` |
| Resolution | 512 | unchanged |
| Train batch size | launcher: 1; paper main text reportedly says 2 | 1 |
| Training steps | launcher: 1500 | 1500 after smoke review |
| Learning rate | `1e-5` | unchanged |
| `scale_lr` | enabled | enabled; single GPU × batch 1 × accumulation 1 leaves LR `1e-5` |
| LR scheduler | parser default `constant` | `constant` |
| Warmup | launcher: 0 | 0 |
| Seed | parser default 42 | 42 |
| CAA/cosine weight | launcher: 0.2 | 0.2 |
| Trainable UNet parameters | Custom Diffusion cross-attention K/V projections | unchanged |
| Text encoder | gradient mask targets modifier rows, while full-parameter AdamW decay may also drift zero-gradient ordinary rows | six shared token rows; literal ordinary-row drift is recorded, not restored, and not a failure criterion |
| Optimizer | AdamW β₁ 0.9, β₂ 0.999, weight decay 0.01, ε `1e-8` | unchanged |
| Gradient accumulation | parser default 1 | 1 |
| Maximum gradient norm | parser default 1.0 | 1.0 |
| Data workers | parser default 2 | 2 |
| Checkpoint interval | parser default 1000; launcher inherits it | unchanged at 1000 |
| Mixed precision | parser default `None`, therefore Accelerate configuration | explicitly `no` |
| xFormers | off unless flag passed | off because CAA needs attention maps |
| `hflip` | launcher passes flag; inspected dataset path does not apply it | retained as an official no-op |
| `center_crop` / `noaug` | parsed, no effective inspected data-path change | not used |
| Prior preservation | default off; inspected path lacks a complete implemented prior loss | off |
| Resume | argument exists; inspected path lacks effective resume implementation | not used |
| Official launcher tokens | `<s1*>+<s2*>+<c1*>+<c2*>+<c3*>+<c4*>` | `<s1*>+<s2*>+<s3*>+<c1*>+<c2*>+<c3*>` |
| Official launcher initializers | `cone+sphere+red+green+blue+yellow` | `cube+sphere+cylinder+red+cyan+gray` |
| Training prompt form | `a photo of <s*> shape in <c*> color` | nine prompts with the corresponding subject/color tokens |
| Training mask | 64×64 latent grid with 62×62 interior valid region at scale 512 | unchanged; no GT mask |
| Test parser | 100 steps, seed 42, CFG 6.0, 20 samples | honor 100/6.0 and use seeds 42–61 |

## Six learned tokens

| Token | Meaning | Initializer |
|---|---|---|
| `<s1*>` | cube | `cube` |
| `<s2*>` | sphere | `sphere` |
| `<s3*>` | cylinder | `cylinder` |
| `<c1*>` | red | `red` |
| `<c2*>` | cyan | `cyan` |
| `<c3*>` | gray | `gray` |

The frozen baseline configured `<c2*>` with `cyan`. A direct read-only check of
the cached SD 1.4 tokenizer found `cyan -> [1470, 550]`; the historical official
path did not enforce that an initializer encoded to exactly one token. The
baseline remains unchanged. Conditional follow-up training rejects multi-piece
initializers and allows a human-selected one-token replacement only for
`<c2*>`: `aqua [18613]`, `teal [22821]`, or `turquoise [19899]`. This is a
scientific initializer ablation, not an official parameter or compatibility
fix, and no candidate/run result is currently claimed.

## Environment compatibility locks

- Python 3.10
- PyTorch 1.13.1 + CUDA 11.7
- torchvision 0.14.1 + CUDA 11.7
- Transformers 4.30.2
- Accelerate 0.20.3
- huggingface-hub 0.15.1

Only Diffusers 0.17.0 and `transformers>=4.25.1` are backed by the inspected public repository guidance/files. The remaining exact versions are non-scientific server compatibility choices and must be recorded in the final environment freeze.

## Important code-versus-plan note

The official code passes the full text-embedding parameter to AdamW with weight decay 0.01. The locked reproduction policy is `literal_official_adamw_decay_allowed`: ordinary vocabulary embedding drift is measured and disclosed, no post-step restoration is added, and nonzero ordinary-row drift alone does not fail a smoke or full run.

Two real training smokes precede the full run. The two-step smoke uses the first two locked samples and imposes no gradient/delta requirement on unseen tokens. The independent nine-step smoke uses all nine cells once; every modifier token must have exposure count 3, at least one nonzero-gradient observation, and nonzero final delta. Launcher dry-runs are preflight only.
