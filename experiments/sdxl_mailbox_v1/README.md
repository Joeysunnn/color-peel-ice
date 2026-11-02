# SDXL 1.0: Mailbox four-branch migration

Train Subject, Color, Material and Texture separately on the official SDXL 1.0
base, then run each branch's own transfer probes. Two methods remain: fullrank
token-local K/V residuals plus learned token vectors, and rank64/alpha64 K/V
residuals plus learned token vectors. Fullrank does not train the full UNet.
No composition and no refiner are included.

| Branch | Frozen input | Token / initializer | Updates / paired snapshots | Transfer |
| --- | --- | --- | --- | --- |
| Subject | Mailbox27: 24 renders + original and blue/green photos | `<S*>` / mailbox | 1000 / 400,600,800,1000 | 9 material/scene/color prompts |
| Color | Orange9: three shapes x three views, rendered gray background | `<C*>` / orange | 150 / 100,150 | 10 unchanged Color prompts |
| Material | Mailbox8: fixed reviewed ROI roughness/metallic medians | `<M*>` / metal | 1000 / 400,600,800,1000 | chair/mug/vase/sofa, native/literal/learned |
| Texture | Mailbox16: reviewed grayscale albedo patch, two nuisance materials | `<T*>` / texture | 1000 / 400,600,800,1000 | chair/mug/vase/sofa, native/literal/learned |

Mailbox Material historically uses `metal`; the later `material` initialization
belongs to the wood experiment. Mailbox literal descriptions are `painted metal`
and `fine speckled`. New S/M/T dose follows the latest common 1000-step recipe;
it is not a claim that SDXL needs the same number of updates as SD2.1.

## Necessary SDXL changes

- Use the official cached base revision `462165984030d82259a11f4367a4eed129e94a7b`.
  `prepare_model` hashes its actual components and creates a new external linked
  bundle. It does not download by default or change old models.
- Learn one target vector in each of SDXL's two text encoders. Concatenate their
  penultimate hidden states (768+1280), retain encoder2's projected EOS pool,
  and provide original-size/crop/target-size micro-conditioning. Added token IDs
  must not replace EOS as the pooling position.
- Use the native SDXL UNet and preserve native self-attention. Only cross-attention
  K/V receives the target-position residual; ordinary text positions retain the
  base projection. Ordinary token embeddings and all frozen weights stay fixed.
- Copy and hash-check the original 512px images, captions and masks. Training
  resizes RGB to 1024 using Lanczos and masks using nearest sampling; this does
  not create new image detail. No new render, recolor, crop or augmentation is
  introduced. Fixed source order is retained. S/M/T use the object mask; Color
  uses the valid full-image region. Retain the old one-latent-pixel valid border
  and channel-sum mask-loss normalization.
- On V100, use mixed precision with frozen UNet in FP16, FP32 trainable adapter
  parameters/token vectors and FP32 VAE. This differs from the old all-FP32
  training. Record actual runtime dtypes, finite losses and gradient checks in
  smoke evidence before formal launch. BF16 is not used.
- Training seed42, batch1/accum1, constant LR, embedding LR1e-5, fullrank KV1e-5,
  rank64 KV5e-5, AdamW betas .9/.999, eps1e-8, weight_decay0, max_grad_norm1.
  No new loss or text-encoder transformer tuning is introduced.

These changes make this an SDXL migration experiment, not a strict base-only
ablation. In particular, an apparently better result can depend on resolution,
conditioning and precision as well as the base model.

## Transfer and evidence

PNDM100, 1024px, no added negative prompt, CFG3.5 for S/M/T and CFG6 for Color.
Use seeds30-40 inclusive for every branch in this first pass. Subject contributes
792 images, Color440, Material1056, Texture1056: **3344 planned images** across
both methods and all snapshots. Native/literal M/T controls use the same loaded
checkpoint with no learned token in the prompt and an all-zero K/V gate. The
learned residual contributes zero at every text position. These are controls
through the adapted pipeline, not independently loaded stock pipelines.

SDXL does not natively include the old checker. The bundle separately links the
same frozen SD1.4 safety checker/feature extractor and inference explicitly runs
it after SDXL decoding. The receipt records that actual provenance.

Each `checkpoint-<step>.pt` pairs that step's adapter with both token vectors,
optimizer and RNG state plus immutable base/config provenance. Frozen base
weights need not be copied into every checkpoint. Transfer loads each matching
pair, not the final token vector for an earlier adapter.
Each config also locks the execution Git commit, model-source receipt SHA256
and copied dataset-manifest SHA256; these locks travel with its checkpoint.

Each branch/method has config, training metrics/audit/checkpoints, transfer
images/ledger/review and `train_outputs/` evidence. The shared root contains
`experiment.json`, per-GPU queue statuses, copied input provenance and an HTML
entry page. Failed directories remain evidence and are never reused.

## Deployment and execution

The new environment is
`/home/r12user5/Documents/Jiawei/colorpeel-assets/sdxl_v1/env`.
Its `borrowed_trellis_runtime.pth` exposes existing TRELLIS and univlr
site-packages read only; the new environment installs Diffusers0.35.1,
Accelerate1.11.0 and PEFT0.17.1 locally. It inherits Torch2.6.0,
Transformers4.57.3, Hugging Face Hub0.36.2 and xformers0.0.29.post3.
Inherited PEFT0.15.2 was incompatible with the new Diffusers; the new local
PEFT installation overrides it without altering the source environment.

This is a separate overlay environment, not a self-contained copy: it depends
on both borrowed directories remaining available and unchanged. Record actual
module paths and versions externally, and retain those directories. The
requirements file lists observed relevant versions; it is not a complete
transitive dependency lock. `colorpeel017`, TRELLIS and univlr remain unchanged.
Local edit/test/commit/push must precede a fresh server checkout/pull; the server
executes only committed source.

```bash
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1
SDXL_PY=/home/r12user5/Documents/Jiawei/colorpeel-assets/sdxl_v1/env/bin/python
$SDXL_PY -B -m experiments.sdxl_mailbox_v1.prepare_model
$SDXL_PY -B -u -m experiments.sdxl_mailbox_v1.run \
  --output "$FRESH_SMOKE_DIRECTORY" --gpus 1 2 3 --smoke --execute
$SDXL_PY -B -u -m experiments.sdxl_mailbox_v1.run \
  --output "$FRESH_FORMAL_DIRECTORY" --gpus 1 2 3 --execute
```

Every invocation consumes a new external directory, including dry runs (omit
`--execute`). Smoke performs all eight two-update trainings, step2 paired exports
and one learned image per pair. Formal launch follows meaningful CPU tests and
successful runtime smoke. GPU indices above are examples and must be checked
against live process ownership before launch. Never terminate other tasks.

The user supplied home494.19GiB and var/tmp30.92GiB and requested no further
capacity query. New outputs live on home; the runner does not query disk space.
Finite GPU queues execute training then transfer and exit. Verify startup and
first finite losses, then disconnect; do not add a monitoring service or resume
the previously paused wood Subject27 transfers. Successful execution does not
establish four-branch disentanglement or transfer quality.

SDXL architecture references: [official model](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
and [Diffusers SDXL training](https://github.com/huggingface/diffusers/blob/v0.35.1/examples/dreambooth/train_dreambooth_lora_sdxl.py).
