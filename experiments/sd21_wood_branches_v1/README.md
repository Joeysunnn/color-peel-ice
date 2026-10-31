# SD2.1-base wood spoon: unchanged single-branch methods

This experiment replaces the SD1.5 base with pinned **SD2.1-base 512**. It reuses
the actual, frozen RGB/mask/caption datasets and the established training code.
It does not load any SD1.5 learned embedding or adapter, regenerate data, combine
branches, or change the loss, optimizer, learning rates, augmentation or ordering.
The shared trainer and all historical experiment entrypoints stay unchanged.

| Branch | Training images | Token / initializer | Methods | Updates / paired checkpoints |
| --- | --- | --- | --- | --- |
| Subject | TRELLIS blue/green72 | `<S*>` / spoon | original, rank64 | 1000 / 400,600,800,1000 |
| Material | natural wood_spoon8 | `<M*>` / material | original, rank64 | 1000 / 400,600,800,1000 |
| Texture | natural wood_spoon16 | `<T*>` / texture | original, rank64 | 1000 / 400,600,800,1000 |

Original means fullrank **token-local K/V residual plus target embedding**, not
full UNet training. Rank64 uses the existing rank64/alpha64 token-local K/V LoRA.
Embedding LR=1e-5; K/V LR=1e-5 original, 5e-5 rank64; batch/accumulation=1/1;
seed42; float32 training; no hflip; constant LR; no attention loss. The existing
object-mask loss, including its channel-sum normalization, is preserved.

The original official SD2.1 HF endpoint was inaccessible during intake. The
server already holds `sd2-community/stable-diffusion-2-1-base` revision
`4e63672c03103b6c636b8fb4119ba982469b2955`. Its 1024-dimension text/cross-attention,
23-layer converted OpenCLIP encoder, linear projections, 64px latent and epsilon
prediction are checked explicitly. The bundle links these cached components and
the same frozen SD1.4 checker/feature extractor used by the SD1.5 experiments.
All source/component bytes are hashed. No CLIP skip is added. This records the
actual mirror provenance; it does not claim a fresh download from Stability AI.

Inference remains PNDM100, 512px, fp16, CFG3.5, checker on, no negative prompt.
Subject uses the existing nine material/scene/color transfer prompts. Material
and Texture each use chair/mug/vase/sofa with native/literal/learned arms, using
the inherited literal words wood / vertical grain. Seeds30–60 inclusive yield **8184**
single-branch transfer images across the four checkpoints. Each snapshot pairs
its own embedding with its own adapter; the final snapshot is checked against
the final trainer exports. Actual training RGB/captions are included in review.

Literal spoon is an independent **stock** SD2.1 pipeline/UNet/attention baseline:
`a photo of a spoon`, CFG3.5/5/7.5 × seeds30–60, 93 images. It has no training
steps or learned checkpoints. No trained token or adapter is loaded. Together
the formal experiment generates 8277 images. Training seed remains42.

## Execution

Source changes must be committed and pushed locally, then cloned/pulled on the
server. Never edit the server checkout. Use fresh output directories, including
for failed runs, smoke runs and dry runs. Source assets and old results are read
only. The new output root uses `/home` because the live `/var/tmp` free space
cannot hold the approximately 50GB of complete SD2.1 training states; no old
results are moved or deleted.

```bash
export PYTHONDONTWRITEBYTECODE=1 TMPDIR=/var/tmp
export COLORPEEL_OUTPUT_ROOT=/home/r12user5/Documents/Jiawei/colorpeel-runs/sd21_wood_branches_v1
TPY=/home/r12user5/miniforge3/envs/colorpeel017/bin/python
$TPY -B -m experiments.sd21_wood_branches_v1.prepare_model
# New directory for each invocation. Smoke = six 2-update runtime checks,
# paired step2 exports, six learned images and one stock image.
$TPY -B -u -m experiments.sd21_wood_branches_v1.run \
  --output "$NEW_SMOKE_DIRECTORY" --gpus 0 2 3 --smoke --execute
$TPY -B -u -m experiments.sd21_wood_branches_v1.run \
  --output "$NEW_EXPERIMENT_DIRECTORY" --gpus 0 2 3 --execute
```

Omitting `--execute` prepares a dry-run matrix only. Each GPU runs a finite
training→paired-export→inference queue. All branches, methods, checkpoints,
literal baseline, configs, logs and review pages are under the same experiment
root. `experiment.json`, `gpu_N.json`, per-method `train_outputs/` and inference
ledgers record actual status. A queued job is not a completed job. After startup
verification, disconnect; do not create monitoring jobs or change this checkout
while queues are active. Training success does not establish disentanglement or
better spoon geometry; those require visual review of the generated results.
