# Subject27 at400/600 + Texture600

The user paused the large standalone Subject transfer sweep after viewing enough
images. Both27-image Subject trainings completed1000 updates successfully.
Only their transfer generation processes were interrupted; no checkpoint or
existing PNG is removed. The older runner may record an intentional interruption
as failed; the separate pause receipt records the actual user-requested reason.
Do not restart that sweep as part of this task.

Run four matched-method combinations, with both same-step embeddings and K/V:

| Subject input | Subject step | Texture source | Methods |
| --- | --- | --- | --- |
| approved27 images,675609d | 400 | previous natural Texture16 SD2.1,step600,19049ce | original,rank64 |
| approved27 images,675609d | 600 | same Texture600 | original,rank64 |

Prompt: `a photo of <S*> spoon with <T*> texture`.
Seeds30–60 inclusive;4×31=124 images;PNDM100/CFG3.5/512px/fp16/common checker.
The historical dual adapter's internal material slot carries `<T*>`; no Material
token or adapter is loaded. No training or checkpoint re-exporting occurs.

Source checks require completed training manifests at their recorded Git/configs,
the static paired-export index/receipts, exact snapshot tensor equality,1024
dimensions, matching methods and the same pinned model. Subject400/600 and
Texture600 all have1000 total training updates. **Do not require the interrupted
transfer sweep to have succeeded** or lock mutable generation status/provenance
as evidence for the learned weights. Static checkpoints and receipts are enough.

Deploy local edits/tests/commit/push, then server clone/pull into a clean separate
checkout. Choose currently free GPUs and use a new external experiment directory:

```bash
export PYTHONDONTWRITEBYTECODE=1 TMPDIR=/var/tmp
export COLORPEEL_OUTPUT_ROOT=/var/tmp/colorpeel-r12user5/runs
/home/r12user5/miniforge3/envs/colorpeel017/bin/python -u -B \
  -m experiments.sd21_wood_subject27_v1.composition \
  --output "$FRESH_COMPOSITION_DIRECTORY" --gpus 2 3 --execute
```

Output: `S400/{fullrank,lora_r64}` and `S600/{fullrank,lora_r64}` under one new
root. `index.html` compares four columns by seed; per-case pages show all actual
27 Subject and16 Texture RGB/captions. Source audits retain historical training
Git separately from the new inference Git. The pause receipt is copied for context.
Each GPU executes a finite queue, without retries or monitors. Verify first
images, then disconnect; do not resume old transfer or overwrite old results.
