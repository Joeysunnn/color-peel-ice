# Material initializer-only comparison

Fresh SD2.1-base512 LoRA64 runs for `painted`, `glossy`, and `material`, each200 steps
with paired adapter/embedding checkpoints150 and200. All reuse the exact Qwen Mailbox
Material8 images, masks, captions and seed42 recipe. No old learned weights are reused.
The explicit ablation manifest preserves and hash-locks the original Qwen decision;
the two alternative words are user-selected interventions, not new VLM predictions.

Each checkpoint automatically generates44 material-only images: chair/vase/sofa/mug,
seeds30-40, PNDM100,512px,FP16,CFG3.5,empty negative,safety checker on. Total264 images.
Initial denoiser input hashes are checked across variants and checkpoints. Training
configs may differ only by initializer/provenance, run identity and GPU assignment;
bitwise training reproducibility is not claimed. No composition or automatic visual review.

Run from a clean committed checkout in `colorpeel017`:

```sh
python -u -B -m experiments.sd21_material_initializer_ablation_v1.run --output FRESH_EXTERNAL_OUTPUT
```

GPUs0/1/2 must be idle; no process is stopped. Each variant is an isolated subprocess.
Open `index.html` for comparisons at150/200; variant folders contain paired `ckpts/`,
training logs, `train_outputs/`, inference manifests and individual galleries.
The parent runs the finite train/export/infer sequence; no agent supervision or rerolls.
