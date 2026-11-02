# SD2.1 Texture background-preservation continuation

This is the authorized limited LoRA experiment following the unsuccessful frozen-weight Texture screen. Both arms restore the same original wood Texture16 **complete checkpoint400** (adapter, text encoder, optimizer, scheduler, RNG and custom state), then train through600 and save500/600. `lambda0` retains the original objective; `lambda1` adds unit-weight teacher consistency outside a 3x3-dilated foreground, excluding the latent border. No fullrank training, new data, or automatic continuation to1000 is included.

The original RGB/masks/captions, fixed data order, seed42, batch1, FP32, rank64/alpha64, adapter LR5e-5, embedding LR1e-5 and constant scheduler are inherited exactly. The source configuration is checked against the original successful run before any change. GPU0 alone is visible; SDXL and GPUs1/2/3 are untouched.

The source400 checkpoint belongs to a completed1000 run. A separate resume package copies all six state files and the exact first400 metric lines. Its audit is explicitly **derived**, not an original final400 audit: prefix exposure/gradient counts are inferred only if the original observed/exposure/nonzero counters all equal1000 and every metric row is finite and continuous. Its embedding drift is recomputed from the actual400 encoder versus the pinned initial model. Source and copied hashes are checked before each subprocess and after evaluation; old files are never modified. Checkpoint400 ends epoch25 for the16-image loader. Accelerator RNG is restored; worker RNG is not checkpointed, so equality with historical600 is measured, not assumed.

Before training, an actual512px T400 batch must produce finite nonzero background gradients for the adapter and Texture embedding, with a detached teacher and zero adapter/T gradients in class-only and unconditional controls. Four LoRA group norms are reported; passing requires at least one adapter group and the T row to be nonzero. Original foreground-loss gradients and background/foreground norm ratios use the same student forward; these diagnose unit-weight balance for this batch only. Controls run sequentially and release their graphs; gradient checkpointing remains disabled, matching the original recipe and avoiding the pinned reentrant-checkpoint/autograd.grad incompatibility. The cache and teacher must preserve Python, NumPy, CPU/CUDA0 RNG and module modes. No optimizer step occurs in the probe. A failed probe blocks both training arms.

The teacher uses the **initial texture-initialized caption** and frozen base UNet with zero modifier gate. It is a model prediction, not a ground-truth natural background. Training already has a foreground reconstruction mask; this experiment adds an outside-foreground constraint. Lambda0 invokes no teacher cache/forward or added metric fields. Lambda0's final adapter/token tensors and file hashes are compared with historicalT600; exactness or drift is reported explicitly. This short paired comparison cannot prove universal background generalization or disentanglement.

After training, each arm at each saved step gets three SCT scenes (street/snow/flowers) and Texture-only mug/vase, all seeds31/38, plus two historical failure anchors: Texture-only chair seed38 and sofa seed31. This gives **48 images**. SCT fixes center11 Mailbox S1000 and orange C150; T-only loads only the selected new Texture adapter and same-step embedding. PNDM100, CFG3.5,512px, fp16 inference and safety checker are unchanged. Every image uses a fresh seeded generator. The manifest, per-image status, gates and HTML comparison are saved; filtering is reported rather than retried.

Run from the fresh audited checkout, using a new external output path and the previous186-image scene source audit:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 python -u -B -m experiments.sd21_texture_bg_preserve_v1.run \
  --output /home/r12user5/Documents/Jiawei/colorpeel-runs/sd21_texture_bg_preserve_v1/NEW_RUN \
  --source-audit /ABSOLUTE/PREVIOUS_SCENE_RUN/experiment/source_audit.json
```

The command runs audit, real gradient probe, both training arms, paired exports, old/new600 comparison and all48 inference images serially. `status.json` records phase failures; `gradient_probe.json` must pass before training. Parent source Git, new training Git and inference Git are recorded separately. CPU contract/prefix/manifest tests: `python -B -m unittest discover -s tests -p test_sd21_texture_bg_preserve.py`.
