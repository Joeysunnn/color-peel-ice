# Mailbox Material and Texture transfer

Use the paired exports from training revision
`16c1e7e12bc99c7d58489bef683e342cfbbf2024`: Material150/200 and Texture300/500.
For each checkpoint, generate all seeds20-60 (inclusive) for chair, mug, vase and
sofa. Each checkpoint has164 images; the four runs total656 images.

Exact templates:

- Material: `a photo of a {object} with a <M*> surface`
- Texture: `a photo of a {object} with <T*> texture`

Each process loads only its selected branch. Use the Subject probe's unchanged
SD2.1 base, PNDM100, CFG3.5,512px, FP16, empty negative prompt and enabled checker.
Every image uses a fresh generator with its seed. Token gates, loaded embeddings,
source hashes and equal same-seed initial noise are audited by the existing
midpoint inference engine. No prompt expansion, rerolls or image selection.

From a tested, pushed commit and fresh clean server checkout, run four independent
processes using available GPUs, for example:

```sh
python -u -B -m experiments.paper_mt_transfer_v1.run \
  --object-root /home/r12user5/Documents/Jiawei/colorpeel-runs/paper_experiments_v1/image_081__mask_130 \
  --branch material --step 150 --gpu 0
```

Other assignments: Material200/GPU1, Texture300/GPU2, Texture500/GPU3. Use the
existing colorpeel017 environment with both output-root variables set to the
existing colorpeel-runs root; no model/dependency downloads are required.

The four fresh `infer/20261107__{branch}{step}_transfer__{git7}__seeds20-60/`
directories are siblings of the existing Subject inference directory. Each saves
raw PNGs under `images/{object}/seed{seed}.png`, a gallery, generation manifest,
per-image ledger, status and source provenance. Existing Subject results and
training exports stay in place. Training Git and inference Git remain distinct.
Check startup only; no ongoing monitoring or automatic visual-quality judgment.

The Python `run()` and `prepare()` APIs also accept explicit keyword arguments
`training_run`, `source_git`, and `source_image_sha` for another audited Mailbox
training run. Defaults retain the original M/T run above. The run name must be a
single directory name, and its resolved source must remain under this object's
`training/` directory. Producer Git, source-image identity, completion, paired
snapshots, initializer provenance, base model and all source hashes are still
verified. This only selects another trained source; prompts and sampling stay
unchanged. The command-line interface retains its original default selection.
