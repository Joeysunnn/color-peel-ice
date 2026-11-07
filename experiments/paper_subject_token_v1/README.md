# Mailbox Subject1000 without a category word

Inference only from the user-selected `cb169c6` Subject step1000 paired export.
The nine exact prompts in `run.py` use `<S*>` without `mailbox`; seeds20-60
inclusive produce369 images. Keep the previous Subject sampling settings:
SD2.1, PNDM100, CFG3.5,512px, FP16, empty negative prompt and enabled checker.
Only Subject is loaded. No prompt expansion, retries, selection or training.

The fresh object folder is named `image_081__mask_130`. It contains a byte-identical
checkpoint copy under `ckpts/subject/cb169c6/step1000`, unchanged producer receipts
under `provenance`, and the new dated inference directory under `infer`.
Original assets stay in place. Producer Git and inference Git remain distinct.
The adapter and paired embedding are verified and loaded from the new copy.

Use a tested, reviewed commit pushed to the existing fork, then a fresh clean
server checkout with the full SHA verified. Use the existing `colorpeel017`
Python and cached models, with both output-root variables pointing to
`/home/r12user5/Documents/Jiawei/colorpeel-runs`. Check GPU ownership before launch.

```sh
python -u -B -m experiments.paper_subject_token_v1.run \
  --object-root /home/r12user5/Documents/Jiawei/colorpeel-runs/paper_experiments_v1/image_081__mask_130 \
  --gpus 0 1 2 3
```

Seeds are partitioned across GPUs; all nine prompts for one seed stay together.
Each image gets a fresh seeded generator. The worker and final merged ledgers
verify equal initial denoiser inputs across same-seed prompts. Each worker has
its own log/status/gallery and writes unique image paths. The final gallery and
status are produced automatically, including retained safety-filtered outputs.
Only bounded startup verification is requested; visual review belongs to the user.

This is a Subject identity and text-editability probe. It does not by itself
establish learned Color/Material/Texture transfer or four-attribute disentanglement.
