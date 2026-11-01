# Mailbox27 continuation:600 to1000

Continue both completed SD2.1 Mailbox27 original/fullrank and rank64 runs from
`20261101-221114__both600__a513371__42`. Keep the same27 inputs, order, masks,
captions, base model, seed42, losses, optimizer and learning rates. No retraining
of the first600 updates. Fullrank remains token-local K/V, not full UNet tuning.

Restore complete Accelerate600 states, including adapter, text encoder/token,
optimizer, constant scheduler, RNG and registered custom state. With27 images,
batch1,accum1 and no shuffle, resume epoch22 after6 batches; update601 sees row7.
Copy verified metrics1-600 into the new log, inherit real embedding audit counters,
and retain the original initialization baseline. Resume receipt records hashes.
Worker RNG was not saved; no broad bitwise-reproduction claim is made.

Train400 additional updates and save only800/1000 in the new run. After each
method completes, export matching adapter/embedding pairs and run the same9
Mailbox probes: metallic, **matte**, glossy plastic, ceramic, wood, snow, flowers,
white, orange. PNDM100,CFG3.5,512,fp16,safety checker; seeds30-40 inclusive.
99 images/checkpoint,198/method,396 total including44 matte images.

Use a fresh `continuations/<timestamp>/experiment` under the original experiment.
Keep the old dataset, checkpoint600 and all old outputs read only. Four new full
states need twice the measured600-state bytes; require another2GiB for finals,
paired exports, images and margin. Do not delete historical data to make room.

Local edit/test/commit/push, then fresh server checkout and pull. Run with the
existing colorpeel017 Python, offline caches and output root:

```bash
python -B -u -m experiments.sd21_mailbox27_resume1000_v1.run --output "$NEW_ROOT/experiment" --gpus 2 3 --execute
```

Preflight validates old completion/provenance and each launch command. Verify
restoration and first new finite losses601-603, then disconnect. Finite queues
include transfer; no monitoring service. Startup does not establish image quality.
