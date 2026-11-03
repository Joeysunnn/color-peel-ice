# Subject1000 / Material150 CFG comparison

Inference only from the current single-image Qwen Mailbox paired exports. Subject1000 and Material150 are tested separately, with CFG3.5/5/7.5 and seeds30-40 inclusive. No composition, retraining, new base/literal controls, rerolls, continuous agent supervision or visual review.

- Subject: white,red,snow,street,flowers;165 images.
- Material: chair,vase,sofa,mug;132 images.
- Total297 images, preserving the previous exact prompt strings, PNDM100,512px,FP16,empty negative and safety checker. Only CFG varies within each same-checkpoint/prompt/seed comparison.

Each branch uses one fresh pipeline on a separately checked idle GPU (Subject0,Material1). Per-prompt token gates are identical for the three CFGvalues because all exceed1; each image still receives its own requested guidance scale. Saved initial denoiser inputs verify same-seed equality across CFGvalues and branches. Checkpoint pairs, initializer provenance, image hashes and filtered flags are recorded. Source weights remain unchanged.

Use the existing `colorpeel017` environment and a reviewed fresh Git clone:

```sh
export COLORPEEL_RUN_ROOT=/home/r12user5/Documents/Jiawei/colorpeel-runs
python -u -B -m experiments.sd21_qwen_cfg_compare_v1.run --output FRESH_OUTPUT
```

`index.html` links two branch galleries with CFGvalues in adjacent columns for each prompt/seed. `status.json`, per-branch logs, `generation_manifest.jsonl`, `source_audit.json` and the per-group generation ledgers record execution. Confirm startup only; the user assesses image quality.
