# Qwen SD2.1 midpoint transfer

Inference only, using the completed Mailbox single-image Qwen round's paired LoRA64 and token exports: S500, M500, T500, C100. Texture is the Mailbox's `uniform` concept, not earlier wood-spoon weights. Source training commit is pinned separately from this inference checkout. Existing results and checkpoint files remain immutable.

Seeds30 through40 inclusive; 242 images across five fresh pipelines:

| Group | Cases | Images |
| --- | --- | --- |
| S500 | white, red, snow, street, flowers | 55 |
| M500 | chair, vase, sofa, mug | 44 |
| T500 | hair, vase, sofa, mug | 44 |
| C100 | bowl, vase, parrot, sofa, rose | 55 |
| S500+C100+M500+T500 | plain, street, snow, flowers | 44 |

`hair` means head hair, as requested. Color uses five exact historical prompts from `natural_image_subject_color_pilot/configs/d1_mailbox_orange_shared_kv_joint_1000_color_transfer_protocol_v1.json`. Material uses the current `with a <M*> surface` grammar for the `painted` initializer; Texture uses `with <T*> texture`. Subject color and scene prompts follow the earlier Subject transfer probes. SCMT uses the current Qwen round's four scene prompts. Full strings are recorded in `generation_manifest.jsonl`.

Keep PNDM100, 512px, FP16, empty negative prompt, enabled safety checker; Color-only CFG6, other groups CFG3.5. Fixed independent GPU queues: GPU0 Subject then SCMT, GPU1 Material, GPU2 Texture, GPU3 Color. Refuse busy GPUs. Audit source checkpoint pairing and hashes, initializer provenance, prompt masks, actual loaded embedding values, same-seed first denoiser inputs, and output PNG hashes. Safety-filtered images remain flagged; no rerolls or automatic visual-quality claims.

Deploy reviewed source through Git to a fresh server clone. In the existing `colorpeel017` environment, with `COLORPEEL_RUN_ROOT=/home/r12user5/Documents/Jiawei/colorpeel-runs`:

```sh
python -B -m experiments.sd21_qwen_midpoint_transfer_v1.run --output FRESH_OUTPUT
```

Output includes a browsable gallery, source audit, exact generation plan, per-group logs/ledgers, and aggregate verification. No training or environment changes.
