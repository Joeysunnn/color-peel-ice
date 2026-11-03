# Mailbox Material250 kickoff

Fresh SD2.1 Material-only training to250 steps, then automatic44-image inference. Reuse the Qwen round's8 Mailbox Material images and masks, `painted` initializer, seed42, LoRA64/alpha64, FP32, batch1, embedding LR1e-5 and KV LR5e-5. No checkpoint resume. The inherited constant learning-rate schedule has no warmup; changing the training horizon does not rescale the first250 learning rates. Save the paired adapter and token at250.

Inference uses the same four prompts as the midpoint study: `a photo of a {chair|vase|sofa|mug} with a <M*> surface`, seeds30–40 inclusive. Material250 only, PNDM100,512px,FP16,CFG3.5, empty negative prompt and enabled safety checker. Retain filtered outputs without rerolls.

```sh
export COLORPEEL_RUN_ROOT=/home/r12user5/Documents/Jiawei/colorpeel-runs
python -u -B -m experiments.sd21_qwen_material250_v1.run --output FRESH_EXPERIMENT --gpu 0
```

Use the existing `colorpeel017` environment and a fresh reviewed Git checkout. The parent runs training, validates/exports the paired250 checkpoint, and then starts inference automatically. It records config/command/log/state in the experiment and `train_outputs/`, and creates `inference/index.html` plus the Material gallery. A failed stage prevents dependent stages; old source assets remain unchanged. Check GPU availability before launch. Only startup verification is requested: no ongoing agent supervision or visual review; the user reviews the outputs manually.
