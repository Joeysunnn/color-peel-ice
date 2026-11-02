# SD2.1 Mailbox background diagnosis

This bounded inference diagnostic investigates the weak scene response of the
existing S1000 / orange C150 / wood T1000 combination. It does not retrain or
modify source checkpoints. Use an isolated checkout; GPU0 only. SDXL jobs and
GPUs1-3 remain reserved.

For each method, keep all three learned embeddings, the full flowers prompt,
PNDM100, CFG3.5 and seeds31/38. Compare the original gates with all residuals off
and with only S, C or T residuals off. Seeds31/38 are the first two seeds with
unfiltered outputs across every method/scene in the previous experiment; they
were not selected for visual quality. A stock ordinary-mailbox control adds two
images:22 total. This control changes the prompt and is not a pure residual
ablation. All-off retains learned embeddings and is not the stock base either.

```bash
python -B -m unittest tests.test_sd21_background_diagnosis tests.test_sd21_mailbox_sct_attention -v
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 python -B -m experiments.sd21_background_diagnosis_v1.run --source-audit PREVIOUS_SCENE_EXPERIMENT/source_audit.json --output FRESH_EXTERNAL_OUTPUT
```

Source adapter/token tensors are audited again, and source hashes are checked
before and after generation. Baseline PNG hashes are compared with the previous
run. The safety checker remains enabled; filtered outputs are retained/labeled.
These two-seed flowers probes locate mechanisms, not a benchmark or evidence of
general success across street/snow/flowers. Any successful intervention must
also retain subject identity, orange color and wood texture before being called
a solution.
