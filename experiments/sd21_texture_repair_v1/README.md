# SD2.1 Texture checkpoint and residual-scale screen

User-authorized first repair stage after the frozen background diagnosis. No training or source modification.
Subject1000 and orange Color150 remain fixed. Fullrank uses three fullrank branches; LoRA uses three rank64/alpha64 branches.

The eight conditions are Texture400/600/800/1000 at K=V=1, Texture1000 at K=V=0.25 or0.5, and Texture1000 at K=0/V=1 or K=1/V=0.
Changing a checkpoint loads both its Texture adapter and same-step embedding. This tests the paired checkpoint, not adapter training dose in isolation.
All scale conditions retain the Texture1000 embedding. Only Texture K/V residual outputs are multiplied; Subject/Color outputs and all token gates remain unchanged.
Scale1 leaves the existing SCT modules and their forward path unchanged. Scales are installed once on fresh processors, without forward hooks.

Each condition uses street, snow and flowers with seeds31/38: eight conditions x three scenes x two seeds x two methods =96 images.
The literal prompt remains `a photo of <S*> mailbox in <C*> color with <T*> texture` plus the original scene suffix.
Sampling remains PNDM100,CFG3.5,512px,FP16,with the safety checker enabled. GPU0 only; the methods and conditions run serially.
Each method loads a fresh pipeline for each of the four Texture checkpoints. The later T1000 scale cases reuse its embeddings,
but reinstall all three processors from the original audited states before applying each scale, avoiding cumulative scaling.

Supply the original completed186-image scene experiment's `source_audit.json`, not the22-image diagnostic audit.
Preparation reuses the SCT source verification, re-audits paired Texture400/600/800 tensors, and records the original training Git separately from the new inference Git.
Source locks are checked before and after generation. Each method/condition has a selected-source receipt in `source_selections.json`.
The complete96-row manifest is saved before model loading. Status and the generation ledger update after every image.
The root HTML links six comparison pages, one per method and scene; each page compares all eight conditions at both seeds.

Texture1000/scale1 baseline PNG SHA is compared with the earlier image at the same method, scene and seed, and all12 comparisons are recorded.
A SHA difference is recorded rather than silently ignored or treated as proof of a scientific change; inspect pixels/environment before interpreting that comparison.
The baseline must reproduce the earlier behavior for this screen to be scientifically comparable.

This is a two-seed screen, not evidence of broad generalization. Judge mailbox identity, orange binding, wood texture and scene response together.
A condition that merely removes Texture or improves flowers while breaking color/identity is not a completed repair.

Run from a fresh committed checkout, with a new output directory outside all code and source experiments:

```bash
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1
export COLORPEEL_OUTPUT_ROOT=/home/r12user5/Documents/Jiawei/colorpeel-runs
/home/r12user5/miniforge3/envs/colorpeel017/bin/python -B -m unittest tests.test_sd21_texture_repair -v
/home/r12user5/miniforge3/envs/colorpeel017/bin/python -B -u \
  -m experiments.sd21_texture_repair_v1.run \
  --source-audit "$ORIGINAL_SCENE_EXPERIMENT/source_audit.json" \
  --output "$NEW_ROOT/experiment"
```
