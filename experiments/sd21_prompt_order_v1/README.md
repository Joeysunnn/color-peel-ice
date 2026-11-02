# Fixed SD2.1 prompt-order diagnostic (prepared, not launched)

LoRA Mailbox S1000 + orange C150 + original wood T400, paired learned embeddings,
PNDM100 / CFG3.5 / FP16 / 512 / checker, seeds31/38 and street/snow/flowers.
Only P0 (original suffix) and P1 (same words, scene before the entire body) exist.
No training, short-prompt variant, separate branch conditioning or new negative
prompt is included.

`python -B -m experiments.sd21_prompt_order_v1.run --screen-audit SCREEN96/experiment/source_audit.json --output FRESH_EXTERNAL_DIRECTORY`

The entrypoint verifies the original screen's source audit, completion status,
ledger, six LoRA T400 baseline images and live source hashes. All six P0 image
SHA256 values and safety states must match before any P1 image is generated.
Each prompt has its own checked token gates; the CFG unconditional gates remain
zero. A read-only forward hook records the actual first UNet latent input, after
scheduler scaling, and checks its identity across prompts with the same seed.

`text_conditioning_audit.json` records actual77-token CLIP/pipeline encoding,
S/C/T hidden max-abs differences when suffix scenes are appended, and the change
at T when S/C are replaced by single-token initializer IDs at the same positions.
If mailbox/orange does not have a single initializer ID, the latter measurement
is explicitly skipped; no BPE piece is silently selected. Normal CLIP causal
behavior is not itself a bug. P1 also changes positional encoding and syntax,
so improvement cannot by itself establish independent branch conditioning.

This directory prepares a possible next experiment. It must not be automatically
launched while the background-preservation comparison is still under review.
