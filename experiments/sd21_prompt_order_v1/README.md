# Fixed SD2.1 prefix / surface wording diagnostics

LoRA Mailbox S1000 + orange C150 + original wood T400, paired learned embeddings,
PNDM100 / CFG3.5 / FP16 / 512 / checker, seeds31/38 and street/snow/flowers.
Each run contains six P0 (original suffix) and six P1 images. The default
`--mode prefix` retains the original P1_prefix comparison: the same words with
the scene before the entire body. `--mode surface` instead uses P1_surface:
only the ordinary word `texture` following `<T*>` becomes `surface`; the scene
stays at the end. It does not add another comparison to the original 12 images.
No training, short-prompt variant, separate branch conditioning or new negative
prompt is included.

`python -B -m experiments.sd21_prompt_order_v1.run --screen-audit SCREEN96/experiment/source_audit.json --output FRESH_EXTERNAL_DIRECTORY`

For the surface comparison, append `--mode surface` and use another fresh output
directory. Status, HTML and text-conditioning audit explicitly record the mode.
The default prefix manifest, prompts, seeds and inference computation are unchanged.

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

Surface mode also requires P0/surface IDs up to and including `<T*>`, all three
modifier positions and all three modifier hidden vectors to be exactly equal.
Any mismatch stops before image generation. It records actual BPE lengths,
tokens and positions, followed by changed downstream token IDs and hidden
max-abs differences. These downstream comparisons are positional, include EOS
and padding slots, and do not claim semantic alignment if BPE lengths differ.
The changed word can also affect later scene/EOS/padding hidden states; an image
change cannot isolate only the ordinary word's attention contribution or prove
that branch conditioning has become independent.

The prefix experiment has been run. Surface mode is prepared for explicit
review and launch; this entrypoint does not automatically launch either mode.
