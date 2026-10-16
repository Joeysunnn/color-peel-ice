# Native SD 1.4 / 1.5 paired generation

Run `run_pair.py --output "$NEW_RUN_DIRECTORY" --sd14-gpu 1 --sd15-gpu 2`
with the existing `colorpeel017` Python after checking those GPUs are free.
It runs both arms, then verifies their ledgers and actual latent hashes before
creating `review/overview.jpg` and one full-size sheet per object. SD1.4 is on
the left, SD1.5 on the right. It writes a parent manifest with child commands,
PIDs and completion/failure status; no agent polling is needed.

This study compares the two unmodified base models before adapter training.
There are 20 images per model: `chair`, `mug`, `vase`, and `sofa`, each with
`a photo of a {object}` and seeds 42–46. Both arms use 512×512, PNDM, 100 steps,
CFG 3.5, CUDA and float16. No learned token or adapter is loaded.

`protocol.json` locks the existing local SD 1.4 snapshot and SD 1.5 download.
The SD 1.5 download revision is
`451f4fe16113bff5a5d2269ed5ad43b0592e9a14`; its
`download_manifest.json` is checked before execution. All model loads use
`local_files_only=True`. Use the existing `colorpeel017` environment with
PyTorch 1.13.x and Diffusers 0.17.x. This study installs or upgrades nothing.

The local SD 1.5 download has no safety checker. **Both arms explicitly load
the same SD 1.4 SafetyChecker and feature extractor.** This common safety
source is recorded separately from each generation model; it is not presented
as an SD 1.5 component. Filtering remains enabled and every image records its
filter result. A filtered black image is not evidence of generation quality.

For each row, a new CPU generator seeded with the row's seed produces float32
noise with shape `[1, 4, 64, 64]`. It is cast to float16 on CPU, hashed, then
copied to the selected GPU and supplied as the pipeline's explicit latents.
Matching `pair_id` rows must have identical `latent_sha256` values across arms.
The base model's PNDM scheduler configuration is recorded in the run manifest.
Different CUDA devices may produce small numerical differences; this is a
paired visual comparison, not a claim of cross-device bitwise reproducibility.

Develop locally, commit and push, then update the clean server checkout using
GitHub fetch/fast-forward. The server only executes tracked code. Each task
refuses a dirty checkout, an existing output directory, and outputs outside
`COLORPEEL_RUN_ROOT` or inside the code checkout. Dry runs consume their own
fresh output directory and never load Torch or allocate a GPU.

```bash
conda activate colorpeel017
export COLORPEEL_RUN_ROOT=/home/r12user5/Documents/Jiawei/colorpeel-runs
commit=$(git rev-parse --short=7 HEAD)
stamp=$(date -u +%Y%m%d-%H%M%S)
python experiments/sd_native_comparison_v1/generate.py --model sd14 \
  --output "$COLORPEEL_RUN_ROOT/sd_native_comparison_v1/${stamp}__sd14_dry__${commit}" \
  --device cuda:0 --dry-run
python experiments/sd_native_comparison_v1/generate.py --model sd15 \
  --output "$COLORPEEL_RUN_ROOT/sd_native_comparison_v1/${stamp}__sd15_dry__${commit}" \
  --device cuda:0 --dry-run
# Use new output directories for execution, after checking GPU availability.
CUDA_VISIBLE_DEVICES=0 python experiments/sd_native_comparison_v1/generate.py \
  --model sd14 --device cuda:0 \
  --output "$COLORPEEL_RUN_ROOT/sd_native_comparison_v1/${stamp}__sd14__${commit}"
CUDA_VISIBLE_DEVICES=1 python experiments/sd_native_comparison_v1/generate.py \
  --model sd15 --device cuda:0 \
  --output "$COLORPEEL_RUN_ROOT/sd_native_comparison_v1/${stamp}__sd15__${commit}"
```

The two model tasks can run on separate free GPUs. Each output preserves
`manifest.json`, the protocol, exact command, installed package versions,
`pip-freeze.txt`, model component inventory, planned rows, PNG files, and
`ledger.jsonl` with latent hashes, PNG hashes, safety results, and errors.
Execution hashes every file under each used model component, including the
common safety components; dry-run inventories record paths and sizes and
explicitly defer weight hashing until execution. Model setup and generation
errors leave a failed manifest and a complete ledger of attempted/skipped
rows, then return a nonzero exit status. A dry run records `dry_run`, never
`succeeded`; success requires all 20 actual rows.
