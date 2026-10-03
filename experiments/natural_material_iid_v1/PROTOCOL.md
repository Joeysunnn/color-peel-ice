# IID multi-hypothesis material extraction

This independent V2 uncertainty baseline follows the [IID paper](https://arxiv.org/abs/2312.12274)
and [official implementation](https://github.com/Peter-Kocsis/IntrinsicImageDiffusion).
Run only the material-diffusion stage. The model returns 6 channels: Albedo
RGB and a 3-channel BRDF packing whose first two channels are Roughness and
Metallic. The third BRDF channel is retained for audit and excluded from M*.
Its released checkpoint is `iid_e250.pth`. The
official stage-2 CLI averages 10 samples before saving, so `run_iid_candidates.py`
calls the unmodified official model and saves all 10 raw six-channel tensors
before any average. Its per-candidate seeds are fixed in `config.json`.

The paper reports 50 DDIM steps for its experiments; the released model config
uses 100 and eta 0. This run follows the released config and records its hash.
The official loader converts sRGB to linear and resizes to 480x640. To avoid
warping the existing 512x512 natural crops, `prepare_inputs.py` centers each
as 480x480 within a 640x480 neutral-gray canvas and transforms the material
mask identically. This is an explicit adaptation to the user's natural images.
IID was trained on synthetic indoor scenes; both source images are domain
shift, so candidate spread is uncertainty evidence, not a calibrated posterior.

The Qwen3-VL semantic priors are copied with hashes from the completed
semantic-constraint run to keep the language judgment fixed while changing
the material estimator. On each IID candidate, aggregate the raw Roughness
and Metallic maps inside the material mask. Apply the predeclared categorical
coated/bare-metal and gloss gates. If candidates pass, take the median of their
region medians as M* and report both all-candidate and accepted-candidate
spreads. If none pass, report `rejected` without inventing M*. Albedo is kept
for audit and excluded from M*; opacity remains untestable with R/M alone.

Keep mailbox and spoon control in separate run subdirectories, preserve source
and checkpoint hashes, model config, raw maps, PNG previews, candidate seeds,
VLM prior, selection checks, renderer metadata, and Git commits. Render only
a small three-light neutral-gray sphere sheet if an M* is accepted. Do not
generate the 96-image grid or train `<M*>`.
