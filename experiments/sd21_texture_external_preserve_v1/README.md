# External-image preservation for SD2.1 Texture

This isolated experiment tests whether teacher consistency on existing external
images can reduce learned Texture damage to scenes. It does not guarantee fine
grain transfer or preserve every concept. No Material branch is added.

## Fixed assets

The authorized source is
`/home/r12user5/Documents/Jiawei/papers/ICE/outputs/preservation_images/`.
The inspected directory contains 1000 JPEGs, each 512x512 RGB, with unique numeric
filename prefixes 0 through 999. The 36-image visual sample was beach-themed and
included people; these are not certified empty backgrounds. There are no masks
or recovered per-image captions/generation metadata.

`a photo at the beach` is the current ICE default and agrees with the sample.
It is the explicit prompt chosen for this experiment, not a claim that every
image's historical generation caption is known. The original generation model,
revision and seed are unknown. Selection seed 42 is not a generation seed.

`assets.prepare(source_dir, output_manifest)` reads and decodes all 1000 source
files without changing or copying images. It checks the exact source root,
numeric IDs, dimensions, RGB mode, JPEG format and the pre-registered canonical
inventory SHA256. That digest includes each numeric ID, filename, file SHA256,
size and mode, sorted numerically; it pins the downloaded 2026-11-03 evidence
without adding the image collection or a large absolute-path inventory to Git.

It writes two new runtime JSON files outside the source:

- `externalinventory.json`: all 1000 records and source evidence hashes.
- The requested manifest: schema version 1, the explicit prompt, selection seed
  42, and 200 records selected without replacement by
  `random.Random(42).sample(numeric_id_sorted_records, 200)`.

Each record is `{path, sha256, size: [512, 512], mode: "RGB", id}`. Paths are
absolute runtime source paths. The returned `source_locks` maps all 1000 source
paths and both JSON paths to SHA256. The runner must recheck these locks before
each arm. No silently substituted, resized, captioned or regenerated image is
accepted. Training consumes one selected image per step in manifest order for
the 200 continuation steps; step401 uses record0 and step600 uses record199.

## Paired protocol

The planned comparison resumes the same original LoRA Texture T400 checkpoint
to T600, preserving the original Texture16 foreground training data and recipe.
It saves T500 and T600. The two arms use external preservation weights 0 and 10.
Both arms share the fixed external manifest and all other settings.

The initial T400 probe at weight0.1 passed the gradient and RNG checks, but its
aggregate auxiliary gradient norm was approximately 0.0102% of the foreground
norm; the largest per-group ratio (V_down) was 0.167%. Before viewing any
generated evaluation images, that preliminary run was stopped and retained.
The weight was adjusted once to10: the same probe predicts about1.02% aggregate
and16.7% for V_down. The final comparison starts both arms fresh from T400 and
repeats the probe. This is a conservative single-sample numerical calibration,
not an optimal-weight claim or a guarantee about Adam update magnitudes.

The new regularizer is teacher prediction consistency with the student's
Texture K/V residual active on all 77 text positions for the external pass.
It is **not** ICE's noise-target prior loss, and **not** the earlier
foreground-mask-complement loss on Texture16 images. External images have no
object masks, so this is whole-image consistency, not a labelled background-only
constraint. Teacher and student must receive exactly matched latent noise and
timesteps. Teacher/reference conditioning remains frozen. The gradient probe
must establish the intended adapter gradients and rule out an inactive-gate
false positive before either full arm runs. The ordinary generation/inference
path retains the original token-local policy.

The auxiliary term has no direct embedding gradient. The shared gradient norm
clipping can nevertheless change the scale of the foreground embedding update;
embedding trajectories are not claimed to remain identical between arms.

The main comparison is the new paired arms. Historical T600 is a reference,
not assumed numerically equivalent to continuation. The fixed beach corpus is
a narrow additional distribution; improvement there does not prove improvement
on street, snow or flowers, and a stronger regularizer may weaken grain.

## Evaluation and implementation boundary

The fixed evaluation contains 56 images: the existing 48-image paired evaluation
(T-only plus SCT, both checkpoints and both arms), plus SCT beach transfer at
seeds31/38 for each arm and each checkpoint, adding 8 images. No additional
scenes or parameter sweep are included.

The asset helper only prepares the immutable runtime contract. It neither
starts training nor accesses the server. Tests use synthetic temporary JPEGs
and a substituted test digest, never the protected source collection.
