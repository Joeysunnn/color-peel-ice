# Five-matte-image Subject LoRA

This experiment trains a rank-64, alpha-64, token-local K/V Subject LoRA from
the five reviewed pure-matte mailbox preview images. The source preview and all
older staging directories are read-only; `prepare_staging.py` copies the five
images and their reviewed masks into a new study-owned staging directory.

The per-image captions are locked as follows:

- `red_matte.png` -> `a photo of <S*> mailbox in red color`
- `green_matte.png` -> `a photo of <S*> mailbox in green color`
- `cyan_matte.png` -> `a photo of <S*> mailbox in cyan color`
- `blue_matte.png` -> `a photo of <S*> mailbox in blue color`
- `magenta_matte.png` -> `a photo of <S*> mailbox in purple color`

These are five recolors of one mailbox pose, background, and mask rather than
five independent views. The experiment therefore carries a deliberate
pose/background memorization risk.

Training keeps the token-embedding learning rate at `1e-5`, uses K/V learning
rate `5e-5`, and saves one 3000-step trajectory at steps 1000, 2000, and 3000.

## Inference

`protocols/inference_v1.json` locks the three Subject snapshots, the rank-64
step-1000 `metal_spoon` and `wood_spoon` Material snapshots, and the historical
140-row Subject-transfer manifest. For each Subject snapshot,
`evaluate.py --task comparison` generates the fixed plain/red/blue grid for
Subject-only, literal metal, and both Material-token checkpoints (60 images);
`--task transfer` replays the historical transfer prompts unchanged (140 images).

The historical transfer manifest treats magenta as seen and purple as unseen.
That labeling is preserved for benchmark comparability even though this run
captions `magenta_matte.png` as purple; it is not the current training cohort's
reconstruction split.
