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
