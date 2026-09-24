# Color and material composition diagnostic

The first target is the orange emission color branch already used by the
subject/color mainline. The project owner selected the ground-reflection
`<M*>` run in
`../material_token_local_pilot_v1/configs/selected_material_source.json`.
This experiment first retrains `<C*>` as a separate token-local K/V adapter
from the existing nine verified orange emission images. No material images
enter color training, and no paired color/material images are used.

The short color run keeps the earlier orange branch's data, initializer,
optimizer, seed, and 100-step budget. `cos_weight` is zero because token-local
training supports one modifier token. The changed adapter architecture means
this is a new diagnostic color branch, not a continuation of the old shared
K/V checkpoint. Check color-only transfer before drawing conclusions from
composition.

After color training, compose the two frozen adapters at inference with
separate `<C*>` and `<M*>` token-position masks. Compare base, color-only,
material-only, and paired prompts at identical seeds and sampling settings.
Record color, metallic appearance, object identity, safety filtering, extra
objects, and framing separately. The orange fruit artifact seen previously
must remain visible. This stage does not train a joint checkpoint.
