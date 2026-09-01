# Two-object training authorization

Date: 2026-09-01

The user completed the renderer/contact-sheet review and accepted the dataset.
A small minority of medium-to-heavy occlusions is present; these samples do not
dominate any cell and are retained as realistic multi-object variation.

Training is authorized in this fixed order:

1. 2-step startup smoke using cube-red metal/rubber object records and their GT
   instance masks.
2. 18-step coverage smoke using one masked record from every subject/color/material
   state.
3. Seed-42, 1500-step full training using all 576 masked object records, only if
   both smokes pass.

This authorization does not change the renderer, eight shared tokens, CAA,
Custom Diffusion K/V, AdamW behavior, learning rate, or full-run step count.
