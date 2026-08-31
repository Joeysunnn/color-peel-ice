# Two-object protocol decision

- 18 semantic states are paired into 9 complementary pairs.
- Each pair is rendered in forward and swapped left/right orientation.
- Each state therefore appears once on each side, removing a fixed-position shortcut.
- Both orientations share camera, light, and Cycles seeds for paired comparison.
- 18 scene cells × 20 views = 360 RGB scenes.
- Views 0–15 create two object-level samples each: 576 training samples total.
- One shared eight-token model is trained; there is no per-object checkpoint.
- GT masks enter only the two-object reconstruction loss. Historical ColorPeel
  baselines remain mask-free and unchanged.
- Stop after the real renderer/contact-sheet gate; Fold or full training is not
  part of this implementation run.
