# Independent polished-metal positive control

The control is a **real photograph** of a bare stainless-steel soup spoon on a
white background. Its Commons page identifies the uploader's own work and
contains iPhone 5s camera metadata; no evidence on that page indicates a
rendered image. The photograph is not a physical BRDF ground-truth sample.

- Source: [Stainless Steel Soup Spoon.jpg](https://commons.wikimedia.org/wiki/File:Stainless_Steel_Soup_Spoon.jpg)
- Author: Paolomarco; CC BY-SA 4.0.
- Original file, crop, masks, RGBA input and SHA256s: `assets/`, `provenance.json`.
- `prepare_control.py` deterministically crops the original to the spoon bowl,
  traces the visible bowl as the SuperMat object alpha, and defines a smaller
  interior metal ROI for scalar aggregation. Review `assets/region_overlay.png`
  before using the values. The ROI contains 57,761 pixels, all inside the
  object mask. The main mailbox run does not use these files.

## Control protocol

Use the **same** SuperMat checkout, checkpoint, SD 2.1 base, seed 42, 512-pixel
size, and inference settings as the mailbox run. Feed `assets/input_rgba.png`.
Read the separate `roughness.png` and `metallic.png` grayscale maps; albedo is
saved for audit but is not a material scalar. Run `analyze_control.py source`
with `assets/material_region.png` to obtain the two medians and IQRs.

Use those measured values without manually forcing metallic to zero. Render a
neutral-gray sphere under the same three lights using the mailbox calibration
renderer and profile, in a **new control run directory**. Feed each rendered
`input_rgba.png` back through SuperMat with the same inference settings. Run
`analyze_control.py roundtrip` to compare per-light medians against the source.
The renderer colors and source image pixels are not compared. Round-trip
agreement is estimator consistency only, not proof of physical parameters.

The natural source has real reflections and some fine marks; if SuperMat maps
show broad or bimodal values over the ROI, record that limitation rather than
silently treating the median as a homogeneous metal material.
