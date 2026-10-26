# Wood wording comparison: seeds42-46

Execution commit: `ce138a7`.

Server result:
`/var/tmp/colorpeel-r12user5/runs/sd15_native_mailbox_wood_wording_v1/20261026-234449__wood_wording15__ce138a7__42`

All15 images completed,0 safety-filtered; all512px PNG hashes match their ledger.
Runtime retains stock Diffusers UNet/attention, frozen SD1.5 weights, PNDM100,
CFG3.5,fp16 and the existing shared checker. No learned token or adapter is loaded.
Fresh CUDA generators use seeds42,43,44,45,46 independently for every wording.

The three columns contain the following complete prompts:

1. `a photo of a wooden mailbox`
2. `a photo of a mailbox made entirely of wood`
3. `a photo of a mailbox with a wood surface`

The original third wording at42/43 exactly reproduces the earlier native10 PNGs:
both prompt equality and byte-identical image hashes are true. See
`baseline_reproduction.json`. Old baseline images were read for comparison only.

## Qualitative observations

- Seed42: the first wording changes the roof/box design; the second produces a
  smooth tan rounded box. Clearly natural wooden appearance is not consistent.
  The third wording again places the pale mailbox against wooden background planks.
- Seed43: the first two wordings introduce vertical weathered/grain-like marks on
  the front, while retaining a red curved side and mixed painted/metal-looking parts.
  The third retains the familiar silver front.
- Seed44: the first two produce brown mailbox bodies with embossed lettering;
  the third has a gray-green weathered enclosure. RGB alone does not establish
  the actual surface material or separate grain from reflectance.
- Seed45: all three retain pale gray curved mailbox shapes without a convincing,
  unambiguous whole-object wooden appearance.
- Seed46: `made entirely of wood` creates the clearest wooden boards/body, but
  changes the object into an open-lid box-like design. The other two mostly retain
  a red mailbox mounted on a wooden support.

Conclusion limited to these samples: stronger wording can increase wooden visual
cues, but none of the three formulations consistently binds wood to the entire
mailbox across all five seeds. Shape, texture, local parts and background also
change. This is an appearance/prompt-binding comparison, not physical BRDF ground
truth or a fixed-identity material replacement test. No stronger CFG or alternative
sampler was tested, and no new Subject training was performed.

The complete3-column/5-row `contact_sheet.png`, all15 raw images, exact runtime,
prompts, safety flags and reproducibility records are saved under the result path.
