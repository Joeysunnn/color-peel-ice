# Controlled two-object generation protocol

The trained shared model is evaluated through prompt composition, matching the
ICE inference convention: multiple complete object token bundles are joined by
`and`. GT training masks are not an inference input.

Each prompt explicitly assigns one full subject/color/material bundle to the
left and another to the right. The locked campaign contains:

- 9 renderer-seen semantic pairings x 2 left/right orientations x 20 seeds;
- 9 disjoint unseen semantic pairings x 2 orientations x 20 seeds;
- 720 images total, seeds 42--61, 100 inference steps, CFG 6.0;
- FP16 on GPU 3, with the SafetyChecker disabled under the existing recorded
  policy to avoid false all-black outputs.

A four-image real smoke covers seen/unseen and forward/swapped once at seed 42.
The full campaign starts only if the smoke is non-black, decodable, and shows
two separable objects. Full output is resumable only when image hashes and the
model/protocol fingerprints match the status ledger.

After 720/720 validation, the bundler creates a randomized human-review CSV,
one 36-scene contact sheet, and nine seen/unseen orientation comparison sheets.
The process stops there; automatic classification is not the primary gate.
