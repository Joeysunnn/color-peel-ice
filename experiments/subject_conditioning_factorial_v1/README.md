# Mailbox and wood spoon: color x recolor support x geometry alpha

User-requested 2026-11-05. Generate eight fresh native meshes per object,
sixteen total, with seed42 and pinned TRELLIS inference. No training, extra
seeds, retries, smoothing, model download or production-policy change.

| Factor | Values |
|---|---|
| RGB color | Original / pink hue0 |
| RGB recolor support | SAM mask>0 / associated BiRefNet alpha>0 |
| Geometry alpha | SAM binary / complete BiRefNet soft values |

Scope means **where pink RGB may change**. It does not separately cut out
RGB. Original RGB and all pink pixels outside the selected recolor scope
retain the frozen base. Original/SAM and original/BiRef scope pairs therefore
have identical conditions for a given geometry alpha. They are still run
separately as requested, displayed and checked as duplicate controls, not
eight independent conditions. The user explicitly confirmed these
recolor-only semantics before execution.

Freeze the existing, audited, uncolored BiRefNet RGBA512 RGB once per object.
Its RGBA resizing happened before this study and is identical in every cell.
This matters for the2560px wood spoon: independent RGB resizing would change
22099 pixels inside soft support. Do not select a different RGB resampling
path according to geometry alpha. Existing SAM is fully covered by the soft
support for both objects. No SAM or BiRefNet re-inference is needed.

Pink uses the established Lab/LCh hue0 conversion and edge blending derived
from the selected support. Exact-white RGB protection is retained. Save the
pre-alpha RGB separately; serialize RGBA with RGB zero only at geometry A=0,
as required by the existing transparent-pixel contract. The final model input
is the official float32 RGB*alpha composite, without further resizing here.

Every cell uses one crop derived once from the frozen soft alpha:

- Mailbox: `[144,253,440,549]`,296x296 including original out-of-canvas padding.
- Wood spoon: `[23,19,487,483]`,464x464.

Consequently the Mailbox SAM cells do not use the previous294 crop. Also,
pink/SAM-support with soft alpha retains original-colored RGB in the extra
soft support; it is not the previous diagnostic that forced this region black.
These differences must remain explicit when comparing with earlier studies.

Code deployment: local tests/review, commit/push to the authorized fork, new
clean GitHub checkout with full SHA, GPU process ownership/availability check,
then fresh external run. Reuse existing `execute` for one active job per GPU,
offline pinned files, independent failure records and no retries.

```sh
python -B -m experiments.subject_conditioning_factorial_v1.run \
  --output NEW_EXTERNAL_RUN --gpus 0 1 2 3
```

After generation, use the existing geometry-only comparison renderer with
one frozen reference camera plus three common canonical views per object.
Also retain independently fitted PBR/clay/normal previews, clearly labeled
as different fitted cameras. A silhouette score or completed run is not user
acceptance of identity, surface quality, hidden geometry or downstream learning.
