# Joint two-object binding v2 authorization

The original shared-token two-object checkpoint completed 1500 steps, but its
four-image composition smoke failed before the 720-image campaign. Seen and
unseen prompts both showed extra objects, wrong shapes, and left/right bundle
binding failures. That checkpoint and smoke remain immutable baselines.

The user authorized an independent correction on 2026-09-01. This variant
changes only the multi-object training semantics:

- both complete object bundles condition every two-object RGB training image;
- left and right diffusion reconstruction losses are mask-normalized separately
  and averaged;
- ColorPeel CAA is evaluated separately inside each three-token object bundle;
- ICE Stage Two mask-to-attention Wasserstein guidance localizes each averaged
  bundle attention map to its corresponding GT mask, using the ICE code values
  `lambda_attention=1e-5`, Sinkhorn regularization `1e-3`, and 200 iterations.

The eight shared token identities, token initializers, SD1.4, trainable Custom
Diffusion K/V parameters, ColorPeel CAA weight 0.2, official AdamW behavior,
learning rate, batch size, seed, resolution, and 1500-step schedule are unchanged.
No learned left/right or object-slot token is introduced.

Execution is gated by 2-step and 18-step real smokes. The 1500-step run may
start only when reconstruction, CAA, Wasserstein, total loss, and gradients are
finite and all eight shared tokens have the expected exposure. The four-image
generation smoke remains the gate before any 720-image campaign.
