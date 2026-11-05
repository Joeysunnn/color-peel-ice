# Mailbox: current SAM identity with full BiRefNet soft alpha

User request, 2026-11-05: show a pink Mailbox mesh using the same soft-alpha
path already verified on the wood spoon, before integrating original/pink
automatic candidate selection. One fixed seed42 mesh; no training or S11.

Frozen Scheme A manifest:
`/home/r12user5/Documents/Jiawei/colorpeel-runs/single_image_four_branch_v1/20261104__mailbox_A_full__da8e990__42/prepared/segmentation/objects.json`.
Manifest SHA256: `09cc03fbc2356b59d290a3dda4c582d47719e81f59ca7514ece14fd73d6fac37`.
Selected instance: `object_003`, mailbox. SAM SHA256:
`44c8a935359f52c97f3cc788168d0103fc8482295be1f9531efff537a2052d0c`.
Native source SHA256:
`644a77fe664bc2d9b0c94ada86259ce5a810a530580e754f1b23a40a82906be5`.

Use fresh semantic inputs, freshly inferred full-native BiRefNet, complete
selected components, historical joint RGBA512 resampling and pink hue0. The
canonical SAM/photo stay unchanged. All preparation and generation use one
clean, verified GitHub checkout and existing offline pinned model files.

## Initial association failure retained

The initial `d5b9d15` attempt stopped before pink preparation or sampling because
its any-contact ownership rule counted a soft boundary touch as a second object.
The Mailbox component contains44831 pixels and covers all42397 Mailbox SAM
pixels. It touches just43 pixels of the34738-pixel neighboring door SAM mask
(0.00123784 coverage; summed opacity1204, mean28/255).

The revised engineering ownership rule requires
`intersection_pixels / min(component_support_pixels, SAM_pixels) >= 0.01`.
This treats substantial overlap with a tiny object as ownership while ignoring
incidental contact with a large adjacent object. It does not clip, normalize or
otherwise change any selected soft-alpha pixels. Significant multiple ownership
still blocks the target, and selected components must cover at least80% of its
SAM mask. The threshold is an explicit association heuristic, not proof of
correct instance separation for all scenes.

The failed attempt, native alpha and traceback are retained under:
`/home/r12user5/Documents/Jiawei/colorpeel-runs/mailbox_soft_alpha_v1/20261105__sam_soft_pink__d5b9d15__42/`.
Subsequent execution uses a new run directory; no failed evidence is overwritten.
