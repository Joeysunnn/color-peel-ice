# Material full-grid generated-image human gate

Decision date: 2026-08-30

## Human findings

- Metal and rubber are stably distinguishable.
- Shape and color remain correct when material changes.
- Some metal outputs show a two-region appearance: the upper region has strong
  metallic reflections while the lower region has weak or absent visible
  gloss.
- Some rubber outputs, especially gray spheres, show noticeable specular
  highlights, but remain identifiable as rubber.

## Decision

The full-grid gate passes as
`passed_with_known_material_appearance_artifacts`. The observations are treated
as material-lighting appearance coupling and material-fidelity limitations,
not current evidence of systematic subject/color/material token entanglement.

The locked Fold A/B/C training campaign for seeds 42/43/44 is authorized. No
loss, renderer, dataset, prompt, optimizer, token, or training parameter may be
changed. The campaign must run serially on GPU 3 and stop on the first failure.

Held-out evaluation must retain separate human labels for metal upper/lower
layering and excessive rubber specular reflection. This approval does not claim
illumination-invariant material representation or solved disentanglement.
