# Emission Color-Branch Decision

## Decision

Emission CLEVR counterfactuals are the provisional operational color branch for
the natural-image subject/color plan. This decision is based on the completed
two-target short learning runs and their expanded transfer-generation outputs;
it does not alter RC-1, RC-2A, RC-2B, RC-2C, or any earlier C-pilot artifact.

## Evidence

- `D1GT:81/130.png` (`C*=83.70`, `h=40.95 degrees`) was trained in
  `20260912-062941__natural_image_subject_color_pilot__emission_color_transfer_orange_short100__89dd1d1__42`.
  Its 10-template by 10-seed transfer generation completed successfully in
  `20260912-074208__natural_image_subject_color_pilot__emission_color_transfer_orange_all100__7dd85d5__42`.
- `D1GT:13/30.png` (`C*=51.71`, `h=290.15 degrees`) was trained in
  `20260912-073922__natural_image_subject_color_pilot__emission_color_transfer_purple_short100__7dd85d5__42`.
  Its matching transfer generation completed successfully in
  `20260912-075720__natural_image_subject_color_pilot__emission_color_transfer_purple_all100__275b294__42`.
- Both expanded runs used the fixed ten canonical `category == "transfer"`
  templates from the supplied manifest, seeds 42--51, and recorded 100/100
  successful PNG outputs with the source-manifest and learned-model hashes in
  their provenance files.

## Known qualitative limitation

Human review observed that the orange-initialized `<C*>` can, in some cases,
produce a literal orange fruit in the generated scene. This is a qualitative
initializer-associated semantic artifact. It does not invalidate the completed
generation ledgers or establish that all orange transfers are correct. It must
remain visible in later subject/color evaluation and must not be hidden through
prompt filtering, sample deletion, or unrecorded resampling.

## Scope released

The next planned batch is a separately specified natural-image recoloring
subject-branch pilot. This record does not authorize mixed shared-checkpoint
training, full counterfactual dataset production, or held-out evaluation.
