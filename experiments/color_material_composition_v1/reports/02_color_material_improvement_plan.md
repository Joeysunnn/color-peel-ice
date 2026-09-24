# C+M improvement plan after owner review

## Diagnosis to resolve

The owner review finds material appearance acceptable in mugs and spheres,
while `<C*>` orange is unreliable even in color-only results and becomes less
clear in paired results. Cube seed 44 fails to be a cube in the base condition,
so retain it as a shape-validity failure rather than evidence of C/M
interference. The orange-fruit artifact is recorded but is lower priority.
These observations do not yet distinguish weak standalone color learning from
competition between the two tokens during inference.

## 1. Matched prompt and adapter-strength diagnosis, without training

Keep the selected `<M*>` checkpoint, base model, seed, CFG, and denoising steps
fixed. For each object and seed, compare the following exact prompt family:

| Condition | Prompt pattern |
| --- | --- |
| Literal controls | `a photo of a {object} with orange color and metal material` |
| Learned color only | `a photo of a {object} with <C*> color and metal material` |
| Learned material only | `a photo of a {object} with orange color and <M*> material` |
| Both learned | `a photo of a {object} with <C*> color and <M*> material` |

The literal-orange plus `<M*>` row tests whether the selected material branch
can express orange metal under this wording. Compare it directly with the
both-learned row. Include more seeds and retain every base-shape failure in
the ledger; classify attribute control only on predeclared shape-valid cases.
Separately vary only the color K/V residual scale around its current value
(for example 0.5, 1.0, 1.5, 2.0) while keeping the `<C*>` embedding and
material branch fixed. This is a diagnostic scale sweep, not a new trained
checkpoint. Record object validity and material retention alongside color.

## 2. Improve standalone C before adapting C+M

If color-only and literal-control comparisons show the learned color branch
is weak, keep `<M*>` frozen and run small, one-change-at-a-time color pilots:

1. Restage the same nine verified emission images with their verified object
   masks. Compare masked and existing full-image training at the same step
   count, seed, and initialization.
2. On the selected mask setting, compare 100, 300, and 500-step checkpoints.
   The present `hflip: true` has no effect because the flip transform is
   commented out; an actual flip would be a separately recorded data change.
3. Evaluate color-only on cube and sphere plus held-out mug before C+M.
   Review orange coverage on the object, shape retention, and background
   leakage; fruit leakage remains a secondary field.

If nine images cannot produce reliable color-only transfer, expand a new
counterfactual color set with the same orange target crossed over shape,
view, lighting, and at least matte and selected metal materials. Hold mug out
to preserve an unseen-object check. Review rendered masks and orange body
color before training. This stage changes the data distribution and needs its
own provenance and visual review.

## 3. Only if C-only passes but C+M still loses color

Test inference strength calibration first. If it does not preserve color and
material together, a later targeted training pilot may freeze the base model
and `<M*>`, update only the `<C*>` embedding and C-local K/V residual, and mix
C-only with paired orange-metal examples. Check the old M-only outputs after
training to detect material degradation. The current trainer supports only one
token-local modifier per run, so this paired update requires an explicit new
training implementation and protocol; it is not a YAML-only change.

## Readout

For each image record shape validity, orange on the object, metallic surface,
requested context, extra objects, fruit leakage, and safety filtering. Keep
the original 36-image run as an unchanged baseline. A successful pilot needs
the paired row to show both orange and metal across several shape-valid seeds
and at least one held-out noun; a single attractive seed is insufficient.
