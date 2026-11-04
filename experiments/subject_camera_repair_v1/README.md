# Original-image Subject camera repair

The single-image integration used complete-mesh centering, while historical
center11 preserved the fitted source camera and crop. Its newly reconstructed
orange Mailbox also has poor source-view alignment (IoU 0.661872). The formal
input remains the original photograph and its Scheme A SAM mask; historical pink
geometry is a diagnostic reference only.

First run a bounded same-mesh probe: 16 predetermined yaw/pitch views and the two
saved current cameras, all with neutral clay appearance. No geometry edits,
camera optimization, model sampling, training or transfer generation occur.
Do not treat a completed probe as image-quality approval.

From a reviewed clean Git checkout in the existing TRELLIS environment:

```bash
python -B -m experiments.subject_camera_repair_v1.probe \
  --asset /absolute/path/trellis/native_asset.pt --asset-sha256 EXPECTED_SHA256 \
  --source-receipt /absolute/path/input/source_receipt.json \
  --subject-manifest /absolute/path/subject/subject.json \
  --output FRESH_EXTERNAL_DIRECTORY --gpu 0
```

The manifest binds the exact geometry, source receipt and original camera review.
The output includes source hashes, immutable geometry checks, all camera matrices,
mask extents, 18 clay images, a contact sheet and `status.json`. Existing inputs
and results are preserved. Later repair decisions must be based on these views,
not an assumption that the original orange photograph cannot work.

The fixed-view probe confirmed elongated geometry in the current native asset.
The next bounded attempt reconstructs exactly one new asset with `--seed 0`,
using the same original-image receipt, Scheme A mask, pinned models and sampler
settings. The legacy reconstruction command keeps its default seed 42. No pink
input or historical mesh is substituted, and this attempt does not start training.

Subject rendering now preserves the saved source camera matrices verbatim.
Both the recorded source-review mask IoU and the first actual training-render
mask IoU must be at least 0.80. Edge crops are allowed only on edges touched by
the original mask within two pixels. A failed check leaves the branch unavailable;
it does not automatically retry reconstruction. Optimizer termination is recorded
as a warning because its success flag alone did not distinguish historical usable
fits. These are engineering checks, not user approval or proof of 3D fidelity.

The seed-0 sample's first source-camera fit reached IoU 0.722472 and exhausted
the 350-evaluation optimization budget. One camera-only continuation is permitted
from its saved best six parameters, with the same objective and bounds and at
most 1000 Powell evaluations / 10 iterations. Keep the original baseline as a
candidate and retain both reviews. Do not generate another mesh, add starting
points, lower the 0.80 guard or extend the budget again. The continuation is an
explicit diagnostic; the normal reconstruction command keeps its original fit
budget until the result has been evaluated.
