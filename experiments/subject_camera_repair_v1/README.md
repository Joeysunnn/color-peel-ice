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
