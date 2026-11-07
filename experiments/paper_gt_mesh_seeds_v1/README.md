# image_000__mask_160: GT mesh candidates

The user requested original/pink RGB x GT hard/BiRef soft alpha x seeds42-45:
16 new meshes, retained for manual Subject selection. This runner composes the
existing preprocessing, TRELLIS.2 generation and review modules; it does not
change their methods or start training.

Inputs are the source image and GT mask locked in this object's REVIEW_STATE.json.
Hard alpha is the unchanged binary GT mask. BiRef runs once on original RGB in
the GT bounding box with the existing 10% padding; soft alpha is clipped to GT.
Pink uses the existing hue0 recolor. Each condition is prepared once and reused
for all four seeds. Hard/soft pre-alpha RGB must match pixel for pixel.
The generator retains its pinned models and official alpha automatic crop.

Each mesh independently uses the existing automatic camera fitter: 72 starting
cameras followed by one Powell optimization (maxfev350, maxiter6), selecting the
lowest recorded loss. The fitting target is that condition's RGBA512 on the
original image canvas (including pink RGB and/or soft alpha when applicable).
The objective remains 1-soft_mask_IoU + 0.05*RGB_BaseColor_MAE. No manual initial
camera or warm start is supplied. Review JSON and the full fitting trajectory
are retained and checked against the four-view renderer's reference matrices.
The other views remain yaw0/+90/-90. Camera fitting changes no mesh geometry.

Run from a clean, exact-revision checkout with the established external runtimes:

```sh
python -B -m experiments.paper_gt_mesh_seeds_v1.run \
  --object-root /path/to/paper_experiments_v1/image_000__mask_160 \
  --output /path/to/paper_experiments_v1/image_000__mask_160/mesh_candidates/NEW_RUN \
  --gpus 0 1 2 3
```

Set COLORPEEL_RUN_ROOT and COLORPEEL_OUTPUT_ROOT to the authorized external run
root and use the existing offline environment. Each GPU processes its assigned
conditions serially; every subprocess checks that its GPU is idle first.
Failures remain visible without automatic retries or quality-based selection.

Outputs: status.json (all16 slots, provenance and commands), index.html, four
condition sheets, and variants/CONDITION/seed_N/{trellis,geometry_views}.
The native mesh is trellis/native_asset.pt; the automatic reference evidence is
trellis/review/{review.json,camera_fit_trajectory.json}. Generated assets remain
under the same object parent as future checkpoints and inference.

Validation: tests/test_paper_gt_mesh_seeds_run.py and the existing GT preparation,
GT batch and automatic-camera tests. Perform finite startup checks only; do not
monitor the full run or accept a mesh on the user's behalf.
