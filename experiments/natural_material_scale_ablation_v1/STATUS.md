# Scale-only metal spoon diagnostic status

Date: 2026-10-04. The full-map metal-spoon object scale was reduced from 2.0
to 1.3; all other grid config fields and renderer inputs were held fixed.
This run uses the original shape-specific captions, not the generic-caption
ablation captions.

Server data root:
`/home/r12user5/Documents/Jiawei/colorpeel-runs/natural_material_scale_ablation_v1/run_20261004_metal_spoon_scale13/`.
The 96-image Cycles grid completed. All 72 train and 24 held-out cone records
are present. The 72 training object masks have 12,482–28,336 foreground
pixels, median 20,750 (about 7.9% of 512×512), versus original full-map
32,011–70,463, median 52,836 (about 20.2%). No training mask touches an image
edge. `mask_occupancy_qa.json`, the render manifest, each image/mask hash,
staging manifest, and `configs/audit.json` preserve the comparison.
The one-image smoke comparison is in `review/smoke_compare.jpg` locally.

The clean-checkout launcher dry-run succeeded at server commit `f78b27e`.
The independent 5000-step token-local K/V training was launched on GPU 1:
`/home/r12user5/Documents/Jiawei/colorpeel-runs/natural_material_scale_ablation_v1/20261004-210138__natural_material_scale_ablation_v1__metal_spoon_scale13_token_local_kv_5000__f78b27e__42/`.
Its first manifest read recorded `running`. Per the user's instruction,
training is not being monitored. The user will report completion, after which
the fixed cone/mailbox plain/photo prompts, seeds 42–46, PNDM 100 and CFG 3.5
should be run against this checkpoint and the original scale-2.0 checkpoint.
