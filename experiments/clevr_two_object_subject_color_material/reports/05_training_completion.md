# Controlled two-object training completion

Date: 2026-09-01  
Launch commit: `0446729a267e4282f64e857e121ffe86214404e3`  
Environment: `colorpeel017`  
GPU: `CUDA_VISIBLE_DEVICES=3`, Tesla V100-SXM2-32GB

## Gates

- Full server test suite: `150 passed in 26.82s`.
- 2-step instance-mask smoke: passed, 2/2 finite metric rows, exact expected
  token exposure, final K/V plus eight tokens saved and reloaded.
- 18-step coverage smoke: passed, subject/color exposure 6 and material
  exposure 9; every modifier token had nonzero gradients and a positive delta.

## Full run

Run:

`/home/r12user5/Documents/Jiawei/colorpeel-runs/20260901-163000__clevr_two_object_subject_color_material__controlled_two_object_seed42__0446729__42`

- Status: succeeded, return code 0.
- Runtime: 2026-09-01 16:27:22 to 16:37:14 HKT.
- Completed steps: 1500/1500; all loss records finite.
- Mean reconstruction / CAA / total loss: `0.429982` / `0.022995` / `0.434581`.
- Step-1000 Accelerator checkpoint: present.
- Final Custom Diffusion K/V and eight token files: present and non-empty.
- Final textual-inversion reload count: 8/8.
- Custom Diffusion SHA-256:
  `8025623456b9789d481db5550e47360bda5204f9a94f1ce3892807f84d5f927d`.
- Metrics SHA-256:
  `86f294a66672502ee2f9905b6ecd7222dc1599c9fc14058be4636e519e78d220`.
- Embedding audit SHA-256:
  `059f7b2684c871dc514d9e523134354830a9a1e43a036db635b6a512ebd6fe7e`.

## Exposure disclosure

The official DataLoader remains sequential. Because 1500 steps equal two full
576-record epochs plus 348 records, the partial third epoch produces subject
exposures `<s1*>=576`, `<s2*>=540`, and `<s3*>=384`. Every token still has
nonzero gradients on every exposure and a positive final delta. This ordering
effect is reported and was not corrected post hoc.

Ordinary-vocabulary AdamW drift remains observation-only: 49,408 changed rows,
mean L2 `6.76598e-05`, maximum L2 `2.11111e-04`.

No image generation or two-object disentanglement claim is included in this
training completion report.
