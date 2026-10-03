# IID multi-hypothesis + VLM semantic selection: pilot result

Run: `/home/r12user5/Documents/Jiawei/colorpeel-runs/natural_material_iid_v1/run_20261003_mailbox_spoon_v2/`.
The earlier `run_20261003_mailbox_spoon_v1/` is retained as a failed attempt:
the adapter expected five channels, while the released model returned six.
No candidate from v1 entered this analysis.

## Method and provenance

- Official IID checkout: `59da5826c4b336a2d5905a9286855c5698778fd6`.
  Released `iid_e250.pth` SHA-256:
  `804c67834d2ce172018bfd954b59361c0a1537a7f2b456e2b9ff679b2ca21def`.
  The run manifest also hashes the OpenCLIP weight, IID model config, code,
  inputs and individual float maps. Project commits: sampling and primary
  aggregation `0b5fd92`; optional larger-mask aggregation `0fb5791`.
- Ten independent `model.sample` calls per image, seeds 42–51, using the
  released config's 100 DDIM steps and eta 0. Each raw six-channel prediction
  was saved before averaging: albedo RGB, roughness, metallic, auxiliary BRDF
  channel. Albedo and auxiliary BRDF are excluded from the canonical material.
- Inputs were square natural crops resized to 480×480 and centered in a
  640×480 neutral-gray canvas to match IID's input size without changing
  aspect ratio. The material mask underwent the same geometric transform.
- The fixed Qwen3-VL semantic priors from the preceding semantic-constraint
  experiment were reused. The categorical gate allows coated/painted mailbox
  hypotheses only when metallic ≤0.15 and roughness is 0.20–0.65; for the
  glossy bare-metal spoon it requires metallic ≥0.50 and roughness ≤0.45.
  The VLM did not assign numeric R/M. For each candidate, the float map's
  median within the material mask is evaluated; the reported R/M is the
  median across passing candidates.
- The primary mailbox ROI is the preliminary painted-lid mask. A second
  aggregation uses the larger subject-eroded mask on exactly the same IID
  candidates, with its own transformed-mask manifest. Spoon uses the same
  bowl-interior ROI in both aggregations. The two subjects have separate
  candidate and renderer subdirectories.

The [paper](https://arxiv.org/abs/2312.12274) reports 50 DDIM steps for its
experiments, while the [released code](https://github.com/Peter-Kocsis/IntrinsicImageDiffusion)
configures 100. This run follows the released code. Saving separate hypotheses
and using VLM selection are deliberate adaptations; the official CLI averages
ten predictions before saving. IID was trained on synthetic indoor scenes,
making both natural image crops out of distribution.

## Primary ROI results

| Source | VLM prior | Passed / 10 | Selected R | Selected M | All-candidate R median [range], IQR | All-candidate M median [range], IQR |
|---|---|---:|---:|---:|---|---|
| Mailbox painted lid | coated/painted, semi-gloss | 2 | 0.3108 | 0.1173 | 0.3451 [0.0232, 0.4504], 0.0693 | 0.1765 [0.0237, 0.2758], 0.0685 |
| Spoon bowl | bare metal, glossy | 1 | 0.3575 | 0.5970 | 0.3268 [0.1931, 0.7502], 0.1301 | 0.3146 [0.1364, 0.7506], 0.2281 |

| Seed | Mailbox R | Mailbox M | Mailbox gate | Spoon R | Spoon M | Spoon gate |
|---:|---:|---:|---|---:|---:|---|
| 42 | 0.365 | 0.144 | pass | 0.257 | 0.136 | fail |
| 43 | 0.348 | 0.173 | fail | 0.634 | 0.475 | fail |
| 44 | 0.256 | 0.091 | pass | 0.351 | 0.227 | fail |
| 45 | 0.374 | 0.235 | fail | 0.411 | 0.382 | fail |
| 46 | 0.450 | 0.224 | fail | 0.302 | 0.402 | fail |
| 47 | 0.291 | 0.172 | fail | 0.221 | 0.247 | fail |
| 48 | 0.023 | 0.024 | fail | 0.193 | 0.234 | fail |
| 49 | 0.335 | 0.205 | fail | 0.750 | 0.751 | fail |
| 50 | 0.343 | 0.180 | fail | 0.298 | 0.193 | fail |
| 51 | 0.418 | 0.276 | fail | 0.357 | 0.597 | pass |

The mailbox's two passing candidates themselves span R=0.256–0.365 and
M=0.091–0.144. The spoon has only one passing candidate: its accepted-set
IQR is mathematically zero but provides **no uncertainty estimate**.
These values are provisional render parameters, not reliable physical ground
truth or a stable canonical M*.

## Mailbox mask sensitivity

| ROI | Passed seeds | R | M | All-candidate R IQR | All-candidate M IQR |
|---|---|---:|---:|---:|---:|
| Preliminary painted lid | 42, 44 | 0.3108 | 0.1173 | 0.0693 | 0.0685 |
| Subject-eroded mask | 42, 44 | 0.3078 | 0.1155 | 0.0631 | 0.0633 |

The change is small (ΔR=−0.0030, ΔM=−0.0017). For this IID model and these
two masks, local mask size does not explain the large variation across
hypotheses. This does not rule out errors in the actual object segmentation.

## Renderer check and decision

The provisional parameters were rendered with the existing Blender 4.2
Cycles/Principled BSDF neutral-gray sphere under soft-front, side-directional,
and top-environment lighting. All six renders completed, with no obvious
shader or exposure failure. The images show changing highlights; the two gray
sphere sheets alone do not establish material fidelity or distinguish the
source surfaces decisively. Renderer metadata and image hashes are in
`review_render/{mailbox,spoon_positive_control}/manifest.json`.

The earlier SuperMat direct estimate was mailbox R≈0.353, M≈0.039 and spoon
R≈0.247, M≈0.847. IID's selected estimates differ, particularly spoon
metallicity; there is no basis yet to call either estimate physically correct.
The low IID pass rates and synthetic-to-natural domain shift mean this pilot
does **not** justify generating the full counterfactual dataset or using these
values to train `<M*>`. No such generation or training was run.

Review sheets:

- `review_sheets/mailbox.png`: source ROI, three renders, all ten albedo/R/M
  hypotheses and pass/fail marks.
- `review_sheets/spoon_positive_control.png`: same for the positive control.

The exact numerical selection checks, per-candidate map statistics, input
hashes, and uncertainty distributions are stored in `canonical/*.json` and
`canonical_subject_eroded/*.json`; `candidates/manifest.json` records model
and sampling provenance. The review sheet script and image hashes are in
`review_sheets/manifest.json`.
