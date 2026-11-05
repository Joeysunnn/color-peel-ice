# TRELLIS conditioning: user observations

## 2026-11-05 user review

The user explicitly asked to retain this observation:

> 据我观测，mesh提取结果和颜色、alpha mask有关。
> mailbox original提取结果不好，pink提取好。
> wood spoon反而是original比pink提取效果好。

| Photograph | User's visual assessment |
| --- | --- |
| Mailbox | Original color gives a poor mesh; pink gives a better result. |
| Wood spoon | Original color gives a better result than pink. |

Record these as the user's relative visual judgments on the examined examples.
Conditioning color and alpha both affect reconstruction; the preferred color
differs across subjects. Do not summarize this as "pink is always best" or
"original is always best."

For the latest wood-spoon comparison, original and pink used the same new soft
alpha, models and seed42. Original reproduced the historical native mesh
exactly. Both avoided the earlier broad hard-alpha bowl facets, while the user
preferred the original-color result overall. This relative preference is a
separate observation from strict four-branch learning or transfer acceptance.

Evidence:

- [Mailbox original/current-SAM pink and historical replay](subject_pink_input_repro_v1/RESULTS.md).
- [Mailbox five-color comparison](subject_five_color_mesh_v1/RESULTS.md).
- [Wood-spoon alpha hardening diagnostic](wood_spoon_input_cause_v1/RESULTS.md).
- [Wood-spoon new soft alpha: original versus pink](wood_spoon_soft_alpha_v1/ORIGINAL_RESULT.md).

This entry records user review only. It does not change saved measurements,
Subject eligibility guards or the temporary production pink default, and does
not authorize an automatic color search or another training run. The Mailbox
review must not be represented as a newly tested BiRefNet-soft-alpha result.
