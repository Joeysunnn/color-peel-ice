# Mailbox 两组官方联合九格

本轮仅生成两组各9张RGB，先展示再由用户决定后续。不扩展30张，不训练。
此前官方toy-car九格已由用户确认符合要求；这里保留相同模型、权重、注入位置、提示词、
Euler、seed42、30steps、CFG5、strength1、ControlNet1、IPAdapter1和既有独立环境。

| stage | Target | Material reference | Generation Mask | Depth |
|---|---|---|---|---|
| `mailbox_bowl` | 原图BICUBIC对齐1024 | 官方原始RGBA metal_bowl | 原样官方rembg | 从该1024输入重新运行官方DPT |
| `mailbox_self` | 同一1024 mailbox | 官方None回退到同一1024 target | 原有mailbox Mask，RGB BICUBIC对齐 | 从该1024输入重新运行官方DPT |

两组使用完整 `M=[-20,0,+20] × R=[-1,0,+1]`，每次同时传入两属性网络，包括零值。
这是相对控制强度，不是PBR标定参数。每张均从同一原图独立编辑，不串联。
两组同时变化Mask和reference，不把组间差异解释成reference单变量因果。

官方函数生成的Mask/Depth仅在各自新运行内缓存；不读取此前512输入的旧Depth。
第一组原样使用官方 `prepare_mask`，不替换为alpha阈值、不修补自动Mask。
两组保存实际生成Mask及单独的 `review_mask.png`：后者只用于一致的mailbox放大视图，
第一组不会用它替换官方自动生成Mask。

旧源码stage、资产、checkpoint、结果均保留，新运行目录必须不存在，所有raw1024
保留且不合成背景。contact sheet顶部显示实际Target、reference、Mask、Depth。

```bash
export PYTHONDONTWRITEBYTECODE=1 TMPDIR=/var/tmp
export HF_HOME=/var/tmp/colorpeel-r12user5/assets/marble_c_v1/hf_home
export COLORPEEL_OUTPUT_ROOT=/var/tmp/colorpeel-r12user5/runs
MPY=/var/tmp/colorpeel-r12user5/assets/marble_c_v1/env/bin/python
# 每个FRESH变量必须是单独、全新的外部运行目录。
"$MPY" -B -u -m experiments.subject_marble_diagnostics_v1.run \
  --assets /var/tmp/colorpeel-r12user5/assets/marble_c_v1 \
  --output "$FRESH_BOWL_RUN" --gpu 2 --stage mailbox_bowl
"$MPY" -B -u -m experiments.subject_marble_diagnostics_v1.run \
  --assets /var/tmp/colorpeel-r12user5/assets/marble_c_v1 \
  --output "$FRESH_SELF_RUN" --gpu 2 --stage mailbox_self
```

查看 `contact_sheet.png`（全画面）、`subject_crop.png`（固定mailbox ROI），并保留
`source/initialization.png`、`source/conditions.json`、`plan.json`、`runtime.json`、
`manifest.jsonl`、`status.json`、`repro_outputs/`。两组Target和新Depth的SHA256应分别一致；
实际生成Mask与reference分别记录，不以色差/梯度统计冒充物理M/R测量。
