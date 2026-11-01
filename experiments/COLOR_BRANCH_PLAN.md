# Color branch 暂定统一方案

日期：2026-11-01（Asia/Shanghai）。依据：用户确认实验完成并人工验收。

## 当前结论

**当前 Color branch 所需训练步数较少，100、150 steps 的结果均可以满足要求。**
后续暂定每组训练 **150 steps**，保存并测试 **100 / 150 steps**。
original/fullrank 与 rank64/alpha64 两组均沿用这一计划。

这是当前 SD2.1、原 Mailbox 橙色、渲染灰背景九图及相应 Color transfer
测试下的用户验收结论；尚未证明所有颜色、数据集、基础模型均适用相同步数。
也不据此断言更长训练必然变差，或所有属性已严格解耦。

## 暂定方法与设置

- 数据：球体、立方体、圆柱体 × 三视角，共九张；环绕式灰色背景直接渲染，
  保留自然明暗与投影，不后期填成统一 RGB128。
- 前景：沿用原图目标 Lab a/b 和已有受限亮度映射（L约42–58）；背景像素不改。
- 模型：既有冻结 SD2.1-base512；`<C*>`，初始化词 `orange`。
- 两种方法：original 全秩 token-local K/V 残差＋目标 embedding；
  rank64/alpha64 token-local K/V LoRA＋目标 embedding。其余基础模型冻结。
- 每组150 updates；配对保存100/150步的 adapter 和 embedding。
- 训练seed42，FP32，batch/accumulation=1/1，constant LR，embedding LR=1e-5，
  original K/V LR=1e-5，rank64 K/V LR=5e-5，weight_decay=0，cos_weight=0。
- 沿用整张有效图像区域的损失，不提供主体mask作损失限制；固定样本顺序，
  150步对应16个完整九图循环加前6张。其余训练行为保持现有实现。
- 沿用十条 Color transfer prompts，PNDM100、CFG6、512px、FP16、checker开启。
  本次验收的生成seeds为30–45（含两端）。

## 验收来源与状态

实现：[150步实验](sd21_color_rendered9_150_v1/README.md)，执行提交
`5f3389aad843ae0eb631cb2422941cef30abb9ff`。

服务器实验：
`/home/r12user5/Documents/Jiawei/colorpeel-runs/sd21_color_rendered9_v1/20261101-183502__rendered9_both800__2fc8981__42/experiment/supplemental150/20261101-193942__both150_save100_150__5f3389a__42/experiment`

用户本轮确认实验完成、100/150步均满足要求。本次只记录结论与后续默认方案，
未重新读取服务器终态、重新生成图片、训练或启动监控。
历史200/800/1000步实验参数、权重、图片和状态均保留，不改写。
现有150步入口已符合该保存计划，无需修改训练器。
