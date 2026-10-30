# Subject branch 暂定统一方案

日期：2026-10-30（Asia/Shanghai）。用户确认：本次先记录统一方案。
本文是后续 Subject 实验的暂定约定，不是已执行实验的配置或结果记录。

## 数据与方法

- 反事实数据构建暂定采用当前已使用的 TRELLIS.2 方法：从去背景原图重建主体，保留生成的原生几何，通过统一 Base Color 与标量 Metallic/Roughness 重新渲染。
- 每个主体使用 **72 张**训练 RGB，采用 **blue、green 两色 × 6 种 M/R 状态 × 3 个视角 × 2 套灯光**。
- M={0,1}，R={0.15,0.50,0.85}；沿用现有视角、studio/courtyard 灯光、512px 分辨率及中性背景。
- 沿用已有图像、mask、caption、来源顺序和来源 hash；不混入其他颜色或原始照片。
- 每个主体分别训练 `<S*>`，保留 **original/fullrank** 与 **rank64/alpha64** 两组对照。original 指 token-local K/V 全秩残差与目标 embedding 更新，不是全 UNet 微调。

## 训练与保存点

| 项目 | 暂定统一设置 |
| --- | --- |
| 每组训练总步数 | **1000 updates** |
| 保存 checkpoint | **400 / 600 / 800 / 1000 steps** |
| 基础模型 | 既有冻结 SD1.5 bundle |
| 初始化词 | 沿用主体类别：Mailbox 用 mailbox，两把勺子用 spoon |
| 训练 seed | 42 |
| batch / gradient accumulation | 1 / 1 |
| embedding LR | 1e-5 |
| original K/V LR | 1e-5 |
| rank64 K/V LR | 5e-5，rank=alpha=64 |
| 其他训练设置 | 沿用现有 optimizer、constant LR、mixed_precision=no、object-mask loss、hflip=false、cos_weight=0 |

沿用固定顺序循环，不更改 shuffle 或重加权。1000 updates 对应 13 个完整的
72 图循环加前 64 张，每图曝光 13 或 14 次。保存点用于比较主体身份保持与迁移效果，
不预先指定哪一个 checkpoint 最佳。

## 沿用的评估与目录约定

- 每个保存点使用同 step 配对的 Subject embedding 和 adapter 进行推断。
- 沿用 metallic、matte、glossy plastic、ceramic、wood 五种材质测试，以及
  `in the snow`、`surrounded by flowers`、`in white color`、`in orange color`
  四种独立 transfer 测试。
- 沿用 seeds42/43、PNDM100、CFG3.5、512px、fp16 和现有 safety checker。
  按两种方法计算，每个 checkpoint 36 张，四个 checkpoint 每主体计划 144 张。
- 每个主体的一次实验放在同一新目录，包含实际训练数据及 caption、两组训练、
  step400/step600/step800/step1000 推断、日志、配置和审计记录。
- 执行时严格遵循本地修改、验证、提交推送，再由服务器拉取运行；旧结果保留，
  工作树保持干净，完成启动检查后不持续监控。

## 状态与适用范围

- confirmed：本方案由用户指定并确认先记录；本次只修改文档。
- pending：将四个新保存点接入可执行训练、配对导出及推断配置；具体启动哪些主体，待后续执行任务确定。
- 现有历史入口仍保留其原设置：例如
  [spoon600](subject_trellis_spoons600_v1/README.md) 为 600 步、保存 200/400/600；
  旧 [TRELLIS Subject](subject_trellis_dataset_v1/README.md) 1000 步实验保存 500/750/1000。
  本文不改写这些实验的历史参数、代码、运行状态或结果。
- TRELLIS.2 数据是生成几何上的合成反事实，不等同于原照片的精确反事实；
  本方案不表示多勺柄问题或 Subject/Material 解耦已经解决。
