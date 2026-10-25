# ZeST Material Counterfactual Generator v1

独立模块，不改Token Learning。首轮只有原始Mailbox，沿用既有SuperMat A/R/M、已审阅区域、
对象Mask和1024 DPT。目标是筛出适合Material Token学习的候选，不把视觉材质迁移直接称为解耦数据。

## 官方Baseline与适配

- ZeST `b71ea174c08c5385f2487cc0ee6d8302a4bdf8f3`；Tencent IP-Adapter
  `62e4af9d0c1ac7d5f8dd386a0ccf2211346af1a2`，完整cross-attention，不能用MARBLE材质块替代。
- 同SDXL、ViT-H、IP-Adapter、Depth ControlNet权重只读共享，旧官方源码和环境不改。
- 官方30steps、seed42、Depth scale0.9、IPAdapter scale1、灰度前景初始化。
  Diffusers0.30下原调用默认strength0.9999、CFG5，本模块显式固定并记录实际步数。
- 使用项目既有修补Mask／冻结DPT属于明确适配，不声称官方DPT预处理逐项复现。
  512源图先BICUBIC适配1024，避免之前初始化尺寸错位。

## 参考库与四组消融

固定Cycles球体、均匀无纹理Albedo、灯光、相机、seed，只改变M=[0,.5,1]、R=[.2,.5,.8]。
参考Albedo来自原SuperMat已审阅lid区域的linear RGB中位数；另渲染估计原M/R的anchor。
PBR数值只属于参考球，不是生成Mailbox的Ground Truth。

| 组 | 初始化 | ControlNet | 能力边界 |
|---|---|---|---|
| baseline | 官方有效编辑区灰度 | Depth .9 | 原始ZeST生成路径，冻结项目Mask／Depth |
| color | 原RGB亮度×估计Albedo色度 | Depth .9 | 输入侧软条件，不是扩散中的可微Albedo损失或硬保证 |
| texture | 官方灰度 | Depth .9＋Albedo Canny .3 | 显式新增边缘条件，非原版ZeST，不能保证完整Texture |
| combined | Albedo色度初始化 | Depth .9＋Albedo Canny .3 | 两者组合，仍须近似QC及人工审阅 |

四组同prompt、seed、Depth、有效Mask、参考和其余推理参数；只改变表中所列因素。
所有图独立生成，保留原生1024 RGB；不把原RGB主体覆盖回输出，也不后贴背景掩盖漂移。
接近1的strength可能削弱初始化软条件的作用，这应由消融验证，不预设Color组有效。

先生成并评估baseline的self Reference和estimated-original参考两个控制。前者仍是扩散重生成，
不是数学恒等；后者有参考几何／照明及估计误差，不能混称“无编辑”。
`grid`要求已存在baseline/self质量记录；Identity未通过时，后续只作为诊断候选，训练资格保持关闭。
其余组也有独立Identity控制；每组九格共36张诊断候选，不自动进入训练集。

## 质量与筛选

原图与生成图都用同一官方SuperMat设置重新提取，512px、seed42、相同alpha。
保存估计A的DeltaE76（非DeltaE00）、Delta ab／L，估计A的三尺度边缘与高频相关，
reviewed ROI的估计M/R median/MAD／目标方向，以及独立DPT预测的尺度／偏移对齐误差。
背景、Mask外及保护区的raw RGB MAE仅作内容保护检查，不作为Albedo泄漏证据。
不使用同一个输入Mask伪造输出IoU；SuperMat和单目Depth都不是物理或几何真值。

`mailbox.json`中的阈值是首版保守工程预设，未获人工验证，不会为让失败Identity通过而放宽。
可改为新配置重跑；null代表未校准，不能自动通过。缺map、空域、NaN/Inf拒收；低纹理明确不可评估。
所有默认要求人工审阅，只有候选各项通过且Identity获得批准，才可能`training_eligible=true`。
当前不会启动Material Token训练，也不会把reference M/R直接当成生成图真实PBR标签。

## 本地→远端→服务器

先本地提交推送，再在服务器独立Git checkout运行。以下路径均为新目录：

```bash
export PYTHONDONTWRITEBYTECODE=1 TMPDIR=/var/tmp
export COLORPEEL_OUTPUT_ROOT=/var/tmp/colorpeel-r12user5/runs
PROJECT_PY=/var/tmp/colorpeel-r12user5/assets/marble_c_v1/env/bin/python
ZEST_ASSETS=/var/tmp/colorpeel-r12user5/assets/zest_material_v1
# 独立源码副本，不把旧SuperMat目录中的权重／缓存纳入源码；权重路径仍只读共享。
git clone --no-hardlinks /home/r12user5/Documents/Jiawei/colorpeel-runs/natural_material_supermat_v1/deps/SuperMat "$ZEST_ASSETS/SuperMat"
git -C "$ZEST_ASSETS/SuperMat" checkout --detach 4fe25bc6cb8cb7a3ed81ce74512f760dd33f80f4
# FRESH_ZEST_RUN 必须是新结果目录；后续续跑使用同一个已准备目录。
$PROJECT_PY -B -m experiments.zest_material_counterfactual_v1.assets \
  --output "$ZEST_ASSETS" --shared-marble-assets /var/tmp/colorpeel-r12user5/assets/marble_c_v1 --with-canny
$PROJECT_PY -B -m experiments.zest_material_counterfactual_v1.prepare \
  --config experiments/zest_material_counterfactual_v1/mailbox.json --output "$FRESH_ZEST_RUN"
CUDA_VISIBLE_DEVICES=3 /home/r12user5/Documents/Jiawei/tools/blender-4.2.11-linux-x64/blender \
  --background --python-exit-code 1 --python experiments/zest_material_counterfactual_v1/render_bank.py -- \
  --plan "$FRESH_ZEST_RUN/bank_plan.json" --output "$FRESH_ZEST_RUN/bank" --gpu 3
$PROJECT_PY -B -m experiments.zest_material_counterfactual_v1.run --run "$FRESH_ZEST_RUN" \
  --assets "$ZEST_ASSETS" --stage identity --gpu 2 --variants baseline
$PROJECT_PY -B -m experiments.zest_material_counterfactual_v1.assess --run "$FRESH_ZEST_RUN" \
  --assets "$ZEST_ASSETS" --stage identity --gpu 2 --resume --supermat-root "$ZEST_ASSETS/SuperMat"
$PROJECT_PY -B -m experiments.zest_material_counterfactual_v1.review --run "$FRESH_ZEST_RUN"
# 查看Identity误差之后继续诊断，训练资格仍关闭：
$PROJECT_PY -B -m experiments.zest_material_counterfactual_v1.run --run "$FRESH_ZEST_RUN" \
  --assets "$ZEST_ASSETS" --stage all --gpu 2 --resume
$PROJECT_PY -B -m experiments.zest_material_counterfactual_v1.assess --run "$FRESH_ZEST_RUN" \
  --assets "$ZEST_ASSETS" --stage all --gpu 2 --resume --supermat-root "$ZEST_ASSETS/SuperMat"
$PROJECT_PY -B -m experiments.zest_material_counterfactual_v1.review --run "$FRESH_ZEST_RUN"
$PROJECT_PY -B -m experiments.zest_material_counterfactual_v1.summarize --run "$FRESH_ZEST_RUN"
```

## 参考构图诊断

首轮球体只占参考图约5.6%，其余为灰色场景。官方CLIP编码整张参考图，不自动定位材质区域；
self参考也含有Mailbox之外的彩色门牌。两者都可能干扰材质表征，尚不能据此确定唯一失败原因。
`focus_reference`将参考对象Mask的方形bbox各边扩展10%后裁切、BICUBIC到1024，
仅改变Reference，不裁切目标图，不改变初始化、Mask、Depth或推理参数。
对baseline及combined各做self／estimated-original，共4张独立诊断，不混入原44张消融。

```bash
$PROJECT_PY -B -m experiments.zest_material_counterfactual_v1.focus_reference \
  --run "$FRESH_ZEST_RUN" --assets "$ZEST_ASSETS" --gpu 3
# FOCUS_RUN取上一命令返回的新diagnostics/reference_focusNNN目录。
$PROJECT_PY -B -m experiments.zest_material_counterfactual_v1.assess --run "$FOCUS_RUN" \
  --assets "$ZEST_ASSETS" --stage identity --gpu 3 --resume --supermat-root "$ZEST_ASSETS/SuperMat"
```

每次新建目录，保留Reference裁切框、占比、原始／裁切参考、原始输出、五列Comparison Sheet与输入hash。
诊断目录复制原prepared文件并保留逐字节hash，可独立复用SuperMat／Depth评估，不修改原消融或批准训练。
`summarize`另存`analysis/analysisNNN`，分开报告生成数、QC状态、自动阈值未拒收数及实际训练资格；
同时记录近似M/R响应、有效区域覆盖、1024背景局部变化和初始化组间RGB差异。它不改变原QC阈值或状态。

`--resume`锁配置、prepared输入、生成实现、模型／参考与已有输出hash；只补缺失项。
中断尝试保留在`attemptNNN`，不覆盖旧结果。原始配置不可原地改后冒充续跑。
新增载体可在独立config的`subjects`列表提供对齐RGB、Mask、保护Mask、SuperMat输入／估计和冻结Depth，
共享同一reference bank；第一版运行仍只Mailbox。固定参考色不一定适配其他颜色载体，必须经过同样QC。

主要输出：`bank/`、`source/`、`cases/*/attempt*/raw.png`、`generation_manifest.json`、
`estimates_manifest.json`、`quality/assessmentNNN/report.json`、`dataset_manifest.json`、
`reviews/`、`repro_outputs/`。Comparison Sheet逐行展示Original／Reference／Generated／估计质量指标。
