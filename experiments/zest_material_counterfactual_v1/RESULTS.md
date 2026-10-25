# Mailbox ZeST v1：生成与质量验证

本轮实现了独立生成、断点续跑、近似属性评估和可视化流程，但Identity未通过；不能将候选称为合格的Material反事实训练集。未启动Token训练。

## 实际执行

- Blender固定场景：9个M/R网格参考球＋估计原材质参考球，共10张。
- ZeST四组：baseline、Albedo色度初始化、Albedo边缘ControlNet、两者组合。每组2张Identity＋9张参数网格，共44张原始1024px输出。
- 独立Reference裁切诊断：baseline／combined各2张，共4张，不混入原消融。
- 固定seed42、名义30步／实际29步、strength0.9999、CFG5、Depth0.9、IPAdapter1；边缘组额外Canny0.3。
- 原图与生成图使用相同SuperMat参数重新提取；独立DPT估计输出几何。原始RGB未回贴到输出。

官方ZeST采用完整Tencent IPAdapterXL，已检查各cross-attention及ControlNet处理器；不是MARBLE去掉属性MLP的替代实现。复用现有1024 Depth和修补Mask属于明确适配，不声称逐项复现官方Depth预处理。

最终44/44张完成QC：44 rejected、0 invalid、0 pending、0 accepted；36张grid的最终训练资格为0/36。
另4张裁切诊断同样全部rejected。它们存在自动指标失败，不是仅因尚未人工审批而暂缓。

## Identity结果

下表是fresh SuperMat估计Albedo的DeltaE76第95百分位，不是物理Albedo真值误差。

| 条件 | 自身参考 | 估计原材质球参考 |
|---|---:|---:|
| baseline | 108.924 | 85.394 |
| color | 68.543 | 28.989 |
| texture | 110.158 | 82.330 |
| combined | 67.008 | 30.409 |

8张均被当前工程阈值拒收。自身参考的高频结构相关从baseline的−0.004改善到combined的0.176，仍不足以说明原纹理保持。颜色初始化有作用，但不能消除新增标记、结构细节重写或凹槽丢失。

estimated-original参考使用均匀、无纹理、估计R/M的球体；它不是原Mailbox真实空间材质的严格无编辑真值。self-reference也仍是扩散重生成，不能当成数学恒等映射。

## M/R网格与保持机制

四组各9张grid全部拒收。combined组的近似材质响应如下；每格为SuperMat估计M/R中位数，列、行只代表参考球的真实输入参数。

| 参考M／参考R | R=0.2 | R=0.5 | R=0.8 |
|---|---|---|---|
| M=0.0 | 0.016 / 0.325 | 0.012 / 0.361 | 0.020 / 0.471 |
| M=0.5 | 0.012 / 0.325 | 0.012 / 0.361 | 0.020 / 0.478 |
| M=1.0 | 0.008 / 0.306 | 0.012 / 0.345 | 0.012 / 0.510 |

固定M时，估计R随参考R增加有一致趋势；估计M却集中在很低范围，没有验证三档Metallic控制。不能把参考参数直接标到生成Mailbox上作为PBR真值。

baseline↔color及texture↔combined共22对同条件对照没有完全重复的RGB图；编辑区RGB MAE约76.1–89.3/255。
本轮颜色初始化确实改变了输出，不能将失败解释为初始化完全未生效。边缘条件也未解决完整纹理保持。

有效alpha覆盖编辑区98.41%，有682个512px编辑像素缺少有效估计支持。原始1024背景像素L1的第99百分位约25.3–26.7/255；平均差异较低不等于背景逐像素不变。

## Reference构图诊断

官方ZeST直接传入整张Reference；Tencent CLIP对整图编码，没有参考对象Mask定位。初始球体仅占5.61%，存在灰色场景干扰风险；自身参考含有Mailbox外的彩色门牌，可能污染颜色及文字特征。这些是有依据的解释，不是已证明的唯一原因。

独立裁切将球体占比提高到54.37%，Mailbox占比由16.38%提高到33.33%；目标图、Mask、Depth、初始化和采样参数保持不变。

| 条件／参考 | 裁切前估计A DeltaE76 p95 | 裁切后 | 裁切后高频相关 | QC |
|---|---:|---:|---:|---|
| baseline／self | 108.924 | 115.244 | 0.003 | rejected |
| baseline／球体 | 85.394 | 74.714 | 0.003 | rejected |
| combined／self | 67.008 | 65.614 | 0.056 | rejected |
| combined／球体 | 30.409 | 32.917 | 0.015 | rejected |

裁切没有解决本轮Identity要求。球体参考可能使目标变成颜色接近、但缺少原纹样的光滑表面；自身参考仍出现新标记。不能据此断言所有Reference构造都无效，也不能将本轮失败归结为Reference未传入模型。

## 解释边界

- 参考球M/R是Blender输入真值；Mailbox输出的SuperMat估计不是目标M/R真值，也不能保证三档单调响应。
- RGB高光变化不直接等于Color泄漏。Color／Texture指标使用估计Albedo；背景RGB指标另作保护检查。
- 当前人工保护Mask为空，不能声称已验证文字／Logo保护。接口支持另外提供保护区域。
- 修补编辑Mask有部分区域超出旧SuperMat有效alpha；该部分没有完整的Albedo／Texture评估支持。覆盖率见独立analysis报告。
- baseline/self Identity失败会关闭全部grid训练资格；人工审阅默认也不批准。不会通过放宽阈值使失败样本进入训练集。
- MARBLE对照沿用历史原始输出，但设置不同，只作诊断展示，不是可归因于架构的公平消融。

ZeST将参考外观编码为隐式特征，不提供“只改变M/R且严格锁定Albedo／纹理”的硬约束。本轮Color初始化与Canny边缘只是软条件，也没有声称实现可微Albedo损失。

## 路径与复现

服务器主结果：

```text
/var/tmp/colorpeel-r12user5/runs/zest_material_counterfactual_v1/20261025-200121__mailbox_v2__8baa9b1__42
```

- `bank/`：10张PBR参考、Mask、场景／参数收据。
- `cases/`：44张raw输出、各项输入和采样记录。
- `quality/latest.json`：最终评估入口；对应report保存逐项指标、阈值和拒收原因。
- `dataset_manifest.json`：最终训练资格，不能用“已生成”代替“已合格”。
- `analysis/analysisNNN/`：原始背景局部差异、近似M/R响应矩阵、覆盖率和初始化对照差异。
- `reviews/`：Original／实际Reference／Generated／估计质量指标，以及MARBLE对照。
- `diagnostics/reference_focus001/`：4张裁切诊断、独立SuperMat／Depth评估及对照图。
- `repro_outputs/`：生成状态、命令、实现适配与可比性说明。

命令见README。34项CPU检查已在服务器通过；四组真实推理、独立评估和原始输出可视化均实际执行。源码、权重、配置及结果有hash收据。

旧SuperMat目录只读复用权重，另建同提交的干净源码副本进行QC。先前因Blender PNG时间戳导致的错误掩码判定已改为比较解码像素；失败尝试原样保留在独立旧目录，不覆盖或删除。

官方依据：[ZeST代码](https://github.com/ttchengab/zest_code/tree/b71ea174c08c5385f2487cc0ee6d8302a4bdf8f3)、[Tencent整图CLIP编码](https://github.com/tencent-ailab/IP-Adapter/blob/62e4af9d0c1ac7d5f8dd386a0ccf2211346af1a2/ip_adapter/ip_adapter.py#L129-L138)、[ZeST论文](https://arxiv.org/html/2404.06425v1)。
