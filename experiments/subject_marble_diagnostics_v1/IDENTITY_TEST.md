# Identity Reconstruction Test

先验证无有效属性控制下的内容保持程度；暂停Counterfactual Dataset扩展，不训练。
原始Mailbox、1024自身Reference、已审阅的冻结Mask、从1024重新提取的官方DPT。
沿用已通过官方样例的模型、Euler、30steps、CFG5、strength1、ControlNet1、IPAdapter1。
新阶段 `mailbox_identity` 不覆盖此前任何运行，不引入损失或改模型架构。

## 十二项最小诊断

| seed | 控制 | 数量 |
|---|---|---:|
| 42 | 空MLP字典、M0/R0、M-only[-20,0,+20]、R-only[-1,0,+1] | 8 |
| 43、44 | 每个seed仅空字典与M0/R0 | 4 |

主对照保留seed42；额外两个seed只用于有限的重生成稳定性检查，不是训练数据扩展，
也不支持统计稳定性保证。每张从同一输入独立生成，raw1024不做背景后合成。

空字典完全不调用属性网络，但官方strength1仍然从噪声重生成；它不等于原图直通。
M0/R0和单独M0/R0仍调用各自网络，零输入不保证零偏移。保存
`zero_control_offsets.json` 测量实际CLIP零控制偏移；其范数不是物理材质参数。

## 评估原则

- 对比纹理图案、原有文字／新增字样、轮廓、投递口、锁及部件位置。
- RGB变化、高光、明暗和图像边缘变化可能由反射造成，不能直接定性Color／Albedo泄漏。
- 没有真实Albedo、几何与光照真值，本轮不能独立验证Albedo保持；明确记录unknown，
  不用灰度图、低高光区域或单图属性提取器的输出冒充真值。
- 新身份评估分支不复用旧HSV色差阈值或 `inspect_color_drift` 标记，
  不把RGB MAE或梯度统计当作身份／物理属性评分。只保留注明解释范围的RGB外观诊断。
- 以新增或被改写的图案、文字、明确部件变形作为可见内容保持问题的证据。
  三个seed分别记录，不预设所有图必然失败或成功。

## 可视化与复现

`seed42_full.png` / `seed42_subject.png` 展示原图与八项主对照。
`stability_subject.png` 每行一个seed，三列为原图／无属性网络／M0R0。
`neutral_upper_subject.png` / `neutral_lid_and_lock.png` 用固定坐标放大原图与
none、joint0、M0、R0，检查背板图案、字样、门缝和锁。所有图都标清条件，
不将裁剪或RGB差异解释为Albedo真值。

```bash
export PYTHONDONTWRITEBYTECODE=1 TMPDIR=/var/tmp
export HF_HOME=/var/tmp/colorpeel-r12user5/assets/marble_c_v1/hf_home
export COLORPEEL_OUTPUT_ROOT=/var/tmp/colorpeel-r12user5/runs
/var/tmp/colorpeel-r12user5/assets/marble_c_v1/env/bin/python -B -u \
  -m experiments.subject_marble_diagnostics_v1.run \
  --assets /var/tmp/colorpeel-r12user5/assets/marble_c_v1 \
  --output "$FRESH_IDENTITY_RUN" --gpu 2 --stage mailbox_identity
```

新目录还保存source、原始RGB、manifest、逐项seed/控制、模型与代码来源及repro_outputs。
已有模型／原图／Mask在生成前后校验，不读取旧Depth缓存，不扩30张或启动训练。
