# 无属性控制：strength-only Identity 对照

用户授权降低strength，排查新增纹样／字样；不扩数据、不训练。
分支`codex/mailbox-marble-strength-v1`，新checkout及新结果目录；旧运行只读。

## 官方源码审计

固定官方提交`0139f2563057e8ac7d8e82055ea891ffd7474bd5`：

- `marble.py:203–218`：主体Mask内先转为原图亮度的灰度，Mask外保留RGB；并非常数灰填充。
- `marble.py:245–261`：自身Reference回退到原target，灰度初始化输入管线，strength硬编码1。
- `ip_adapter_instantstyle/ip_adapter.py:235–255`：空属性字典不增加M/R偏移，但仍使用参考图CLIP全局特征及投影tokens。
- 同文件`:505–598`：官方生成函数继续运行参考引导扩散，将strength传至Diffusers。

空MLP不等于关闭生成。现场Diffusers的`prepare_latents`在strength1时使用噪声；
较低strength时使用带噪初始化latents，实际执行步数由`get_timesteps`裁切。
Mask与估计Depth提供编辑区域和深度条件，参考图为全局特征，不构成文字和纹样的逐点硬约束。
这解释了默认路径为什么允许内容改写；具体新增文字是否来自背景招牌、CLIP语义或SDXL先验
尚未被独立消融证实，不能归因于某个单一来源。

## 四项最小对照

固定seed42、空属性字典、原1024目标＝自身Reference、已有Mask、新1024官方DPT、
相同灰度主体初始化、官方模型及提示、Euler、名义30steps、CFG5、ControlNet1、IPAdapter1。
仅strength取`1.0 / 0.75 / 0.5 / 0.25`；每张独立生成，不连续编辑，不做背景后合成。

不改官方源码或权重。`StrengthOverride`仅在官方caller与adapter间替换硬编码strength；
AST回归执行实际官方caller，验证传给adapter的其他参数和图像不变。
生成后只读`pipe.num_timesteps`记录实际步数，不额外调用scheduler改变状态。
Euler预期实际步数为30/22/15/7；这属于strength本身的影响，不提高名义steps补偿。
strength1应与上一轮seed42空属性raw逐字节一致，否则先解释复现差异。

降低strength更保留的是**灰度初始化**，不能自动解释为保持原始彩色RGB。
接近灰度、反射或高光变化不直接判为Color泄漏。Albedo缺乏独立真值，仍记未验证。
人工逐项检查新增纹样／文字、原有细节、投递口及锁的可见形态，不把RGB MAE当身份评分。

## 执行

严格本地提交→推远端→服务器Git克隆／拉取后运行：

```bash
export PYTHONDONTWRITEBYTECODE=1 TMPDIR=/var/tmp
export HF_HOME=/var/tmp/colorpeel-r12user5/assets/marble_c_v1/hf_home
export COLORPEEL_OUTPUT_ROOT=/var/tmp/colorpeel-r12user5/runs
/var/tmp/colorpeel-r12user5/assets/marble_c_v1/env/bin/python -B -u \
  -m experiments.subject_marble_diagnostics_v1.run \
  --assets /var/tmp/colorpeel-r12user5/assets/marble_c_v1 \
  --output "$FRESH_STRENGTH_RUN" --gpu 2 --stage mailbox_strength
```

输出包含原生1024raw、manifest中实际步数、source及哈希、runtime、status、repro_outputs。
`strength_subject.png`与`strength_full.png`展示原图、实际初始化及四项结果，顶部标明真实输入条件。
生成结束后只作结果展示；不以此次单seed诊断推断材质控制能力或训练收益。
