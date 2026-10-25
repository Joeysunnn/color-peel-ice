# 原始RGB vs 官方灰度初始化：Identity 对照

用户授权只增加初始化方式这一因素；沿用前一轮四档strength，不扩展数据、不训练。
新分支`codex/mailbox-marble-rgb-init-v1`，新checkout和结果根目录
`/var/tmp/colorpeel-r12user5/runs/subject_marble_rgb_init_v1`，所有旧运行只读。

## 八项最小矩阵

| 初始化 | strength | seed | 属性控制 |
|---|---|---|---|
| 官方prepare_init_image（Mask内灰度、外部原RGB） | 1/.75/.5/.25 | 42 | 空MLP字典 |
| 原始RGB（同一对齐后的1024 target） | 1/.75/.5/.25 | 42 | 空MLP字典 |

原图、1024自身Reference、冻结Mask、重新提取的1024 DPT、官方模型权重、Euler、名义30steps、
CFG5、ControlNet1、IPAdapter1、正负提示与前轮相同。每张独立生成，未改变Reference、
未连续编辑、未进行背景后合成。实际步数预计30/22/15/7，生成后只读管线记录值。

保留官方源码和caller不变，扩展原有`StrengthOverride`：gray继续原样传入官方初始化；
RGB分支只在adapter边界替换`image=target.copy()`。这是本实验授权的初始化适配，
不声称它是官方默认。`texture_image=None`仍回退到原target，参考图不随初始化变灰。

保存`source/initialization.png`（官方gray）和`source/initialization_rgb.png`，每张manifest
明确标记初始化、实际文件及SHA256。RGB初始化须与target一致；原有七项输入哈希须复现前轮。
AST回归执行真实官方caller与真实prepare_init_image，证明只替换初始化与strength。

灰度四张须与前轮strength四档逐张比对哈希；RGB与gray在strength1不预设字节一致。
现场管线为4通道UNet，strength1虽从噪声初始化，但Mask外仍混合image latents，VAE具有
空间感受野，因此两种图像初始化并不意味着整个采样路径相同。

## 评估与展示

`rgb_init_comparison_subject.png`及`rgb_init_full.png`：上排gray，下排RGB，列为1/.75/.5/.25；
顶部列出实际输入RGB、gray初始化、Mask、Depth。
另保存`gray_init_subject.png`与`rgb_init_subject.png`，分别展示原图、实际初始化、四项输出。

判断新增纹样／字样、原有可见细节、部件布局；RGB像素差异、高光或颜色变化不直接判为
Color／Albedo泄漏。无独立Albedo／几何／光照真值，Albedo保持仍未验证。单seed不构成
稳定性保证，本轮也不评价材质控制能力或训练收益。所有输出默认不进入训练。

```bash
export PYTHONDONTWRITEBYTECODE=1 TMPDIR=/var/tmp
export HF_HOME=/var/tmp/colorpeel-r12user5/assets/marble_c_v1/hf_home
export COLORPEEL_OUTPUT_ROOT=/var/tmp/colorpeel-r12user5/runs
/var/tmp/colorpeel-r12user5/assets/marble_c_v1/env/bin/python -B -u \
  -m experiments.subject_marble_diagnostics_v1.run \
  --assets /var/tmp/colorpeel-r12user5/assets/marble_c_v1 \
  --output "$FRESH_RGB_INIT_RUN" --gpu 2 --stage mailbox_rgb_init
```

严格本地提交推送、服务器Git获取后运行，输出原生1024 RGB、manifest、runtime、status、
输入与哈希、`repro_outputs/`。先展示结果，由用户决定后续工作。
