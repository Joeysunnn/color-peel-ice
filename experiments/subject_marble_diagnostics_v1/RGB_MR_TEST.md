# RGB初始化、strength0.25：M/R完整九格

用户授权在上一轮最佳可见身份候选上验证M/R控制，先展示，不扩30张、不训练。
独立分支`codex/mailbox-marble-rgb-mr-v1`和独立服务器checkout，输出根目录
`/var/tmp/colorpeel-r12user5/runs/subject_marble_rgb_mr_v1`；历史运行只读。

## 固定条件与矩阵

原1024 RGB同时用作实际初始化和自身Reference；保留已有Mask，从相同1024原图重新提取官方DPT。
同官方模型、权重、Euler、seed42、名义30steps、CFG5、ControlNet1、IPAdapter1、正负提示。
strength固定0.25；实际去噪预计7步，每张结束后读取`pipe.num_timesteps`记录。
没有连续编辑、参考替换或背景后合成，每张独立从原图生成。

| M相对控制／R相对控制 | -1 | 0 | +1 |
|---|---|---|---|
| -20 | M-20/R-1 | M-20/R0 | M-20/R+1 |
| 0 | M0/R-1 | M0/R0 | M0/R+1 |
| +20 | M+20/R-1 | M+20/R0 | M+20/R+1 |

每格均调用M和R两个官方MLP，包括零值。中心格不是之前空MLP的Identity基线，
因为零输入仍可能产生非零特征偏移。控制是相对学习方向，不是精确PBR数值；
沿用原协议的M符号解释（负方向对应更金属），但实际图像仍需审阅，不预设成功。
复用官方九格顺序与现有StrengthOverride，只在caller边界传RGB初始化和strength0.25。
官方源码、网络、权重不改。原有stage行为不变。

## 核验与判断

- 九格控制字典、顺序、seed、初始化文件及SHA、输出SHA、实际步数逐项保存。
- 八项输入条件应与上一轮RGB-vs-gray实验相同；RGB初始化、target、Reference须相同。
- AST回归执行真实官方caller，验证九格的两个MLP包括零值均传入，且RGB/.25/30/seed42正确。
- 先看同M行的R变化：高光宽窄、清晰度与反射平滑趋势；再看同R列的M反射差异。
  仅明暗变化不直接证明Metallic控制有效，RGB像素差异也不构成物理属性真值测量。
- 单列新增文字／纹样及背板边框、压纹、投递口、锁的可见保持程度；与原图和上一轮
  RGB/.25空MLP基线对照，不能把中心0/0当作无编辑原图。
- RGB颜色与高光变化不直接判为Color／Albedo泄漏。无独立Albedo、几何、光照真值，
  不声称已验证Albedo或精确BRDF；单seed不保证稳定性。

`rgb_mr_full.png`及`rgb_mr_subject.png`为完整九格和固定主体裁剪；顶部展示真实原图、
实际RGB初始化／自身Reference、Mask、Depth。保留原生1024raw及全部参数和repro_outputs。

```bash
export PYTHONDONTWRITEBYTECODE=1 TMPDIR=/var/tmp
export HF_HOME=/var/tmp/colorpeel-r12user5/assets/marble_c_v1/hf_home
export COLORPEEL_OUTPUT_ROOT=/var/tmp/colorpeel-r12user5/runs
/var/tmp/colorpeel-r12user5/assets/marble_c_v1/env/bin/python -B -u \
  -m experiments.subject_marble_diagnostics_v1.run \
  --assets /var/tmp/colorpeel-r12user5/assets/marble_c_v1 \
  --output "$FRESH_RGB_MR_RUN" --gpu 2 --stage mailbox_rgb_mr
```

本地提交推远端后，服务器Git获取运行；旧checkpoint、配置、图片及报告均不覆盖。
所有输出仍标记`training_eligible=false`，结果由用户审阅后再决定下一步。
