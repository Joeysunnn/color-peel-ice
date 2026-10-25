# 无属性strength对照结果（2026-10-25）

四档已完成。单seed42下，降低至0.25明显减少新增纹样／文字，部件可见布局更接近输入；
1、0.75、0.5仍发生内容改写。0.25主要保留官方灰度初始化，不能宣称原始彩色图被稳定、
精确重建，也没有验证Albedo或材质控制能力。本轮停在展示结果，不扩展数据、不训练。

## 为什么空属性控制仍会新增内容

固定官方提交`0139f2563057e8ac7d8e82055ea891ffd7474bd5`，现场核对官方源码及实际Diffusers：

1. [官方初始化](https://github.com/Stability-AI/marble/blob/0139f2563057e8ac7d8e82055ea891ffd7474bd5/marble.py#L203)
   将Mask内主体转成保留亮度结构的灰度，Mask外保留RGB；不是直接把原始彩色主体送入重建。
2. [官方属性控制入口](https://github.com/Stability-AI/marble/blob/0139f2563057e8ac7d8e82055ea891ffd7474bd5/marble.py#L248)
   固定strength1。服务器实际管线`prepare_latents:919–933`在这一档使用噪声latents，
   Mask内仍需扩散再生成；4通道UNet管线每步的原图latents混合约束主要作用在Mask外。
3. [空属性字典的真实作用](https://github.com/Stability-AI/marble/blob/0139f2563057e8ac7d8e82055ea891ffd7474bd5/ip_adapter_instantstyle/ip_adapter.py#L223)
   是不加M/R网络偏移，参考条件和扩散仍然运行。参考图经过CLIP全局特征，投影成4个image
   prompt tokens，并非原图纹样、文字和部件的空间逐点对应。Depth提供估计深度，Mask定义
   编辑区域，二者都没有硬性约束原始文字与纹理。
4. 因此，默认路径允许模型依据参考语义和扩散先验重绘表面。具体新增文字是否源于上方135
   招牌、CLIP语义或SDXL先验，尚未单独消融证实，不能将其归因于某一个模块或来源。

第1–3项为代码事实，第4项为有边界的解释。降低strength的结果表明默认重生成幅度是本例
内容保持问题的重要影响因素，但不证明它是唯一原因；并未通过修改prompt或参考图来追求通过。

## 实际四档结果

所有图片同seed42、空M/R字典、同1024原图与自身Reference、同Mask、同Depth和同灰度初始化。
官方模型、Euler、名义30steps、CFG5、ControlNet1、IPAdapter1及正负提示完全不变。
仅strength为授权改变。官方源码、权重不改，通过薄代理替换caller中的硬编码参数。

| strength | 实际去噪步数 | 人工直接观察 |
|---|---:|---|
| 1.00 | 30 | 重现旧结果：新增草写字样、彩色／分支状纹样，投递口和锁的可见细节被改写。 |
| 0.75 | 22 | 轮廓更接近输入，但背板出现大号蓝色伪文字，盖板也有蓝色字样。 |
| 0.50 | 15 | 盖板仍有蓝色伪字／图案，背板模压边框出现额外凸起装饰。 |
| 0.25 | 7 | 当前可见尺度下未见上述明显新增文字／花纹；背板边框、盖板平行压纹、投递口与锁布局明显更接近输入及实际初始化。 |

名义采样steps保持30；实际步数变化是strength在官方管线中的固有效果，未增加名义步数作补偿。
0.25的细节仍不逐像素一致，仅一张单seed图，不能称稳定或完美重建。

0.25接近灰度来自保留官方灰度主体初始化，不直接判为Color泄漏。反射、高光和RGB像素差异
也不直接判定Color／Albedo泄漏。无独立Albedo、几何、光照真值，本轮Albedo保持仍未验证。
不能仅凭RGB断言3D几何变化；这里只评估可见轮廓和部件细节。

## 核验与复现

- 新分支：`codex/mailbox-marble-strength-v1`；生成提交：`0791447a403f67479a22499a6b998a5886ac9a6d`。
- 新服务器checkout：`/var/tmp/colorpeel-r12user5/checkouts/mailbox-marble-strength-v1`。
- 新运行目录：`/var/tmp/colorpeel-r12user5/runs/subject_marble_strength_v1/20261025-144911__strength4__0791447__42`。
- 严格本地修改、推远端，服务器独立Git clone后运行；原Identity运行和其他旧结果均只读。
- 4/4张raw哈希和7项输入条件哈希验证通过；Target、Reference、Mask、Depth、初始化及原Mask
  均与上一轮Identity测试完全一致。Depth在本轮从1024重新提取，输出哈希与上一轮一致。
- strength1与上一轮`seed42__none.png`逐字节一致：
  `d83f7b91d2241529ecc6d8dfd7de32ae8ac5d98f74ff71f204011476f6de9abb`。
- runtime除strength列表和本轮只使用seed42的seeds列表外，与上一轮一致；实际步数30/22/15/7。
- 本地及服务器7项CPU回归通过。AST测试执行官方caller，验证除strength外的所有调用参数、
  图像内容一致。生成后只读`pipe.num_timesteps`，没有额外scheduler重置。
- 生成前后模型／官方源码校验通过，生成期间项目Git未变化。4张均未批准加入训练集。

可视化：`strength_subject.png`、`strength_full.png`，按原图／实际初始化／1／.75／.5／.25顺序。
保存`source/`、`raw/`、manifest、runtime、status及`repro_outputs/`。
原始stdout在同级`20261025-144911__strength4__0791447__42.launch.log`。
自动状态`generated_pending_visual_review`保留为原始记录，本文件补充人工审阅。
执行命令和边界见[STRENGTH_TEST.md](STRENGTH_TEST.md)。

若继续排查，下一项应将“初始化是否使用原RGB”作为独立因素，而不能混在此次strength对照里。
本轮未实施RGB初始化替换、后合成、额外seed扫描、材质编辑扩展或训练。
