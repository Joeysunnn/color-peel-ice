# RGB vs 官方灰度初始化：Identity 结果（2026-10-25）

8张完成。单seed42的RGB初始化＋strength0.25是本轮最接近原图可见内容的候选：
保留原始橙色外观，在当前审阅尺度未见其他档出现的明显新增字样和装饰，部件布局更接近输入。
这不是稳定性、Albedo保持或M/R可编辑性验证；本轮未扩数据、未训练。

## 实际对照

| strength | 实际步数 | 官方灰度初始化 | 原始RGB初始化 |
|---|---:|---|---|
| 1.00 | 30 | 新增草写字样、蓝白纹样及彩色端面装饰。 | 视觉上同类改写仍明显，不能靠替换RGB初始化解决。 |
| 0.75 | 22 | 背板／盖板新增蓝色伪文字。 | 原橙色外观更接近输入，但盖板仍有蓝色伪字，背板边框新增铆钉／装饰。 |
| 0.50 | 15 | 盖板仍有伪字，边框有额外凸起。 | 盖板仍有蓝色伪字和表面标记，边框也未完全保持原貌。 |
| 0.25 | 7 | 未见上述明显新增字样，接近官方灰度初始化。 | 未见上述明显新增字样／花纹；原有边框、盖板平行压纹、投递口和锁的布局更接近原图，原橙色RGB外观也保留得更好。 |

0.25仍有细节重建差异，不称逐像素恒等或完美重建。RGB外观更接近原图是直接可见的现象，
不是Albedo真值测量。RGB像素、高光、颜色差异不直接判定Color／Albedo泄漏。
没有独立几何真值，也不把可见形态差异直接解释成真实3D几何变化。

## 有边界的解释

原始RGB分支只在adapter入口替换`image`，Reference仍为同一原图；灰度组保持官方流程。
低strength能够保留更多初始化信息，因此0.25不再被迫保留灰度主体。
0.75和0.5虽保住橙色，仍新增内容，说明仅替换初始化并不足以在这些档位满足身份保持要求。

strength1两种初始化的输出视觉上非常接近，但raw文件不逐字节一致，未预设一致。
现场4通道管线在strength1以噪声开始，Mask外仍混合图像latents；两种初始化经VAE的空间影响
允许采样路径有差异。本轮没有进行模块级归因，不能把这点差异直接定为额外参数变动。

本轮只支持这一输入、seed42下的无属性重建比较。未验证多seed稳定性，也未验证低strength
是否仍有足够的M/R编辑能力；不能将无编辑保持较好直接等同于反事实训练数据已经合格。

## 执行与完整性

- 分支：`codex/mailbox-marble-rgb-init-v1`；生成提交：`a4adeb091a813334449ac0fed45d23e6421f1728`。
- 新服务器checkout：`/var/tmp/colorpeel-r12user5/checkouts/mailbox-marble-rgb-init-v1`。
- 新运行目录：`/var/tmp/colorpeel-r12user5/runs/subject_marble_rgb_init_v1/20261025-150952__init8__a4adeb0__42`。
- 严格本地修改提交推送，服务器独立Git clone后运行；旧运行目录均只读。
- seed42、空MLP字典、原图＝1024自身Reference、同Mask、同1024官方DPT，Euler、名义30steps、
  CFG5、ControlNet1、IPAdapter1及官方正负提示均不变。每张独立生成，无背景后合成。
- 8/8张1024 RGB及8项输入条件SHA核验通过。原有7项条件与上一轮完全一致；新增RGB初始化
  与target和self Reference的PNG哈希相同。每行manifest保存实际初始化文件及哈希。
- 灰度4/4张raw与上一轮strength4逐张字节一致；两组实际步数均为30/22/15/7。
- runtime除两组strength列表外与前轮一致。官方源码、权重和冻结输入在生成前后校验通过。
- 本地和服务器8项CPU回归通过；AST执行官方真实caller与prepare_init_image，验证仅初始化
  和strength改变。独立代码审查及图像审阅完成。
- 四列图初版页脚过长，随后只缩短页脚并重绘本轮contact sheets；未重生成或改动raw、
  source、manifest、runtime及原始运行状态。旧实验contact sheets未修改。
- 自动状态保留`generated_pending_visual_review`，本文件记录人工结论；8张均未批准用于训练。

## 展示与复现

- `rgb_init_comparison_subject.png`：主体固定裁剪，上排gray、下排RGB，列为1/.75/.5/.25。
- `rgb_init_full.png`：相同顺序的全图。
- `gray_init_subject.png`、`rgb_init_subject.png`：各自显示原图、实际初始化及四档结果。
- `source/`、`raw/`、manifest、runtime、status、`repro_outputs/`保存实际输入和参数。
- stdout位于同级`20261025-150952__init8__a4adeb0__42.launch.log`。

命令和协议见[RGB_INIT_TEST.md](RGB_INIT_TEST.md)。停在结果展示，后续由用户决定。
