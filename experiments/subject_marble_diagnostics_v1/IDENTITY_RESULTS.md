# Mailbox Identity Reconstruction Test（2026-10-25）

12张诊断图已生成并逐图审阅。当前配置在三个seed的无属性基线中均新增了原图没有的
纹样或字样，未通过这次有限的可见身份保持测试。停止在诊断阶段，不扩展反事实数据，
不启动训练。此前官方样例的用户验收不受本结论影响。

本结论不以RGB变色或像素差异为依据。Metallic和Roughness本身可以改变高光、反射和
RGB外观；本轮没有独立Albedo真值，不能判断Albedo是否保持，也不将RGB差异称为Color泄漏。

## 运行与协议

- 分支：`codex/mailbox-marble-identity-v1`。
- 生成提交：`59b9c398dea3eecf8eb674537ed213145c5a40de`。
- 新服务器checkout：`/var/tmp/colorpeel-r12user5/checkouts/mailbox-marble-identity-v1`。
- 新结果：`/var/tmp/colorpeel-r12user5/runs/subject_marble_identity_v1/20261025-134217__identity12__59b9c39__42_43_44`。
- 严格本地修改、提交、推远端，服务器Git获取后运行。旧运行目录未写入。

原始512 mailbox以BICUBIC适配1024；使用官方`texture_image=None`回退，将同一个1024
图像作为自身Reference。沿用已审阅的原有mailbox Mask，按官方RGB BICUBIC规则适配，
从1024原图重新运行官方DPT，不读取旧Depth缓存。每项独立从同一个输入生成，
保存原生1024 RGB，不进行背景后合成。

沿用已通过官方样例的SDXL、官方Depth ControlNet、IPAdapter、M/R网络及材质块。
Euler、30steps、CFG5、strength1、ControlNet1、IPAdapter1，官方通用正负提示。

| seed | 对照 | 数量 |
|---|---|---:|
| 42 | 空属性字典；M0/R0；M-only[-20,0,+20]；R-only[-1,0,+1] | 8 |
| 43、44 | 各自空属性字典及M0/R0 | 4 |

这里的M/R是官方相对控制值，不是精确PBR参数。空字典完全不调用属性MLP，但仍然是
strength1的参考引导扩散重生成，不是原图直通，也不是数学上的恒等重建。
因此本轮检验的是该配置是否适合身份固定的反事实数据生成，而非宣称官方支持精确反演。

## 直接观察

下表记录每个seed的两张中性基线；每行内无属性网络与M0/R0视觉上非常接近。

| seed | 无属性网络与M0/R0共同出现的内容保持问题 |
|---|---|
| 42 | 背板新增草写字样，箱盖出现蓝白分支状纹样，端面出现花卉／彩色点状装饰。 |
| 43 | 端面新增花卉，箱盖新增字样，侧面出现大号金色字符，背板出现圆形徽饰／螺钉状细节。 |
| 44 | 背板出现砖格及星形装饰，箱盖出现花枝／字样，端面和侧面出现密集纹样。 |

原图较素的表面被改写，原投递口、锁等部件的可见形态也未忠实保留。大体轮廓仍可辨认，
但没有3D几何真值，不能据此断言真实几何发生改变。
seed42的单属性控制也存在新增字样／纹样；部分控制可见反射或高光平滑趋势，
不足以消除无属性基线已经存在的内容保持问题。

这些证据支持“当前配置未保持原有可见纹理图案、文字和部件细节”；
不支持“RGB变化证明Albedo或Color泄漏”。也无法由这些RGB独立确定新图案来自哪个内在属性层。
三个seed仅为有限检查，不构成对所有MARBLE输入或配置的统计结论。

## 零控制的实际含义

以本次自身Reference实际计算CLIP特征偏移L2范数：

| 项目 | L2 |
|---|---:|
| Reference embedding | 20.27240372 |
| M0 offset | 0.47445843 |
| R0 offset | 0.16747597 |
| 联合M0/R0 offset | 0.50176120 |

零输入经过有偏置的MLP，实际偏移并非零。这些是特征范数，不是物理属性测量。
但空属性字典完全不增加该偏移时也出现相同类型的改写，所以不能把本轮失败归因于零控制偏移。

## 核验与产物

- 12/12张raw文件SHA、7项输入条件SHA、控制字典与seed记录均匹配。
- Target和Reference的1024 PNG完全一致；冻结512 Mask原件哈希与原先资产一致。
- 新1024 Depth与上一轮相同1024输入的DPT输出一致，未使用旧512缓存。
- seed42联合M0/R0与上一轮自身Reference九格的对应raw文件逐字节一致，确认该路径未改变。
- 六项CPU回归在本地与服务器通过；官方源码和模型哈希在运行前后校验。
- 身份评估分支不使用旧HSV色差标记；RGB MAE仅作外观诊断，不用于Albedo或身份判定。
- 自动生成状态`generated_pending_visual_review`保留为原始运行记录；本文件补充人工审阅结论。
  所有12张均未获训练批准，未扩30张，未启动训练。

结果目录内：

| 文件 | 内容 |
|---|---|
| `stability_subject.png` | 顶部实际输入条件；三行seed42/43/44，三列原图／无属性网络／M0R0。 |
| `seed42_subject.png`、`seed42_full.png` | seed42原图及八项主对照，分别为固定主体裁剪与全图。 |
| `neutral_upper_subject.png`、`neutral_lid_and_lock.png` | 原图、none、joint0、M0、R0的固定区域放大。 |
| `zero_control_offsets.json` | 实测零控制CLIP特征偏移。 |
| `source/`、`raw/`、`manifest.jsonl`、`repro_outputs/` | 实际条件、原始图、逐项参数及代码／模型来源。 |

复现命令、测试边界与评估原则见[IDENTITY_TEST.md](IDENTITY_TEST.md)。
下一步应先解决中性重生成下的内容保持，再讨论扩大反事实数据；本轮未实施新配置扫描。
