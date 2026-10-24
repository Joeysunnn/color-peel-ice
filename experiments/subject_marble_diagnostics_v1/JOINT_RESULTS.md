# Mailbox 两组联合九格结果（2026-10-24）

两组各9张已完成，共18张原生1024 RGB；先展示结果，由用户决定是否继续。
未扩展30张训练数据，未启动训练。此前官方toy-car九格已由用户确认符合要求，
本轮复用其模型、权重、环境及采样配置，只执行用户授权的两组mailbox设置。

## 运行位置与代码

分支：`codex/mailbox-marble-joint-v1`。
执行提交：`9cd67e7f19206f2776bf0dff137b5d86a3563371`。
新服务器checkout：`/var/tmp/colorpeel-r12user5/checkouts/mailbox-marble-joint-v1`。
严格本地修改、提交并推远端，服务器Git克隆后运行；旧checkout未切换或修改。

输出根目录：`/var/tmp/colorpeel-r12user5/runs/subject_marble_diagnostics_v1`。

| 组别 | 新运行子目录 | 设置 |
|---|---|---|
| 第一组 | `20261024-150849__mailbox_bowl_grid__9cd67e7__42` | 官方metal_bowl参考、官方rembg自动Mask |
| 第二组 | `20261024-150849__mailbox_self_grid__9cd67e7__42` | 1024 mailbox自身参考、原有mailbox Mask |

两组均从同一个对齐后的1024原图重新运行官方DPT；没有读取旧512输入的Depth缓存。
为节省重复计算，各自新Depth仅在该组九次生成内复用。第一组保留官方Mask原样，
第二组通过官方 `texture_image=None` 回退到1024 target自身，不使用旧512参考。

网格行顺序为M=[-20,0,+20]，列顺序R=[-1,0,+1]。每次同时调用M/R两个网络，包括零值，
每张从原图独立编辑。M/R为相对控制强度，不是物理参数标定值。

## 已核验条件

- 下载的18/18张raw SHA256与生成manifest一致，均为1024×1024 RGB；各自source文件哈希均与各自条件记录匹配。
- 两组Target字节一致：`598e39369e341b72c022628a53b0a5c2bf207fda39c16b4cde9db074b937f791`。
- 两组新Depth字节一致：`6e921a6557e482accb8bc4631d733eb627dca61dce25479f2107e382057acc04`。
  它不同于此前512输入缓存的`00c5d04ec3411073279135e66911d3ae19a740725101a682eb6db3f5c4c9964f`。
- 第二组reference与1024 target的PNG哈希完全一致。
- 第一组metal_bowl保留官方696×696 RGBA，保存副本与官方输入逐像素一致。
- 第一组实际Mask前景占39.4570%；第二组为16.3845%，与原有mailbox Mask对齐后相同。
- 两组runtime与已通过官方例子一致：SDXL、Euler、seed42、30steps、CFG5、strength1、
  ControlNet1、IPAdapter1、官方材质块及通用正负提示。Scheduler的默认字段枚举顺序
  不影响实际参数；归一化这一顺序后配置一致。
- 本地与服务器4项CPU回归全部通过。官方源码、模型及冻结输入在生成前后校验。

## 直接看图观察

第一组的官方自动Mask同时选中了mailbox和上方135标牌，故标牌也被编辑。
这不是把生成越过Mask当作背景改变；实际自动Mask在contact sheet顶部完整展示，
没有为了只编辑mailbox而手工替换它。另存的 `review_mask.png` 仅定义两组一致的
mailbox放大区域，不参与第一组生成。

第一组在M=0/+20的部分档位可见银色反射及R增大时高光变宽、变柔的趋势；
M=-20行出现蓝金区域、新颗粒和结构改写，三档R差异较小。Mailbox轮廓大体保留，
投递口、锁扣等细节有所简化或丢失。低Metallic仍可能偏银灰，不能将它视为物理M=0。
独立metal_bowl参考本来就执行材质迁移，不将原橙色改变本身作为这组违背官方流程的证据。

第二组只将已有mailbox Mask作为编辑区域，门牌未被纳入编辑。高Metallic出现蓝金镜面，
M=0/+20出现原图没有的白蓝纹理、文字和装饰；部分R变化伴随反射变柔，但纹理、
字样也一起变化。它显示了属性响应，同时仍存在颜色和内容耦合。

两组同时改变reference和Mask，因此组间差异不能单独归因于reference。
当前只有单图、单seed；不将这些观察推广成MARBLE整体有效或无效的结论。
所有样本仍保留 `training_eligible=false`，没有创建训练概念配置或训练集。

## 查看与复现

每组目录包含：

- `contact_sheet.png`：完整九格全画面；顶部展示实际Target、reference、generation Mask、Depth。
- `subject_crop.png`：同一固定mailbox区域的九格放大视图，不改变raw。
- `source/initialization.png`、`source/conditions.json`：实际初始化与输入条件。
- `raw/`、`manifest.jsonl`、`plan.json`、`runtime.json`、`status.json`、`repro_outputs/`：
  原始输出、文件哈希、代码来源、采样参数和执行状态。

完整命令见 [JOINT_GRIDS.md](JOINT_GRIDS.md)，每次复现均需使用新输出目录。
通用 `repro_outputs/PATCHES.md` 中部分描述沿用旧诊断模板；本轮实际设置以
本报告及各组 `plan.json` / `source/conditions.json` 为准，两组均不存在旧尺寸生成项。

最初推送遇到连接重置，未启动生成；网络恢复后推送成功，再由服务器克隆运行。
没有通过SCP上传代码。SCP仅将新结果下载用于本地核验展示，旧结果未写入或覆盖。
