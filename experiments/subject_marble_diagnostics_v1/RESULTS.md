# MARBLE 官方样例与 Mailbox 诊断结果（2026-10-24）

结论：完成官方样例9张、mailbox诊断18张；未扩展30张训练集，未启动训练。
尺寸适配已修复，但不足以解决当前内容漂移。仅移除材质参考背景后，新增字样和
装饰明显减少；颜色、纹理与独立R控制仍不符合此前反事实训练数据的要求。
全部图片保持 `training_eligible=false`，本轮止于结果展示与人工决策。

## 实际运行与协议

分支：`codex/marble-diagnostics-v1`。本地修改、提交并推送后，服务器Git拉取运行。
使用全新checkout：`/var/tmp/colorpeel-r12user5/checkouts/marble-diagnostics-v1`。

输出根目录：`/var/tmp/colorpeel-r12user5/runs/subject_marble_diagnostics_v1`。

| 阶段 | 不可覆盖的运行子目录 | 执行代码 |
|---|---|---|
| 官方9格 | `20261024-121907__official_grid__95f1ecd__42` | `95f1ecdd05c379ecd5cb0e841926e580c458b82b` |
| Mailbox 16+2 | `20261024-122956__mailbox_controls__323ef06__42` | `323ef065abbb238a32fbde43ec279052848345a7` |

官方源码固定为 `0139f2563057e8ac7d8e82055ea891ffd7474bd5`；官方源码未修改。
两阶段均为SDXL、Euler、seed42、30steps、CFG5、strength1、ControlNet1、IPAdapter1，
仅向官方材质块注入，使用相同通用正/负提示。没有加载Subject token、K/V或LoRA。
保留原生1024输出，未做背景恢复。全部raw下载后按manifest验证：官方9/9，mailbox18/18。

## 官方样例：部分趋势复现，存在明显质量限制

按照固定官方 `parametric_control.ipynb`，toy_car为目标，metal_bowl为独立材质参考；
M按[-20,0,+20]排列，R按[-1,0,+1]排列。Mask沿用官方rembg操作，Depth由官方DPT
方法提取，不使用仓库预存depth；仅将相同条件缓存一次以避免重复推理。

M=0和M=+20行能看见反光及哑光变化，但M=-20行出现额外金色装饰和颗粒，R响应较弱。
高R/低Metallic组合也丢失轮毂、车牌等细节。只能称为“官方样例在当前环境已完成运行，
复现部分属性趋势”，不能称已复现论文全部图像质量。独立金属碗参考允许改变原车蓝色，
所以这里不将蓝色变化本身判为固定Albedo失败。

明确的环境差异：Python3.11、Torch2.5.1+cu124、CPU offload；官方README测试Python3.9.7，
requirements指定Torch2.3。Diffusers0.30等核心推理库遵照官方版本。不能排除环境差异，
也没有证据将伪影归因于Torch版本。Notebook无已保存输出PNG，无法做官方像素级对照。

rembg默认U2Net文件本次SHA256：
`8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491`。

## 尺寸适配：问题确认，修复不足以消除漂移

旧调用先将Mask放大1024，却仍用512 target执行ImageChops；实际Mask裁取区全黑，
前景灰度化未发生。新调用在ImageChops前统一target/Mask为1024，保留官方RGB
BICUBIC插值。前景灰度化、Mask外像素与输入不变性均通过CPU回归检查。

为避免同时改变CLIP输入，full reference显式保留原512照片；Depth使用旧六图实验
的同一缓存，未重新估计。额外生成两项旧512错误对照，与新1024调用仅改变target尺寸。

| 配对（其余条件固定） | 新旧尺寸前景MAE，0–255 | 看图结论 |
|---|---:|---|
| 无属性MLP | 5.9678 | 两者都有新增文字、紫棕圆筒及背板饰纹 |
| 仅M=-20 | 6.2200 | 两者都有蓝金配色、强反光和新纹理 |

修复确实影响输出，不能说完全无效；但不是这些失真的充分解释。两项legacy是本轮新生成的
错误尺寸对照，不是旧六张联合M/R条件的直接复跑。MAE只量化图像差异，不代表质量提升。

## Mailbox属性与材质参考对照

两种reference：原512全图；同512画布、Mask外替换RGB127的图。主体像素、尺度不变，
不裁剪、不放大、不归一化颜色。目标场景、Mask、Depth和采样参数全部固定。

每种reference的8项：空属性字典、M0+R0、M-only[-20,0,+20]、R-only[-1,0,+1]。
空字典仍执行参考引导的扩散重生成，不能理解成原图恒等复制；零控制仍运行带bias的MLP。

直接观察：

- full reference、空属性MLP时已经出现大号白字与背板书法状饰纹，圆筒偏紫棕。
  因此漂移在属性偏移之前就存在，不能全部归因于M/R联合控制或控制正负号。
- masked-gray reference明显减少上述字样和饰纹，支持参考背景语义是本例泄漏的一项来源。
  无属性控制下，仅改变reference背景的前景MAE为22.6424/255。
- masked-gray仍有紫棕颜色和表面变化；M=-20仍变成蓝紫强反光，并产生背板金色颗粒，
  不能认为去背景已实现材质独立编辑。
- R-only三档在当前图中差异较弱，未验证稳定、清晰的三档高光控制。
- 空MLP与M0/R0结果也非像素相同：full前景MAE3.2348，masked-gray为0.6376。

Raw背景相对输入MAE：官方3.9508–4.1268/255，mailbox3.5221–3.8041/255，
不能声称背景像素完全不变。新图并未用背景复制掩盖这一点。

单图、单seed诊断不能推广为MARBLE整体能力结论。灰背景只改变CLIP材质参考；
target背景的蓝金标牌仍在，因此残留颜色也可能受context影响，尚未分离这些因素。

## 查看与复现

每阶段有 `plan.json`、`runtime.json`、`source/`、`raw/`、`manifest.jsonl`、
`status.json` 和 `repro_outputs/`。Mailbox另有：

- `contact_sheet_full.png`、`contact_sheet_masked_gray.png`：全部16张的全画面视图。
- `subject_crop_full.png`、`subject_crop_masked_gray.png`：同一固定坐标的主体放大图。
- `size_comparison.png`、`size_pair_metrics.json`：仅尺寸变化的两项成对对照。

`review.py`在新目录生成紧凑 `reference_comparison.png`、完整主体图与
`review_metrics.json`；它只读取既有RGB，不运行扩散或修改原始输出。
紧凑图摘选空MLP、M=-20、R=+1；完整8项对照保留，未隐藏其他输出。

按[README](README.md)中的命令重新运行必须选择新目录。CPU三项回归在本地和服务器均通过。
新review辅助脚本在本地对实际18张下载输出完成预检，正式review由服务器拉取后执行。

旧六张实验目录未写入；其6张raw及6张背景合成图的SHA256仍与此前保存manifest一致。
旧输入、Mask、Depth及模型在生成前后校验。未改现有B、PBR C、MARBLE旧实验或训练参数。
