# Mailbox：五色 × MARBLE M/R，30＋5张候选

按用户选定的历史第二组`mailbox_self`展开。当前阶段生成展示，不启动Subject训练。
保留官方灰度前景初始化和strength1，不能混用后来的RGB初始化／strength0.25实验。

## 固定设置

- 五色：pink、green、cyan、blue、purple；原资产的red／magenta仅改标签，不重跑RGB-LCh算法。
- 每色六张：M=[-20,+20]，R=[-1,0,+1]。负M更金属，正M更非金属；数值是相对CLIP编辑强度，非PBR参数。
- Target为各色原512图BICUBIC到1024；官方`texture_image=None`回退同一1024 Target自身作为Reference。
- 严格复用历史mailbox Mask，不使用rembg或原五色目录里另一版本Mask。两者实测有830像素差异，源RGB保持原样。
- 每色从1024 Target新提官方DPT，该色六张复用；不串联编辑，不从上一张生成结果提Depth。
- Euler、seed42、30steps、CFG5、strength1、Depth ControlNet1、IPAdapter1、官方通用正负提示及材质块均不变。
- 复用已有独立MARBLE环境／权重，只读校验；不改Token Learning、LoRA或旧实验。

程序核对历史`mailbox_self`的Mask1024哈希、全部采样设置、scheduler参数及库版本。
30张原始1024输出完整保存，不回贴主体或背景，不对生成图再次LCh调色。

## 35张未来训练候选

`dataset/`包括30张raw输出LANCZOS到512的候选，加5张逐字节复制的原改色图。
五色均来自原orange图，额外五张中不另加orange。
固定512二值历史Mask独立复制；每条concept各有一张图片与一个Caption，避免同目录随机匹配错Caption。

生成图Caption注明请求的Color／Metallic／Roughness，原图Caption沿用已有glossy painted metal描述。
它们是拟议监督词，尚不等于实际外观或物理参数验证。所有候选待用户审阅；代码没有训练开关。
此前该配置存在内容改写，本轮按用户明确选择完整生成30张，不自动过滤、替换或掩盖失败结果。

## 本地提交、远端推送、服务器拉取

在新的干净checkout执行；MARBLE资产不重新安装。以下`NEW_RUN`必须是新的外部目录：

```bash
export PYTHONDONTWRITEBYTECODE=1 TMPDIR=/var/tmp
export COLORPEEL_OUTPUT_ROOT=/var/tmp/colorpeel-r12user5/runs
MPY=/var/tmp/colorpeel-r12user5/assets/marble_c_v1/env/bin/python
$MPY -B -u -m experiments.subject_marble_color30_v1.run \
  --assets /var/tmp/colorpeel-r12user5/assets/marble_c_v1 \
  --output "$NEW_RUN" --gpu 2
```

中断后在代码、配置、原图、条件和模型哈希不变时加`--resume`；已完成图片校验后跳过。
失败生成的孤立PNG保留，新尝试写不同attempt编号。Depth／初始化若在保存收据前中断，
保留该目录，改用新目录，避免猜测条件内容或覆盖旧文件。

## 输出

- `source/{color}/`：原改色图、1024 Target／自身Reference、新Depth、实际灰度初始化。
- `raw/{id}/attemptNNN.png`：30张原生1024输出。
- `dataset/manifest.json`：30＋5张实际候选、哈希、Caption与待审阅状态。
- `dataset/images/{id}/image.png`、`masks/{id}/image.png`、`concepts.json`：35条未来训练输入，未执行训练。
- `review/overview35.png`：五行七列，展示实际35张512候选，首列是另外加入的五张原改色图。
- `review/six_{color}_{full,subject}.png`：每色六格原生输出／同一主体裁剪；顶部标实际原图、Reference、Mask和Depth。
- `review/index.html`、`captions.csv`：完整图集与35条Caption。
- `generation_manifest.json`、`conditions_manifest.json`、`runtime.json`、`repro_outputs/`：来源、参数、质量诊断与复现收据。

RGB差异仅作外观描述，不能直接判定Albedo泄漏；反光与高光允许变化。新增文字、原纹样丢失、结构变化需逐图审阅。

CPU检查：`python -B -m unittest discover -s tests -p 'test_marble_color30_*.py' -v`。

## 用户批准后的35图Subject训练

生成阶段与训练阶段分开；用户“生成完成了，开始训练”批准本轮35张数据用于Subject学习。
`stage_training.py`将30张MARBLE输出和5张原改色图逐字节复制到新的训练目录，保留原候选目录的待审标记与所有文件。
沿用已保存Caption：生成图标请求的颜色／相对M/R外观，额外原图标glossy painted metal；这些词不代表真实BRDF测量。

`training.py`只包装Experiment B既有训练器：SD1.5、`<S*>`、`mailbox`初始化、seed42、1000更新，
original fullrank和rank64 alpha64各一项，保存500／750／1000。学习率、优化器、精度、Mask与损失均沿用B/C。
DataLoader仍按原有固定顺序循环：1000步=28个完整35图周期＋前20张，逐图28或29次；不增加shuffle或采样权重。

```bash
export PYTHONDONTWRITEBYTECODE=1 TMPDIR=/var/tmp
export COLORPEEL_OUTPUT_ROOT=/var/tmp/colorpeel-r12user5/runs
PYTHON=/home/r12user5/miniforge3/envs/colorpeel017/bin/python
$PYTHON -B -u -m experiments.subject_marble_color30_v1.train_pipeline \
  --candidates /var/tmp/colorpeel-r12user5/runs/subject_marble_color30_v1/20261025-224336__color30_plus5__5a75bce__42/dataset \
  --output "$NEW_TRAIN_RUN" --gpu 2 --execute
```

省略`--execute`仅构建独立训练数据和两种配置，不运行模型。训练执行使用新的干净服务器checkout，代码由本地提交推送后拉取。
串行完成original、rank64及既有训练审计后，自动生成`step500/`、`step750/`、`step1000/`。
五种提示仍为metallic、matte、glossy plastic、ceramic、wood；seeds42／43、PNDM100、CFG3.5、512px、fp16及safety checker不变。
每step20张结果，每份contact sheet显示全部35张实际训练图片、ID及完整Caption。
精确命令、配置差异、来源哈希和运行状态保存在新目录`train_outputs/`及`training_plan/`；不修改任何旧结果。
