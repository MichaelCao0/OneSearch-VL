# AV observer 输出协议与音视频时间网格：执行结果

日期：2026-10-10。开发环境：coder ssh 3 `/home/jovyan/OneSearch-VL`。仓库：MichaelCao0/OneSearch-VL，分支 `research/av-deepresearch`。本次完成真实 H20Z GPU 诊断、音频 PTS 放置、视频取帧网格修正和记录核验；第一篇仍是多源 AV search、证据核验与研究报告。当前结果不构成完整search系统、论文benchmark或训练效果。

## 结果与固定分母

始终使用上一轮相同的六个自然AV输入，每个source-local `[0.5,2.5)`，每轮请求12秒。模型为 `Qwen/Qwen2.5-Omni-7B`，revision `ae9e1690543ffd5c0221dc27f79834d0294cba00`，环境由锁定的 `runtime/av-observer/pixi.toml` 管理。所有素材和binding保持原hash，未重写媒体。固定分母6与真实generation数分开记录。

| 阶段 | 实际generation | 接受结构/固定6 | Backend失败 | Parse问题 | Tokens记录 | Generation秒数 | Job保守秒数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| protocol-v1 |5|1/6|1|4|584|13.142|82|
| protocol-v2 |5|2/6|1|3|540|11.006|79|
| protocol-v3 |5|5/6|1|0|535|11.219|81|
| clock-v1 |3|3/6|3|0|338|7.273|73|
| avgrid-v1 |3|3/6|3|0|335|7.752|75|

protocol-v3 的条件结构通过率5/5，avgrid-v1为3/3。`failed_clip_count`只统计backend失败；parse abstention也属于没有可接受结构的输入。例如protocol-v1共有5/6没有可接受结构，不能只报告1个backend失败。新时钟策略拒绝真实音频重叠，原strict_origin只检查声明的原点，所以5/6与3/6不能作为同条件质量变化。

## 输出协议修正

上一轮五份raw有结构引号转义，全部被拒绝。本次提供literal JSON空events示例、完整字段和uncertainty规则，避免示例引入自然场景事件；protocol-v1中转义问题消失，但4/5 raw以 `}.` 结束。protocol-v2再明确“最后字符必须为 }”后仍有3/5尾随句点。

pinned tokenizer把 `}.` 编为单一token7810，因此常规stopping不能在同一token内切开。新增显式 `output_policy=json_object_v1`：仅在生成时抑制该compound token，并在全部新生成文本首次严格解码为JSON object时停止。raw生成后不去句点、不抽取对象、不去转义、不补字段，严格parser仍验收schema、重复key、非有限数、clip边界及coverage。`unconstrained`仍为默认。

这是一项窄生成控制，不是完整grammar/schema约束；全局抑制也会影响字符串里的相同token，必须在后续baseline中记录策略。protocol-v3的5份、clock-v1的3份和avgrid-v1的3份raw直接通过严格结构检查；不能据此宣称模型无约束协议能力或语义可靠性。avgrid新增保留完整生成token IDs、解码raw和hash；解析异常时已生成内容也可由私有cost sidecar保留。前三个协议阶段及clock-v1仅有当时token长度记录，不能独立恢复完整token流。

## 音频时钟与coverage

`audio_pts_adapter.py`读取实际decoded PTS和native resampler output PTS，把PCM放到共享clip轴。真实gap拆分并reset/flush resampler；不可证明的material overlap拒绝。容器tick及sample rounding内的连续性修正逐项留receipt。原文件和binding在处理前后重新核对hash。

01音轨在1.012秒开始，输出32000个16kHz样本。记录到的coverage是 `[1.012,1.9998125)`；前16192 samples及最后3 samples是缺失媒体的transport filler，不能作为录得静音或事件证据。接受的audio/joint事件必须全部位于coverage，判定在clip-local轴完成再映射source offset；没有2ms宽容。

02、06可在完整 `[0,2)` 上处理。03、04、05的实际音频frame存在约9–21ms跳变/重叠，明确 `AudioPTSAdapterError`，未执行generation，不把这三片段排除分母。默认 `strict_origin` 与更严格策略的支持范围不同。

## 视频时间网格修正

独立审查发现，官方helper端点linspace取帧与processor scalar2fps网格存在系统偏差。01旧取帧实际为 `[0,.64,1.32,1.96]`，02/06为 `[0,.667,1.3,1.967]`，processor却按 `[0,.5,1,1.5]` 处理。旧最大frame残差460/467ms、第二temporal patch首帧残差320/300ms。原始视频CFR证明不能替代模型输入时间网格证明。

新增显式 `clock_policy=av_pts_grid_v1`，以 `j/sample_fps` 为目标取最近真实decoded PTS，有理数比较、平局选早帧、禁止重复选择；仅支持已经证明的有界CFR、整数偶数frame count。官方pinned helper接收选定PIL RGB帧做resize，跳过端点取帧和奇数末帧复制。每份receipt保留全部PTS/timebase、indices、目标/实际时间、量化误差、tensor hash及helper源码身份。

| 输入 | 修正后实际PTS | 最大frame量化误差 | temporal patch首帧残差 |
|---|---|---:|---:|
| 01 |[0,.48,1,1.48]|20ms|0ms|
| 02 |[0,.5,1,1.5]|0ms|0ms|
| 06 |[0,.5,1,1.5]|0ms|0ms|

synthetic FFV1/PCM同步音频/视觉pulse从1.000秒开始，真实processor消费两类tensor，pulse保留在模型grid1秒位置。25fps半帧量化明确披露20ms误差/21ms上界；VFR、掉帧、奇数/非整数预算、重复帧、binding/hash改变均拒绝。该CPU检查未加载权重或做模型语义定位。

avgrid-v1随后在H20Z真实执行01、02、06三次generation，全部通过结构与audio coverage；输入tensor/array及源媒体hash留存。仍存在source-frame量化和两帧temporal pooling，不能称一般VFR支持或精确语义事件时间准确。第一份raw重复描述了两次相同jingle且没有视觉事件，显示schema通过仍可能语义冗余。保留原文，人工语义状态仍pending。

## 验证、成本与来源

- 本次最终完整CPU validation discovery **71/71通过，无skip**；包括真实pinned tokenizer/template/processor、audio/video fixtures、官方thinker参数转发、coverage边界与异常后原文保留。compileall、launcher shell syntax和diff检查通过。
- 每个Job单节点、单H20Z、8CPU request/limit、64Gi request/limit、700秒active deadline、620秒脚本timeout、backoffLimit0。实际CUDA UUID→PCI→ordinal映射唯一核对并执行matmul后加载模型。
- 五个本次Job均终态Failed，无active/terminating Pods、零自动重试。终态来自CLI对预期backend clock拒绝返回1；完整六次attempt与summary保留，并非作业未跑。
- 本次30次backend attempts、21次实际generation、2332记录tokens、generation总计50.392秒；显存峰值约17.1GiB（torch allocator，排除非PyTorch分配）。
- 按Job startTime至terminal transition向上取整，本次390秒；加历史337秒，总计 **727/2100秒**，剩1373秒。预算未重置；未扩数据或训练。
- 历史protocol-v1/v2完整源码经逆向已知修改后逐项匹配当时冻结hash才保存；v3及后续在提交前保存完整source bytes。实验base HEAD是67f3f8b；实际改动由frozen-source与digest确定。runroot `research/runs/av-observer-repair-20261010-v1/` 保留私有输入、raw、cost、receipts和所有阶段日志。
- adapter内 `receipt_sha256`排除自身字段；backend `clock_receipt_sha256`包含adapter自hash；video receipt同样两个独立口径。不能直接比较不同口径digest。

## 独立审查读回

独立reviewer重算所有record/raw/receipt digests，重新解码实际媒体、resample和resize，完整audio/video receipts与GPU版本逐项相同；使用实际官方processor重跑的input tokens与GPU一致。新3份完整token流119+105+111=335，独立decode逐字等于保留raw。七项执行源码/锁文件live与frozen-source匹配，完整validation discovery独立71/71通过，无skip。

机器审查结果 `integrity_errors=[]`；avgrid audit SHA256 `4492d33c0858e6f173c28ec35739cc76f0388bfaa455162be99d5764a8513873`，中文review SHA256 `7050a1ab964eabb34df4e6ca900203a51b4049d9bb95f4156b79984181edfaf0`。这些审查是记录与代码验证，不是独立人类语义标注。

## 接下来

下一步按照 `AV-Observer-Grid-Semantic-Gate-2026-10-10.md` 建立八类语义probe（先16个development、再8个冻结holdout），reference和split按source family隔离。先比较audio-only、video-only、naive joint与coverage/grid joint，在相同工具次数、AV秒数、像素和token预算下检查观察、错位、信息不足、重复实例及多源冲突。

人工人审目前0，G0、inspect_av搜索trace、独立oracle和跨源研究报告闭环仍未通过。只有这些门限得到实际证据，才扩大合成数据与推进SFT/RL。当前阶段完成的是可审计的observer基础，不能作为论文主要方法提升。
