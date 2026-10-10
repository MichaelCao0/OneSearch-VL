# 总体 Research Proposal v2：AV-ResearchBench 与训练型 OneSearch-AV

**候选论文题目：AV-ResearchBench: Evaluating Evidence-Seeking Research over Audio-Visual Sources**

**中文题目：面向多源音视频证据搜索与研究的基准、基线与能力分析**

日期：2026-10-10。第一篇定位：**Benchmark + AV Training Data Engine + Trained OneSearch-AV Baseline + Capability Analysis**，配套一个以性能提升为目标的增量方案。主开发基于 [MichaelCao0/OneSearch-VL](https://github.com/MichaelCao0/OneSearch-VL)，在 `coder ssh 3` 的 `/home/jovyan/OneSearch-VL` 上开展，使用 Pixi 管理环境。本文整合 benchmark、训练数据、SFT baseline 与分析，替代 v1 作为当前总体研究方案；历史版本保留。`AV-ResearchBench` 为暂定名称。

> 本文给出拟开展的研究、数据规模和实验，不包含尚未发生的性能结果。已有工作与执行状态集中在第 14 节。Search for Generation 属于第二篇，不进入本篇主实验。

## 1. 研究摘要

音视频研究问题常要求模型在多份录制之间寻找证据：发现相关来源，定位具体事件，比较可听与可见过程，再结合其他材料完成回答。直接观看一个视频的问答，以及从音频或图像识别实体后查找网页，覆盖了这条流程的部分能力。我们关注完整的 **来源发现—事件定位—跨源组合—研究回答**，并将每个环节独立测量。

本项目拟构建 AV-ResearchBench：基于自然音视频与相关文本的可搜索基准，提供来源、事件区间、答案事实和可接受证据组合标注。同时建设独立来源的 AV research 训练集，在真实工具环境中合成并核验多轮轨迹，训练 OneSearch-AV-SFT。以冻结 corpus 为主评测环境，统一工具和预算，比较常规 ASR/frames 工作流、原生 AV agents、OneSearch-AV-Prompted、OneSearch-AV-SFT 和强 omni 模型。通过模态对照、阶段性 oracle 和规模/预算实验，明确主要瓶颈及不同系统的优势。根据 development 结果，进一步验证实例级 AV reranking 或局部观察选择的性能增量。

第一篇的核心交付是一个有实际用途的新评测对象、一套来源隔离的训练数据与可训练 baseline，以及解释搜索能力和训练收益的实验分析。训练 baseline 是正式工作包，额外的 reranking/行动机制由实际瓶颈证据支持。

## 2. 研究动机、价值与问题

在技术演示比较、现场表演研究、动物/环境记录分析等场景中，用户不仅想知道“视频里有什么”，还想找到其他录制，定位符合条件的事件，整理不同来源共同支持的事实。例如，寻找采用相同操作步骤的演示，比较声音变化与动作顺序，或者检索另一份记录补齐当前素材缺少的信息。

研究的关键是 **任务需要的具体证据能否被发现和使用**。相关视频可以包含错误的事件实例；正确录制可以包含多个相似时刻；模型也可能找到正确片段却没有组合成完整答案。最终答案分数之外，需要能区分这些情况的标注与测量。

| Research question | 要检验的能力 | 对应实验 |
|---|---|---|
| RQ1：声画联合线索能带来什么？ | 声音和画面是否帮助缩小来源、实例与片段候选 | 模态对照、任务信息路径标注 |
| RQ2：AV research 的瓶颈在哪里？ | 来源召回、定位、感知、证据组合各自的影响 | 来源/片段/事实 oracle 诊断 |
| RQ3：主动搜索与观察如何改善结果？ | 相同预算下，交互式检索/窗口选择相对固定工作流的收益 | Workflow vs Agent、预算曲线 |
| RQ4：怎样训练出更强的 AV research policy？ | 真实 AV 工具轨迹监督能否改善查询、定位、证据组合；实例信息是否有进一步增量 | Prompted vs SFT、数据规模/轨迹消融、实例 reranking |
| RQ5：能力是否随任务复杂度保持稳定？ | 更多候选、重复实例、长媒体与跨源组合的影响 | 同题扩 corpus、分层统计与泛化分析 |

这些问题对应待检验的假设：联合线索能帮助判别自然相似候选；搜索与定位可能构成重要瓶颈；实例信息可能改善同预算下的候选选择。所有结论由实际实验确定。

## 3. 与相关工作的定位

本方案依据当前已读取的论文版本：OneSearch-VL v1、Omni-DeepSearch v1、OmniGAIA v3、VideoDR、VideoSearcher 和 HiEviDR-Bench。现阶段的定位是下面列出的任务/协议组合，正式论文前继续核对 AV retrieval、moment retrieval、跨视频问答与 agentic search。

| 工作 | 主要覆盖 | 本项目的研究重点 |
|---|---|---|
| [OneSearch-VL](https://arxiv.org/abs/2610.12419) | 视觉 anchor、外部事实与答案操作；VGEG 数据和训练 | 原始 AV 来源搜索、事件实例和片段级测量 |
| [Omni-DeepSearch](https://arxiv.org/abs/2605.08762) | 音频起始，搜索 text/image/video，含视频匹配和时序视觉推理 | 联合 AV 查询，具体实例定位，多份原始 AV 证据组合 |
| [OmniGAIA / OmniAtlas](https://arxiv.org/abs/2602.22897) | 原生 AV、多跳工具、跨模态事件图扩展、主动局部听看 | 测试时 AV evidence search 的可搜索语料、source/span 与多证据包评测 |
| [VideoDR](https://arxiv.org/abs/2601.06943) | 视频 anchor 与开放网页事实搜索 | 搜索和读取外部 AV 内容，并利用非语言声音和声画关系 |
| [VideoSearcher](https://arxiv.org/abs/2607.02927) | 视频定位、细看与外部搜索策略 | 联合声画证据及跨原始录制的组合任务 |
| [HiEviDR-Bench](https://arxiv.org/abs/2607.25151) | 文本/视觉材料的层次证据聚合与研究报告 | AV 来源/时间片段与最终回答之间的能力分解 |

本篇拟贡献三个方面：

1. **任务与数据：** 在共享可搜索的 AV corpus 中发现来源、定位事件、组合证据，提供自然任务和分层标注。
2. **评测与分析：** 将 retrieval、localization、perception 和 composition 连接到最终研究成功率，并在统一预算下比较系统。
3. **训练与性能：** 提供 AV research 数据引擎和 OneSearch-AV-SFT，测量轨迹训练对搜索与研究的收益，并根据瓶颈验证实例 reranking 等增量。

OmniGAIA 已包含 AV、多源、多跳和主动感知；论文的比较重点放在本 benchmark 的任务、标注和测量对象。新增音频工具、query rewrite、evidence graph 或标准 ranking loss 本身不单独承担机制贡献。

## 4. Benchmark：任务定义与环境

### 4.1 输入与输出

每个任务表示为：

\[
T_i=(q_i,x_i,\mathcal C_i,B_i;\mathcal Y_i,\mathcal P_i),
\]

其中 `q` 是问题，`x` 是初始声画线索，可为空；`C` 是该 split 的共享 corpus；`B` 是执行预算；`Y` 是接受的答案事实；`P` 是参考证据包集合。分号后内容只用于标注与评分。

模型输出事实答案或简短比较表，并提交证据列表：`source_id`、源时间轴上的区间、对应的答案事实；文本引用另保存 quote 和位置。另提交最终有序的 `ranked_sources` 与 `ranked_windows`，用于检索/定位主指标。短研究报告作为呈现方式，不以篇幅决定成绩。

主设置从 AV 线索出发；文本问题起始设置另行标记。不同初始信息设置分别统计，避免把问题难度与 modality access 混在一起。

### 4.2 四类任务

| 任务族 | 定义 | 示意任务 |
|---|---|---|
| Source discovery | 声画条件共同约束要查找的录制 | 找到采用指定动作并伴随指定声音变化的演示 |
| Event-instance localization | 定位同类事件中的特定一次 | 在重复操作中找到“声音变化后立即发生指定动作”的实例 |
| Cross-source comparison | 搜索并比较不同原始 AV 录制 | 比较两种操作的动作顺序、可听变化或次数 |
| Research composition | 用多个来源补齐必要事实 | 从演示确定步骤，再查另一录制/资料完成解释或比较 |

示意问题用来指导采集；正式题目从真实素材、可核对的事实和自然研究需求生成。首批领域包括技术演示/机械、现场表演、动物与环境。后续领域根据素材与标注可行性扩展。

多源 AV 子集要求至少两份外部原始 AV 录制的内容共同支撑答案；AV+text 子集单独标记。相同录像的转载归为同一录制家族；网页说明不计为第二份 AV 观察。正式版目标至少一半题目属于跨原始 AV 来源的组合/比较，原型后根据实际材料校准。

### 4.3 主赛道与诊断赛道

**Frozen-corpus policy track，主赛道。** 该 split 的所有题共享 AV 与网页/文档库，采用固定官方索引与检索 API。系统可以自主生成查询、读取任意返回来源、选择时间窗口和收集证据。Corpus 包含有效证据、自然相似候选与背景材料，不暴露每题正确候选列表。这个设置主要比较控制与研究策略。

**Retrieval-system track，扩展。** 系统可在同一 corpus 上构建自己的索引/retriever。报告预处理、索引大小和查询/读取成本，与 policy track 分榜。统一工具名称不代表索引信息相同，不能把两个赛道混为同一公平对照。

**Given-evidence diagnostics。** 分别提供参考来源、参考区间或人工事实，测搜索、定位与理解组合的诊断表现，不进入自主搜索主榜。

**Open-web extension。** 后续增加真实在线搜索；按运行日期和快照独立报告。第一版主要结论适用于冻结语料，open-web 是外部效度扩展。

### 4.4 工具与官方索引

核心工具为 `search_sources`、`inspect_av`、`read_text` 和提交答案/证据。`inspect_av` 接受来源与原时间轴区间，返回可供模型消费的媒体或明确标记的 observer 结果。预算内可以多次查看同一来源。`compare_sources` 如果实现语义比较，要显式调用和记录感知/推理，metadata 比较不替代它。

第一版官方索引先实现 metadata/正文全文检索，再建立固定的 ASR、音频事件描述和视觉描述索引，采用公开可复跑的 hybrid retrieval。描述由锁定的模型从媒体产生，不读取题目参考答案。区间来自实际来源时钟。原始媒体读取仍由工具完成。

索引字段与 snippets 的可见内容按赛道固定。模态分析使用相应索引变体：去音频时同时屏蔽 ASR、音频描述和音频派生 snippets；去视觉时屏蔽视觉描述/图像与视觉派生信息。可访问的自然网页仍可能提供线索，需另报 metadata/text 捷径表现。

## 5. 数据管线与标注

```mermaid
flowchart LR
  A[自然 AV 来源组与资料] --> B[来源登记和时间区间]
  B --> C[研究问题与事实候选]
  C --> D[人工回放与证据包标注]
  D --> E[按录制和事件家族划分]
  E --> F[冻结 corpus 和 reference]
  F --> G[Baseline 搜索与评分]
  G --> H[能力分析与性能改进]
```

### 5.1 来源组驱动的构建

先围绕自然专题采集来源组：多份同类操作、演示或记录，配套技术/背景资料及自然相似候选。再提出可由这些来源共同回答的研究问题。目标是每个独立专题/事件组有足够材料支持搜索与比较，避免从单个短视频批量生成大量近重复题。

优先使用可重新分发的素材，保存来源、许可、原媒体与元数据；正式 corpus 的权利记录在数据卡中明确。保留现实中的片段长度与来源差异，不为当前 2 秒 observer smoke 限制重定义 benchmark。

### 5.2 每题标注内容

| 内容 | 要求 |
|---|---|
| 问题与初始线索 | 自然需求、输入版本、必要的查询线索 |
| 答案事实 | 数值/集合/比较项/可接受别名，必要项与补充项分开 |
| 来源和时间 | `source_id`、有效事件区间、具体实例，以及文本 quote |
| 证据包 | 完成各必要事实的支持项、多个有效替代组合 |
| 信息路径 | 音频、视觉、联合关系与外部搜索分别发挥什么作用 |
| 依赖家族 | 专题、录制、事件、近重复/转载关系，用于 split 与统计 |

候选描述/问题可以由模型辅助生成；reference 经实际审核者回放、核实来源和裁决。两名审核者独立检查核心事实与区间，分歧由第三人裁决或修订题目；模型输出不充当人工标注。记录一致性和标注时间，用于确定正式规模。

### 5.3 多个有效证据包

设每题有 `P_i={p_i1,p_i2,...}`，每个包可充分支持答案。允许模型找到等价来源和不同检索路径。评分以一个完整有效包为单位，不要求收集所有等价录制。

使用多种 baseline 的 retrieval pooling 和人工搜索补充候选，核验其他有效来源。正式测试参考在主评测前冻结；新发现的有效包通过统一裁决加入新版本，并对所有系统重新计分。参考覆盖状态在数据卡中记录，不把未审过的来源自动判成语义错误。

### 5.4 Split 与构建标准

Development/test 按录制、事件与近重复家族隔离。同来源多题归于同一 split；多源问题共享关键录制时，将来源关系连通分量作为划分单位。可训练数据另采来源，其 teacher、负例挖掘和轨迹合成不读取 test reference。

划分落实到 corpus：开发只搜索 `C_dev`，方法与配置冻结后再访问共享 `C_test`。只隔离问题标签而让开发过程读取所有测试媒体，不构成来源隔离。泛用背景文本可以按公开规则共享，跨 split 的关键 gold evidence 则归到同一依赖分量。

保留标准围绕问题清晰、答案可核对、证据可获取、事件可区分和真实信息路径。Question-only、metadata-only、AV-only/no-search 和模态对照帮助测量任务需求；单个模型答错不是“模态必要性”的唯一判断。任务与保留标准在方法对比前确定，不按某个 proposed 的失败/成功筛选正式题。

### 5.5 建议规模

| 阶段 | 任务目标 | 交付与作用 |
|---|---:|---|
| 原型 | 16，每类 4 | Reference、搜索环境、评分器与逐题 baseline |
| 构建验证 | 80，覆盖更多独立来源组 | 标注一致性、信息路径、成本、初步瓶颈 |
| 正式版 | 约 400，暂定 80 dev/320 test | 冻结数据、完整 baseline 和论文分析 |

规模是计划目标。暂按每类 100 题设计，dev 每类 20、test 每类 80，来源连通分量隔离优先于精确配额。400 题建议覆盖至少约 150 个独立专题/事件家族，并报告每类的家族数；正式来源数量由自然证据组决定，按数百至千级 AV/文本来源准备 shared corpus，而不是以每题几份固定候选代替检索。

最早的 16/80 题作为 development，正式 test 来自后续独立家族。80 题后用实际有效任务比例、审核工时和推理成本校准规模。核心结果同时给出按任务与按来源家族的统计。

## 6. Baseline：比较什么、如何实现

### 6.1 最小但完整的 baseline 套件

| ID | Baseline | 输入与流程 | 实验职责 |
|---|---|---|---|
| B0 | Question-only / 初始 AV-only，无搜索 | 不访问 corpus | 测已有知识与初始素材覆盖 |
| B1 | Metadata/text search | 使用自然 metadata/正文，不含 AV 派生描述；不读取原始 AV | 测文本线索能覆盖多少目标，属于信息对照 |
| B2 | ASR+frames workflow | 固定检索、分段转写/抽帧、统一 reader、回答 | 常见可部署基线 |
| B3 | Joint-AV workflow | 与 B2 相同检索框架，使用原生 AV observer | 感知路线比较 |
| B4 | Generic joint-AV agent | 自主 query、检索、选窗、迭代与回答 | 重要强对照，包含普通查询改写 |
| B5a | OneSearch-AV-Prompted | 发布 OneSearch-VL controller + 固定 AV observer，不做新增训练 | 训练前基线 |
| B5b | OneSearch-AV-SFT | 同一 controller/observer/tool/index，controller 用独立 AV research 轨迹 SFT | 主要训练型 baseline，测新数据的收益 |
| B5c | OneSearch-AV-SFT + 实例 reranking | 同一 SFT policy，增加同预算的候选评分 | 可选机制增量 |
| B6 | 强 omni 系统/OmniAtlas adaptation | 相同 corpus、工具和预算，固定模型版本 | 能力参照与系统上界探索 |
| D | 来源/区间/事实 oracle | 只用于阶段诊断 | 定位搜索、观察和组合瓶颈 |

B2/B3 比较时清晰列出 ASR、帧与原生 AV 的实际信息、模型和成本，不能把 perception 与 policy 变化混成一个机制结论。B4/B5 的策略比较共享 observer、索引与工具。闭源模型只在服务、版本和预算实际配置后运行，计划中的名字不计作结果。

### 6.2 推荐实现：OneSearch-AV baseline

```text
问题与初始 AV
  → controller 生成查询
  → 官方检索器返回来源/候选区间
  → controller 选择来源和观察窗口
  → AV observer 或 ASR+frames reader
  → 收集事实与证据，继续检索/读取
  → 提交答案和 source/span 引用
  → 独立 evaluator 评分
```

Controller、感知和评分解耦。首先接通当前 `inspect_av` 与真实 observer，保存模型看到的内容、后续查询和最终引用。原生 AV backend 的 2 秒/CFR smoke 支持需要扩为可计量的多窗口或更长区间读取，先做实际感知检查再锁定正式配置。多次单片段 observation 要经过 controller 合成回答，才形成完整 episode。

先完成标准 prompted agent 作为对照，同时将训练数据引擎和 SFT 列为必要工作包。OneSearch-AV-Prompted 与 OneSearch-AV-SFT 使用同一感知与检索栈，主差异是 controller 权重。所有 baseline 输出同样的事实/证据格式，由独立 evaluator 执行。既有私有 reference 不进入 controller 的提示词。

### 6.3 同预算规则

主表的策略比较固定 corpus/index、工具、最大调用数、观察秒数、生成 tokens 和总超时。B0/B1 及模态对照的索引信息差异显式列出，作为信息需求分析。感知模型、controller 与 official index 版本逐项记录。缓存策略一致；离线建索引/共享观察缓存的工作单列，并报告摊销后的总成本。

端到端 episode 按实际可访问的每份 candidate observation 计逻辑 AV 请求秒数，不因媒体结果已缓存而免费；controller、scorer 与 observer 的生成 tokens 均计入 episode 生成预算。缓存降低物理运行开销，不扩大允许观察的信息。

建议原型校准的三档逻辑预算：

| 档位 | 工具调用上限 | 候选 AV 累计请求秒数 | Episode 生成 token 上限 |
|---|---:|---:|---:|
| Small | 8 | 30 | 4,000 |
| Medium，候选主表 | 16 | 90 | 8,000 |
| Large | 32 | 180 | 16,000 |

这些是待原型测量后冻结的研究设置，不是现已获配的正式计算额度。AV 请求秒数按每次访问累计，重复请求不免费；ASR-only、video-only、joint-AV 模式分别记录。初始线索的读取单列但限额一致。实际 decode、感知 tokens、wall/GPU cost 另报，不把逻辑秒数等同于 GPU 时间。

## 7. 训练数据引擎与 OneSearch-AV-SFT

OneSearch-AV 的训练目标是学习 **读过实际 AV 观察后怎样搜索、选窗、比较、补事实并提交答案**。只把已有问答配上音视频路径，不能训练出完整 research policy；数据单元应是可回放的多轮工具轨迹。

### 7.1 继承 OneSearch-VL 的哪些部分

公开论文 Appendix A.5/D.1 描述：已核验任务 → expert 在真实工具环境 rollout → 工具执行并把 observation 加入历史 → 答案与过程筛选 → SFT。训练目标为 assistant reasoning/tool-command/final-answer，输入和工具 observation 的 loss mask 为 0。公开 SFT 基于 LLaMA-Factory/ShareGPT；原 recipe 为 Qwen3-VL-8B language fine-tuning、freeze vision/projector、ZeRO-2。

可以继承真实工具交互、resource handles、轨迹格式、loss masking 和 checkpoint/评测框架。我们重做 AV train corpus、自然多源任务、音频/声画 observations 和证据完整性筛选。论文报告约 110K SFT、64 H800、约四天，这不是本项目首轮的规模要求；原 paper 的 learning rate 为 2e-5，公开 YAML 为 1e-5，训练时锁定实际 recipe，而不声称复现同一超参数结果。

### 7.2 两条模型路线

| 路线 | 模型输入与训练对象 | 首轮定位 |
|---|---|---|
| **Modular OneSearch-AV-SFT，推荐先做** | 发布 OneSearch-VL-8B controller 读取固定 AV observer 结果及可用帧/文本；LoRA 更新 controller language policy，observer 不训练 | 最快完成新训练集和 before/after 对照；称模块化 AV research agent |
| Native-Omni OneSearch-AV | 原生 Omni policy 消费 waveform+video、生成工具调用与答案；freeze perception 后训练 thinker/policy 或 LoRA | 后续升级/参照，需要验证数据加载、声画 processor、loss mask 和 backward |

模块化路线中的 controller 并没有原生音频 encoder，音频理解来自固定 observer；不能将其 checkpoint 描述为单模型原生 AV policy。原生路线优先研究已 pin 的 Qwen2.5-Omni-7B，是否采用它或其他 Omni backbone 由训练兼容性/性能验证确定。无需输出语音，生成侧 talker 不属于本篇目标。

Coder3 的 bundled trainer 源码已有 `qwen2_omni`/`qwen3_omni` template、`Qwen2OmniPlugin`、`use_audio_in_video` 和 thinker loader，但当前正式注册只有 image/multi-image/video。对 joint-AV 的可执行训练支持要经过实际 batch、gradient、optimizer 和 checkpoint reload 验证。Omni plugin 目前分别处理 video/audio，不会自动调用已有 PTS adapters；训练与评测的 clipping、声画时间网格和 processor 输入必须统一验证。原训练 runtime 与现有 observer runtime 分开建 Pixi lock，避免替换已经跑通的推理依赖。

### 7.3 训练集与 benchmark 分工

建立 `C_train`、train-validation、`C_dev` 和 `C_test` 的来源家族隔离。第一步先划家族，再扩图/造问题/合成轨迹。不同划分的真实录制、转载、近重复与关键支持来源不交叉。正式 test 的题、证据包、查询和实际媒体不进入 teacher 合成、负例挖掘或训练缓存。

Benchmark 约 400 题服务测量；训练集目标为 10–20K 条保留的成功 AV research trajectories。不是拿 benchmark 400 题增强到 20K；16/80 个 development cases 继续作为开发评测，tiny SFT 也用独立 train 来源。不同 family 上的同一任务模板可共享，具体实例和内容隔离。

四类任务在训练集里近似均衡：source discovery、instance localization、cross-source comparison、research composition。目标轨迹包含有效 AV inspection 和真正需要的后续搜索/比较；自然声音、speech、music、animal/ambient 的比例按来源分布记录。可增加少量基本工具操作与视觉任务 replay，权重通过 train-validation/dev 选择，避免将旧公开数据未审计混入新 benchmark。

### 7.4 AV training data engine

```text
独立 train 来源组
 → 实际媒体事件/区间和外部事实登记
 → 任务、答案及充分证据包候选
 → 核实可解性和参考证据
 → teacher 在同一真实搜索/读取环境 rollout
 → 验证最终答案、source/span 与实际 observation 支持
 → 按来源/题/轨迹去重，形成 SFT conversations
 → token/media/loss-mask 检查
 → SFT、checkpoint reload、独立任务评测
```

用于生成任务的 authoring 模型可以读取来源和参考材料；用于产生示范轨迹的 teacher 只拿问题、初始线索与工具，不拿答案/正确来源包。筛选器读取 reference 与执行日志，要求回答正确、关键事实受支持、所需 AV 片段确实读取且无虚构工具返回。

Teacher 推荐强工具使用模型；优先接已配置服务，或用可部署 open teacher。当前 Gemini endpoint/key 尚未确认可用，不能将其当成已部署。可采用强文本/视觉 teacher controller + 固定 AV observer 的同一模块化栈；同一来源观察随真实工具执行进入历史，不能让 teacher 自己编写 observation。Teacher 的每一步动作都要能由 student 可见 history 支持，避免只有 teacher 看得到的原始 AV/特权信息形成不可学监督。每题原型最多 2–4 条 rollout，保留成功但有不同有效路径的示范，报告 pass/retention rate。

保存结构化 episode trace：question/initial media、工具 definitions、assistant 查询/决策、实际 tool arguments、executed observation、media handles、成本、final answer 和引用。私有 reference/筛选得分只给过滤器。导出 ShareGPT conversations，监督 assistant 工具调用、简洁决策说明与答案；工具 observation、user/system 和参考字段不作为生成 target。显式配置 `train_on_prompt: false`、`mask_history: false`，逐轮抽样核对 assistant labels 与 `IGNORE_INDEX`。路径和工具 handle 在多轮中一致。

训练集自动 QC 全量覆盖执行/绑定/字段/答案与支持关系，人工分层抽检至少约 200 条或保留集的 5%（取较大者，规模后校准），重点看实际声画内容和跨源对应。Benchmark reference 则逐题人工核验。两种质量流程职责不同，不以全部训练数据人工重标作为扩量的必需流程。

### 7.5 SFT 阶段与规模

| 阶段 | 数据 | 训练与交付 |
|---|---|---|
| Connectivity | 64–256 条独立 train 轨迹 | 数据 exporter/多轮 loss mask 验证；16–32 条短 batch 的真实 optimizer smoke，保存并 reload |
| Pilot SFT | 约 2K retained trajectories | OneSearch-AV-SFT-LoRA，初步 1–2 epochs，train-validation/dev 上比较 Prompted vs SFT |
| Main SFT | 约 10–20K retained trajectories | 依据学习曲线锁定配置，完成正式 checkpoint 与 benchmark 主表 |
| 可选强化训练 | SFT 后存在明确的序列行动瓶颈时 | 再考虑 preference/ranking 或在线 RL，单独登记数据、成本和对照 |

Connectivity 的作用是训练接通；不能把一条 loss 下降当成 AV search 提升。Pilot 用约 0.5K/1K/2K 子集学习曲线，主阶段用 2K/5K/10K 或实际可获得规模。子集按来源家族取样并嵌套，重复同一题的多个 rollout 不算独立任务数量。

模块化 pilot 从 OneSearch-VL 发布 8B 开始，LoRA 更新 language policy，冻结 vision/projector 与 AV observer。候选初始配置：BF16、gradient checkpointing、microbatch 1、累积后有效 batch 16、context 8K–16K、LoRA rank 16/32，学习率在 train-validation 上比较 2e-5 与 1e-4；输入轨迹长度由实际 tokens/媒体测量决定。以上是待 tiny train 验证的研究配置，不是已验证 launcher。

长轨迹采用保留真实前缀的 assistant-turn/action-prefix training 样本，或增加 context；不任意截掉 observation 后续工具依赖或最终 response。全参数 full SFT 和原 32K recipe 保留为资源允许时的扩展，首轮不要求复刻 64 卡配方。

Tiny train 必须看到真实 backward/gradient、优化器更新后的权重或 LoRA 变化、checkpoint 保存和 reload；可续训 checkpoint 保存 optimizer/scheduler/RNG。正式收益用同一 observer/index/budget 下独立任务的来源、片段、答案和 EBS 成绩证明。

### 7.6 训练对照与 analysis

主要配对是 **同一 checkpoint 起点 + 相同 AV 工具的 Prompted vs AV-trajectory SFT**。另比较：

- Answer-only SFT vs 多轮 tool-trajectory SFT：训练任务和 backbone 相同，分别报告监督 tokens，增加 token-matched 子集对照。
- 通用工具轨迹 vs AV research 轨迹：匹配任务数/更新预算，检查是否学到声画驱动的搜索/选窗/组合。
- 完整 joint-AV observation vs 缺少音频派生信息的训练对照，测试保持一致：分析音频训练信息的增量。
- 实际执行后成功轨迹 vs task/reference 导出的静态规划：只在资源允许时作数据引擎消融，静态样本明确不当作真实执行示范。
- Train size/来源多样性/trajectory 长度曲线：辨别重复问法与新增来源带来的不同效果。

主表同时保留 B5a、B5b，清晰报告 SFT 对查询、读取位置、证据完整性与最终成绩的改变。额外 reranker 在同一个 SFT baseline 上比较，不把其效果与整个 SFT 收益混算。

## 8. 可选增量性能方案：实例级 AV reranking

核心假设：候选已经相关时，题目所需的局部声画关系与具体实例信息可以改善选择。提出一个小模块，接收问题、候选区间和实际 AV observations，评分并重排来源/片段，后续 controller 使用同样的工具完成研究。

### 8.1 两轮实验

**第一轮：相同 observation cache 的排序比较。** 普通 query-aware scorer 与实例 scorer 接收相同候选、AV observations 和 token 上限。实例 scorer 显式评估可听事件、可见动作/对象、局部先后/共现是否满足问题。先用同一 backbone 的 prompted scorer，观察候选排序与下游回答变化。该实验作为 controlled-ranking diagnostic；如果共同观察池超出主赛道 episode 限额，结果进入诊断表。进入端到端主表时，观察、scorer 与后续回答的全部访问和生成成本需落在同一预算内。

**第二轮：按瓶颈选择训练或主动检查。** 如果排序信号明确，构造自然实例 hard negatives，采用标准 pairwise ranking/SFT。负例包括相似画面但声音不同、相似声音但动作不同、同一录制另一相似实例。若失败主要因为窗口没有覆盖事件，则优先实验候选差异驱动的下一窗口选择。两种升级分别评测，不同时加入多个模块后只报总收益。

标准 ranking loss 可采用：

\[
\mathcal L_{rank}=-\log\sigma\big(s_\theta(q,o^+)-s_\theta(q,o^-)\big),
\]

这里拟研究的是实例数据/表示对 AV search 的作用，不把该标准 loss 表述为新优化算法。训练正负数量、来源和更新预算与普通负例对照相同。

### 8.2 关键消融

- 普通 scorer vs 实例 scorer：同一信息、同一 backbone，隔离评分方式。
- 普通 negatives vs 实例 hard negatives：同样样本数和训练预算，隔离数据作用。
- 保留事件词、移除局部关系/顺序：检查是否超过关键词匹配。
- 相同总预算的固定观察 vs 自适应窗口：检查行动选择的收益。
- 候选内 oracle 与全任务结果：区分排序可解决的错误和初始召回缺失。

Benchmark 的正式任务由自然需求和预设构建标准确定，所有方法均可参加。Reranker 是候选改进，最终是否成为论文方法段落由 development 证据决定。

## 9. Evaluation：主指标与评分规则

### 8.1 研究完成

主表同时报告 `Answer Accuracy` 与 `Evidence-backed Success (EBS)`。

\[
EBS_i=\mathbf 1[\text{必要答案事实正确}]
\cdot\mathbf 1[\exists p\in\mathcal P_i:\text{提交证据充分支持 }p].
\]

答案包含相互矛盾的关键结论不能通过事实正确性；无关文字不加分。对开放解释先拆成可核对事实项，报告 necessary-fact coverage 与 all-correct。充分证据检查来源/区间/quote 的支持关系及有效替代包，不规定唯一推理顺序。

### 8.2 来源检索与事件定位

单目标检索报告 Source Recall@k、MRR。多源任务报告 **Complete-package source recall@k**：前 k 个检索来源是否至少覆盖一个完整有效包所需的来源集合；另报 **Best-valid-package coverage@k**，即各合法包中的最高来源覆盖比例。把所有替代包来源取并集会惩罚合理的替代选择，因此不将该并集 recall 作为主指标。

事件层报告正确实例率与 Moment Recall@k，在 IoU 0.3/0.5/0.7 下匹配可接受的事件区间；multi-event 采用同来源的一对一匹配并计事实覆盖。复合必要包单列完整包定位成功率。Reference 保存事件核心和可接受上下文窗口；区间和阈值由标注规则在正式评测前固定，过长区间受到 IoU 和读取预算限制。

主表使用 actor 最终提交的有序 `ranked_sources` / `ranked_windows`，按列表顺序去重，统一评测 k=1/5/10；缺少 ranked list 按空列表计。来源检索列表可以包含搜索发现但尚未读取的候选，事件窗口必须对应实际 observation；最终 EBS 的引用仍要求有效读取和支持。Retriever 的 per-query Recall@k 另列为工具诊断，episode 的首次发现/读取顺序另用于 discovery curve，不与最终排名混算。

Whole-corpus 的指标和“参考来源已经进入候选池”的条件指标同时报告，不能把召回缺失从主分母删掉。模型的 source/span 证据提交与实际读取日志对应；搜索中看到标题不等于读取了该片段。

### 8.3 成本与统计

报告 tool calls、逻辑 AV 秒数、真实 prepared frames/audio samples、observer/controller tokens、wall time 和可得到的 GPU 工作量。离线索引/预计算成本单列；无法精确得到的资源项标为 unknown。失败和超时保留在任务分母，另给执行完成率。

正式结果按来源/事件家族进行 paired cluster bootstrap，给 95% 区间；必要时以配对置换检验辅助比较。多题同源不按独立样本计算。跨预算和任务族比较保留预先设定的主比较，避免从大量切片中挑显著结果。

### 8.4 答案与证据 judge

数值/集合/选择事实优先规则匹配与别名；语义等价和证据支持使用固定 judge recipe，在分层人工子集上测一致性。人工集覆盖各任务族、不同系统与正确/错误输出，不只看高分结果。Human reference 不由同一个生成模型自动自证，judge 版本、提示词和裁决记录可复跑。

## 10. Analysis：论文要回答的实证问题

| 分析 | 操作 | 测量 | 能回答什么 |
|---|---|---|---|
| A1 模态互补 | question/text、audio、video、joint 的对应输入/索引 | Source/span、Answer、EBS | 声画联合信息在哪些任务有用 |
| A2 阶段瓶颈 | 正常运行；给来源；给来源+区间；给人工事实 | Answer/fact coverage；可引用媒体设置另报 EBS | 搜索、定位/感知、组合可能影响多少 |
| A3 交互策略 | 固定 workflow 与 generic agent，同预算 | 成功率、有效读取比例、调用轨迹 | 主动检索/观察的实际收益 |
| A4 候选规模 | 同题同有效来源，增加随机/自然近似干扰 | 性能—corpus size 曲线 | 检索系统对规模与相似候选的敏感性 |
| A5 实例难度 | 自然重复次数、事件间距、clip 长度分层 | 实例命中、定位、回答 | 是否只识别类别而未定位具体事件 |
| A6 多源组合 | 按必要来源数、事实数和证据包形态分层 | Package recall、fact coverage、all-correct | 来源已找到后是否完成组合 |
| A7 预算与效率 | Small/Medium/Large 与早停行为 | Answer/EBS—cost 曲线 | 更多预算改善哪一环节 |
| A8 训练与改进归因 | Prompted/SFT、trajectory/answer-only、规模曲线、第 8 节 scorer 消融 | 排名与下游成功、matched cost | 训练数据与额外机制分别贡献什么 |

模态结果同时按人工标注的 complementary、redundant、audio-dominant、video-dominant 信息路径分解。画面内字幕属于实际视觉信息，可保留并标记；移除原始 waveform 而仍提供 ASR snippet 不称为无音频实验。

Oracle 干预分别定义：给来源只给正确 ID，不给事件区间/答案；给区间提供实际媒体，不给人工事实；给事实提供核验过的观察项。Oracle 主分析用 Answer Accuracy/necessary-fact coverage；EBS 只在提供可引用来源且实际读取媒体的设置另报，fact oracle 不按自主搜索 EBS 计分。它们改变可用信息，是阶段诊断参考，分数差值不能直接当成可相加的因果误差比例，也不混入主榜。

增加 corpus 干扰时保持 gold 不变，随机和近似候选分别报告；初始索引 recipe 与预算相同。自然难度分层给出家族数和域分布，不能只凭“重复多的题更难”推导因果结论。

逐题错误分析使用简洁分类：初始来源未召回、错误实例/区间、AV 观察错误、外部事实缺失、组合不完整、执行/预算失败。多种错误可以共存；先由日志与 reference 自动建议，再在人审子集校验分类。错位、缺失和转载混淆作为辅助切片。

## 11. 论文实验表与图

预先规划下面的证据结构，数值由真实实验填入：

1. **主表：** 各系统 Answer Accuracy、EBS、package-source recall、instance/moment 指标和成本，统一主预算。
2. **任务族表：** 四类任务及跨原始 AV/AV+text 子集表现，展示目标能力。
3. **模态与阶段表：** 单模态/联合信息、来源/区间/事实 oracle，解释能力瓶颈。
4. **训练与改进表：** Prompted vs SFT、轨迹 vs answer-only、训练规模与 AV 信息消融；普通与实例 reranking 作进一步方法比较。
5. **图：** benchmark 能力链与自然案例；accuracy/EBS–cost；corpus size/实例密度变化。

Dataset 统计表报告题数、独立家族、来源数、媒体长度、领域、语言、必要来源/事实数、标注一致性。附录给工具、数据卡、judge、成本规则与更多案例。主文围绕任务价值和实证发现组织，不按工程修复过程展开。

## 12. 实现工作包与交付

Benchmark 测量、运行环境和训练轨迹合成分别管理，在同一 repo 上共享来源与媒体读取基础。

| 工作包 | 主要交付 | 依赖 |
|---|---|---|
| WP1：任务/reference | 16 题完整包、区间/事实/证据包、来源家族 | 自然材料与实际审核 |
| WP2：搜索环境 | 官方索引、共享 corpus、语义 inspect_av 与完整 episode 日志 | WP1 的素材；AV runtime |
| WP3：独立评测 | 来源/片段/package/答案评分、judge 校准、结果聚合 | WP1 标注与 WP2 输出 |
| WP4：核心 prompted baselines | B0–B5a 的逐题运行，感知/索引/预算一致性 | WP2–WP3 |
| WP4b：训练集与 SFT | 独立 train corpus、256 条接通样本、2K pilot、10–20K 目标集、真实 SFT checkpoint | WP2–WP3；训练 adapter/CUDA smoke |
| WP5：正式数据 | 80 题验证；独立家族的正式 test；数据卡 | 标注成本、协议与任务需求验证 |
| WP6：分析与改进 | A1–A8、训练前后成绩、强模型参照、候选性能方案 | development 选择、正式 frozen test、WP4b |

复用 `research/av_data` 的来源/媒体工具、`research/av_search` 的本地搜索和 `research/av_agent` 的 observer/controller 接口。新增 benchmark task/reference/evaluator/episode runner 模块，文件布局以现有代码接口审查后确定。线上推理 actor 只访问允许的 corpus/索引，reference evaluator 独立运行。

原型完成标准是 16 题可以从共享搜索接口开始，提交来源/区间与答案，由独立评分器产出逐题成绩；人工审核记录和不能成立的候选题同时保留。构建验证阶段再冻结任务标准、赛道、预算、指标和正式 test split。

## 13. 时间、资源与扩量安排

建议安排在标注人员和算力可用的前提下：第 1 周完成 16 题包与首条搜索评分闭环；第 2–3 周推进 80 题、core prompted baseline、训练轨迹接通和 tiny SFT；第 4–6 周并行建设正式 benchmark、2K pilot/10–20K 训练集、SFT 与运行矩阵；第 7–8 周完成分析、候选改进、论文与发布材料。此为依赖资源的研究排期，正式规模在 80 题后校准。

用原型测量预算：

\[
Cost_{eval}\approx N_{tasks}\times N_{systems}\times N_{repeats}
\times \overline{Cost_{episode}},
\]

再加入索引预处理、感知缓存、judge、人审与可选训练。按任务族测平均和长尾 episode 成本，预估 disk/GPU/API/审核工时，登记正式 ceiling 后扩大。Baseline 越多不意味着每个都执行全部分析矩阵：先完成核心主表，再按 RQ 选择相关系统运行配对分析。训练成本另按 retained trajectories、有效监督 tokens、epochs/steps、optimizer 和实际 GPU 计量；teacher rollout 和 judge 成本也独立登记，不能藏在 SFT GPU 时长里。

既有诊断预算累计 727/2100 Job 秒，余 1373 秒，适合有界原型感知准备；正式 benchmark 数百题、完整 baselines、训练和强模型服务不计入该小额余额。单个后续 observation bootstrap 最多 700 秒 deadline/一 GPU，setup 时间继续从历史总额扣减。

当前没有 Serper 也能构建主赛道；闭源服务按实际配置启用。权重、原媒体和 private test reference 留在远端受控目录；数据公开时按许可与 test 答案策略发布代码、任务/语料、评测说明和可复跑 recipes。

## 14. 当前已有基础与下一步

截至 2026-10-10 的代码/记录检查：

- 发布 OneSearch-VL 8B 完成自定义图像/视频的 CPU 最小运行；未测完整官方论文 benchmark。
- 主开发迁至 Coder 3；Qwen2.5-Omni-7B 在 Kjob H20Z 真实运行，完成 observer 输入网格与输出协议修复；本轮 21 次 generation 和 observer validation 71/71 有记录。
- 已有 10 条自然 AV 素材和相关网页、可回放的本地搜索/读取/裁切基础。
- 16 行 performance-oriented 候选已设计，reference 尚未填齐；没有正式 benchmark、端到端 baseline 成绩或新方法增量结果。
- `inspect_av` 尚未完成与语义 observer 的完整 episode 集成，`compare_sources` 主要是 metadata 比较；目前的 AV backend 支持范围是短片段 smoke。
- 总 proposal v1 已推送至 `04e35f93ba1c7cd4673f5aeac293189e7d8ea00c`；本 v2 加入独立训练集与 SFT 工作包。
- 远端 `SFT/data/dataset_info.json` 注册的是 image/multi-image/video；没有新的 AV research SFT exports。训练栈源码含 Omni template/loader 支持，不等于 joint-AV backward 已验证。
- 尚未开始 AV teacher 批量合成、SFT optimizer update 或新 checkpoint 训练。本版定义的是接下来要实施的训练路径。

下一份实质交付是 **16 个可评分任务 + 搜索环境 + Prompted baseline 逐题结果**，并行建立 **独立 train 来源上的 256 条 trajectory seed 与 tiny SFT**。随后完成 2K pilot 的训练前后评测，再据此扩到正式 benchmark 和 10–20K SFT。

## 15. 最终交付与两篇论文的关系

第一篇交付 AV-ResearchBench 数据/环境与 evaluator、独立 AV research 训练集与 OneSearch-AV-SFT checkpoint、强 baseline recipes、来源/片段/答案成绩、模态/阶段/训练分析，以及有结果支持的额外增量方案。它建立“能否完成 AV evidence search 与研究”的评测和性能证据。

第二篇再研究搜索结果如何支持生成：检索证据怎样进入生成条件，怎样影响素材选择、生成结果与用户目标。第二篇使用第一篇的能力基础，但具有独立 generation 任务、评测和实验，不把搜索成功直接等同于生成质量。
