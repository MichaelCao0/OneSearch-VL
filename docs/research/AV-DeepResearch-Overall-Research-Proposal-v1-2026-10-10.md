# 总体 Research Proposal：多源 Audio-Video Search 与研究

**候选论文题目：AV-ResearchBench: Evaluating Evidence-Seeking Research over Audio-Visual Sources**

**中文题目：面向多源音视频证据搜索与研究的基准、基线与能力分析**

日期：2026-10-10。第一篇定位：**Benchmark + Strong Baselines + Capability Analysis**，配套一个以性能提升为目标的增量方案。主开发基于 [MichaelCao0/OneSearch-VL](https://github.com/MichaelCao0/OneSearch-VL)，在 `coder ssh 3` 的 `/home/jovyan/OneSearch-VL` 上开展，使用 Pixi 管理环境。本文整合 benchmark 提案与性能导向研究计划，作为当前总体研究方案；历史版本保留。`AV-ResearchBench` 为暂定名称。

> 本文给出拟开展的研究、数据规模和实验，不包含尚未发生的性能结果。已有工作与执行状态集中在第 13 节。Search for Generation 属于第二篇，不进入本篇主实验。

## 1. 研究摘要

音视频研究问题常要求模型在多份录制之间寻找证据：发现相关来源，定位具体事件，比较可听与可见过程，再结合其他材料完成回答。直接观看一个视频的问答，以及从音频或图像识别实体后查找网页，覆盖了这条流程的部分能力。我们关注完整的 **来源发现—事件定位—跨源组合—研究回答**，并将每个环节独立测量。

本项目拟构建 AV-ResearchBench：基于自然音视频与相关文本的可搜索基准，提供来源、事件区间、答案事实和可接受证据组合标注。以冻结 corpus 为主评测环境，统一工具和预算，比较常规 ASR/frames 工作流、原生 AV agents、OneSearch-VL 的 AV 扩展和强 omni 模型。通过模态对照、阶段性 oracle 和规模/预算实验，明确主要瓶颈及不同系统的优势。根据 development 结果，进一步验证实例级 AV reranking 或局部观察选择的性能增量。

第一篇的核心交付是一个有实际用途的新评测对象、一套可复跑的 benchmark 与 baseline，以及解释搜索能力的实验分析。方法改进由瓶颈证据支持，不预先绑定某种模型架构或训练算法。

## 2. 研究动机、价值与问题

在技术演示比较、现场表演研究、动物/环境记录分析等场景中，用户不仅想知道“视频里有什么”，还想找到其他录制，定位符合条件的事件，整理不同来源共同支持的事实。例如，寻找采用相同操作步骤的演示，比较声音变化与动作顺序，或者检索另一份记录补齐当前素材缺少的信息。

研究的关键是 **任务需要的具体证据能否被发现和使用**。相关视频可以包含错误的事件实例；正确录制可以包含多个相似时刻；模型也可能找到正确片段却没有组合成完整答案。最终答案分数之外，需要能区分这些情况的标注与测量。

| Research question | 要检验的能力 | 对应实验 |
|---|---|---|
| RQ1：声画联合线索能带来什么？ | 声音和画面是否帮助缩小来源、实例与片段候选 | 模态对照、任务信息路径标注 |
| RQ2：AV research 的瓶颈在哪里？ | 来源召回、定位、感知、证据组合各自的影响 | 来源/片段/事实 oracle 诊断 |
| RQ3：主动搜索与观察如何改善结果？ | 相同预算下，交互式检索/窗口选择相对固定工作流的收益 | Workflow vs Agent、预算曲线 |
| RQ4：如何利用具体事件信息增长性能？ | 实例级声画关系能否提高正确候选排序与下游回答 | 普通 vs 实例 reranking、关系与负例消融 |
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
3. **性能方案：** 提供强 AV baseline，根据实际瓶颈验证一个具体增量，例如实例级 AV reranking。

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
| B5 | OneSearch-VL + AV observer | 现有 controller 接入相同研究工具与 AV 结果 | 复用用户 codebase 的研究基线 |
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

先提供标准 prompted agent，不依赖重新训练。所有 baseline 输出同样的事实/证据格式，由独立 evaluator 执行。既有私有 reference 不进入 controller 的提示词。

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

## 7. 增量性能方案：实例级 AV reranking

核心假设：候选已经相关时，题目所需的局部声画关系与具体实例信息可以改善选择。提出一个小模块，接收问题、候选区间和实际 AV observations，评分并重排来源/片段，后续 controller 使用同样的工具完成研究。

### 7.1 两轮实验

**第一轮：相同 observation cache 的排序比较。** 普通 query-aware scorer 与实例 scorer 接收相同候选、AV observations 和 token 上限。实例 scorer 显式评估可听事件、可见动作/对象、局部先后/共现是否满足问题。先用同一 backbone 的 prompted scorer，观察候选排序与下游回答变化。该实验作为 controlled-ranking diagnostic；如果共同观察池超出主赛道 episode 限额，结果进入诊断表。进入端到端主表时，观察、scorer 与后续回答的全部访问和生成成本需落在同一预算内。

**第二轮：按瓶颈选择训练或主动检查。** 如果排序信号明确，构造自然实例 hard negatives，采用标准 pairwise ranking/SFT。负例包括相似画面但声音不同、相似声音但动作不同、同一录制另一相似实例。若失败主要因为窗口没有覆盖事件，则优先实验候选差异驱动的下一窗口选择。两种升级分别评测，不同时加入多个模块后只报总收益。

标准 ranking loss 可采用：

\[
\mathcal L_{rank}=-\log\sigma\big(s_\theta(q,o^+)-s_\theta(q,o^-)\big),
\]

这里拟研究的是实例数据/表示对 AV search 的作用，不把该标准 loss 表述为新优化算法。训练正负数量、来源和更新预算与普通负例对照相同。

### 7.2 关键消融

- 普通 scorer vs 实例 scorer：同一信息、同一 backbone，隔离评分方式。
- 普通 negatives vs 实例 hard negatives：同样样本数和训练预算，隔离数据作用。
- 保留事件词、移除局部关系/顺序：检查是否超过关键词匹配。
- 相同总预算的固定观察 vs 自适应窗口：检查行动选择的收益。
- 候选内 oracle 与全任务结果：区分排序可解决的错误和初始召回缺失。

Benchmark 的正式任务由自然需求和预设构建标准确定，所有方法均可参加。Reranker 是候选改进，最终是否成为论文方法段落由 development 证据决定。

## 8. Evaluation：主指标与评分规则

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

## 9. Analysis：论文要回答的实证问题

| 分析 | 操作 | 测量 | 能回答什么 |
|---|---|---|---|
| A1 模态互补 | question/text、audio、video、joint 的对应输入/索引 | Source/span、Answer、EBS | 声画联合信息在哪些任务有用 |
| A2 阶段瓶颈 | 正常运行；给来源；给来源+区间；给人工事实 | Answer/fact coverage；可引用媒体设置另报 EBS | 搜索、定位/感知、组合可能影响多少 |
| A3 交互策略 | 固定 workflow 与 generic agent，同预算 | 成功率、有效读取比例、调用轨迹 | 主动检索/观察的实际收益 |
| A4 候选规模 | 同题同有效来源，增加随机/自然近似干扰 | 性能—corpus size 曲线 | 检索系统对规模与相似候选的敏感性 |
| A5 实例难度 | 自然重复次数、事件间距、clip 长度分层 | 实例命中、定位、回答 | 是否只识别类别而未定位具体事件 |
| A6 多源组合 | 按必要来源数、事实数和证据包形态分层 | Package recall、fact coverage、all-correct | 来源已找到后是否完成组合 |
| A7 预算与效率 | Small/Medium/Large 与早停行为 | Answer/EBS—cost 曲线 | 更多预算改善哪一环节 |
| A8 改进归因 | 第 7 节 scorer/negative/relation/窗口消融 | 排名与下游成功、matched cost | 性能增长来自什么增量 |

模态结果同时按人工标注的 complementary、redundant、audio-dominant、video-dominant 信息路径分解。画面内字幕属于实际视觉信息，可保留并标记；移除原始 waveform 而仍提供 ASR snippet 不称为无音频实验。

Oracle 干预分别定义：给来源只给正确 ID，不给事件区间/答案；给区间提供实际媒体，不给人工事实；给事实提供核验过的观察项。Oracle 主分析用 Answer Accuracy/necessary-fact coverage；EBS 只在提供可引用来源且实际读取媒体的设置另报，fact oracle 不按自主搜索 EBS 计分。它们改变可用信息，是阶段诊断参考，分数差值不能直接当成可相加的因果误差比例，也不混入主榜。

增加 corpus 干扰时保持 gold 不变，随机和近似候选分别报告；初始索引 recipe 与预算相同。自然难度分层给出家族数和域分布，不能只凭“重复多的题更难”推导因果结论。

逐题错误分析使用简洁分类：初始来源未召回、错误实例/区间、AV 观察错误、外部事实缺失、组合不完整、执行/预算失败。多种错误可以共存；先由日志与 reference 自动建议，再在人审子集校验分类。错位、缺失和转载混淆作为辅助切片。

## 10. 论文实验表与图

预先规划下面的证据结构，数值由真实实验填入：

1. **主表：** 各系统 Answer Accuracy、EBS、package-source recall、instance/moment 指标和成本，统一主预算。
2. **任务族表：** 四类任务及跨原始 AV/AV+text 子集表现，展示目标能力。
3. **模态与阶段表：** 单模态/联合信息、来源/区间/事实 oracle，解释能力瓶颈。
4. **改进表：** 普通与实例 reranking、数据/关系/窗口消融，固定预算。
5. **图：** benchmark 能力链与自然案例；accuracy/EBS–cost；corpus size/实例密度变化。

Dataset 统计表报告题数、独立家族、来源数、媒体长度、领域、语言、必要来源/事实数、标注一致性。附录给工具、数据卡、judge、成本规则与更多案例。主文围绕任务价值和实证发现组织，不按工程修复过程展开。

## 11. 实现工作包与交付

Benchmark 测量、运行环境和训练轨迹合成分别管理，在同一 repo 上共享来源与媒体读取基础。

| 工作包 | 主要交付 | 依赖 |
|---|---|---|
| WP1：任务/reference | 16 题完整包、区间/事实/证据包、来源家族 | 自然材料与实际审核 |
| WP2：搜索环境 | 官方索引、共享 corpus、语义 inspect_av 与完整 episode 日志 | WP1 的素材；AV runtime |
| WP3：独立评测 | 来源/片段/package/答案评分、judge 校准、结果聚合 | WP1 标注与 WP2 输出 |
| WP4：核心 baselines | B0–B5 的逐题运行，感知/索引/预算一致性 | WP2–WP3 |
| WP5：正式数据 | 80 题验证；独立家族的正式 test；数据卡 | 标注成本、协议与任务需求验证 |
| WP6：分析与改进 | A1–A8、强模型参照、候选性能方案 | development 成绩与正式 frozen test |

复用 `research/av_data` 的来源/媒体工具、`research/av_search` 的本地搜索和 `research/av_agent` 的 observer/controller 接口。新增 benchmark task/reference/evaluator/episode runner 模块，文件布局以现有代码接口审查后确定。线上推理 actor 只访问允许的 corpus/索引，reference evaluator 独立运行。

原型完成标准是 16 题可以从共享搜索接口开始，提交来源/区间与答案，由独立评分器产出逐题成绩；人工审核记录和不能成立的候选题同时保留。构建验证阶段再冻结任务标准、赛道、预算、指标和正式 test split。

## 12. 时间、资源与扩量安排

建议安排在标注人员和算力可用的前提下：第 1 周完成 16 题包与首条搜索评分闭环；第 2–3 周推进 80 题、核心 baseline 和索引/感知校准；第 4–6 周并行建设正式数据与运行矩阵；第 7–8 周完成分析、候选改进、论文与发布材料。此为依赖资源的研究排期，正式规模在 80 题后校准。

用原型测量预算：

\[
Cost_{eval}\approx N_{tasks}\times N_{systems}\times N_{repeats}
\times \overline{Cost_{episode}},
\]

再加入索引预处理、感知缓存、judge、人审与可选训练。按任务族测平均和长尾 episode 成本，预估 disk/GPU/API/审核工时，登记正式 ceiling 后扩大。Baseline 越多不意味着每个都执行全部分析矩阵：先完成核心主表，再按 RQ 选择相关系统运行配对分析。

既有诊断预算累计 727/2100 Job 秒，余 1373 秒，适合有界原型感知准备；正式 benchmark 数百题、完整 baselines、训练和强模型服务不计入该小额余额。单个后续 observation bootstrap 最多 700 秒 deadline/一 GPU，setup 时间继续从历史总额扣减。

当前没有 Serper 也能构建主赛道；闭源服务按实际配置启用。权重、原媒体和 private test reference 留在远端受控目录；数据公开时按许可与 test 答案策略发布代码、任务/语料、评测说明和可复跑 recipes。

## 13. 当前已有基础与下一步

截至 2026-10-10 的代码/记录检查：

- 发布 OneSearch-VL 8B 完成自定义图像/视频的 CPU 最小运行；未测完整官方论文 benchmark。
- 主开发迁至 Coder 3；Qwen2.5-Omni-7B 在 Kjob H20Z 真实运行，完成 observer 输入网格与输出协议修复；本轮 21 次 generation 和 observer validation 71/71 有记录。
- 已有 10 条自然 AV 素材和相关网页、可回放的本地搜索/读取/裁切基础。
- 16 行 performance-oriented 候选已设计，reference 尚未填齐；没有正式 benchmark、端到端 baseline 成绩或新方法增量结果。
- `inspect_av` 尚未完成与语义 observer 的完整 episode 集成，`compare_sources` 主要是 metadata 比较；目前的 AV backend 支持范围是短片段 smoke。
- 已推送性能计划提交 `852f21edcbeb9b57673a6410da3179ac0e3fe7e3`；本 proposal 是新增研究文档。

下一份实质交付是 **16 个可评分任务 + 搜索环境 + 核心 baseline 逐题结果**。这一步同时验证任务是否自然成立、指标是否有效，以及主要瓶颈，再据此确定正式规模与性能方法。

## 14. 最终交付与两篇论文的关系

第一篇交付 AV-ResearchBench 数据/环境与 evaluator、强 baseline recipes、来源/片段/答案成绩、模态与阶段分析，以及有结果支持的增量方案。它建立“能否完成 AV evidence search 与研究”的评测和性能证据。

第二篇再研究搜索结果如何支持生成：检索证据怎样进入生成条件，怎样影响素材选择、生成结果与用户目标。第二篇使用第一篇的能力基础，但具有独立 generation 任务、评测和实验，不把搜索成功直接等同于生成质量。
