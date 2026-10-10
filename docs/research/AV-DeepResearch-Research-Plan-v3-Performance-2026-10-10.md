# Research Plan v3：以性能提升为主线的 Audio-Video Deep Research

日期：2026-10-10。开发位置：`coder ssh 3`，`/home/jovyan/OneSearch-VL`；用户 fork：`MichaelCao0/OneSearch-VL`，分支 `research/av-deepresearch`，环境继续由 Pixi 管理。第一篇研究多源音视频搜索、证据补齐和回答；Search for Generation 留给第二篇。

本版根据用户“可以做增量创新，主要为了增长性能”的要求调整 v2。方法和数据实验均是下一阶段方案，不能把本版设计当作已实现结果。v1/v2 和历史运行证据保留。

## 1. 研究判断与优先级调整

前期实现的输出格式、音视频时间网格、来源登记、引用回放和成本记录，为公平实验提供基础。它们可以支撑论文的工程描述，但不承担主要方法贡献。前一版以结构化后果预测、取证约束和多级验收为中心，方法复杂度走在性能证据前面。现在先实现一个能直接改善候选选择的模块，并检查其下游收益。

核心问题：**初始检索已经返回相关音视频时，怎样利用局部声画互补和事件关系，把题目需要的正确来源、正确事件实例和正确片段排到前面，从而提高固定预算下的回答成功率？**

推荐候选方法：**面向事件实例的 Audio-Visual reranking**。沿用现有 controller、检索和 observer，增加任务条件的候选评分。方法创新可以是已有组件上的具体增量；不要求新模型架构或新 RL 算法。需要用强对照证明收益来自该增量。

查询改写、追加检查和研究报告保留在流程中，但第一轮不同时引入完整后果预测、复杂 evidence graph、source dependence 模型和新 RL reward。v2 的 structured consequences vs same-label Q 研究降为备选，不再作为当前必须完成的主实验。

## 2. 已完成与尚未完成

以下为本次读取代码和运行报告后确认的状态。

| 层面 | 已发生的结果 | 当前边界 |
|---|---|---|
| OneSearch-VL 基础 | 发布 8B 的 CPU BF16 图像/视频最小运行；视频中实际调用 select_frame 后完成回答 | 自定义短任务 smoke；没有复现完整论文 benchmark，也没有在线搜索指标 |
| 研究环境 | 主开发已转到 Coder 3，Pixi 独立 AV runtime，Kjob 实际 H20Z 推理 | controller 与 AV observer 的完整搜索闭环仍未接通 |
| 数据与工具 | 10 条自然 AV 原件；三案候选的冻结探索包含 6 AV + 6 网页；本地文本搜索、读取与媒体裁切可回放 | 元数据和机械工具回放不是 AV 检索性能；自然任务 reference 尚待审核 |
| 本轮 observer 修复 | 五个 GPU 诊断阶段、30 次 backend attempts、21 次实际 generation；observer validation 71/71，独立复验通过 | 最终固定 6 输入里 3 个生成，3 个因音频 PTS 重叠拒绝；成功生成的 3 个结构通过，语义尚未验收 |
| 代码发布 | 已推送 `e8b704d16fce3ef0dc29d7f77a084a65a72f2c5b`，Coder HEAD 与 tracking 一致 | 这是 observer 修复提交，没有 reranker、训练或性能增量结果 |
| 16 development cases | 用户已选择规模，已做源素材 inventory 和计划 | 尚未完成 16 题数据包、reference、runner 或新 GPU 运行 |

本輪五个 Job 均已终态，无正在运行的该批 Job。终态 Failed 来自诊断 CLI 对预期输入拒绝返回非零；应同时看完整 attempts 和 summary。GPU 成本按 Job start 到 terminal 向上取整累计 **727/2100 秒**，剩余 **1373 秒**，包含历史成本，不能重置。

没有检索 Recall/MRR、正式 answer accuracy 或新方法收益的当前结果。人工语义审核记录仍为 0。可看到一个结构通过的输出重复报告了同一 jingle，这也说明格式通过并不等于内容正确。

## 3. 方法的最小计算流程

`问题／输入 AV 线索 → 初始搜索 → 候选来源与片段 → 固定预算 AV observation → 事件实例 reranking → 选择来源／片段 → 外部事实补齐 → OneSearch 回答或短研究报告`

1. 初始检索产生固定的候选池。输入可为问题及已有 AV 线索；先用本地冻结 corpus 做受控比较，后扩展到真实在线搜索。
2. 对候选提取可听事件、可见对象/动作和局部关系，包括先后、共现、动作邻接。观察来自真实素材，reference 只给评测器，不给 observer/controller。
3. 普通 baseline 判断“这个候选与问题是否相关”；候选方法进一步判断“它是否是问题要求的那个具体实例”。第一轮使用同一 backbone 的 prompted scorer，保存完整分数和排序；信号成立后再训练 LoRA 或小型 ranker。
4. 所有方法消费同一候选池和 observation cache。第一轮固定 observation 集，只比较排序，便于将排序收益与观察成本分开。候选差异驱动的追加 inspect/query 改写放到下一轮，单独消融，消耗同等行动预算。
5. 排序后的来源与片段进入相同的事实检索、证据补齐和报告流程。报告语言质量不是核心指标，优先检查答案和关键证据是否正确。

假设性例子：题目问“金属撞击声后操作者打开阀门的那次操作，相关技术说明如何解释该步骤”。初始检索可能找到多段蒸汽机操作；只看机器类别仍不足以选出题目所指片段。联合撞击声、开阀动作和局部顺序的评分，有机会先选对片段，再查对应技术材料。该例仅说明目标机制，不是已经标注的自然案例或当前模型成功案例。

## 4. 增量来自什么

候选贡献分成三个可测部分，不靠扩大术语来包装创新。

- **实例级声画评分。** 不只匹配实体名或类别词，还匹配题目所需的局部音频事件、视觉事件及关系。需验证其超过直接 joint-AV caption 和普通 query-aware reranking。
- **自然实例 hard negatives。** 正例和负例可能具有相似画面、相似声音或同类动作，但只有正例满足题目要求。负例包括同类画面/不同声学事件、同类声音/不同动作、同一视频另一重复实例。采用标准 pairwise ranking/contrastive loss 即可，训练目标本身不宣称首创。
- **对 Deep Research 的下游收益。** 正确候选前移应减少预算浪费，提高必需证据命中和最终答案准确率。仅排序指标上升而回答不变，论文主张应限制为检索模块收益。

机制重合必须继续核对：OneSearch-VL 已有 VGEG、grounding 和 evidence reward；Omni-DeepSearch 的正文已包含音频实体推断、候选视频粗查/细查、查询重试及检索反馈。因此“增加音频工具”“音频驱动 query rewrite”“新增 evidence graph/GRPO”均不足以单独承担方法贡献。上述实例 reranking 是候选研究点；还需补查 audiovisual retrieval、temporal grounding、hard-negative learning 和 active retrieval 近邻，当前不声称首创。

## 5. 16 个 development cases

目标是找性能信号和错误来源；这 16 题会用于选择方法，不能充当最终论文测试集。按来源/事件簇记录依赖，16 题不自动等于 16 个独立事件。

| 组别 | 数量 | 任务和主要信号 |
|---|---:|---|
| 音频补充视觉检索 | 4 | 相似画面候选中，声音能否帮助正确来源/片段进入前几名 |
| 视觉补充音频检索 | 4 | 相似声音候选中，可见对象/动作能否帮助正确来源/片段进入前几名 |
| 声画局部关系与重复实例 | 4 | 事件类别相同，先后/邻接或发生实例不同；是否选对具体一次 |
| 多源证据补齐 | 4 | 选对 AV 片段后，是否检索另一来源，完成两个以上目标项 |

错位、音轨缺失和信息不足作为这些任务的辅助诊断，不占主体。优先在自然、可解码、可裁决的数据上做实验；不要求先解决所有容器格式和边界情况。确实存在的执行失败仍进入固定任务分母并单列，不能事后删除来制造性能提升。

现有 10 条素材可作为起点，但重复实例和必要的多源路径没有确认。素材不足以支持某组时，按明确角色补充自然来源或先报告空缺；不用合成拼接假装真实多源事件。合成变体用于诊断，与自然任务分开报告。Reference 包含正确来源、有效事件区间、关键事实和可接受的证据组合。

当前 backend 的 joint-AV 单次 2 秒、CFR 和 PTS 条件是 smoke 边界。短窗 observation 或分别看两个来源，不等于完整检索成功。完整题目可需要多个窗口、检索和后续推理，要由 episode runner 执行和计量。

## 6. 强基线与消融

| 方法 | 作用 |
|---|---|
| OneSearch-VL 的现有视觉/文本能力 | 量化增加 AV 系统的整体收益；因观察能力不同，与其他对照分开解释 |
| OneSearch + 同一 AV observer，自由选择候选 | 排除“只要增加音频就提高”的替代解释 |
| 同一 observations 的普通 query-aware AV reranker | 排除一般 reranking 和额外计算的作用，是最重要的方法对照 |
| 实例 reranker：普通 negatives vs 实例 hard negatives | 正负数量、训练样本和优化预算相同，检验数据增量 |
| 实例 reranker：保留事件词但去掉局部关系 | 检验收益是否来自声画关系，而非更多关键词 |
| Oracle candidate selection | 只作上限诊断，区分初始召回、排序、观察和后续推理瓶颈 |

机制比较共享初始候选池、候选数量、observer 模型/revision、可见信息、观察窗口和缓存规则。固定 inspected AV seconds、controller tokens 和工具调用上限，同时报告实际 GPU/wall cost。主动追加检查的比较再单列同预算实验；不能给 proposed 更多观察后称排序更好。

候选池内 ranking 与初始检索分开报告：gold 未召回时，reranker 没有机会找回。报告全题分母指标和 gold 在候选池内的条件指标，避免把任务召回错误藏到排序结果之外。

## 7. 指标与判断顺序

1. **检索与定位：** source Recall@k、MRR/nDCG、正确事件实例比例；时间定位按人工接受区间/预设 IoU 规则计分，阈值在结果前固定。
2. **下游答案：** objective answer accuracy；多目标任务报告关键事实覆盖和 all-correct。短报告检查必要事实与支持证据，流畅度只作辅助。
3. **预算效果：** 固定预算成功率，以及 accuracy–cost 曲线。统一候选数、AV 秒数和 token 规则，执行失败保留。
4. **辅助可靠性：** 无依据结论、错误声画对应、缺失输入表现，作为错误分析。

16 题先报逐题结果、错误类型和 paired difference，不能凭几题提升做统计结论。之后扩到来源隔离的正式开发/测试集合，按来源/事件簇计算区间。最终集合冻结后再评测，不能继续用它选 prompt 或挖负例。

## 8. 数据管线和训练顺序

`公开自然 AV/相关文本 → 来源登记与预处理 → 候选来源/片段池 → 问题与 reference 审核 → 自然 hard negatives → 相同 observer observations → baseline 排序/episode traces → ranking 或 action SFT → 冻结测试`

数据包保留来源、事件簇和有效时间区间。合成诊断单列；音频静音消融同时屏蔽音频派生信息。候选排序输入不暴露 private reference、负例类型或只有正确来源才有的标注字段。

先验证 prompted scorer 的可用信号及 oracle 上限，再采用标准 ranking/SFT。没有增量时检查表示、正负例可分性和初始召回，而不直接扩大数据或加 RL。若普通 reranker 已达到相同收益，就用结果缩小方法主张或另选明确错误类型。RL 只在监督训练后仍有明确的序列行动瓶颈时考虑。

## 9. 接下来执行什么

1. 完成按四组组织的 16 题候选包和来源/事件 inventory，标出哪些只是 observer probe，哪些已经具备可判定的搜索路径。
2. 接通真实 AV observations 与 episode runner 的 `inspect_av`；第一轮复用相同 observation cache，减少重复 GPU 开销。现有 `inspect_av` 仍是裁切，`compare_sources` 仍主要返回 metadata，不能把它们写成语义推理已完成。
3. 实现普通 query-aware scorer 与实例 scorer 的相同接口，保存每题完整候选排名；独立评测器读取 reference。
4. 跑 16 题小实验，先交付逐题的来源/片段排名、正确实例与答案结果，然后决定扩大数据和训练哪一项。

剩余已登记 GPU 预算为 1373 秒。下一次 observation bootstrap 最多一项 700 秒 deadline、单 GPU 的有界 Job，完整 setup 时间扣入原预算。它用于构建共享真实 observation 缓存，未必能完成所有方法的全流程比较。所有方法的完整训练/正式测试需另行登记实际预算，不能从“16 题 observation 跑过”推导 benchmark 已完成。本次进度回答时尚未提交该新 Job。

当前最应该交付的下一张表，是 **相同预算下 baseline 与 proposed 在正确来源、正确片段和最终答案上的逐题差异**。基础可靠性检查保持够用、可追溯，新增研究工作围绕这张性能表推进。
