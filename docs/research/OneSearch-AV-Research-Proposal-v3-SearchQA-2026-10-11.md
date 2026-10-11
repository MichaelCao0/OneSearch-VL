# OneSearch-AV：面向音视频线索的搜索问答

日期：2026-10-11。按用户确认的任务范围更新；本文件替代此前 proposal 中把跨来源比较与研究报告设为必选任务的部分。原版文件保留作为历史记录。

## 研究目标与任务

沿用 OneSearch-VL 的范式：给定用户问题及可选的音视频线索，agent 按需搜索、检查媒体、读取网页，回答问题。来源数量由题目决定；单个来源足以回答时不要求增加来源。Search for Generation 是后续独立工作。

研究问题是：通过音视频搜索轨迹训练，能否让 agent 更有效地利用声音、动作与时序线索，提高需要外部信息的问答正确率？核心结果应是训练后性能增长，而非更多审核流程。

原版搜索工具依赖 Serper，目前未配置。第一轮使用冻结语料检索完成可重复 pilot，随后增加真实外部检索适配；冻结语料结果不能直接视为开放互联网搜索表现。

## Benchmark

围绕自然搜索问答建立题库，而非强制比较和报告模板。包含音频线索检索、动作与时序线索检索、声画联合识别、媒体线索与外部知识问答。纯描述题用于感知诊断，不作为核心外部搜索能力的主要证据。

每题保存问题、初始媒体、答案与可接受变体、支持来源与时间区间，以及是否必须检索外部信息。答案和私有标注不进入 actor 输入。按原始录制、事件和近重复家族划分 train/dev/test。

主指标是 Answer Accuracy。辅助报告 source recall、有效媒体检查率、工具调用数、媒体检查秒数及失败类型。时间定位只在题目要求时计分；跨来源能力作为可选分析维度。

现有 16 个 development cases 应重新筛选和改题，属于开发候选；没有经审核的答案之前不报告正式 accuracy。现有机械、音乐、降雨三组素材用于工程 pilot；后续扩展到更依赖外部知识的识别题，避免题目全部退化为标题匹配或描述视频。

## 数据与训练

数据流：来源采集 → 内容与线索整理 → 自然问题及私有答案候选 → 真实 agent rollout → 质量筛选 → ShareGPT 导出 → SFT。

Benchmark gold 严格人审；训练探索可以使用明确标为 silver 的数据。缺少 benchmark gold 不阻止真实 rollout。训练来源与 reserved dev/test 的字节交集必须先检查；尚未完成录制家族审核的数据仅用于探索，不进入正式泛化结论。

真实模型决定查询、工具和回答；脚本连接性 fixture 不算 teacher 轨迹。固定观察回放必须来自真实模型运行，精确绑定来源和时间区间。回放与动态 observer 分别记录。语义未核验的完整轨迹称为候选，不因为 JSON 正确或引用区间有效就称为正确训练样本。

首轮：发布 OneSearch-VL-8B controller + 固定 Qwen2.5-Omni-7B observer；LoRA 更新 controller。原生音频模型作为后续对照。先跑 3–12 个真实候选 episode，定位失败；形成可筛选的 seed 数据后做 optimizer/checkpoint smoke，再扩展到 128–256 条、2K pilot。这些是实验目标，不是已有数据量。

## Baselines 与方法

主比较：OneSearch-AV-Prompted、Answer-only SFT、AV trajectory SFT。固定 controller 起点、observer、搜索索引、工具和预算。增加无搜索、无音频观察、固定窗口检查对照，以确认收益来自哪些信息与策略。

第一阶段先建立经过训练的 OneSearch-AV baseline。方法候选是查询改写与按需媒体检查：根据音频/动作线索改写查询，根据检索和观察结果选择下一段检查位置。它们的论文贡献由配对实验与消融结果决定，不提前声称有效。

## 执行顺序与产物

1. 更新仓库范围约定和 prompt；把三组 12 个题目改为直接搜索问答，保留历史候选。
2. 在 coder ssh 3 使用 Pixi 和有界 GPU Job，真实运行发布 8B controller。第一轮复用已有真实 Omni 固定观察，采集模型自主工具轨迹；随后接动态检查。
3. 汇总逐题工具使用、回答和失败原因；用可查看的轨迹页审核候选。先修实际执行问题，再增加数据规模。
4. 筛选与导出 silver seed，验证 trainer labels，运行 LoRA optimizer/checkpoint smoke。
5. 扩展数据并做同预算训练前后比较；单独完成 benchmark 答案审核和来源家族划分。

工作地点：coder ssh 3，/home/jovyan/OneSearch-VL，分支 research/av-deepresearch。环境用 Pixi。原媒体、私有答案和权重保持在远端忽略目录；代码、测试、公开方案可提交到用户 fork。
