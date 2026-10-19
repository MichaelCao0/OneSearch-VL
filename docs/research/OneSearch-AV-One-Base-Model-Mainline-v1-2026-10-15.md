# OneSearch-AV 主线收敛与投稿执行方案 v1

更新：2026-10-15T08:13:25.738228+00:00。本文件落实用户“调一个基座模型，主要 follow OneSearch-VL”，并作为当前研究入口。实际开发均在 coder ssh 3 的 /home/jovyan/OneSearch-VL；本地保存报告和审核页面。

## 1. 研究主张与贡献

**用真实音视频线索构造可搜索问题，再用真实工具轨迹训练同一个 Omni 模型，提高外部知识搜索问答成功率。**

沿 OneSearch-VL 的基座→teacher trajectories→SFT→search QA evaluation 主线展开。三项相互支撑的交付为 AV-grounded 数据引擎、独立 AVSearch-Bench、训练后的 OneSearch-AV。数据管线贡献由构造消融、人工质量和下游性能验证；benchmark 贡献由新的模态/检索诊断维度与强模型比较验证。第一篇保持直接 Search QA 任务，不强制多源报告，Search for Generation 留后续。

## 2. 一个基座与原训练主干

主模型固定 **Qwen/Qwen2.5-Omni-7B**，revision ae9e1690543ffd5c0221dc27f79834d0294cba00。原生 AV 进入同一个 Thinker；初始片段和 inspect_av 的后续媒体均保留真实音轨。复用 onesearch_vl/onesearch_infer/pipeline.py 与原 SFT/src/train.py；omni_runner.py 负责原生模型接口，不增加文字 controller 作为主架构。

原版发布 recipe SFT/examples/onesearch/qwen3_vl_8b_joint_110k.yaml 使用 Qwen3-VL-8B-Instruct、full SFT、冻结视觉塔/projector、110K joint data、batch128、LR1e-5、8epochs、32K cutoff、2FPS。AV 版采用相同搜索轨迹监督及冻结感知/训练语言的原则，因模型与数据量变化单独选择训练预算。

主性能路线为 **full Thinker SFT**：冻结 audio_tower、visual、projector，训练语言推理部分，Talker 不参与。既有 LoRA 作为已经验证的 pilot 和低成本对照。先对 full 方案做实际权重/有限梯度/保存重载/显存 smoke，再按 dev 成功率推进，失败时分析资源原因。

## 3. 最小核心实验

| 条件 | 目的 | 执行状态 |
|---|---|---|
| Direct QA base | 内部知识与感知参照 | 需与新独立 benchmark 对齐 |
| Strong prompted-search base | 充分提示后的搜索能力 | 旧 base 协议差，须补 dev prompt 调整 |
| Same-question answer-only SFT | 学会答题与学会搜索的差异 | 同一500题候选配置已写，尚未训练 |
| Trajectory SFT | 学习真实搜索决策的增益 | 已有500题 LoRA pilot；2K full Thinker 候选已写 |
| 相同权重 AV/静音/audio-only/video+ASR/no-search | 诊断声音、视觉和工具贡献 | 已有少量 AV/静音 silver，其余待真实重跑 |

测试时给 answer-only 与 trajectory 相同搜索工具、prompt 和推理预算；另列 no-search 条件。训练时固定题目/初始化，并报告 supervised/context tokens 与 GPU 成本。旧 base 低分不能替代强提示基线。官方 OneSearch-VL-8B、video+ASR 搜索及闭源 AV 模型作为外部参照；本轮只调一个 Omni 主基座。

候选配置：records/2026-10-15-onesearch-mainline/answer-only-500.candidate.yaml 与 full-thinker-2k.candidate.yaml。前者与实际500轨迹 LoRA 保持rank8、LR1e-4、2epochs、250updates；后者初始 LR1e-5、2epochs、4GPU×microbatch1×accumulation4=batch16、ZeRO2，dev 可比较 LR2e-5。GPU smoke 后按现场容量分配。2K路径为未冻结占位，候选文件不代表可直接正式执行。两文件通过原trainer字段/schema及非device参数构造检查；GPU TrainingArguments、真实权重与训练尚未验证。

## 4. 数据管线与规模

公开真实 AV/发布者事实→录制家族分组→先预留benchmark来源→原生AV出题→匿名线索任务→teacher实际搜索/网页读取/片段检查→silver语义验收→paired trajectory/answer-only→原trainer CPU媒体/label检查→冻结训练集。

完整目标为 **20K不同训练问题 + 500 train-validation；200 benchmark dev + 1K test**。先完成2K pilot；后续按500/2K/5K/10K/20K从同一基座训练。answer-only是同题对照，不重复计数。最终论文填写实际release数量。

现场合格池为 **904 / 904 retained**，2K生产进程live=True，新cohort已采得164条可用AV录制；这些是素材，尚不是新增合格问题。benchmark producer live=True，正等待cohort完成预留，之后消费最多160个未尝试录制。既有19题/14录制是dev候选，human gold=0。

## 5. Benchmark与质量建设

主搜索集要求从片段获得可搜索线索，回答需要检索的具体外部事实；熟悉物种/人物的直接识别另列控制。隐藏时间地点问题仍须有从片段到正确来源的可辨别路径。类别依据真实内容核实，采集query的bird/music等只保留为来源标签。

24例真实AV自动抽查：21完成、3失败不重试；一致规则得到16保留建议、5感知控制、3unknown。其中train10例为9保留/1unknown，benchmark14录制为7保留/5控制/2unknown。模态角色为模型建议，尚未代替静音/ASR实际干预或人工判断。

已启动独立100录制训练质量抽查，冻结当前合格池、排除既有24录制的付费抽查输入，从884可用题中按采集query类别确定性选择，不按student表现选择。先渲染原生输入和60–90秒来源上下文，再以4worker、最多100API请求评估实际类别、声音/画面线索、答案文本支持与搜索必要性。该分层样本不是均匀抽样，不用它直接估总体精度。结果只作silver建议；原训练池与人审选择均不自动改写。

Benchmark按100题构建pilot→200dev/1Ktest推进；正式gold核对答案、来源、关键区间和模态作用。来源隔离在训练前预留，按原始录制、媒体hash和真实training observations检查；不按模型成功率筛测试题。

## 6. 当前性能证据与接下来执行

已有500轨迹 LoRA 完成250真实updates；silver heldout39可判题成功23（59.0%），静音21；dev15可判题成功8，静音13。它支持继续做训练，但声音优势尚不稳定。主表接下来补强prompted base和same-question answer-only，然后在冻结2K上训练full Thinker与对应对照，每次训练结束立即同题评测。

近期顺序：持续CPU扩到2K→100例真实质量结果与benchmark首批人审→修正下一版构造→冻结2K和matched controls→GPU smoke→full SFT→任务成功率/模态/搜索错误分析。扩到20K和1K test时同步保存采集产率、保留率、API调用/费用、tokens、GPU小时与逐题结果。

## 7. 论文结构与尚缺证据

Introduction以一个声画线索→搜索来源→外部事实的完整案例引出任务；Task/Benchmark介绍新评价维度；Data Construction介绍模态线索驱动构造和真实轨迹；Training介绍沿用原训练主干的OneSearch-AV；Experiments回答训练收益、声音作用、数据构造与规模三个核心问题；Analysis定位线索识别/query/来源选择/媒体检查/答案提取的失败。

按benchmark五项框架审查：任务与评价框架已有设计，数据管线已有真实执行，模型已有小规模训练；提交前仍需完成人工gold、核实直接近邻比较、强prompt/answer-only主表、声音及构造消融和正式规模统计。当前不是已完成投稿实验。

## 8. 入口和证据

- [投稿总计划 v5](OneSearch-AV-Paper-Master-Plan-v5-2026-10-15.md)
- [数据管线 v2](OneSearch-AV-Dataset-Pipeline-v2-2026-10-15.md)
- [Benchmark blueprint v2](OneSearch-AV-Benchmark-Blueprint-v2-2026-10-15.md)
- [原生AV质量抽查](OneSearch-AV-Native-Content-Quality-Audit-v1-2026-10-15.md)
- [研究总账](OneSearch-AV-Research-Ledger-2026-10-15.md)
- records/2026-10-15-onesearch-mainline/：训练候选、原trainer参数验证、当前进程身份与计数snapshot。

本次未交新GPU任务；没有改写运行中的production源码freeze，核对结果unchanged=true。旧文档、付费失败记录与私有数据保留。
