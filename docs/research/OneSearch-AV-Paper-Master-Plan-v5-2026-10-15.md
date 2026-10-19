# OneSearch-AV 投稿主计划 v5：数据管线、Benchmark 与原生 Omni 搜索训练

日期：2026-10-15。主开发：coder ssh 3，/home/jovyan/OneSearch-VL，Pixi。本文收敛此前计划，作为下一阶段研究入口；历史方案、失败运行和原始结果保留。文中的规模、排期与实验矩阵是目标，当前完成量以附带 evidence snapshot 为准。

## 1. 论文主线与基座

**让原生音视频模型从片段中发现可搜索的线索，通过真实搜索和媒体检查取得外部知识，并正确回答问题。**

主线 follow OneSearch-VL：基座模型 → teacher 搜索轨迹 → SFT → 独立搜索问答 benchmark。第一篇围绕 Search QA；不强制跨来源比较或研究报告，Search for Generation 留作第二篇。

主基座为已经完成真实训练与评测的 **Qwen2.5-Omni-7B**。它原生接收视频与录制音轨，复用 OneSearch-VL agent loop、工具协议和原 SFT/src/train.py，先采用 Thinker LoRA。不同数据规模从同一 pinned base 初始化，不从上一个规模的 adapter 继续训练后冒充同条件学习曲线。模块化 observer/controller 保留历史对照，不作为主架构。是否换更新 Omni 基座在主结果形成后决定，不先铺开模型库。

候选论文标题：**OneSearch-AV: Learning Audio-Visual Search from Grounded Tool Trajectories**。这是工作标题，尚不主张“首次”；最终与 OneSearch-VL、OmniGAIA/OmniAtlas、Omni-DeepSearch 等直接近邻核对任务、数据构造、训练与评测边界。

## 2. 找回最初的 target

| 时间与文档 | 当时目标 | 本次处理 |
|---|---|---|
| 2026-10-10 Overall Proposal v2-Train | 10–20K 成功训练轨迹；约 400 benchmark，80 dev/320 test；2K pilot | 保留训练阶段逻辑；跨源研究报告任务已被用户后续修正 |
| 2026-10-11 SearchQA v3 | 沿用 OneSearch-VL 直接搜索问答；先 128–256，再 2K | 保留任务定义；旧模块化架构被原生 Omni 替代 |
| 2026-10-12 Teacher-SFT Plan v1 | 15K–20K 成功轨迹；约 1K 正式测试问题；1K/5K/10K/15K/20K 曲线 | 作为正式规模依据 |
| 2026-10-13 2K Data Execution | 2,000 个不同问题、自动审核保留并通过原 trainer CPU 检查的轨迹 | 只是 pilot，不是论文最终数据规模 |
| 本文 v5 | **20K train + 500 train-validation；1K benchmark test + 200 benchmark dev** | 明确区分训练、调参和正式测试；200 benchmark dev、500 train-validation 是本次新增规划 |

10 月 12 日旧计划还估过 10K–12K 原始素材、24K–36K rollout 候选。它们是资源估算，不是已经取得的素材或独立题目。新方案优先扩大来源多样性，按实际合格产率重新估算需要采集的录制。

## 3. 三个论文贡献与各自需要的证据

| 贡献 | 要交付什么 | 论文中如何证实 |
|---|---|---|
| AV-grounded trajectory data pipeline | 从真实声画线索和可核实外部事实构造问题，执行真实搜索，导出可学习的多轮 AV 轨迹 | 数据规模、多样性、人工抽查、构造/过滤消融、单位成本的下游收益 |
| 独立 AV Search benchmark | 1,000 test + 200 dev；自然 AV 输入、答案、来源、关键区间、模态依赖标签与冻结搜索环境 | 人工双审、替代来源裁决、单模态与无搜索对照、强 baseline 和稳定评分 |
| Trained OneSearch-AV | 同一 Omni 基座经轨迹 SFT 学会提取线索、构造查询、选择来源、检查媒体与回答 | 对充分调好的 prompted base、同题 answer-only、VL/ASR 方案的任务成功率和成本提升 |

数据管线的学术价值在于 **把声画线索变成需要外部知识的问题，并让 student 学到实际有用的搜索行动**。请求重试、哈希和格式检查支撑复现；主贡献应由数据及训练收益证明。原生 AV 接口接通、loss 下降、更多工具调用都不是论文主结果。

计划增加的关键环节是 **模态依赖分层与同类干扰来源**：区分声音关键、声画联合、视觉为主和文本/语音控制问题，在相同题数与训练预算下比较自然随机采样和分层采样。这个环节尚未完成，不能提前写成已有方法收益。

## 4. 当前事实：已有学习收益，尚未形成正式 benchmark

本轮现场核对：869 个去重后自动审核保留问题，其中 **855 个通过原 trainer CPU 媒体/标签检查，14 个待查**。这些是 silver training data，人工 benchmark gold 仍为 0。数据生产旧进程已退出；state.json 的 running 为陈旧状态，退出原因未确认。当前没有正在扩容的生产进程。

clean500 来自当时 629 个合格问题的冻结池，含 500 问题、494 原始录制、495 自动候选家族；已排除固定评测的已知参考 URL、媒体哈希和候选来源家族。自动家族不是完整的人审近重复保证。当前 profile 的 family 键数 495 高于 recording 数 494，统计口径尚待归一；正式数据卡不将 495 写为已确认独立家族数。

训练：原 /home/jovyan/OneSearch-VL/SFT/src/train.py；Thinker LoRA rank 8、alpha 16、dropout 0.05；LR 1e-4、effective batch 4、2 epochs、250 次实际 optimizer updates；平均 train loss 0.8977。权重、optimizer/scheduler/RNG 与重载检查通过。

评测已完成六组各 56 个 rollout；匿名配对自动 judge 成功判定 54/56 问题。当前使用固定 prompt、greedy、8 turn、每 turn 512 输出 token。这是 pilot silver，不是最终 gold benchmark。

| 同题条件 | heldout：成功/已判定 39 题 | dev：成功/已判定 15 题 |
|---|---:|---:|
| 未训练 base，完整 AV | 0/39，0.0% | 0/15，0.0% |
| 历史 old172，完整 AV | 15/39，38.5% | 9/15，60.0% |
| clean500，完整 AV | **23/39，59.0%** | 8/15，53.3% |
| clean500，静音视频 | 21/39，53.8% | **13/15，86.7%** |

heldout 计划 40 题，其中 1 题 judge unknown；dev 计划 16 题，也有 1 题 unknown。clean500 heldout 的全 40 题成功率完成上下界为 57.5%–60.0%，这不是置信区间。heldout AV 比静音高 5.1 pp，家族 bootstrap 95% 区间 [-8.1, 19.4] pp；dev 上 AV 比静音低 33.3 pp。因此目前可以说 **轨迹训练产生了搜索问答收益**，尚不能说 **声音带来稳定收益**。

old172 有 18/40 heldout 参考页面历史暴露，只作历史参照，不作为干净的 scaling control。base 目前几乎不会按搜索协议行动，需要补充分调好的 prompted baseline；现有结果不能替代与强提示、同题 answer-only 的正式比较。500 条 answer-only 正式对照尚未训练。主分析保留 heldout/dev 的差异，不用合并数字掩盖负向音频结果。

## 5. 最终数据交付与计数

| 交付物 | 目标 | 计数单位与约束 |
|---|---:|---|
| OneSearch-AV-Train | **20,000** | 不同问题的合格成功轨迹；目标至少 15,000 个不同录制家族，每录制最多 2 个不同事实问题 |
| Train-validation | 500 | 独立来源家族，供数据/训练配置选择，不计入 20K |
| Paired answer-only | 与 train 同题 20,000 行 | 初始问题 + 原生 AV + 最终答案；是对照导出，不新增 20K 问题 |
| AVSearch-Bench development | 200 | 人工核实，供 prompt、预算、judge 校准 |
| AVSearch-Bench test | **1,000** | 独立人工核实；与 train、train-validation、benchmark dev 家族隔离 |
| Benchmark 全集 | 1,200 | 目标至少 1,000 个不同录制家族；现有 56 题不自动升级为 gold |
| Frozen retrieval snapshot | 暂按至少 10K 媒体来源及配套网页规划 | 目标及同类干扰项进入同一可搜索库；在 100 题 pilot 后按搜索难度和许可调整 |

15K 是旧计划下沿和生产里程碑，20K 是本次按原目标上沿设定的完整交付。最终论文按实际 release 统计，不预写 20K 已完成。独立问题、候选 rollout、成功轨迹、录制、转码、answer-only 行数分别列出；多候选可用于路径分析，不增加独立问题数。

当前距 2K CPU 合格目标还差 **1,145**，距 20K 还差 **19,145**。14 条待 CPU 检查先做无 API 的核验。继续扩容前先预留 benchmark 来源，再从未尝试录制建立新队列。

## 6. 研究问题与实验矩阵

**RQ1：真实工具轨迹监督能否提高搜索问答成功率？**

主表：原 Omni direct QA、充分调好的 Omni prompted search、同题 answer-only SFT、trajectory SFT。主基座统一 Qwen2.5-Omni-7B，工具、索引、媒体预处理和测试预算一致。Prompt 在 benchmark dev 上选择并冻结，示例仅来自独立 train；不能携带测试答案或来源。

**RQ2：收益来自音视频搜索能力的哪部分？**

同题实际重跑 AV、静音视频、audio-only、video + ASR、no-search。静音必须覆盖初始 AV 与后续 inspect_av 返回，删除派生声学摘要；ASR 仅是转写，不包含非语言音效描述。分别报告语音、非语音与声画联合子集。合成换音只作干预分析，不混入自然主测试。

**RQ3：数据规模与数据构造如何影响性能？**

嵌套 500/2K/5K/10K/20K；15K 保留为生产里程碑或补充曲线点。固定成员、录制家族分层、基座和 recipe；记录真实 context/supervised tokens、updates、GPU 小时。比较随机采样 vs 模态分层、publisher-text-only 出题 vs 原生 AV 出题、无同类干扰 vs 有干扰。数据构造 ablation 先在 2K 做，避免把所有组合扩到 20K。

**RQ4：相对原 OneSearch-VL 和常见替代方案如何？**

官方 OneSearch-VL-8B 作为视觉搜索参照；其不能原生听音，音频缺失须明确。增加 video + ASR 模块化搜索；强闭源 AV teacher 在同环境下作为能力参照。不同规模和训练历史单列，不能将跨模型差异全归因于训练。

**RQ5：在哪里失败，怎样提高性能？**

按线索识别 → query → 来源选择 → 媒体检查 → 事实提取 → final 六阶段定位失败；报告来源命中但回答错误、重复搜索、年份/地点混淆、无依据附加陈述。用 dev 的错误分布补充对应训练数据，再以新的冻结 checkpoint 验证，不根据 test 成败筛题。

主方法最终至少 3 个训练 seed；核心 matched-data answer-only 也以 3 seed 为目标。规模曲线与工程 pilot 可先 1 seed，但须明确。主指标为每题 task success，辅以逐字段正确、source retrieval、supported success、时延和费用；未知 judge 不算错误，也不从报告中消失。

SFT 是当前主训练阶段。OneSearch-VL 的后续 reward/RL 路线在 SFT、gold benchmark 和奖励校准完成后评估：如已有清楚的搜索决策瓶颈，可在同一基座上增加 SFT+RL 对照。原版 EVGR 偏视觉，AV 奖励须用真实感知与来源支持校准，不能直接把自动 silver 成功当成已验证 AV reward。本阶段不先扩出另一套算法框架。

## 7. 数据管线作为贡献的具体设计

现有链路：公开 AV 与网页 → 录制分组与来源预留 → 原生 AV 出题及私有事实候选 → 匿名 AV 搜索任务 → teacher 在原 agent loop 实际调用工具 → 独立调用的自动 silver review → paired 导出 → 原 trainer CPU 检查 → 冻结训练集。

升级工作：

1. 出题时要求可定位的 audio cue、visual cue、外部事实与来源支持，增加“为何需要搜索”和模态角色字段；这些只用于构造/审核，不能直接交给 actor。
2. 建立相似机器、相似声景、相似乐器/表演和同类动物干扰池，使搜索需要区分具体实例；不把题目退化为读取文件名或标题。
3. 对分层候选做模态遮蔽审核，优先补充真实声音或声画联合帮助区分来源的问题。训练集可先使用自动 silver 标签，benchmark 须人工确认。
4. 用实际 retained/hour、CPU-qualified/hour、阶段延迟、token/API 用量诊断吞吐；按阶段分别配置并发，做 4/8 worker 的固定任务测试后再考虑 16。
5. 用等题数/近似训练计算量的 ablation 验证分层和干扰项是否带来收益。所有 proposed 组件分别标注实现、运行和效果状态。

详见 Dataset Pipeline v2。数据管线与 benchmark 同时推进，避免到 20K 才发现评测不测声音。

## 8. 下一阶段执行顺序和工作量

| 阶段 | 主要工作 | 完成交付 |
|---|---|---|
| A：当前结果归档，优先完成 | 固化 500 训练/六组评测；审阅 AV 输给静音的案例；补强 prompted base | 实验 ledger、错误标签、强提示基线配置 |
| B：2K pilot 与 benchmark 100 题并行 | 审完 14 条；新版本恢复直接 CPU 产数；预留新 benchmark 家族；双审 100 题构建 pilot | 2K 合格 train、2K answer-only、100 人审候选与构建报告 |
| C：验证训练与音频收益 | 同题 trajectory vs answer-only；强提示 base；模态对照；数据策略 2K ablation | 可用于选 recipe 的独立 development 结果 |
| D：正式规模 | train 达 5K/10K/15K/20K；benchmark 达 200 dev + 1K test；冻结 release | 数据卡、gold、搜索快照、主训练 checkpoint |
| E：论文实验与写作 | 主方法与关键对照多 seed；完整 benchmark 主表、成本曲线、能力分析 | 论文、图表、补充材料、可复现代码/数据说明 |

排期沿用原先约 7–8 周工作量框架，但从当前阶段重新按资源排：约 1 周完成 A/B pilot，随后 2–3 周扩数据并推进人工标注，再 2 周主实验，1–2 周分析/写作。机器生成和人工标注并行；这是工作量估计，不是当前吞吐下的完成承诺。

人工工作按 1,200 题 × 2 reviewer × 12–20 分钟估计为 **480–800 人时**；若 20% 题需第三人 10 分钟裁决，再约 40 人时。先用 100 题测实际耗时和保留率。人工 baseline 另取 100 test 题盲搜，每题最多 10 分钟，两人约 33 人时。标注人力是正式 benchmark 的关键资源。

训练暂按已测 500 题/2 epochs/约 1 H20Z 小时线性外推：20K 同配方约 40 卡时/seed，3 seed 约 120 卡时；这是算术估计，长度、吞吐和有效 batch 改变后需重测，不包含 answer-only、评测和其他消融。按当前 clean500 native 约 31.6 秒/题外推，1K test 约 8.8 单卡小时/条件，仅作容量规划。

数据生产按实际合格问题/录制估算：若为 0.5、0.8、1.2，20K 分别需要约 40K、25K、16.7K 次候选录制处理；不是默认每条录制都会生成两个成功问题。API 金额从实际 usage/账单另算，当前不虚报成本。

## 9. 论文组织与图表

正文组织：Introduction（AV 线索如何帮助找外部事实）→ Task & Benchmark → Data Construction → OneSearch-AV Training → Experiments → Analysis。

- Figure 1：自然 AV 问题、听看线索、搜索、正确来源、外部事实答案的完整例子。
- Figure 2：数据引擎与训练/benchmark 分支；明确 actor 可见信息。
- Figure 3：训练规模—task success；AV/静音/ASR 对照与成本。
- Table 1：benchmark 与直接近邻的任务/模态/检索/标注比较，逐项查证。
- Table 2：数据规模、家族、领域、语言、时长、模态依赖、保留率、标注一致性。
- Table 3：强 baselines 与 OneSearch-AV 的正式主结果。
- Table 4：轨迹/答案监督、数据管线与模态 ablation。
- Appendix：运行 ledger、完整 recipe、失败/恢复、judge 校准、更多案例和数据发布边界。

论文主张由最终 gold 和强 baseline 结果确定；工程过程完整留档，主文围绕贡献与发现展开。

## 10. 文档与证据入口

- [数据管线与数据卡](OneSearch-AV-Dataset-Pipeline-v2-2026-10-15.md)
- [Benchmark 建设与评分方案](OneSearch-AV-Benchmark-Blueprint-v2-2026-10-15.md)
- [过程与实验总账](OneSearch-AV-Research-Ledger-2026-10-15.md)
- 机器可读目标、当前计数、原计划引用与 artifact hashes：records/2026-10-15-paper-plan-v5/
- 原始训练：/home/jovyan/OneSearch-VL/research/runs/av-learning-500-20261015-v1/
- 本次完成评测：/home/jovyan/OneSearch-VL/research/runs/av-learning-500-eval-20261015-v2/

本轮整理文档与结果，不等于新 2K 训练、数据生产重启或正式 benchmark 已完成。

## 11. 本计划之后的执行与基座训练补充

本文件第4–5节的855+14、旧worker停止为初次核查snapshot。随后14条已核查并重启直接CPU生产；最新计数、benchmark候选与播放验证见OneSearch-AV-Execution-Update-v1-2026-10-15.md。基座固定Qwen2.5-Omni-7B，除LoRA pilot外增加更贴近官方训练策略的full Thinker SFT：冻结音视觉塔与projector、训练语言部分，仍使用原SFT/src/train.py，具体baseline、调参预算和CPU meta预检见OneSearch-AV-Base-Model-Training-v1-2026-10-15.md。目标20K train/200dev/1Ktest不变。
