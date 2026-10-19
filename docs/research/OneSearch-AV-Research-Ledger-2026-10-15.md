# OneSearch-AV 研究过程与实验总账

日期：2026-10-15。目标是持续积累可用于论文的方法、数据、实验与失败记录。正文叙事见 Paper Master Plan v5；这里保存实际过程，历史错误不覆盖为成功。

## 1. 阶段沿革

| 阶段 | 已做尝试 | 得到的结论/后续变化 |
|---|---|---|
| 10/09，OneSearch-VL 审计与最小复现 | fork 代码、公开权重/数据、Pixi、无 Serper 的受限执行 | 原版视频路径不原生保留音频；受限 smoke 不是论文完整复现 |
| 10/10，AV 工具与模块化 observer | 真实裁剪、时间信息、原生声画 observer、16 development cases | 形成工程输入与工具通路；此路线随后保留为对照 |
| 10/11，任务收敛 | 用户要求像 OneSearch-VL 一样做直接搜索问答 | 移除强制跨来源研究报告；generation 留后续 |
| 10/12，原生 Omni 与 teacher | Qwen2.5-Omni-7B 主 actor；原 agent loop；原 trainer；原生 AV API 探针、真实 rollout、13 题小训练 | 接通原生 AV 轨迹和训练；接口成功与训练接通不等于性能提升 |
| 10/13，172 问题学习 pilot | trajectory/answer-only 各 86 updates；silver heldout/dev 评测 | 有初步行为收益；发现旧训练 observations 中 18/40 heldout 参考页面暴露 |
| 10/14，来源隔离与扩容 | 建立 clean pool、过滤固定评测页面/媒体；直接 CPU 并发生产 | 按去重合格轨迹计数，不把题目候选/answer-only 重复计数 |
| 10/15，clean500 | 500 问题、250 updates、六组同题评测与匿名 judge | heldout 有训练收益；音频收益不稳定，需要正式独立 benchmark |
| 本次整理 | 恢复 15K–20K/1K test 目标，明确 20K train + 200 dev/1K test、贡献和实验矩阵 | 文档与规划已落地，未新启动生产/训练 |

主仓库：/home/jovyan/OneSearch-VL；原 fork MichaelCao0/OneSearch-VL；branch research/av-deepresearch。snapshot 记录当前 HEAD、工作树 diff hash 和运行 artifact hashes。HEAD 不代表未提交工作树的全部内容；实际 run 的 source freeze 优先。

## 2. 实验注册表

| 实验 | 数据/模型/训练 | 评测与状态 | Artifact |
|---|---|---|---|
| Historical172 | 172 unique train，trajectory/answer-only 各 86 updates | 历史参考，已知页面暴露，不作为 clean scaling control | /home/jovyan/OneSearch-VL/research/runs/av-learning-pilot-20261013-v4/ |
| Clean500 train | Qwen2.5-Omni-7B，Thinker LoRA，rank8/alpha16/dropout.05，LR1e-4，batch4，2epochs，250updates | 完成；权重及 optimizer/scheduler/RNG、有限梯度已验证 | /home/jovyan/OneSearch-VL/research/runs/av-learning-500-20261015-v1/ |
| 500 first evaluation | 同一 500 adapter 与固定 56 题 | 训练完成后，base muted 的 16 个 FFV1 dev clip 因容器兼容性失败；不是模型答错 | 上一行原 run 的 gpu-v1/ |
| Evaluation recovery | 不重训；复用 96 个 hash-bound base 成功 rollout，补 240 个 rollout | 六组均 56/56 完成；54/56 个匿名问题判分，2 unknown | /home/jovyan/OneSearch-VL/research/runs/av-learning-500-eval-20261015-v2/ |
| Production v4 | teacher 出题→实际工具→silver→CPU audit，直接 workspace CPU | 869 retained，855 CPU-qualified，14 pending；旧 worker 不存在，生产未运行 | /home/jovyan/OneSearch-VL/research/runs/av-next-phase-20261015-v4/ |
| 2K/20K/正式 benchmark | 见 workload-targets.json | 规划，尚未完成；500 answer-only 对照尚未训练 | 后续新 run 注册 |

500 profile 的 recordings=494、family keys=495，口径尚待统一，不能将 family keys 直接当作独立人工家族统计。

500 adapter SHA-256：166598d30fc5376f45a18a1b72afa57580eb6bcea42d358104513cc7d52e5301。
500 run frozen source revision：ae9e1690543ffd5c0221dc27f79834d0294cba00。
训练耗时约 3,600 秒；含预处理包装约 4,259.6 秒。train loss 0.8977 是优化记录，不是任务成功率。

## 3. 最新同题语义结果

judge：现有服务的 gemini-3.8-flash 标识；单题匿名比较 3 模型 × 2 模态；56 请求槽位、54 完成。参考和 judge 均为 silver。原始结果完成于北京时间 2026-10-15 12:44 左右。

| 条件 | heldout 成功/39 | dev 成功/15 | 全部成功/54 |
|---|---:|---:|---:|
| base native | 0 | 0 | 0 |
| base muted | 1 | 0 | 1 |
| old172 native | 15 | 9 | 24 |
| old172 muted | 15 | 11 | 26 |
| clean500 native | 23 | 8 | 31 |
| clean500 muted | 21 | 13 | 34 |

Supported success 与上表相同，只有 old172 native 的 dev 为 8/15、all 为 23/54。计划 heldout40/dev16，分别各有 1 个 judge unknown。正式报告包含全分母完成上下界；不得把 39 或 15 偷换成已规划题数。

clean500 native − base native，heldout +59.0 pp；clean500 native − old172 native，heldout +20.5 pp，但 old172 暴露使其不适合严格 scaling 归因。clean500 native − muted：heldout +5.1 pp，95% family bootstrap [-8.1,19.4] pp；dev −33.3 pp。当前不能作稳定音频优势主张。

未训练 base 的协议遵循很差；下一步补 strong prompted baseline、同题 answer-only 及 independent gold，不以 base 0 分作为唯一论文证据。

## 4. 失败与修复记录

- FFmpeg：GPU Pixi 环境不能将 FFV1 视频 stream-copy 到 MP4，错误为 Could not find tag for codec ffv1 in stream #0, codec not currently supported in container。默认 CPU 环境能过，不能替代目标 runtime 检查。
- 修复：MP4/MOV/M4V 保持 MP4，其余采用 MKV；复制视频包、移除全部音轨。初始与未来 inspect_av clips 同样处理。56 个输入在 CUDA Pixi 环境检查；真实 dev 的视频 tensors/grid/FPS 一致，audio count=0。
- 恢复 GPU job slurm-profile-slurm-v5dc4：eval-only、1×H20Z、6h cap、backoff0；completion-state 为 report_generated，returncode0。原先训练 job slurm-profile-slurm-kz479 的失败发生在训练结束后的评测。
- 旧数据泄漏：18/40 heldout 参考页曾进入 old172 training observations。新 500 已过滤已知保留页面/哈希/候选家族；预训练污染与自动家族漏检仍未知。
- API 失败或中断：保留原始 attempt，不自动付费 replay；source queue 恢复只消费未尝试录制。
- 陈旧状态：production state.json 仍写 running，但 worker-start.json 的 PID 1499336 在本次 snapshot 已不存在，退出原因未确认。

## 5. 当前能写与待证实内容

可据 artifact 写：原生 AV 端到端工程、真实 teacher 工具轨迹、855 合格训练题的计数、clean500 实际训练与 silver 同题收益，以及音频对照的正负结果。

待证实：20K 数据 release、1K 正式 gold test、强 prompted baseline 上的训练增益、同题 answer-only 的差距、稳定的声音/联合模态增益、AV-conditioned pipeline 的独立价值、跨种子稳定性及公开环境泛化。

## 6. 后续每次实验必须追加的记录

保存研究问题、父实验、数据成员/来源 split、基座和 adapter hash、代码/环境身份、实际训练 updates/tokens/GPU 时间、推理预算、逐题结果、judge/human 状态、错误/unknown、主表及可复现命令。数据生产记录各阶段数量/产率/吞吐与实际费用；新版本保留原有失败和未知。

记录写进新的日期/版本文件，并更新研究入口。旧 snapshot 不改。完整问答、答案、teacher 历史和私有媒体仍保存在远端 run；本文及本次本地镜像只包含汇总、目标与 artifact hashes。

配套机器可读目录：/home/jovyan/OneSearch-VL/docs/research/records/2026-10-15-paper-plan-v5/。

## 7. 2026-10-15 后续执行更新

最新计数为904 CPU-qualified/904 retained；14条待查已完成，v5直接CPU生产已启动并核对进程身份。新增19题/14录制的benchmark dev候选、10个训练与14个benchmark可播放审核案例，human gold仍0。发现采集query标签与真实内容不一致，已更正UI语义并保留样例。详见OneSearch-AV-Execution-Update-v1-2026-10-15.md和OneSearch-AV-Base-Model-Training-v1-2026-10-15.md；此前表格是历史snapshot。

## 8. 2026-10-15 单基座主线与100录制质量检查

- [单基座主线与投稿执行方案](OneSearch-AV-One-Base-Model-Mainline-v1-2026-10-15.md)：固定Qwen2.5-Omni-7B，原agent loop/原trainer，full Thinker主性能路线与matched answer-only。
- [24例原生AV质量检查](OneSearch-AV-Native-Content-Quality-Audit-v1-2026-10-15.md)：21完成/3unknown，16主搜索保留建议/5感知控制；人工gold仍0。
- 2026-10-15T08:13:25.738228+00:00 合格训练题904；持续CPU生产、预留benchmark producer及新100录制训练质量检查的身份和进展见records/2026-10-15-onesearch-mainline/execution-snapshot.json。候选YAML通过原trainer schema检查，尚未运行新GPU训练。

## 2026-10-15 100录制内容抽查与模态审核

- [100录制抽查与模态审核包 v2](OneSearch-AV-100-Recording-Quality-and-Modality-Review-v2-2026-10-15.md)：91完成/9unknown，66keep/15control/4revise/6reject建议；训练14例与benchmark14例提供真实AV、去音轨视频、仅音频和上下文。
- 未自动筛改904训练池；人工gold仍0。媒体与逐包时间验证通过；浏览器播放单独记录。


## 2026-10-15 用户确认音乐/赛事/现场活动优先扩产

已启动 v7 CPU 数据生产和 v4 benchmark 多批次 worker。起点 904 CPU-qualified；复用 295 新素材；采样目标 30/30/20/20，首批最多预留 60 benchmark 家族、后续每批最多 50；benchmark 总上限 160 author attempts。21 项相关测试通过。v6 在付费调用前发现 sparse registry metadata 不匹配，失败证据保留；v7 按 source_id 和 exact media hash 回连 33 行源元数据。实际状态见 records/2026-10-15-priority-expansion/execution-snapshot.json。


## 2026-10-15 17:15 论文材料归档与数据进展

新增 OneSearch-AV-Paper-Dossier-v1-2026-10-15.md，按问题、方法和证据组织 motivation、related work、贡献、数据、实验和投稿路线。完整目标保持20K train、500 train-validation、200 dev/1K test；CVPR 2027为建议投稿窗口，未代用户选择venue。9项参考来源与24组主张完成独立核验；所有pilot数值与原始learning-results.private.json复核。

verified_pool()重新审核成对数据和哈希：938 retained、938 CPU-qualified、0 pending；比16:57快照新增34。v7 train producer和v4 reserved benchmark的host/PID/start-ticks/boot身份均匹配；benchmark新增43dev候选，human gold仍0。数据生产继续，归档未提交新GPU任务。快照为records/2026-10-15-paper-dossier-v1/progress-snapshot-03.json，文档校验和引用证据同目录。此次只新增和归档文档，未覆盖旧实验或修改运行源码。

## 2026-10-15 新 benchmark 审核与 matched 500 对照

- [OneSearch-AV-Benchmark-Review-and-Matched-Control-v1-2026-10-15.md](OneSearch-AV-Benchmark-Review-and-Matched-Control-v1-2026-10-15.md)：新增24题可播放声画审核，23完成/1unknown的自动意见，120文件哈希/96 Range/四模式实际播放；500同题answer-only数据与原trainer配置已冻结，GPU训练未执行。
- 17:31现场池为979 retained/938 CPU-qualified/41 pending，43新增dev候选，human gold仍0。见records/2026-10-15-benchmark-review-v3/execution-snapshot.json；生产保持运行。


## 2026-10-15 论文整理 v2

完成 OneSearch-AV-Paper-Dossier-v2-2026-10-15.md：更新 17:40 现场快照（979 retained / 979 CPU-qualified / 0 pending），整理 500 trajectory 结果、same-question answer-only CPU 准备和 24 题 benchmark 四模式审核；保持 20K train+500 validation、200 dev+1K test 目标及已独立核验引用。核对原始数字和正文版本，远端归档后逐文件 hash 回读。本轮整理没有新增 GPU/API 任务，没有改动生产 worker。证据：records/2026-10-15-paper-dossier-v2/。


## 2026-10-15 新批次数据与审核闭环

- [数据构建与审核执行记录 v1](OneSearch-AV-Data-Construction-and-Review-v1-2026-10-15.md)：988 道合格训练题，新增24训练/5 benchmark四模式案例；审核记录导入模块与11项测试、机器可读benchmark设计、29次原生AV审核及浏览器播放收据。音乐/赛事/活动新采集继续运行；参考现场快照，完整目标保持20K train、500 validation、200 dev/1K test。


## 2026-10-15 500条同题answer-only训练与评测已提交

- [OneSearch-AV-Matched500-Answer-Control-v2-2026-10-15.md](OneSearch-AV-Matched500-Answer-Control-v2-2026-10-15.md)：job `slurm-profile-slurm-g4hbg` 已实际Running，1×H20Z/6h上限/无自动重试；同500问题、250更新，训练后接112个新rollout和匿名配对评分。该时点训练及成绩尚未完成。数据扩产与benchmark producer保持运行。


## 2026-10-15 18:44 论文材料 v3 与数据/基线分析

新增 [论文构想与投稿计划 v3](OneSearch-AV-Paper-Dossier-v3-2026-10-15.md)。现场 988 retained/988 CPU-qualified，124 个精确输入内容已查成员（92 主搜索候选、21 control、4 revise、7 exclude），864 未内容审核。三版 prompt dev 均 0/16，48 首轮输入确认实际 AV 和问题送达；不将该结果称为有效 strong baseline。matched500 answer-only 165/250 更新、checkpoint125，训练后自动接112 rollout/最多56同题评分。新250素材中预留50 benchmark家族；train首48录制构造48 accepted问题，新bench50 attempts构造42 accepted问题，均按阶段统计。另启动128此前未付费审核录制的CPU原生AV抽查，4workers；原exports不改。完整目标20K/500/200/1K继续。见 records/2026-10-15-paper-dossier-v3/。


## 2026-10-15 19:20 matched500结果与新批次

同题answer-only完成250实际更新、checkpoint/optimizer/scheduler/RNG核验及112新rollout；六输出同题匿名silver评分54/56完成。Heldout共同38题：baseAV0、answerAV3、trajectoryAV24、trajectoryMuted22；dev16为0/0/9/13。Trajectory-answer +55.3pp，家族bootstrap[38.6,73.5]；音频heldout+5.3pp区间跨0，dev−25pp。新池1030qualified、85bench候选；128新录制审核121完成、7unknown，主规则92keep/9control/4revise/16reject。新优先12train+12bench四模式render-only修复完成，无付费replay。见论文整理v4及records/2026-10-15-paper-dossier-v4/；human gold0，完整目标20K/500/200/1K继续。

正文发布前将v4合格池计数与execution-snapshot核对为1069；1030为较早现场读数，旧正文及校验留档。
