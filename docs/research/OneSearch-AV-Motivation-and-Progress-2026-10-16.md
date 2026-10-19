# OneSearch-AV：Motivation、贡献与当前证据

现场时间：北京时间 **2026-10-16 16:35**。工作标题：**OneSearch-AV: Learning to Search from Audio-Visual Clues**。

## 1. 核心 motivation

真实音视频搜索往往始于一段尚不知道实体名称的片段。用户能提供演出、赛事或现场活动，却未必知道作品、赛事或活动名称，因而难以直接写出准确搜索词。声音和画面提供辨认入口，外部来源提供片段之外的知识。我们研究如何让同一个原生 Omni 模型将声画线索转化为查询、证据读取和必要的片段再检查，通过训练提高搜索问答的成功率。

OneSearch-VL 提供了视觉线索驱动搜索的数据、训练和评测基础，其公开 Video-Bench 是静音采样序列。OmniGAIA 等工作已经研究原生 Omni agent。我们的价值集中在可规模化的 AV 搜索数据、相应 benchmark 和受控学习证据。

可用于论文讨论的表述：

> 音视频中的身份线索可能分布于画面、语音和非语音声音，而回答用户的问题还需要检索片段之外的知识。获得原生声画感知能力后，模型还需要学习如何选择有辨识力的线索、形成可检索查询、根据返回证据修正判断并完成回答。OneSearch-AV 通过真实音视频条件下构造的搜索任务和实际执行轨迹，训练统一 Omni 搜索策略，并构建可做模态干预的独立 benchmark，以检验轨迹监督、数据规模和声画互补对搜索成功率的贡献。

## 2. 问题、方法和贡献

| 挑战 | 方法 | 对应贡献及验证 |
|---|---|---|
| 片段需要包含可辨认线索，问题需要有可检索事实 | 原生 AV 条件出题、发布材料事实支持、真实搜索与筛选 | 数据引擎；质量、有效产率、构造消融和训练收益 |
| Omni 感知线索需要进入实际工具决策 | 同一 Omni 接收初始及工具返回 AV，学习完整搜索轨迹 | Learned search model；同基座 prompted、answer-only、trajectory 和规模对照 |
| 总分需要解释感知、检索、组合及音频的作用 | 来源独立 benchmark、操作分类、模态角色和干预 | Benchmark 与分析；事实人审、强 baseline、AV/muted/audio-only/video+ASR |

定位为沿 OneSearch 搜索问答范式的跨模态数据与训练方法研究，benchmark 提供独立验证。问题—目标、目标—挑战、挑战—模块、模块—贡献四项逻辑对应通过；方法有效性由实验建立。

当前已经跑通 AV 条件出题、真实工具轨迹、silver 语义保留、成对导出及原 trainer CPU 检查。完整 AV 版 VGEG、EVGR/RL 尚未实现；多跳、join、count、arithmetic、comparison 构造和模态采样消融属于后续数据方法实验。

## 3. 最新执行状态

| 项目 | 16:35 实际状态 |
|---|---|
| CPU 合格不同训练问题 | **1,359** |
| 已知排除或暂缓 | 146 |
| 可选 pilot silver | **1,213**；距 2K 差 **787** |
| 本批出题 | 144 录制输入，111 完成/33 失败，接受 **114 问题**；79 录制进入共享检索语料构建 |
| 下一批素材 | 采集中，77 complete assets；尚未形成最终新增录制清单 |
| Benchmark dev | **192 候选**；human gold **0** |
| 新版 500 子集 | 已冻结，用于同质量规则的 500/2K 嵌套比较 |
| 2K 冻结 | 等待达到 2,000 可选问题 |
| Full Thinker | CPU 参数策略已验证；真实 GPU 更新尚未执行 |

生产、接续等待、benchmark、freezer 和素材预取身份均核验存活；v12/v13 没有新的 provider stop。磁盘可用约 1,455 GiB。114 候选尚须经过轨迹、保留和 CPU 检查，未计入 1,359。

主代码位于 `coder ssh 3:/home/jovyan/OneSearch-VL`。固定 `Qwen/Qwen2.5-Omni-7B`，沿原 OneSearch agent loop 和 `SFT/src/train.py`。初始媒体和 `inspect_av` 返回都送入同一个原生 Omni Thinker。当前检索为冻结语料。

完整目标保持 **20K train、500 train-validation、200 dev、1K test**；2K 为下一训练里程碑。

## 4. 已完成训练的证据

历史 500 题 LoRA：trajectory 和同题 answer-only 各完成 250 optimizer updates。共同完成评分的 38 道 heldout silver：

| 条件 | 答案成功 | 有证据支持的成功 |
|---|---:|---:|
| 当前 base，原声 AV | 0/38 | 0/38 |
| 同题 answer-only SFT，原声 AV | 3/38 | 2/38 |
| 完整 trajectory SFT，原声 AV | **24/38** | **22/38** |
| 完整 trajectory SFT，静音 | 22/38 | 22/38 |

Trajectory 相比 answer-only 提高 **55.3 pp**。两者同题、同更新数，监督 tokens 不同。这支持完整轨迹训练提高搜索任务完成。当前 base 大部分未有效使用工具，正式主表还需充分调好的 prompted-search baseline。

声音相对静音的收益尚不稳定：heldout AV 多成功 2 题（+5.3 pp，bootstrap 区间跨 0）；dev16 AV 为 9/16，muted 为 13/16。下一轮在独立核实的模态分层上检验。

上述为冻结语料上的自动 silver 评价。旧 500 中后续有 71 题被标为排除或暂缓，新版 500 已替换；后续规模曲线使用统一质量口径，保留历史结果。

## 5. 下一批关键证据

1. **学习与规模：** 新 500/2K full Thinker、同题 answer-only、有效 prompted base；训练后立即比较任务成功率、来源命中和工具行为。
2. **声音与互补：** 完成 dev 的事实、区间与模态角色人审，运行 AV、muted、audio-only、video+ASR，分析声音是否改善查询和检索。
3. **数据方法：** 同题数或匹配监督预算下，比较 AV 条件构造与 publisher-text 出题、自然采样与模态分层，确认哪些数据策略产生性能增长。

## 6. 来源与证据

- [OneSearch-VL](https://arxiv.org/abs/2610.12419)。
- [OneSearch-Video-Bench](https://huggingface.co/datasets/OneSearch-VL/OneSearch-Video-Bench)。
- [OmniGAIA](https://arxiv.org/abs/2602.22897)。
- 本轮快照：本机 `reports/motivation-progress-2026-10-16/execution-snapshot.json`。
- 历史实验源：远端 `research/runs/av-learning-500-controls-20261015-v2/learning-results.private.json`。
