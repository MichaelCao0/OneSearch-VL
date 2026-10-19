# OneSearch-AV 数据管线 v2：从真实音视频到可训练搜索轨迹

日期：2026-10-15。配套 Paper Master Plan v5。目标为 20K 不同问题的合格轨迹；当前 855 CPU 合格、14 待查。现有数据为自动 silver，不能称人工 gold。

## 1. 数据对象与贡献假设

数据单位是一道自然问题、匿名原生 AV 输入、可核实外部事实、实际执行的工具轨迹和最终回答。每题绑定原始录制家族；同录制的转码、裁剪与同题候选不扩大问题数。最终 paired answer-only 与 trajectory 共用题目，不能把 20K + 20K 行称为 40K 问题。

论文要检验的贡献假设：**以实际声画线索约束问题构造，并保留可执行、来源支持的搜索示范，可以比文本元数据出题或只学习最终答案更有效地教会 Omni 搜索。** 当前已实现 AV 出题、真实 rollout 和 silver 筛选；模态依赖分层、困难同类来源与完整 ablation 是下一阶段。

## 2. 已实现的端到端链路

| 阶段 | 输入与操作 | 输出 | 当前检查 |
|---|---|---|---|
| S0 公开素材采集 | 公开 AV 与发布页；获取实际 bytes、音视频流、页面快照 | 原始媒体、网页、来源索引 | 解码、时长、大小、来源和采集时间 |
| S1 分组与预留 | URL/媒体哈希/视觉指纹及候选家族；合并历史已用来源 | train 候选与 reserved registry | 已知 dev/eval 来源及页面排除；自动家族仍需人工补查 |
| S2 AV 条件出题 | teacher 真正接收完整选定 AV 与 publisher text | 每录制 0–2 个问题、私有 audio/visual cue、答案字段、支持原文 | 答案须有原文支持；不同题要问不同字段；问题不得泄露答案/发布者/文件名 |
| S3 匿名任务与检索库 | 随机 clip ID、真实录制音轨、冻结网页/AV 来源 | actor 题目、AV handles、冻结搜索 corpus | 私有答案、lineage、原始来源路径不进入 actor 输入；匿名 clip 不进入检索索引 |
| S4 实际搜索 rollout | teacher 在原 OneSearch agent loop 自行选择动作 | 多轮消息、text_search/visit_webpage/inspect_av 返回、final | 真工具执行，不能让 teacher 编造 observation；当前一题一候选、最多 8 turn |
| S5 自动 silver review | AV、问题、候选答案、私有参考、实际历史 | 独立调用的保留/拒绝 verdict | 同一 teacher 服务的另一次调用，不是独立人工或独立模型；检查事实、媒体与无实质幻觉 |
| S6 paired 导出 | 实际消息快照与保留 verdict | trajectory.json、answer_only.json、lineage | 原生媒体完整保留，禁止 stripped 媒体、未来 observation 和答案泄漏 |
| S7 原 trainer CPU 核验 | LLaMA-Factory converter、Omni plugin、processor、collator | label/media audit、token profile | 真音轨和视频覆盖，assistant labels，user/tool mask，序列长度，导出哈希 |
| S8 去重与冻结 | 各版本合格导出 | verified pool、固定 train members、训练配置 | task ID 与 recording/question 双去重；只统计有对应完整审计的成对导出 |

Teacher 生产使用现有 gateway 的原生 Gemini generateContent 路径，并非假设 OpenAI-compatible video_url 能看懂音视频。此前 video_url 探针未通过，不能把 HTTP 200 当媒体理解。生产凭据仅远端私有配置读取；数据卡记录模型/接口标识和时间，不含凭据或媒体 base64。

## 3. 代码与真实实现边界

主仓库根目录：/home/jovyan/OneSearch-VL。

| 路径（均在主仓库下） | 职责 |
|---|---|
| /home/jovyan/OneSearch-VL/research/av_data/collect_commons.py | 素材与公开文本采集 |
| /home/jovyan/OneSearch-VL/research/av_data/source_inventory.py | 媒体指纹、候选录制家族与缓存 |
| /home/jovyan/OneSearch-VL/research/av_train/question_authoring.py | AV + publisher text 出题、事实候选、匿名任务 materialization |
| /home/jovyan/OneSearch-VL/research/av_train/teacher_media.py | 保留完整选定片段与真实音轨的传输媒体 |
| /home/jovyan/OneSearch-VL/research/av_train/teacher_rollouts.py | 实际 agent loop、候选队列和请求预算 |
| /home/jovyan/OneSearch-VL/onesearch_vl/onesearch_infer/pipeline.py | 继承 OneSearch-VL 的工具循环 |
| /home/jovyan/OneSearch-VL/onesearch_vl/onesearch_infer/omni_runner.py | 原生 Omni student 多轮音视频输入 |
| /home/jovyan/OneSearch-VL/onesearch_vl/onesearch_infer/gemini_av_runner.py | 原生 AV teacher 请求与生成记录 |
| /home/jovyan/OneSearch-VL/research/av_train/teacher_silver_review.py | 自动 silver 事实/轨迹审核 |
| /home/jovyan/OneSearch-VL/research/av_train/native_sft.py | paired 导出、实际媒体与 label audit、训练配置 |
| /home/jovyan/OneSearch-VL/research/av_train/scale_searchqa.py | 分片生产、共享 corpus、verified_pool() |
| /home/jovyan/OneSearch-VL/research/av_train/source_isolation.py | 训练可见文档的保留来源隔离 |
| /home/jovyan/OneSearch-VL/SFT/src/train.py | 原版训练入口，当前 Thinker LoRA |

代码已核对：native_sft.answer_only_row() 仅保留 system、初始 user AV 和实际 final。它没有保留完整工具历史。因此它衡量初始输入直接答案蒸馏 vs 真实轨迹学习。旧 10 月 12 日文档中“同一完整上下文只屏蔽动作”的描述不能用于当前实际导出。若要单独隔离 action loss，应新增 context-matched masked-action 对照；这个新对照尚未实现。

## 4. 下一阶段的 AV 依赖分层

先以目标配比组织候选：40% 声画联合、20% 声音为主要识别线索、30% 视觉为主、10% ASR/文本可解控制。配比在 100 题人工 pilot 后依据自然来源可得性冻结；不能按本方法在测试上的胜负挑题。

- 声音主要线索：声学事件、乐曲/乐器线索、动物叫声、设备声音帮助定位具体实体或来源；答案本身需要外部查询。
- 声画联合：单一模态有多个合理候选，联合观察缩小到可检索实例。
- 视觉为主：真实控制组，防止系统总是多听、多查。
- ASR/文本控制：测语言内容和简单 metadata 能解释多少收益，明确标签，不混称非语音优势。

新增构造字段拟包括 modality_role、cue_span、external_fact_type、search_required、ambiguity_candidates、supported_source_quote。作者生成的标签是候选，reviewer 要独立确认；cue 描述不直接送给 actor，student 仍看到真实 AV。

每录制最多两个不同事实问题，避免同素材几十个属性题。主要输入保留足够事件上下文，后续以约 10–60 秒、常见 15–40 秒候选探索，不以 2 秒作为默认。超过当前训练/teacher 媒体限制前先实际核验 processor、token 长度和传输，保存原录制预览和所选区间。

## 5. 训练数据质量验证

自动筛选之外，从每个主要版本分层随机抽查至少 200 题，记录问题自然性、媒体与来源匹配、事实支持、查询是否由已见线索支持、无依据附加陈述、模态角色和路径有效性。抽样方案在看模型结果前固定，报告合格率及家族聚类区间；人工发现的问题修订为新版本，不改写历史成绩。

Benchmark 采用独立来源和双审，不从“teacher/student 答对的训练题”挑选。训练集 silver 可以大规模；gold 用更高成本核实。

发布数据卡应包含：候选录制数、问题数、候选/成功 rollout 数、review 保留数、CPU 合格数、独立家族数、各阶段拒绝原因、teacher 标识、领域、语言、时长、答案字段数、轨迹轮数、context/label tokens、媒体与页面可用性、采集截止时间、许可/可分发性。不能分发原始媒体时提供可复现的来源与获取/裁剪说明，最终发布范围按实际权利核验决定。

## 6. 证明管线价值的消融

| 对照 | 固定条件 | 检验 |
|---|---|---|
| publisher-text-only authoring vs AV-conditioned authoring | 来源候选、问题数、student、训练计算 | AV 约束是否提高可解性、搜索价值和声音依赖 |
| 随机采样 vs 模态分层 | 合格题数与预算 | 分层是否提高声音/联合子集及总体表现 |
| 普通 corpus vs 同类困难干扰 | 题目与主要检索预算 | 是否学会区分具体来源，避免粗类搜索 |
| 无 silver review vs review | 相同候选池；另报最终保留数与 token | 筛选的质量/数量/成本折中 |
| answer-only vs trajectory | 同题、同基座；同时报告相同 epochs 与近似算力版本 | 搜索过程监督是否有额外价值 |

先在 2K 做小矩阵；正式 20K 只保留有独立 dev 证据的构造策略。不能把不同生成模型、不同题目难度同时变化的结果解释为单一组件收益。

## 7. 吞吐与成本记录

生产直接运行在 coder workspace 的 CPU 后台，不使用 kjob；GPU 训练/推理才用已授权的 kjob。

待补充的阶段指标：准备/下载/转码时间，author/rollout/review API 延迟及 p50/p95，输入/输出 tokens、媒体上传 bytes，CPU audit 时间、队列等待、重用 corpus/审计缓存次数、retained/hour、qualified/hour、单位合格问题调用与费用。

现有代码已有并发、共享 corpus 和部分缓存，但没有完整的跨阶段瓶颈报告。先对固定未尝试来源做 4/8 worker 小实验；16 worker 需要代码上限及服务实测支持，不能直接越过当前 reviewer 的 8 worker 校验。并发实验处理不重叠的等分层任务，报告工作量/来源差异，不能为测吞吐重放已付费失败请求。

生产旧 worker PID 1499336 已不存在。下一次使用新版本计划，保留原请求记录、未知请求与 14 条待查导出。新队列只消费未尝试来源，先完成只需 CPU 的既有审核；没有自动收费重试。

## 8. 可复现交付

每次冻结保存代码版本/工作树快照、Pixi lock、基座与 tokenizer revision、来源 registry、题目成员、训练导出、实际审计、API/生成标识、筛选规则、tokens/成本和 split。一题的链路可追踪到原 AV 区间、网页支持与实际工具返回。

正文用构造流程、产率/质量统计和训练消融解释贡献；底层运行记录与恢复详情进入 appendix/ledger。最终规模由 verified_pool() 重算，目标不能替代实际 release 数。
