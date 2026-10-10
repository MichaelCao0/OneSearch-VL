# AV train 来源组、12 个候选与 rollout 入口

日期：2026-10-10。本阶段落实总体 proposal v2 的来源组驱动数据构建，保留 16 个 benchmark development 候选，另外建设 train authoring corpus。当前任务均待语义审核，不计为可训练数据。

## 已生成的候选

新增采集 7 份带非静音音轨的视频，与上一批 4 份 train 候选合并。Authoring corpus 共 11 AV + 11 个说明页，约 429.81 秒 AV。3 个专题组覆盖 9 份 AV，另有 2 份有配乐的城市/园艺影片作为潜在自然干扰。采集检索词、网页与媒体身份、许可、原时间轴、冻结说明页分别记录。

| 来源组 | AV 候选 | 任务数 | 要检查的能力 |
|---|---:|---:|---|
| 机械运转 | 3 | 4 | 查找另一运转录制、周期实例、声音节律比较、说明页与观察组合 |
| 现场音乐 | 2 | 4 | 区分演奏画面与背景配乐、动作片段、跨演奏比较、城市资料组合 |
| 降雨环境 | 4 | 4 | 选择不同环境/表面、事件窗口、连续/离散声音比较、地点资料组合 |

每组各有 source discovery、event localization、cross-source comparison、research composition 一题。-v2 任务将组合的必要答案项拆开，窗口本身交给 timed evidence 表达；必要事实对应 source/span/package/answer/EBS 评测。模型/指标安排仍遵循总体 proposal：Question/metadata 对照、ASR+frames workflow、joint-AV workflow/agent、OneSearch-AV-Prompted、OneSearch-AV-SFT、强 omni 系统和阶段 oracle。

原始来源相同字节与 dev 重叠为 0。录制/事件/近重复家族仍待裁决，不能据此宣称所有录制独立。3 个专题组上的 12 个问题不是 12 个独立家族，来源规模在正式研究中需要扩大。

审核页展示全部 11 个 AV 来源、各任务候选、原始 PTS 抽帧、原片播放/许可链接和冻结说明页。每题可记保留/修改/淘汰及模态需求并导出 JSON。展示模板的 localStorage key 绑定任务/source 身份，防止不同批次复用旧意见。适合度意见与语义 reference 分开。

机械周期、首次完整拨弦动作组、水面扰动等定位候选必须能稳定标边界后才保留。Source discovery/comparison 允许多个有效来源，正式 reference 要覆盖不同合法完整证据包，不能只取作者随手选择的一对。

## 管线与 CLI

新模块 research.av_train.prepare_groups 接收 acquisition manifests、reserved source registry 和显式 task specs，支持任意数量来源/任务，取代扩量时“四份文件、四题”的约束。保留旧 prepare_pool 以复跑历史批次。

接口参数为 --manifest（可重复）、--reserved-sources、--task-specs、--out-dir。当前 12 题的完整复跑输入包括 v0 train-acquisition/1 到 /4 的 manifests、v2 acquisition-technical_mechanisms、acquisition-field_performance、acquisition-environment_animals、acquisition-live-performance 的 manifests，以及 task-specs-v2.private.json。Reserved registry 使用 v0/sources.private.jsonl。所有输出使用新的目录。

完整来源/规格 hash 保存在 train-groups/input-lineage.private.json，修订继承关系保存在 train-groups-v1/input-lineage.private.json。如来源筛选后缺少任务指定的 page，builder 会在写入前报错，不静默把跨源题降成单源题。

research.av_train.rollout 新增三个 observer modes：

- unavailable：默认原行为；没有语义 observer，适用于连通性检查。
- native：显式提供 model manifest、backend code revision、runtime manifest/lock 和 device。在固定 runtime 校验后加载 Qwen2.5-Omni，现有短窗上限为 2 秒；该约束进入 actor 可见的 tool definitions。
- recorded：读取独立执行记录与可信 expected backend identity，核验绑定后回放。当前是精确 cache-window replay，适用于回放/诊断；不是任意窗口的完整在线 observer。

Native runtime 的 CLI 检查命令（从 repo 根目录执行）：

    pixi run --locked --manifest-path runtime/av-observer/pixi.toml python -m research.av_train.rollout --help

真正 rollout 还要求已审核的 batch、明确 teacher model/revision，以及进程内的 AV_TEACHER_BASE_URL / AV_TEACHER_API_KEY。不得将 API key 写入命令、日志或仓库。Teacher 服务尚未核实可用，所以本轮不声称执行了真实 teacher。

目前 authoring builder 有意不产出 tasks-public.jsonl 和 scope.json。正式 teacher batch 需完成家族 split、任务可解性与证据审核后另行生成。

## 实际检索与运行验证

在新 authoring corpus 做了 6 次 metadata-search diagnostics：3 组中文查询均 no_results；来源语言关键词分别返回 4/5/5 项。音乐查询 music musician guitar 的第 1 项是 Washington 城市影片，现场演奏候选在后续排名中出现。这是对真实当前索引的结果，不是 benchmark 正确率；语义相关性尚未评分。

这项观察决定接下来的数据内容：teacher/student 需要根据来源语言改写查询，并检查候选的声画内容。扩展官方索引时加入固定 ASR/AV 描述与跨语言检索，保存相同信息与版本的 policy 对照；metadata-only 同时保留作为 baseline。

在本页原始来源组准备阶段，Native 独立 runtime 的 live receipt status=verified、errors=[]、CUDA wheel build 12.6；未加载模型、未验证新 GPU inference。根目录 CPU runtime 被 native CLI 在模型加载前拒绝，是实测的配置边界。

测试记录：source-group builder 7/7、observer CLI 6/6；research regression 201/201 通过。旧 CLI fixture 实际执行 2 次 metadata 工具调用。来源组准备阶段无新增 GPU job；后续真实观察见本文更新。当前仍无 teacher 模型调用、无 retained SFT、无 optimizer update。

## 后续实质交付

三组方向已由用户保留；继续完成原片事实/区间核验，再冻结来源家族与 teacher batch。之后通过真实工具合成并筛选 64–256 条 seed，运行真实 trainer labels/length audit，接通 CUDA backward、optimizer 与 checkpoint reload，再做约 2K pilot 的 Prompted/SFT/answer-only 配对比较。

原数据与私有 evidence 留在 Coder3；GitHub 只提交构建器、CLI、可复跑测试和文档。完整 pipeline 仍待上述数据与模型执行环节完成。

## 2026-10-10 用户反馈与事实核实更新

用户已回复“保留三组，继续核实事实和区间”，三组方向通过。该意见不填入语义 gold，也不裁决来源家族。最新作者任务为 train-groups-v2 中的 -v3 revision，继承同一 source registry 与 corpus。

本次真实 Qwen2.5-Omni-7B GPU 观察：首轮 6 段中 2 段执行、4 段在媒体适配阶段拒绝；CPU 复现确认三段的剪辑首帧晚于窗口起点，音乐片头存在原始缺帧。保留首轮记录，按实际源帧对齐共同 AV 裁剪窗口并换到清楚演奏片段。第二轮 6/6 模型执行、5/6 有效结构化输出、1 段 JSON 解析失败；没有新增人工 reference。

音乐题 006 改为观众移开后的首次无遮挡演奏片段。降雨题 010 改用近处金属栏杆进入/退出远景的自然变化，取代不稳定的第一滴水面扰动。机械题 002 移除题目已给出的实例序号，要求可见相位作为周期边界，继续做原帧高帧率审核。

两项新 Job 共 162 秒，沿用历史 727 秒，累计 889/2100 秒，余 1211 秒，均已终态。新的事实候选审核包为 fact-verification-v2/authoring-review-v1/index.html，包含视频窗口、说明页逐字引用、原始 PTS 帧图与未人工确认的模型输出。原片与私有模型记录继续保存在 Coder3。
