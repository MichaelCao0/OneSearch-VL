# AV-DeepResearch 第一阶段进展与验收

> 2026-10-10 工作位置更新：用户改为 `coder ssh 3`，同一项目与持久目录已验证完整；本文的 workspace 2 是历史运行位置。后续入口见 [研究指南](README.md)。


日期：2026-10-10（Asia/Shanghai；远端 run 保留 20261009 UTC 标识）。研究范围是多源音视频检索、证据核验与研究报告。Search for Generation 留作第二篇。主开发位置为 `coder ssh 2` 的 `/home/jovyan/OneSearch-VL`，branch 为 `research/av-deepresearch`，基线 `0b7b01ed4872aaccd9af0f6f3d23a81814cf55bd`。本轮未 commit、push、创建 PR 或发布数据。

## 1. 当前结论

已从研究计划进入 Phase 1 实现：真实音视频采集、原始时间轴与字节溯源、冻结文本检索、actor/public/private 隔离、原始声画片段读取及收据已串联；10 个候选的确定性基础设施回放全部成功。

这不是 10 案完整 benchmark，也不是 AV 模型效果验证。现在有 10 份 `draft_unreviewed` 任务草稿，没有人工双标 gold、已确认的独立事件簇或音频必要性结论。G0 仍为 pending。没有 teacher episode、640 分支数据或 SFT/RL 训练。

## 2. 素材取得了什么

使用 Wikimedia Commons 公开 API 取得原始文件，保留 API/extmetadata、描述页 HTML/API/text、原始 URL、取得时间、许可/署名声明、字节大小和 SHA-256。描述页是发布者陈述，不能当另一份独立录制。

| 项目 | 实际结果 |
|---|---:|
| 已考虑的来源页 | 19 |
| 下载尝试 | 17，预算上限 25 |
| 传输字节，含失败收到部分 | 317,439,254 bytes |
| 完整原始媒体 | 13 |
| 有视频和非静音音轨 | 11 |
| 最终待审核 seed | 10 |
| canonical 原始资产大小 | 176,816,733 bytes |
| canonical 原始展示轴总时长 | 299.363491 秒 |
| 领域数量 | 技术/机械 4，现场/表演 4，环境/动物 2 |
| 完整下载资产独立 hash readback | 13/13 匹配 |
| canonical 原始资产摄取 | 10/10 成功 |

失败与排除保留在 acquisition manifest：4 个 HTTP 429、2 个超过单资产 100 MB 上限、2 个实际静音、1 个 metadata concern。环境候选中的 Holiday Lights/frog jump 在筛选时被替换，不能将灯光装置当真实动物声源。

十个初始素材：三条 steam-engine 录制、印刷机录制、Tchêlistas 大提琴表演、Stroh violin、Gong and Drum、Cardio Drumming、Shila Waterfall、little corella feeding。标题是公开 metadata，尚未转化为观察真值。三条蒸汽机可能带来任务结构重复；Cardio Drumming 可能有编辑或配乐，需要重点人工核验。

## 3. 已实现的模块

### 原始媒体与时间

`research/av_data/{schemas,ingest,media_clock,validate}.py` 定义 SourceRecord/DerivedRecord、内容地址存储和 hash/bytes 验证。保存原始 container origin、stream time_base/start_pts、decoded PTS、音频样本映射与 VFR；统一使用原始 source-local `[start_s,end_s)`。

局部读取使用 FFV1 + PCM float Matroska；采用共同时间原点，不各自从 STARTPTS 清零，也不强制固定帧率或音频重采样。合成 fixture 覆盖 container 起点 5 秒、延迟音轨、VFR、非零裁切起点及 H264/AAC。实测声画事件差约 0.0625 ms，证明这些 fixture 上的时间映射正确；不代表自然媒体本身同步，也不代表长/不连续或多音轨素材已完全验证。

### 公私隔离与来源关系

`research/av_data/{tasks,gold,split,audit}.py` 实现 TaskPublic、私有 ClaimGold/LineageGold 验证及 event/recording components。gold、答案、binding、来源私有路径与未来 observation 进入 actor 导出时失败；公开标题保留。关系 unknown 不自动变 independent。合法证据包支持包内 AND、包间 OR。

本轮只冻结 exploration 来源池，没有判定自然事件身份。每个任务搜索同一个包含全部 10 条媒体的池；这些素材不能称作 10 个已确认独立事件。

### 冻结检索

`research/av_search/{corpus,retrieval}.py` 使用 SQLite FTS5/BM25，索引实际公开 publisher metadata；当前无 ASR、OCR 或非语言音频总结。保存 corpus、documents、index、retriever 和 manifest 的版本与 hash，返回可定位的原文 excerpt/quote offsets。

独立审查发现并修复 manifest 私有字段、数据库/JSONL 不一致、FTS/正文不一致及非有限数值边界。加载时验证公共字段、manifest identity、DB primary ID/payload 与 JSONL、FTS source/text 和重复数量。episode 固定启动时的可信版本，后续一致性重建也不能悄悄替换。

### 工具适配器

`research/av_agent/{observer,tools,adapter,cli,run_pilot}.py` 支持 `search_sources`、`provenance`、`inspect_av`、`compare_sources`。actor 只能给封闭参数，不能改 corpus/split/source allowlist。严格 JSON 拒绝 duplicate keys、非有限数值和越界请求；限制工具调用数与请求 AV 秒数。内部路径只存 trusted receipt，actor receipt 不带原始/派生文件路径。

`inspect_av` 真实解码原始声画并生成映射片段，记录 input/output hash、时间区间、实际音频样本/视频帧/像素和成本。当前返回 `media_ready_observer_unavailable`，`raw_av_observer_ready=false`，不生成语义描述或观察答案。来源比较当前仅展示公开内容，关系结论保持 unknown。

## 4. 真实回放与验证

确定性脚本对每案执行一次搜索、来源查看、2 秒原始 AV 读取和公开信息比较。

| 指标 | 实际结果 |
|---|---:|
| 回放候选 | 10 |
| 按公开标题 query 返回 seed | 10/10 |
| 真实原始声画读取/裁切 | 10/10 |
| 原始与派生 hash/时间记录验证 | 10/10 |
| 工具调用 | 40 |
| 请求原始 AV 时长 | 20 秒 |
| 外部 provider 请求 | 0 |
| 工具失败 | 0 |
| 工具调用累计 wall time | 115.683 秒 |
| 完整回放 wall time | 491.066 秒 |
| teacher episodes / 方法实验 | 0 / 0 |

搜索 query 使用公开标题，10/10 只说明索引和回放可用，不能当研究检索召回率或任务解决率。完整 wall time 包含重复来源验证、解码、hash 和 Python 端检查，不能用工具时间替代整体成本。`decoded_audio_samples`、video frame/pixel 计数当前只覆盖准备出的片段；完整原始媒体解码工作尚未计量，`max_av_seconds` 只是请求区间预算，不能称实际解码工作预算。Phase 6 的完整解码成本与扩量估算仍 pending。

最终全套研究测试 84/84 通过；官方检索开关回归 7/7 通过。`pixi install --locked` 已成功。历史 `natural-replay-v2` 中 10/40 收据保存了 manifest hash，其他 30 条以及 episode header 只有 corpus/index hashes。这一归档缺口保留，不回写历史结果。最新 adapter 已为各工具和 episode header 保存 manifest hash，并通过新 `final-cli-replay-v2` 的真实搜索与 0.5 秒原始 AV 读取验证；新收据也明确标记 `count_scope=prepared_clip_only`。

已有 8B CPU 最小视觉复现保留，不在本轮重复加载；本轮不声称原生 audio-video 模型或论文完整 benchmark 复现。

## 5. 人工审核准备与尚未完成的 gate

已准备中文独立审核说明、A/B 空白表和裁决表。针对 10 案分别生成 60 个文件，30 份 JSON 全为 `blank_unreviewed`，human signatures 和 gold records 均为 0。具体私有表放在 ignored `research/private/annotation/natural-pilot-v1`；公共文档只是说明与空模板。

审核必须检查：题干是否预设无依据事实、同期声/画外声/旁白/编辑、声音变化与可见动作的时间依据、共现与声源/因果的区别、metadata-only/ASR-only/ASR+frames/raw AV 的充分性、替代合法证据路径、同录制/同事件/独立观察、`within_snapshot` 的证据不足。当前 ASR 和语义 AV backend 均 unavailable，不得虚填对照结果。

下一步顺序仍按 v2 计划：

1. 对 10 案做实际独立语义审核、改写或替换不可裁决任务，补充同事件关联或真正的外部佐证；当前来源页不算独立证据。
2. 部署并验证实际读取原始 AV 的 observer，完成至少 8 项人工语义 probe；记录不可用和失败分母。
3. G0 至少 8 案形成可裁决 evidence packages 且关键引用可回放后，才安排 40 案 exploration。
4. 40 案通过 G1 并冻结 gate-config 后，才做 640 suffix 信号试验和 Proposed vs scalar Q + 同辅助标签比较。

当前没有可访问 GPU，Serper/Gemini 未配置。数据和工具工作已经推进到上述边界；没有越过人工 gold、observer 或扩量 gate。

## 6. 远端可复跑入口

```bash
cd /home/jovyan/OneSearch-VL
pixi install --locked
pixi run validate-retrieval
pixi run validate-research
pixi run av-prepare-pilot --help
pixi run av-replay --help
pixi run av-pilot-smoke --help
pixi run av-review-forms --help
```

新 run 必须使用新输出目录，避免覆盖历史证据。主要实际产物：

- `research/assets/commons-pilot-20261009/acquisition-manifest-v2.json`：采集分母与 canonical seeds。
- `research/assets/originals-v1`：内容地址原件。
- `research/runs/phase1-20261009/natural-pilot-v1`：SourceRecord、TaskPublic、scope、冻结 corpus 与草稿状态。
- `research/runs/phase1-20261009/natural-replay-v2`：10 案工具动作、actor/trusted receipts、clips、成本和 summary。
- `research/private/annotation/natural-pilot-v1`：私有 A/B 与裁决空白包。
- `research/runs/phase1-20261009/progress.md`：执行 ledger。

完整研究计划仍以 `AV-DeepResearch-Research-Plan-v2-2026-10-09.md` 为准，本文件记录实际执行结果。
