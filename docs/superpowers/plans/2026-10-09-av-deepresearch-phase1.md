# AV-DeepResearch 第一阶段执行计划

> 工作位置更新（2026-10-10）：用户最新指定 `coder ssh 3`，仍使用 `/home/jovyan/OneSearch-VL`。本文的 workspace 2 及当时能力状态是历史计划/运行快照；最新实况见 [研究入口](../../research/README.md) 和 Phase 1 验收报告。研究方法与 gate 仍按 v2 执行。


> For agentic workers：按工作包执行和独立验收。用户已同意研究方向及计划撰写；此文件规划后续实现，不代表已经采集自然数据、部署AV observer或运行研究实验。

**Goal:** 在coder ssh 2的OneSearch fork上建立10案可裁决的自然多源AV证据包、冻结可搜索来源池和真实同步AV工具，并由G0/G1决定是否扩至40案。

**Architecture:** 原始媒体/时间/来源registry为唯一资产依据；public corpus与private gold隔离；observer实际读取原始同步片段，receipt验证与语义审核分开。先验证数据和工具，再生成teacher/branch训练数据。

**Tech Stack:** Pixi、Python、ffmpeg/ffprobe、JSONL、SQLite/全文检索；视觉/音频encoder和原生AV observer必须完成设备与真实输入probe后锁定。

**Spec:** /home/jovyan/OneSearch-VL/docs/research/AV-DeepResearch-Research-Plan-v2-2026-10-09.md，尤其§6–§11、§16–§18。

## Global Constraints

- 主开发目录 /home/jovyan/OneSearch-VL，coder ssh 2，research/av-deepresearch。
- baseline 0b7b01ed4872aaccd9af0f6f3d23a81814cf55bd；发布视觉路径保留。
- CPU default已验证；当前没有可访问GPU，不能报告raw AV/训练能力就绪。
- 本阶段不依赖Serper/Gemini、全量官方benchmark或SFT/RL。
- 区间统一source-local [start_s,end_s)，原始AV不重定时、不去掉音轨。
- private gold不进入actor索引/消息/文件名；实际公开metadata保留。
- 收据只验证执行/字节/区间，语义与关系必须独立审核。
- 10事件簇、≤60源资产、≤20小时原始AV、≤50GB持久资产、≤25GB临时cache；达到任一上限不自动扩量。
- 10案建议人工≤40小时；收费API校准≤USD100等值为预算方案，不代表已付费；teacher≤30episodes，snapshot probe≤8suffixes，无训练job。
- 版本化产物不覆盖；没有用户公共动作授权，不commit/push/PR/发布。

## Task 1：媒体schema、原始clock与摄取

**Files:** research/av_data/{schemas,ingest,media_clock,validate}.py；validation/research/test_media_clock.py；对应schema与样例manifest。

**Consumes:** 原始source URL/本地文件与ffprobe stream/PTS。
**Produces:** SourceRecord、DerivedRecord；原始/派生hash及[source-local interval→输出]映射。

- [x] 定义stream/time_base/start_time、timestamp_origin、offset、VFR与原始采样映射字段。
- [x] 构造已知声画脉冲/offset与VFR测试素材，先验证错误实现/缺失映射被拒绝。
- [ ] 摄取保留原始AV，派生ASR音频和clips保持可逆时间定位，不沿用1-fps grid当source clock。
- [x] 校验越界/空区间/不存在hash失败；目标工具同步error≤100ms，记录实测而非只读metadata。
- [x] 保存来源/访问/许可/bytes/hash receipt，cache可删除后重建。

**验收:** 可回放的同步片段、原始时间映射、资产receipt；原媒体原有不同步明确标注。

## Task 2：source lineage与public/private隔离

**Files:** research/av_data/{tasks,gold,split,audit}.py；validation/research/test_split_and_visibility.py。

**Consumes:** SourceRecord、近重复候选、人工关系依据。
**Produces:** recording/event components、split manifest、TaskPublic/ClaimGold/LineageGold分离。

- [x] 区分URL/uploader/recording/event/independent-observation，支持relation unknown。
- [x] 用共享音频、裁切/转载fixtures测试同谱系跨split失败，通用知识页不连接所有事件。
- [x] split先冻结，再生成问题变体/轨迹/分支；exploration不进入test。
- [x] private evidence sets/标签/审核摘要/未来观察进入actor导出时明确失败。
- [x] 实际公开标题保留，opaque ID不能编码答案或关系。

**验收:** 泄漏validator、谱系隔离、来源关系可审核；hash/URL去重不替代事件隔离。

## Task 3：冻结corpus与最小可执行搜索

**Files:** research/av_search/{corpus,index,retrieval}.py；validation/research/test_frozen_search.py。

**Consumes:** split后的公共source内容、ASR/OCR/网页原文。
**Produces:** corpus/index/retriever manifests、SearchReceipt。

- [x] 先实现BM25/全文检索，无外部搜索API也返回实际可访问来源。
- [ ] 设置同类不同录制、转载和不足以裁决的自然干扰；禁止按task只返回gold小池。
- [x] 对固定query核验返回source/hash/quote范围，空结果与坏资产显式error。
- [x] 保存rank/score/config；同版本重放对应同内容。
- [x] 视觉/音频encoder是后续锁定能力，只有实际probe通过才写ready；非语言声学描述首版不进text index。

**验收:** 全局来源池可搜索、全部返回可回放、gold不泄漏、真实retrieval coverage报告。

## Task 4：同步AV observer与tools

**Files:** research/av_agent/{observer,tools,adapter}.py；validation/research/test_av_tool_receipts.py。

**Consumes:** SourceRecord/原始interval、公共corpus。
**Produces:** ObservationReceipt及失败/成本；研究adapter独立于发布视觉工具。

- [ ] 保存实际audio/video bytes身份、model/revision/config、输入范围、raw output和uncertainty。
- [ ] 在真实部署可用时运行≥8个人工语义probes：speech、non-speech、画外声、旁白、同步、静音/遮蔽、错区间、来源歧义。
- [ ] 无原生AV时mode=ASR+frames明确标记；不得把CPU ASR或HTTP200算joint AV通过。
- [x] compare/provenance只消费可见内容，不访问LineageGold，容许unknown。
- [ ] 逐调用记录logical/actual cost、AV秒数/帧像素/latency/retries。

**验收:** 原始音视频确被读取、定位正确、语义表现与人工对照可核验；不成立时保留partial并先完成独立数据工作。

## Task 5：10个自然case与双标gold

**Files / outputs:** research/av_data/{tasks,gold,audit}.py；版本化assets/manifests；独立annotation forms与私有裁决日志。

**Consumes:** 自然问题草稿、public sources与真实原始AV。
**Produces:** ≤10 TaskPublic及ClaimGold的AND/OR evidence packages、模态需求/来源关系审核。

- [ ] 覆盖技术操作、活动现场、环境声景，先保证可裁决，不能为填quota造事实。
- [ ] A/B独立标3–6目标项的decision/status/qualifiers/evidence sets/bindings/missing evidence。
- [ ] 将声画共现与声源/因果结论分开；unknown限定within_snapshot。
- [ ] 审核metadata-only、ASR-only、ASR+frames、raw AV充分性，音频遮蔽连派生字段一起移除。
- [ ] 裁决分歧，记录agreement、工时、major disagreements和替代合法证据路径。
- [ ] G0：至少8案可形成包，100%关键引用source/hash/time或quote可回放；否则修任务不扩大。

**验收:** 人工可裁决的case packages；不是teacher自己认可自己答案。

## Task 6：有界pilot、成本和下一阶段gate

**Consumes:** 已验收registry/corpus/tools/tasks。
**Produces:** 有界真实trajectory、cost ledger、G0/G1决定及40案stage-budget。

- [x] 保留当前已验证命令：pixi install --locked；pixi run validate-retrieval；pixi run minimal。新研究CLI在实现验证前不写成已可运行。
- [ ] teacher不见gold、实际执行工具，严格JSON拒绝duplicate keys；observation不由teacher编造。
- [ ] 每案最多3teacher episodes，总≤30；provider缺失/工具失败保留在分母。
- [ ] 仅做≤8suffix snapshot probe验证clone与budget/cache不污染，不提前启动640分支或训练。
- [ ] 测每case bytes/AV秒/取源与双标小时、每episode cost/latency，按p95×数量×1.2生成下一stage预算并锁定资源。
- [ ] 记录G1候选信号：raw acoustic/joint AV必要性与controls；通过10案G0后，经40案扩量预算和真实observer核验，才冻结正式G1 gate-config。

**验收:** 10案证据与成本报告，能力/失败分母清楚；40案扩量有依据。未经gate不得自动扩大数据或排正式训练。

## 后续任务依赖

Task1–5完成10案并通过G0后，按已批准的40案exploration预算扩量；40案与真实observer满足G1并冻结gate-config后，才按v2的WP5完整snapshot/branch与WP6 same-label Q/Proposed进入640suffix信号试验。G2后冻结test blueprint；G3后才投入正式IID300/OOD120完整gold与主要比较。完整teacher/RL、公开互联网与系统榜单均条件执行。

## 实际执行状态（2026-10-10）

Task1–3 的核心数据、clock、隔离和metadata检索基础设施已实际验证；84项研究测试和7项官方检索开关回归通过。原始AV裁切有真实输入/输出hash、clock映射及重放收据。10个素材是draft_unreviewed seeds，所有任务使用全局探索来源池。

Task4只达到media_receipt_only：raw_av_observer_ready=false，ASR/视觉语义observer也未部署；没有8项人工语义probe。Task5有A/B与裁决空白表，human signatures=0、gold=0、G0 pending。Task6完成10案确定性基础设施回放，未执行teacher/suffix。完整解码成本尚未计量，不能把requested AV秒数当actual工作预算。原始声源、独立事件及必要性仍需人工审核。未进入40案/G1、640suffix或训练。

详见 docs/research/AV-DeepResearch-Phase1-Progress-2026-10-10.md 与 research/runs/phase1-20261009/delivery-verification-v1.json。
