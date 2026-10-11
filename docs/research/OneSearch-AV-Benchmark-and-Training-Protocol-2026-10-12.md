# OneSearch-AV：Benchmark 与训练数据执行协议

日期：2026-10-12。任务沿用 OneSearch-VL：给定问题及音视频片段，检查线索、搜索来源、读取外部信息并直接回答。主代码和环境位于 coder ssh 3 的 `/home/jovyan/OneSearch-VL`，由 Pixi 管理。

## 1. 本轮已完成的数据

**Development：16 道直接搜索问答，8 个原始录制，16 个真实裁剪片段。** 所有片段都有音轨和视频轨，并已生成可播放的预览。索引包含 10 个音视频来源、6 个冻结网页；同类素材保留为候选干扰项。

| 场景 | 题数 | 请求的外部信息 | 音视频线索的预期作用 |
|---|---:|---|---|
| 机械运转 | 4 | 制造商、年份、展示地点、功率、设备归属与重量 | 区分蒸汽机、印刷设备及同类机器 |
| 现场弦乐 | 4 | 乐器、演奏团队、城市、国家、日期 | 使用乐器外观、人数、演奏动作和声音构造查询 |
| 打击乐与节奏活动 | 4 | 地点、仪式、活动名称、参与者、发布信息 | 区分传统锣鼓与健身节奏活动 |
| 自然环境 | 2 | 景观名称、地区、相邻城镇与距离 | 从水流场景与环境声搜索具体录制 |
| 动物 | 2 | 鸟类名称、发布者与页面日期 | 从动物外观、动作及可能的声音寻找对应录制 |

33 个候选答案字段都有同一发布来源的冻结文本支持语句。候选答案及支持区间存入私有文件；题目与音视频片段交给 actor。它们是 development candidates，人工确认的 benchmark gold 目前为 0。发布日期、页面日期和录制日期不能混用，审核时需要逐题确认。

**Train：另生成 6 个匿名片段搜索问答任务，来自 3 个训练录制。** 使用与 dev 相同的输入及工具协议，涵盖机械、音乐和降雨。这 6 个任务尚未产生匿名输入下的真实 teacher rollout，不能新增计为 SFT 轨迹。

上一阶段已有的 6 条 assisted silver 和 8-step LoRA checkpoint 保留。它们使用旧的来源可见输入，只作为训练工程与行为诊断证据；新协议的性能比较需要重新采集训练轨迹。

## 2. 问题与输入如何构建

输入是约 2 秒的匿名音视频片段和自然问题。片段 ID 独立随机生成，不使用原始来源 ID。片段从零开始计时，私有 lineage 保存原始区间、AV 时间映射、父录制与哈希；actor 不接收这些对应关系。

匿名片段不进入检索索引，也不携带发布者标题、URL、说明文本或素材哈希。`inspect_av` 对真实裁剪素材执行；`read_text` 不能读取匿名片段。目标来源只有实际搜索返回后，才能被读取或检查。

工具路径因此是：

```text
匿名片段 + 问题
    → inspect_av 得到可观察线索
    → search_sources 构造和改写查询
    → 检查候选 / read_text 读取支持信息
    → 直接回答
```

题目不要求跨来源比较或研究报告。正式扩展时，增加同类录制、不同机型、不同地点和相似声音，提升“识别正确来源”的难度；来源数量仍由题目决定。

## 3. 训练数据构建

训练与 benchmark 共用匿名片段输入和工具接口，答案候选的用途不同：训练探索允许明确标记的 silver，benchmark 的正式答案须经人工确认。

采集流程为来源收集、场景归类、匿名片段生成、问题与私有答案候选、真实 teacher rollout、筛选、ShareGPT 导出、token-label 核对、SFT。脚本指定路径的 curriculum 单独标记；主要训练集应包含模型实际选择的查询、工具动作和最终回答。

下一批目标是 **64 条有效训练轨迹**，优先覆盖：

- 利用声音事件、可见实体和动作构造搜索词。
- 空检索后改写关键词及查询语言。
- 从多个同类结果中选择正确来源。
- 在合法 AV 窗口内检查，并从工具错误恢复。
- 获得足够信息后直接回答，停止重复调用。

64 条是后续采集目标，不是当前完成量。不要通过重复一个录制的许多属性题代替来源多样性。继续扩展机械、音乐、自然与动物来源后，再进入 128–256 条轨迹的训练 pilot。

观察文本只作为上下文，所有 observation tokens mask；assistant 动作和 final 参与监督。Answer-only SFT 保持同一上下文，屏蔽 assistant 的 action targets；Trajectory SFT 保留动作及 final targets。两组从同一个 base 初始化，以相同题目和训练 token 预算比较，记录动作监督带来的额外 token 和 optimizer 工作量。

## 4. 评测哪些模型

主比较固定 controller 起点、AV observer、检索库、工具约束和推理预算。当前可用 controller 是公开 OneSearch-VL-8B，revision `2623603767b9fef7bab05ad0c3cd4d464301e8df`；AV observer 使用 pinned Qwen2.5-Omni-7B。冻结检索与开放 web 检索分别报告。

| 系统 | 比较目的 | 当前状态 |
|---|---|---|
| OneSearch-AV-Prompted | 原始 8B + AV 工具，测未增加 AV SFT 的起点 | 已有旧协议试跑；新匿名任务尚未执行模型评测 |
| OneSearch-AV Answer-only SFT | 测仅训练最终答案的收益 | 需要在新协议下构建训练集并训练 |
| OneSearch-AV Trajectory SFT | 测查询与工具动作监督的收益 | 已有旧协议 6 样本/8-step 工程 checkpoint；正式比较须重新训练 |
| 无搜索对照 | 检查问题是否真正需要外部信息 | 在同一批题及 observer 上执行，记录推理预算 |
| 去音频 / 去视觉 | 测声音、画面分别贡献多少 | 必须实际重新运行相应模态 observer，不能只删文字描述冒充模态消融 |
| Qwen2.5-Omni controller 对照 | 区分 controller 起点的影响 | 有模型资产及旧协议探索；新协议未执行 |
| 强闭源工具 teacher / judge | 提供更强轨迹及语义评分参考 | 服务配置尚未验证；不能计入已有结果 |

第一阶段优先完成前三组，不同时扩散到很多不同规模的模型。后续扩展 Qwen 系列原始模型时，需单独固定模型 revision、prompt 和工具能力。

## 5. 评测维度与计算方式

| 维度 | 主体 | 计分方式 |
|---|---|---|
| Answer Accuracy | 正式主指标 | 在人工确认的 reference 上评价自由回答是否正确；每题一个总体正确性结果，并保存逐字段判断 |
| 来源发现 | 检索过程 | 目标来源在实际返回结果中的 Recall@1/3/5，以及最终是否读取了目标页面；来源网页和媒体的同源关系以私有 family 映射折叠 |
| 任务完成率 | 全 episode | 在预算内返回有效最终回答的题目比例；失败、截断和解析错误均计入分母 |
| 工具使用 | 每次决策与调用 | 合法动作比例、成功调用、重复调用、错误恢复、检查区间是否合法 |
| 成本 | 实际运行 | 工具调用数、请求 AV 秒数、observer 执行/回放次数、输入和生成 tokens、运行时间 |
| 引用支持 | 回答与真实 observation | 被引用的信息是否实际读取，身份/地点等文本属性是否由文本支持；不能用 AV 区间支持不可见的发布者属性 |
| 声画贡献 | 模态消融 | AV、video-only、audio-only 的 paired difference；按相同任务及成本预算报告 |

当前自动诊断实现 `score_candidate_diagnostic`：请求字段匹配、目标是否检索/读取、工具状态、媒体准备与语义观察次数、预算。它返回 `answer_accuracy=null`，因为 pending publisher 字段匹配无法判断否定、矛盾、额外错误陈述或完整语义。短字段按词边界匹配，避免把 `Al` 在 `California` 中误判为人名命中。

Formal accuracy、source recall 和 tool validity 分开看：答案碰巧正确但没有发现来源，与正确查到来源并回答，代表不同的策略能力。消融以 performance 和成本解释方法，不以更多协议检查充当研究贡献。

## 6. 分割与审核

Dev 保留原有预留录制，Train 使用独立的训练素材池。本轮核对原始文件和公开 origin：两个集合没有相同字节或相同发布 origin。一个录制及其网页、裁剪片段归入同一 family；本批 16 个 dev 问题对应 8 个父录制，不是 16 个独立录制。

更广泛的近重复、转载或同一事件关系仍需内容审查。暂不把当前划分称为正式 source-family test split。独立 test 尚未构建。

审核页支持：先盲看片段，再展开候选答案及原文支持；记录保留/修改/删除、声音必要/辅助/冗余、是否必须搜索和自由修改建议。下载的 JSON 是审核草稿，不能自动成为 gold；实际 reviewer 身份和独立答案复核记录需进入后续 reference 冻结流程。

审核入口：[16 个可播放 development cases](http://127.0.0.1:8781/OneSearch-AV-Anonymous-Dev-Review-2026-10-12/review/index.html)。

## 7. 已执行验证与后续运行

所有 dev 查询素材核对了实际 bytes 和 AV stream，预览保留音视频；浏览器逐个确认 16 个视频已加载、无媒体错误。真实 CPU 连接性执行包含匿名片段媒体准备、搜索和文本读取；controller 是明确标记的 fixture，没有生成模型回答或语义观察。

264 个研究测试通过后，又扩展了 train builder 的同协议支持；最终 research 回归为 **265/265**。本轮未修改 trainer labels 或 optimizer，上一轮 SFT tests 6/6 的结果保留。本轮新增 GPU Job 时间为 0；最近已执行的累计预算为 1969/2100 Job 秒。

新增 CLI 可选择 `replay`、`media` 或 `native` observer，并选择 `canonical` 或 `native` controller 协议。动态 native 配置及 CLI 解析通过测试；本批匿名片段尚未执行 GPU observer/controller 推理。

后续动态 AV 模型运行使用 `runtime/av-observer/pixi.toml`，模型 manifest 为 `research/runs/av-native-smoke-20261010-v1/runtime-setup/model-manifest.json`，runtime identity 为 `runtime/av-observer/runtime-manifest.json`，配套 lock 为 `runtime/av-observer/pixi.lock`。GPU launcher 必须先核对分配设备，并在单独的有界 run 中执行。

先完成新协议 teacher 的小批成功轨迹，再扩充到 64 条和三个主 baseline 的真实比较。当前任务继续保持 active；题目审查、数据规模、新协议训练与正式性能结果仍需继续完成。
