# OneSearch-AV：数据构建、benchmark 设计与审核执行记录

日期：2026-10-15。主开发位置：`coder ssh 3:/home/jovyan/OneSearch-VL`。本文补充《论文构想与投稿计划 v2》，记录已经运行的数据流程、可审查案例和下一组训练对照。

## 1. 研究主线与交付规模

输入匿名音视频片段和问题，原生 Omni 模型通过检查片段、构造查询、搜索与读取来源，回答片段之外的事实问题。第一篇沿用 OneSearch-VL 搜索问答范式；主基座为 `Qwen/Qwen2.5-Omni-7B`，训练复用原 `SFT/src/train.py`。

论文围绕三项预期贡献展开：真实 AV 条件下的搜索轨迹数据引擎、独立 AVSearch-Bench，以及同基座上可归因的搜索能力增长。规模目标为 20K 不同训练问题、500 train-validation、200 benchmark dev、1K test；下一训练节点为 2K。

北京时间约 18:05 的现场计数为 **988 个不同训练问题**，全部通过成对导出及原 trainer 的 CPU 媒体/监督标签检查。trajectory 与 answer-only 是同题两种监督，不合计为两道题。该计数表示 silver 数据生产和训练输入检查完成；内容质量另由真实 AV 抽查、分层标签与人工记录描述。

## 2. 当前实际运行的数据引擎

生产步骤为：采集真实媒体与发布者事实材料 → 归并录制家族 → 预留 benchmark 来源 → 原生 AV 条件出题 → 匿名片段上的实际工具搜索 → 独立 silver 答案审核 → trajectory / answer-only 成对导出 → 原 trainer CPU 检查。

当前训练生产 run 为 `research/runs/av-data-production-20261015-v7`；预留 benchmark run 为 `research/runs/av-reserved-benchmark-20261015-v4`。两个 worker 的 host、PID、进程启动 tick 与 boot ID 均已核实。正在运行的生产源码与冻结哈希一致。

新一轮采样优先级为音乐 30%、赛事 30%、现场活动 20%、其他 20%。该比例是采样目标。现场已接收 67 段同时具有音轨和画面的素材，按采集查询类别分别为音乐 10、赛事 37、活动 9、其他 11；真实内容类别将在后续 AV 审核中标注。素材数量和题目数量分别记录。

该采集批次目标为 250 段，下载上限 700 次、总字节上限 15GB、单素材上限 150MB。采集按查询批次推进，出题和 rollout 配置 8 路并发。本轮新增训练题仍来自前一采集批次；音乐/赛事/活动新素材尚未计入 988 条合格题数。

API 失败或结果不确定的原始 attempt 保留；生产不自动重放付费请求。CPU 数据生产直接在 workspace 后台运行。

## 3. 新增 29 个真实声画案例

本轮从新增且 CPU 检查通过的训练数据中，选择 24 个不同录制；排除此前已付费内容抽查的素材。另补齐 cohort-00 benchmark 中尚未展示的 5 个录制。29 次有界原生 AV 自动审核全部返回；本轮审核没有重放付费请求。

| 案例包 | 案例/录制家族 | 自动处理建议 | 模态角色 |
|---|---:|---|---|
| 新增训练 | 24 / 24 | 17 主搜索、6 感知对照、1 删除 | 18 视觉为主、4 声画联合、2 语音/文本为主 |
| 补充 benchmark | 5 / 5 | 4 保留、1 修改 | 3 视觉为主、2 声画联合 |

训练抽样的真实内容为：机械/交通 9、动物/鸟类 7、环境/事件 4、演讲/访谈 3、其他 1。这一抽样来自复用素材，不能用于描述新一轮音乐/赛事/活动采样配额的完成情况。

这些结果已写入独立 `automatic-quality-partition.private.json`，用于后续主搜索、感知对照与修订队列。原训练导出和历史检查结果保留。当前样本规模适合发现问题类型和指导采集，正式质量比例将以分层抽样和人工标注统计。

新 benchmark 包与此前 24 题包合计展示 29 个录制，各展示一个代表问题；该 cohort 一共已生成 43 道 dev 候选题。展示覆盖与候选总量分别计数。正式 human gold 当前为 0。

## 4. 可播放审核包与验收

本地入口：<http://127.0.0.1:8781/OneSearch-AV-New-Data-Review-2026-10-15/index.html>。

每题提供实际模型输入的原声 AV、物理移除音轨的视频、隐藏画面的音频和来源上下文。来源上下文供审核，模型仍使用匿名输入片段。候选答案与自动意见折叠显示，先看听实际片段，再判断可辨认线索、搜索必要性、答案和时间区间。

已完成的交付验证：145 个媒体/静帧文件哈希一致；116 次 HTTP Range 请求全部返回 206。浏览器实际抽查了训练首题的四种模式与补充 benchmark 首题的原声模式，时间推进、可见画面及播放错误状态已记录。训练画面为 608×1080，benchmark 首题为 640×360，均无播放器错误。

浏览器验收是播放抽查，逐题语义判断保留在审核记录中。本轮没有改写审核人的选择，页面状态仍为训练 0/24、补充 benchmark 0/5。

## 5. 从页面审核记录到版本化标注

新增模块：`research/av_data/review_decisions.py`。输入为页面导出的 `av-searchqa-authoring-review-draft.v2` JSON；可导入多位审核人。

```bash
pixi run --locked python -m research.av_data.review_decisions \
  --package <审核包目录> \
  --review <审核人A导出.json> \
  --review <审核人B导出.json> \
  --out <新的版本化输出目录>
```

导入时核对题目、参考答案、媒体和来源 manifest 的版本；拒绝未知题号、过期答案、被替换媒体、重复题目及重复审核身份。输出逐题审核记录和汇总，分别表示待审核、待第二人、需裁决、需修改、排除、感知对照和一致保留候选。

两人一致项进入候选冻结队列；随后完成事实来源、关键区间、录制家族隔离与分歧裁决，形成正式 benchmark。最新 11 项测试全部通过，覆盖声学角色分歧、单审核人、重复身份、媒体替换、过期参考、空记录和伪造 gold 标志。

用户对题型方向的“可以的”反馈用于继续构建；逐题记录以页面导出和事实核实为准。

## 6. Benchmark 的场景、模型和评价

机器可读设计文件为 `research/configs/avsearch_benchmark.v1.json`，状态为 development candidate。音乐、赛事、现场活动为主要场景；构造 pilot 后校准模态角色配比。计划覆盖声画联合、声音为主、视觉为主与语音/文本线索，并在冻结 test 前确定定义和预算。

| 条件 | 模型/监督 | 实验职责 |
|---|---|---|
| B0 | Qwen2.5-Omni-7B direct QA | 感知和已有知识基线 |
| B1 | 同基座 strong prompted search | 检验训练收益；提示词只用 dev 选择 |
| B2 | 同题 answer-only SFT + 相同工具 | 区分答案监督与轨迹监督 |
| B3 | 同题 trajectory SFT | 主训练比较 |
| B4 | 经过验证的数据策略 + trajectory SFT | 检验数据引擎的独立价值 |
| B5 | 官方 OneSearch-VL-8B，静音视频 | 视觉搜索 agent 参照 |
| B6 | Video + ASR search | 区分原生声学信息与转写文本 |
| B7 | Gemini 原生 AV + 同工具 | 闭源能力参照，标明与 teacher 的模型家族重合 |

主指标为预算内的 task success；同时记录字段正确率、实际已读证据支持、来源检索/读取命中、模态干预差异和成本。现有 `supported_success` 是发布者参考支持代理指标，正式实际观察证据评分将绑定 actor 已看到的工具输出。

比较采用相同 corpus、工具、推理预算与 dev 选定提示词。当前 pilot 为最多 8 轮、每轮 512 输出 token、greedy；正式预算在 dev 上确定后冻结。模态分析采用原声/静音/仅声音/video+ASR；误差按线索辨认、查询、来源选择、片段检查、事实提取和最终回答拆分。统计按录制家族做配对 bootstrap，模型结果不参与 test 选题。

## 7. 训练证据与下一实验

已经完成的 clean500 trajectory pilot 使用 500 道题、494 个录制，Thinker LoRA rank 8 / alpha 16，学习率 1e-4、有效 batch 4、两轮，共 250 次真实优化更新。模型权重和 optimizer/scheduler/RNG 状态已检查。

在 39 道已判定 heldout silver 问题中，原声 clean500 完成 23 题（59.0%），静音完成 21 题（53.8%）；dev 已判定 15 题中分别完成 8 与 13 题。既有 base 的协议完成较弱，因此下一组关键实验同时补 strong prompted base 和同题 answer-only，检验收益来源。

500 条 answer-only 数据与配置已冻结：与 trajectory 相同的问题、初始音视频和实际最终回答；省去工具历史及最终 think 前缀。训练步数、epochs、学习率、LoRA 配置和 seed 相同，输入/监督 token 量分别报告。该对照从同一基座初始化。训练后紧接原声与静音的同题评测；2K full Thinker 路线在新数据冻结后继续。

## 8. 证据入口

- 主论文材料：`docs/research/OneSearch-AV-Paper-Dossier-v2-2026-10-15.md`。
- 本次完整证据：`docs/research/records/2026-10-15-new-data-review-v1/`。
- 现场计数：`execution-snapshot.json`，由 `verified_pool()` 从实际成对导出计算。
- 29 次内容审核与质量分组：`research/runs/av-new-data-review-20261015-v1/`。
- 页面传输、Range、浏览器播放收据与截图已归档；页面空审核状态也保留。
- 本记录未改变生产 worker、旧数据或既有训练结果。后续规模、模型成绩和 gold 数随新的实际产物追加。
