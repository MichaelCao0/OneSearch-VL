# OneSearch-AV：2026-10-16 执行记录

证据时间：北京时间 2026-10-16 10:55:11（UTC 2026-10-16T02:55:11.961148+00:00）。

当前 1,125 个不同问题完成 silver 保留与 CPU 检查，待 CPU 审核 0。下一训练节点 2K，完整目标 20K train + 500 train-validation + 200 dev + 1K test。

本轮推进了以下实际工作：

- 恢复训练生产：保留 v7 的 502/503/transport 停机证据，复用 24 个录制已成功出题的 37 个候选，未重新请求失败或未知出题。v8 的 smoke 分片缺少 corpus 路径，在 API 前停止；v9 连接原冻结 corpus 并先验证 `load_frozen()`，随后单题搜索、silver review 和 CPU audit 全部通过，训练池由 1,124 增至 1,125。其余 36 个候选按 8 workers 继续。
- 扩展 benchmark：已有 139 个 dev 候选，v6 worker 目标累计 200，最多 160 个新录制 author attempts，只消费训练前预留的家族。当前等待新预留批次。human gold 为 0。
- 形成精确输入质量清单：冻结 1,124 题中，257 题完成自动原生 AV 审核（194 main search、32 control、8 revise、23 exclude），17 unknown，850 未审。257 份原始生成、verdict 和实际 AV receipt 均一致。
- 新增 256 录制内容审核：输入、题干和训练行检查 256/256 通过，此前付费 task、输入媒体和父录制交集均为 0。快照完成 44、失败或未知 3、正在执行 4；这是执行快照，非最终审核结论。
- 交付新增 12 train + 12 benchmark 四模式审核包：rsync partial 传输完成，120 文件哈希、96 HTTP Range 检查和两页面抽样播放通过。来源上下文约 60 秒，短录制显示真实长度；原声、无音轨视频及纯音频均可切换。没有代填人审结论。

500 同题控制的已完成结果保持：heldout 共同 38 题中 base AV 0/38、answer-only AV 3/38、trajectory AV 24/38、trajectory muted 22/38。Trajectory 对 answer-only 为 +55.3 pp，自动家族 bootstrap 95% [38.6,73.5] pp；同题同更新数，监督 token 数不同。正式结论继续由独立 benchmark 和有效 prompted baseline 检验。

本轮数据任务直接运行在 coder ssh 3 的 CPU 工作区，新增 GPU job 为 0。当前三个 worker 的 host、PID、start ticks 和 boot ID 均核验匹配。主代码冻结保持一致，原始失败、未知和历史 exports 保留。

## 活动任务

| 任务 | 远端运行目录 | 当前职责 |
|---|---|---|
| 训练生产 | `/home/jovyan/OneSearch-VL/research/runs/av-data-production-20261016-v9/` | 恢复候选、消费未请求素材，扩到 2K |
| Benchmark | `/home/jovyan/OneSearch-VL/research/runs/av-reserved-benchmark-20261016-v6/` | 从新预留家族补至 200 dev 候选 |
| 内容审核 | `/home/jovyan/OneSearch-VL/research/runs/av-train-content-audit-256-20261016-v2/` | 256 个不同录制，4 workers |

## 可播放交付

- [12 个新增训练案例](http://127.0.0.1:8781/OneSearch-AV-Priority-Review-2026-10-15/train-modality-review/index.html)：重点可看第 5、10–12 题。
- [12 个新增 benchmark 案例](http://127.0.0.1:8781/OneSearch-AV-Priority-Review-2026-10-15/benchmark-modality-review/index.html)：重点可看第 7、10、12 题。

下一步由扩产和审核结果冻结 2K 清单，随后进行 full Thinker SFT 与立即评测。同题 answer-only、基座、静音以及实际监督 tokens 和推理预算一并记录。2K 是中间训练节点，后续继续推进 20K 数据和正式 benchmark。

完整研究与投稿方案见《OneSearch-AV-论文构想与投稿计划-v5-2026-10-16.md》。18 份远端证据已逐文件回读校验，保存在 `reports/continuation-2026-10-16-v1/evidence/`。


## 2K 扩产执行（2026-10-16T03:25:47.016648+00:00）

启动 av-data-production-20261016-v10：835 组 query、并行采集预取、每 cohort 一次 materialize；目标池 2200、最终筛选 2000。当前 CPU 合格 1151；精确内容抽检合并后主搜索 384、未抽检 silver 621、已知排除或暂缓 146。质量视图 av-train-quality-view-20261016-v4。Benchmark v7 等待新预留来源。具体预算与口径见 OneSearch-AV-2K-Expansion-v1-2026-10-16.md。未提交新 GPU Job，未 push。

11:28 北京时间验证：v10 单题完整链路新增 1 条 CPU 合格问题，总池 1152；19 题生成中；新素材 62 段。30 项回归通过；freeze hash 一致。证据 records/2026-10-16-expansion-2k-v1/final-live-snapshot.json。


## 2K 扩产与 full Thinker 训练准备 v2（2026-10-16 12:07 北京时间）

CPU 合格不同问题 1175，按质量视图排除已知 146 题后可选池 1029，距最终 2000 差 971。新采集 cohort-01 为 250 录制，预留 benchmark 25 家族；153 录制进入训练出题，现场 45 接受候选尚未计入合格池。benchmark 旧 dev 候选 139，新批 18 接受出题候选尚未完成发布。四个 worker 身份均存活，冻结源码一致，未触发 provider stop。

新增 research/av_train/pilot_freeze.py，6 项新测试及合计 36 项回归通过。新版 clean500 保留旧500中的429题、替换71题，包含185内容审核主搜索题及315未独立审核silver。自动 watcher 在2000个可选问题达到后冻结 clean2000 并检查 clean500 嵌套。CPU meta 调用原 adapter 确认 full Thinker 7615616512 可训练参数、1316197376 冻结参数；尚未执行真实 full Thinker GPU 更新。候选资源4GPU/ZeRO-2，eff batch4须按world size调整accumulation。

具体过程、token量、失败请求诊断与训练/评测安排见 [2K扩产与训练准备 v2](OneSearch-AV-2K-Continuation-v2-2026-10-16.md)。证据 records/2026-10-16-2k-preparation-v2/。本轮数据CPU后台运行，未新交GPU、未push、未重放付费失败调用。

12:09回读：benchmark dev候选157（本批新增18）；训练本批64接受出题候选尚未计入合格池，1175 CPU合格/1029可选保持；四个worker存活、冻结一致。见records/2026-10-16-2k-preparation-v2/handoff-live.json。


## 2026-10-16 14:17:26 扩产恢复v11

CPU合格1317，排除已知问题后1171；benchmark dev候选173。v10于13:46因Teacher HTTP503 burst停止。本轮启动v11回收29成功回执并仅消费14未请求问题及下一批250未出题录制；预取、benchmark v8和freezer v2已接回。29条成功回执正在CPU检查，尚未计入合格池。主代码未修改、无付费失败重放、无新GPU、未push。详见 [扩产恢复进度](OneSearch-AV-Production-Recovery-v11-2026-10-16.md)。


## 16:00 扩产恢复v12

1359 CPU合格、1213可选、192 dev候选。v11新增42合格轨迹后因400/503 burst退出；v12复用144准备媒体，保留6付费回执，仅请求138未调用出题来源，2 workers、暂停新增下载、80GiB空间下限。三worker存活，freeze一致；已出现新成功出题候选。详见OneSearch-AV-2K-Recovery-v12-2026-10-16.md。


## 16:22 持续扩产接续v13

1359 CPU合格/1213可选，本批82接受出题候选尚未验收；benchmark192。磁盘实测1457.12GiB可用，恢复最多2个cohort缓存的素材预取。v13等待v12正常完成后导入实际exports和预留来源再调用Teacher，异常则停止不重放。接续预算1900 author/10cohort，沿用绝对deadline；最低空间150GiB。worker身份与源码freeze匹配，无新GPU、未push。详见OneSearch-AV-Continuation-v13-2026-10-16.md。


## 16:35 Motivation与当前证据

Learning to Search from Audio-Visual Clues：1359 CPU合格、1213可选、本批114候选正在构建共享语料、192 dev候选。历史500轨迹24/38对同题answer-only3/38；声音增益待独立模态评价。full Thinker真实GPU更新尚未执行。见OneSearch-AV-Motivation-and-Progress-2026-10-16.md及records/2026-10-16-motivation-progress/。


## 17:32 三条实验主线：学习、模态、数据方法

已批准A学习与规模、B声音与互补、C数据方法。新实验模块experiment_suite.py、4组full Thinker模板及26项相关回归已完成。当前1405不同CPU合格题，其中新增46来自v12已完成子批次；排除已知146后1259可选，距2K差741。192 dev候选/126录制对1405训练题的录制、家族、媒体与实际轨迹target URL检查通过。

两个CPU/API小试已完成：模态预审31完成/1失败，28keep、3reject自动建议；publisher-only对照28完成/4次HTTP503，56格式与字面支持候选。合计64新调用，无付费失败重放。构造诊断复用历史AV成功来源且prompt/schema有差异，后续共同prompt消融再估计AV输入作用。人工gold0；2K正式训练和full Thinker GPU smoke尚未执行。本轮无新GPU、未push。

详见[三条实验主线与执行计划](OneSearch-AV-Three-Axis-Experiments-v1-2026-10-16.md)，证据records/2026-10-16-three-axis-experiments-v1/。35个远端证据文件已回读并SHA256核验。


## 21:59 Full Thinker与数据方法诊断

1652 CPU合格不同训练题，排除已知146后1506可选，距2K差494；当前79候选进入轨迹/审核/CPU。v13生产、预取、benchmark v12与freezer v4身份存活，20项生产冻结源码一致。Benchmark累计240候选，其中217已冻结，208自动审核为192keep/2revise/14reject，人工gold0；候选目标提高到300供200dev人审筛选。

独立Pixi环境runtime/sft-full新增DeepSpeed0.16.9与CUDA12.6 compiler。4×H20Z、ZeRO2、12个完整极值样本、最长20014 tokens，原SFT/src/train.py实际完成3步，7615616512 trainable/1316197376 frozen；peak allocated69–78GiB/reserved102–119GiB。外置HF scheduler被初版验证器误判，保留Kubernetes Failed状态，修正后CPU回读权重、4 optimizer分片、scheduler与4 RNG通过，无重训。

完整checkpoint原生加载与2题×AV/muted共4条件generation通过；旧静音helper将FFV1放入MP4的问题已用共享modality_inputs.py修复，packet hash一致、缓存收据复用及2类编解码测试通过，只补跑缺失条件。三个GPU作业均终止，job时钟估计0.5717 GPUh；新代码18项回归通过。

28同来源盲审94问题：AV生成37/38可用搜索题，publisher生成7/56；46/56文本题有身份线索泄漏。结果为历史成功来源上的自动诊断，两侧生成prompt/schema不同；正式消融将统一实体隐藏规则与共同prompt。无付费失败重放，未push。详见执行报告及records/2026-10-16-full-thinker-continuation-v1/。
