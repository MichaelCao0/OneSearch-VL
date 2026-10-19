# OneSearch-AV Research Ledger 2026-10-17

# OneSearch-AV 执行进度（2026-10-17 15:55，北京时间）

当前训练池为 **1,974 个不同问题**，已通过成对 trajectory/answer-only 绑定与原训练器 CPU 媒体、监督标签检查。扣除冻结质量视图中的 146 个已知排除项，剩 **1,828 条 eligible silver**，距离 2K 训练里程碑还差 **172 条**。相对昨天 21:59 的 1,652 条 CPU 合格数据新增 322 条。

这些是自动验收的 silver；eligible 不表示所有问题都经过独立内容审核或人工确认。同题 answer-only 导出不重复计入数据量。

## 三条实验主线的实际进展

| 主线 | 已完成 | 下一项交付 |
| --- | --- | --- |
| A：轨迹训练与规模收益 | clean500 已冻结；完整 Thinker 在 4×H20Z、ZeRO2 上实际完成 3 个 optimizer updates，并回读权重、optimizer、scheduler、RNG；完整权重的 2题×AV/muted 推理接口验证通过 | 补齐 eligible 2K，冻结嵌套的 500/2000，完成 trajectory/answer-only 正式训练与相同问题集任务成功率评测 |
| B：音频贡献与 benchmark | 实际累计 **304 道候选题，198 个不同录制**；其中此前冻结的 217 题完成 208 条自动内容审核，192 keep、2 revise、14 reject，9 条 unknown/unreviewed | 对新增 87 题做统一内容与来源隔离审核，整理供 200 dev 人审筛选的题集；完成 native AV/muted/audio-only/video+ASR 对照 |
| C：造数方法 | 已完成 28 个同来源、94 道问题的混合盲审诊断；AV 生成可用搜索题 37/38，publisher-only 7/56；后者 46/56 存在身份线索泄漏 | 统一 prompt/schema 和实体隐藏规则，在新来源上进行正式构造消融，再验证训练收益 |

Benchmark 人工 gold 当前仍为 **0**。新 87 题已经从预留来源生成并按 task_id 去重，但不能把旧 217 题的完整隔离检查自动外推到全部 304 题。

C 线结果是历史成功来源上的自动诊断，生成 prompt/schema 不同，不能解释为 AV 输入的因果增益，也不能解释为学生训练收益。

## 当前任务和瓶颈

上一轮 `av-data-production-20261016-v13` 于今天 **09:51** 跑完预设 10 批后正常结束。没有 provider-stop，付费请求没有自动重放。Benchmark v12 已正常达到 300 候选目标，实际产出 304；旧 2K watcher 如实记录 `producer_finished_below_target`。

已发现来源获取的边际效率下降：旧 collector 每个查询最多取得 20 个结果，查询库被反复访问后，新增 usable AV 从每批 250 段降到 24、12、6、5 段。第 10 批扫完 835 个查询只得到 5 段新录制。代码也没有使用结果翻页，因此仅更换查询顺序不足以保证扩产。

本轮使用原 collector 已支持的参数，将 `per_query_limit` 从 10 调到 25，对应 `gsrlimit` 从 20 提高到 50；新版本不修改原采集或训练主干。已启动新的 CPU 后台产数与 freezer，排除全部已有收费 source IDs，保留旧失败或不确定收据，不重放收费失败请求。

- 新产数：`/home/jovyan/OneSearch-VL/research/runs/av-data-production-20261017-v1`
- 新冻结监视：`/home/jovyan/OneSearch-VL/research/runs/av-learning-2k-preparation-20261017-v1`
- 预算：8 小时、最多 4 个采集批次、700 次新 author 调用、1,000 个新 rollout candidate slots；teacher 并发 2，磁盘下限 150 GiB。
- 2026-10-17 15:55 回读：producer、prefetch、freezer 的 host/PID/start ticks/boot ID 均匹配存活；20 项生产冻结文件和 5 项 freezer 冻结文件均一致。producer 正在等待第一批新来源，尚无本轮新 teacher 调用；首批采集已开始，新数据尚未计入 1,974。
- 最近集群审计：没有运行中的 GPU Job；磁盘剩余约 1,227 GiB。本轮没有提交新 GPU Job。

新来源获取的实际增量仍需从本轮产物验证；提高结果数是已经启动的修正，尚不能报告为解决了全部来源瓶颈。

## 训练效果与下一步

此前历史 500 题 LoRA 在共同 38 题上的 answer success 为：基座 0/38、answer-only 3/38、trajectory 24/38，muted trajectory 22/38。在该设置下轨迹训练有收益，音频的稳定增益仍需更大、模态分层的评测。历史 500 与新 clean500 的成员和训练方式不同，不能混为正式规模对照。

**正式 Full Thinker 的 500/2K 训练尚未启动，新 2K 性能结果尚不存在。** 已完成的是资源和推理接口验证。Full Thinker 训练 smoke 的 Kubernetes Job 因初版 scheduler 检查器误判保留为 Failed；修正检查器后 CPU 回读通过，没有重跑训练。

下一步按已批准的三条主线推进：先达到 2K eligible 并冻结，再依据实际 2K 的长度/媒体极值补充资源检查，运行原 `SFT/src/train.py` 的正式训练和即时成功率评测；同时推进 304 候选的 benchmark 审核与统一规则的造数消融。Freezer 只冻结数据，不会自行把配置准备标为训练完成。

本次远端审计、启动前检查及身份回读保存在本地 `reports/progress-2026-10-17/`。主代码和数据仍位于 coder ssh 3 的 `/home/jovyan/OneSearch-VL`，未 commit/push。
