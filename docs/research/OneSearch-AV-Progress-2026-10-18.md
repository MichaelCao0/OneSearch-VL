# OneSearch-AV 执行进度（2026-10-18 20:26，北京时间）

目前共有 **2,092 个不同训练问题**完成成对导出、媒体与监督标签 CPU 检查，比昨日的 1,974 条增加 **118 条**。扣除冻结质量视图的 146 个排除项，当前 **1,946 条 eligible silver**，距离正式冻结 2K 还差 **54 条**。eligible silver 包含尚未独立内容审核的问题，不代表人工 gold。

## 数据生产

将既有 collector 的单次查询返回上限由 20 提高到 50 后，已实际采集三批共 **750 段候选音视频**。前两批进入 authoring / rollout / review，第三批 250 段已下载保留，未进入本轮训练计数。

上一轮 `av-data-production-20261017-v1` 于 **2026-10-17 19:56** 因 `provider_transport` 触发保护退出，freezer 如实记录 `producer_finished_below_target`。旧 producer、prefetch 和 freezer 均已核对进程身份，确认不在运行。付费失败或不确定请求保留原记录，没有重放。

已完成逐任务恢复检查：停下的 chunk 有 25 条 producer 记录完整的已完成轨迹、8 条失败轨迹和 15 道未建立 attempt state 的 pending 题；同一 authoring batch 还有 33 道从未进入 rollout chunk 的题。已校验原 plan、轨迹完成记录和源码哈希。

当前已启动新的 CPU 后台恢复：

- producer：`/home/jovyan/OneSearch-VL/research/runs/av-data-production-20261018-v1`
- freezer：`/home/jovyan/OneSearch-VL/research/runs/av-learning-2k-preparation-20261018-v1`
- 首先为 25 条已完成轨迹做首次独立自动审核，随后只对 15+33=48 道未执行题目做新 rollout。
- 8 条旧失败轨迹明确排除。新恢复预算为 6 小时、最多 25 次补审和 48 个新 rollout slots，沿用每题最多 8 次 actor 请求；自动收费失败重试为 0。
- 审核并发 1，后续 rollout 并发 2；不重新采集素材，也不提交数据生产 kjob。

2026-10-18 20:26 回读，producer 与 freezer 身份均存活；11 项恢复冻结文件与 5 项 freezer 冻结文件一致。新补审已有 **3 条 completed，2 条 success**，另 1 条 running，未出现新的 provider-stop。这些结果尚未完成导出和原训练器 CPU 检查，**没有提前计入 2,092 / 1,946**。

## Benchmark 与训练

Benchmark 本轮重新按实际任务与来源记录核对，仍为 **304 道候选题，198 个不同录制**。其中此前冻结的 217 题已有 208 条自动审核：192 keep、2 revise、14 reject，9 条 unknown/unreviewed；新增 87 题尚未纳入同一完整审核。人工 gold 为 0，后续从候选池中核实并筛出 200 道 dev。

**新的 Full Thinker 500 / 2K 正式训练尚未启动，没有新的性能分数。** 已完成的资源与接口验证包括：4×H20Z、ZeRO2、原 `SFT/src/train.py` 实际 3 个 optimizer updates，以及完整 checkpoint 的 native AV / muted 推理接口验证。该 smoke 不能作为正式学习收益。

历史 500 题 LoRA 在共同 38 道 silver / frozen-corpus 评测题上的 answer success：base 0/38、answer-only 3/38、trajectory 24/38、muted trajectory 22/38。历史数据集与新 clean500、Full Thinker route 不同，后续正式规模对照使用冻结的嵌套 500/2000，并报告训练 tokens 和 GPU 成本。

最新集群审计没有 active GPU Job，昨天之后没有新增 GPU Job。远端剩余磁盘约 **1,185 GiB**。

## 下一项交付

优先完成最后 54 条可选数据、产出 clean2000 冻结与嵌套成员校验，再按实际长度和媒体极值准备正式 Full Thinker 训练与即时任务成功率评测。Freezer 负责冻结数据，不会自动提交 GPU 训练。

本轮恢复采用原生产函数和训练器，主代码在 `coder ssh 3:/home/jovyan/OneSearch-VL`；没有改动活跃的冻结源码，没有 commit/push。实际回读与启动前检查保存在本地 `reports/progress-2026-10-18/`。
