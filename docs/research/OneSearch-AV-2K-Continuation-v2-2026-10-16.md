# OneSearch-AV：2K 扩产与训练准备进度 v2

执行快照：北京时间 **2026-10-16 12:07:03**（UTC `2026-10-16T04:07:03.172433+00:00`）。主代码和生产继续位于 `coder ssh 3:/home/jovyan/OneSearch-VL`，分支 `research/av-deepresearch`。

本轮已接通“扩产 → 按质量口径选题 → 自动冻结 2K → 原 trainer 训练准备”的链路。后台生产继续运行，2K 尚未完成。第一篇任务保持 OneSearch-VL 风格的音视频搜索问答；完整论文目标仍为 **20K train、500 train-validation、200 benchmark dev、1K test**。

## 1. 当前实际产出

| 项目 | 实际数量或状态 |
|---|---:|
| 轨迹保留且通过原 trainer CPU 媒体/标签检查的不同问题 | **1,175** |
| 已知应排除或暂缓的问题 | 146 |
| 排除上述问题后的 pilot silver 可选池 | **1,029** |
| 到最终 2K 可选问题的缺口 | **971** |
| 本批新增采集且完成去重登记的录制 | 250 |
| 本批预留 benchmark 候选家族 | 25 |
| 本批已准备、正在出题的训练录制 | 153 |
| 本批训练出题结果 | 44 完成、11 失败、4 执行中；45 个接受的问题候选 |
| 此前累计发布的 benchmark dev 候选 | 139 |
| 本批 benchmark 出题 | 20 完成、5 失败；19 个原始问题、18 个接受的问题候选 |
| 下一采集批次 | 运行中；manifest 显示 87 个 complete assets，尚未形成最终新增录制清单 |
| human gold | 0 |

45 个训练问题仍须经过搜索轨迹、语义保留和 CPU 检查，才能计入 1,175。18 个 benchmark 接受候选也尚未计入正式发布的 dev 清单。配套 answer-only 与轨迹是同题对照，不重复计数。

生产、预取、benchmark 和自动冻结四个 worker 的 host、PID、start ticks、boot ID 均核验存活；生产与冻结模块的源码哈希一致。该时点没有触发 `provider-stop.json`，磁盘可用约 550.3 GiB。

## 2. 新增代码和自动冻结

主模块：`/home/jovyan/OneSearch-VL/research/av_train/pilot_freeze.py`。

这个模块直接读取现有 `verified_pool()`、原生 trainer 的逐行 CPU 检查回执和实际训练导出，完成以下操作：

1. 将问题、输入媒体 SHA、trajectory 行 SHA 与 answer-only 行 SHA 精确绑定到质量视图。
2. 排除已知 reject、revise、perception control 和审核 unknown；未做独立内容审核的题继续标记为 `not_content_audited` silver。
3. 确定性选择问题，避免同录制同问题重复；优先增加录制及来源家族覆盖。
4. 检查固定评测来源、媒体及实际工具 observation 与训练来源的隔离。
5. 导出原始轨迹和同题 answer-only，继承完全相同行的既有 CPU 解码/标签回执，记录媒体哈希及 token 数。
6. 生成原 `SFT/src/train.py` 使用的 full Thinker 配置。

自动冻结 run：`/home/jovyan/OneSearch-VL/research/runs/av-learning-2k-preparation-20261016-v1/`。

其 `watch.py` 每 60 秒检查实际合格池，在达到 2,000 个可选问题后写入 `clean2000/`，验证已冻结 `clean500/` 是其子集。watcher 不调用 Teacher、不提交 GPU。若生产结束但题数不足，会记录实际缺口。生产本轮的 CPU 合格池目标为 2,200，留出筛选空间。

继承已有 CPU 回执避免反复解码完全相同的训练行；本轮不将这种组合回执称为重新完整解码。GPU 训练准备状态仍为 `training_ready: false`，等待真实 full Thinker 资源预检。

## 3. 新版 500 题与规模对照

旧 500 题实验中有 **71 题**被后续质量视图标记为排除或暂缓：21 exclude、9 revise、19 control、22 unknown。本轮保留旧实验和数据，在新目录中保留其余 429 题，再补入 71 题，形成新版 500 题。

新版 `clean500/`：

| 项目 | 统计 |
|---|---:|
| 不同问题 | 500 |
| 独立内容审核支持的主搜索题 | 185 |
| 尚未独立内容审核的 silver | 315 |
| 轨迹 context tokens / epoch | 4,014,576 |
| 轨迹 supervised tokens / epoch | 110,243 |
| answer-only context tokens / epoch | 2,458,745 |
| answer-only supervised tokens / epoch | 30,097 |
| 最大 context tokens | 20,014 |

最终新 2K 将包含这 500 题。后续比较使用同一质量规则、同一基座和同一 full Thinker 训练方式。旧 500 LoRA 结果仍可报告训练有效性的先导证据；由于题目和训练方式均有区别，它不能单独作为新 2K 的数据规模对照。

新版 500 membership SHA-256：`9f33ba430df2b561b0fcec5317a36aaaef1fc060b6b81f228283a436d0ce7815`。

## 4. Full Thinker 已验证与待验证部分

本轮使用 pinned `Qwen/Qwen2.5-Omni-7B` 配置，在 CPU meta tensors 上调用原版 `SFT/src/llamafactory/model/adapter.py::init_adapter`，实际确认了参数选择：

- `model.layers.*` 和 `lm_head` 可训练，共 **7,615,616,512** 参数。
- 音视频 towers/projectors 冻结，共 **1,316,197,376** 参数。
- Thinker 不含 Talker，未使用 LoRA。

这是原 trainer 参数策略的验证，尚未加载真实权重执行 forward/backward。按 FP32 的训练参数、梯度及 Adam 状态记账，未分片部分约 113.48 GiB，另需冻结参数、激活和临时缓冲。full Thinker 需要单独的真实 GPU smoke。

已保存候选资源计划：**4 GPU、每卡至少 80 GiB、ZeRO-2**；资源和安装情况须在提交前检查，显存是否足够由实际 smoke 确定。配置按有效 batch 4、两轮、LR `1e-5` 准备；四卡执行时每卡 batch 1、accumulation 应设为 1。冻结数据里的单设备模板 accumulation 为 4，提交时要据实际 world size 生成新运行配置。预计新 500 为 250 updates，新 2K 为 1,000 updates。

真实资源 smoke 需检查最长输入、至少 3 次有限梯度更新、decoder 参数变化、AV 冻结参数不变、完整 checkpoint 回读、峰值显存及实际 step 时间。通过后再开展正式训练与立即评测。本轮未新增 GPU Job。

## 5. 已完成验证与生产诊断

新增 `validation/test_pilot_freeze.py` 的 6 项测试覆盖嵌套子集、确定性、已知排除项、质量哈希篡改、重复问题及题量不足。它们与出题、扩产、provider resilience、来源优先级的回归合计 **36 项通过**。

新批次完成的 55 个训练出题请求中，11 个失败；抽查失败记录返回 `Native teacher did not return a complete text action/answer`，benchmark 也有同类失败。runner 当前会在 finish reason 非 STOP 或 content parts 不满足文本要求时返回该错误，并丢弃细分结束原因。因此现有证据无法区分输出 token 截断、非文本返回或其他原因，尚不能把失败归因于并发或服务限流。

现有成功结果继续流入后续流程，失败/未知调用保留且不重放。后续新版本应保存经过筛选的 finish reason、token usage 及 part 类型诊断，再决定是否调整输出预算；当前运行的冻结模块保持一致。

## 6. 下一实验产出

1. 扩产到筛选后 2K，自动保存 membership、媒体/行哈希、来源隔离及 token 统计。
2. 验证原 trainer 的 full Thinker GPU 资源与实际更新。
3. 同基座完成新版 500 与 2K 的 full Thinker 训练，以及同题 answer-only 对照；报告实际更新数、监督 tokens 和计算量。
4. 每次训练后立即在固定题集运行任务成功率评测，并完成 native AV / muted 对照及检索行为分析。
5. benchmark 继续补至 200 dev candidates；内容审核和人工 gold 分别计数。论文主表还需有效的 prompted-search base 及独立 benchmark。

## 7. 证据入口

远端：`/home/jovyan/OneSearch-VL/docs/research/records/2026-10-16-2k-preparation-v2/`。

本机：`/Users/michaelcao/Documents/Codex/2026-10-09/https-arxiv-org-pdf-2610-12419/reports/continuation-2k-2026-10-16-v2/`。

其中保存 production 与 watcher 的启动身份、源码冻结、原始测试日志、新版 500 冻结和 token 统计、CPU meta 预检、候选资源计划与现场快照。22 份首批证据已逐文件 SHA-256 回读；更新快照单独保存。本轮没有 push。


## 8. 12:09 最后回读

UTC `2026-10-16T04:09:33.769597+00:00`：本批 benchmark 已完成发布 18 个候选，累计 dev candidates 从139增至 **157**，继续等待下一预留批次；human gold仍为0。训练出题已接受 **64 个候选问题**，尚未进入CPU合格计数；该批出题61完成、16失败、4执行中。合格池仍为1175、排除已知问题后1029。四个worker继续存活，冻结源码一致，未触发provider stop。证据见 `handoff-live.json`。
