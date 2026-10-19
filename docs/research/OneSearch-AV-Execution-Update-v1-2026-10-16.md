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
