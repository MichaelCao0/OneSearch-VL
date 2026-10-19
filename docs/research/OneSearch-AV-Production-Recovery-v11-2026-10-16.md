# OneSearch-AV：扩产恢复与实时进度

现场时间：北京时间 2026-10-16 14:17:26。

| 项目 | 当前状态 |
|---|---|
| 轨迹保留并通过原trainer CPU检查的不同问题 | **1317** |
| 已知排除或暂缓 | 146 |
| 最终2K可选池 | **1171**；还差 **829** |
| Benchmark dev候选 | **173**，目标200；human gold 0 |
| 新版500题子集 | 已冻结；锚点membership未变 |
| 2K full Thinker训练与新评测成绩 | 尚未开始 |

上次12:09的合格池1175。之后v10完成cohort-01的102条及cohort-02第一块的40条，池增至1317。13:46出现一分钟3次Teacher HTTP503，触发provider stop；producer、benchmark与freezer随后退出，旧记录保留。29条成功回执正在CPU检查，尚未计入合格池。

本轮恢复run为 `/home/jovyan/OneSearch-VL/research/runs/av-data-production-20261016-v11/`。先复用已经成功的29份语义保留回执，导出原始轨迹和同题answer-only，执行原trainer CPU媒体/标签检查；随后仅从14个从未发起轨迹请求的问题中取1题做完整服务检查，再处理余下题。旧cohort-03已经采集的250段素材继续接入出题与预留benchmark流程。旧失败/未知请求不重放；未修改主代码模块。

并发从8降为4，旧24小时及请求预算按剩余量继续使用。单次服务检查是新问题的首次请求，其结果需实际通过后才能认定服务恢复。新生产与预取已启动；benchmark v8继续从173向200扩；freezer v2保持原500题锚点并等待2000可选问题。现场四个worker身份均存活，源码freeze一致。

下一训练节点保持：筛选后2K冻结 → full Thinker真实GPU smoke → 新500/2K及同题answer-only训练 → 立即固定题集任务成功率、AV/muted和检索行为评测。本轮无新GPU Job、未push。

现场机器证据：远端 `/home/jovyan/OneSearch-VL/docs/research/records/2026-10-16-production-recovery-v11/execution-snapshot.json`；本地 `reports/production-recovery-2026-10-16-v11/final-live-snapshot.json`。恢复任务的启动不等同于已达到2K。
