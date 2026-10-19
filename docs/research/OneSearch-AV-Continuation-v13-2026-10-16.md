# OneSearch-AV：持续扩产与批次接续 v13

北京时间2026-10-16 16:22现场核验。

训练池1359个不同问题通过CPU媒体/标签检查，已知146题排除或暂缓后，1213题可选；最终2K还差787。Benchmark dev候选192，human gold 0。当前v12出题实时结果79完成、28失败、2执行中，接受82个问题候选；这些候选尚未进入合格训练计数。

当前服务没有新的provider stop。生产、benchmark和2K自动冻结任务仍在执行；下一轮v13已完成启动检查：新素材预取并行进行，但模型调用必须等待v12正常完成，再导入其实际合格exports和最新预留来源。若v12异常结束，v13停止并记录，不自动重放失败或未知请求。当前v13的Teacher新增调用为0。

磁盘实测可用1457.12GiB，较上次约232GiB已增加。本轮据此恢复素材采集，最多预取2个cohort，每cohort目标250个AV素材；候选批数上限10、接续出题请求上限1900、累计下载字节上限150GB、最少可用空间150GiB。截止时间沿用当前生产预算的绝对deadline，不将剩余预算重新计满24小时。目标仍为2200 CPU合格后筛出2000。上述均为执行上限，不是已产数据。

主run：`/home/jovyan/OneSearch-VL/research/runs/av-data-production-20261016-v13/`。当前阶段waiting_for_predecessor_normal_completion，prefetch运行。Benchmark接续run `av-reserved-benchmark-20261016-v11` 等前一worker完成，再读取最新累计数；若已达200则不新增出题。Freezer接续run `av-learning-2k-preparation-20261016-v4` 仍锚定原新版clean500题集，达到2000可选后冻结。当前各运行worker身份和源码hash均匹配。

本轮未修改主代码模块；新增版本化的生产接续脚本与计划通过Python语法编译检查。未新增GPU任务，未push。2K full Thinker训练及新性能评测尚未开始，后续节点仍为数据冻结→真实GPU smoke→训练→立即固定题集任务成功率、同题answer-only和native AV/muted评测。

证据：远端`docs/research/records/2026-10-16-continuation-v13/execution-snapshot.json`；本机`reports/production-continuation-2026-10-16-v13/`。
