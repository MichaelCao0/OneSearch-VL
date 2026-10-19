# OneSearch-AV：2K扩产恢复v12

现场时间：北京时间2026-10-16 16:00。

当前训练池1359个不同问题通过原trainer CPU媒体/标签检查，排除已知146题后1213可选，距2000差787。Benchmark dev累计192候选，human gold 0。新版500题锚点保持。

v11已回收29条成功语义保留轨迹并完成CPU验收；14个从未请求过轨迹的问题新增13条合格轨迹。随后author阶段2个503和1个400在60秒窗口内触发停止。400的安全诊断为FAILED_PRECONDITION，具体原因未保留，不能认定为限流或媒体损坏。

v12直接复用144段已经裁剪和规范化的输入及其绑定，保留全部6份既有付费出题回执，只处理138个从未出题请求的来源，后续再消费已经采集的250段素材。并发降为2，不重放失败/未知请求；原始run不改写。首个新出题请求失败，随后另一新来源成功产出1个接受候选，因此只能确认服务存在成功返回，尚未证明稳定性。

因剩余空间约232GiB，本轮不新增下载，并加入80GiB可用空间下限。后台运行：av-data-production-20261016-v12、av-reserved-benchmark-20261016-v10、av-learning-2k-preparation-20261016-v3。16:00三worker的host/PID/start ticks/boot ID匹配，源码freeze一致，无新provider stop。Benchmark启动时曾因producer身份文件尚未落盘，在API前退出；等依赖就绪后在新版本v10恢复，无重复付费。

数据生产目标2200 CPU合格，自动冻结最终2000可选题。此轮只消费有限库存，可能不足2K；以实际产率及磁盘余量决定后续扩产。到2K后继续full Thinker真实GPU smoke、新500/2K同基座训练、answer-only及native AV/muted任务成功率评测。尚未提交新的GPU Job，未push。

主开发：coder ssh 3:/home/jovyan/OneSearch-VL。证据：本机reports/production-recovery-2026-10-16-v12/，含启动检查、生产恢复快照和进程身份核验。
