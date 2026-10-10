# 16 个 development cases：性能导向执行计划

日期：2026-10-10。根据用户最新要求，主线调整为性能提升。完整方法与评测依据见 `AV-DeepResearch-Research-Plan-v3-Performance-2026-10-10.md`。本文件取代原来八类 observer probe 作为主实验的计划；旧草稿保存在私有 run 的 `observer-only-plan-superseded.md`。

## 主体任务

四组各 4 题：音频补充视觉检索、视觉补充音频检索、声画局部关系与重复实例、多源证据补齐。错位、缺失和信息不足作为辅助诊断。现有自然素材尚未完成这些任务的 reference，16 题尚未构建或运行；同来源多题需要报告事件簇依赖。

## 方法与对照

先比较相同 observations 的普通 query-aware AV reranker 与任务条件的实例 AV reranker，另外报告 OneSearch 视觉 baseline 和朴素 AV 工具增强 baseline。主要指标为正确来源/片段排名、实例准确率和下游答案准确率。训练先用标准 ranking/SFT；原 structured consequences 方案降为备选。

## 执行次序

1. 绑定自然任务、来源/片段池和 reference；已有 inventory 不等于可判定任务。
2. 有界生成共享 observation cache，接入 episode runner 的 inspect_av；单片段观察和 paired 单独观察不得算完整 search 成功。
3. 普通与实例 scorer 共享初始候选池、observer、观察窗口、tokens 和工具预算；先出逐题性能表。
4. 有信号后扩来源隔离的正式集合，再训练和冻结评测。

## 预算与记录

沿用 727/2100 秒累计 GPU Job 成本，剩 1373 秒，不重置。下一项 observation bootstrap 最多一项 700 秒 deadline 的单 GPU Job，setup 时间计入预算。2 秒 joint-AV 输入仍是当前 smoke 支持范围，不能伪装完整任务能力；完整 search 可能需要多窗和后续检索，另行如实计量。保存所有任务失败，reference 不传入 actor。媒体、权重和 private gold 保留私有。

本次只更新设计，没有完成新 runner、16 题数据或 GPU 提交。
