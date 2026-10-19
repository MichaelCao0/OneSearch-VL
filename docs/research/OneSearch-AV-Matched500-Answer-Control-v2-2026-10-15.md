# OneSearch-AV：500 条同题 answer-only 对照启动记录

时间：2026-10-15T10:13:10.673803+00:00。主开发与执行位置为 coder ssh 3，/home/jovyan/OneSearch-VL。

本组实验检验搜索轨迹监督的作用。clean500 trajectory 已完成 250 次更新和同题 silver 评测；新增 answer-only 从同一 pinned Qwen2.5-Omni-7B 基座初始化，使用完全相同的 500 个问题、初始音视频与实际最终回答，省去工具历史及最终 think 前缀。学习率 1e-4、有效 batch 4、两轮、LoRA rank8/alpha16/dropout0.05、seed42 与 trajectory 一致，共计划 250 次更新。两种训练的输入与监督 token 不同，在成本表中分别报告。

已提交 Kjob：`slurm-profile-slurm-g4hbg`，namespace `user-michaelcao0`。资源为 1×NVIDIA H20Z、8 CPU、96Gi，activeDeadlineSeconds=21600，backoffLimit=0。本次只创建一个 GPU job，不自动重试。实际调度状态：{"active": 1, "conditions": [{"lastProbeTime": "2026-10-15T10:10:54Z", "lastTransitionTime": "2026-10-15T10:10:54Z", "message": "Job resumed", "reason": "JobResumed", "status": "False", "type": "Suspended"}], "ready": 1, "startTime": "2026-10-15T10:10:54Z", "terminating": 0, "uncountedTerminatedPods": {}}。GPU 已通过分配设备与真实矩阵运算检查，22,379,297,323 字节 pinned 模型文件重新核验。训练尚未完成；以之后的 checkpoint validation 判断实际更新和权重状态。

GPU 流程：原 SFT/src/train.py → 验证 250 次更新、梯度、权重与 optimizer/scheduler/RNG → 原声 56 题 → 静音 56 题。静音在初始及所有 inspect_av 返回媒体中物理移除音轨并保持视频包一致，实际输入检查 audio_count=0。

训练后评测与上一组保持相同问题、提示词、固定 corpus、工具和 8 轮×每轮512 token预算。复用 base/clean500 两模型×两模态共224个成功原始 rollout，保存来源/目的哈希；新增 answer-only 112 个 rollout。完成后由 workspace CPU worker 发起最多56次新的匿名配对评分，每题同时比较三个模型×两种模态，4路并发，不自动重放失败或结果不确定的请求。原始旧结果保留；新比较报告单列，使用相同已判定题集合计算 paired differences 和录制家族 bootstrap。

主指标要求合法最终答案、全部请求字段正确且无矛盾；任务成功与发布者参考支持代理指标分开报告。judge/参考仍为silver。strong prompted base 将在dev上选提示词后单列比较，本组不以base原提示词的0分承担全部训练收益论证。

数据生产并行推进：当前 988 条CPU-qualified / 988 条retained，0 条pending。新采集已接收 138 段实际AV；按查询类别为 {"music": 23, "sports": 72, "live": 20, "other": 23}，真实内容在后续AV审核标注。训练与benchmark producer身份均重新核实。完整目标保持20K train、500 train-validation、200 dev/1K test；2K是下一训练节点。正式human gold为0。

本次CPU提交前检查包含：500条canonical row摘要、同题eval成员、相同recipe、224个复用raw rollout文件、脚本编译、judge闭合schema、bash语法、空闲项目job、磁盘和source freeze。首次将文件字节hash与canonical row hash混比时在提交前停止，修正检查约定后原始500行摘要完全一致，未修改数据。Kjob原始profile把GPU/内存写到limits、CPU写到requests，提交manifest显式固定全部requests/limits后通过server dry-run。错误与修复记录保留。

运行目录：`research/runs/av-learning-500-controls-20261015-v2/`。现场快照、budget、manifest、source freeze、模型/GPU验证与脚本归档在 `docs/research/records/2026-10-15-matched500-answer-control-v2/`。CPU completion monitor带有host/PID/start-ticks/boot身份；实际结果写入该run的learning-results.private.json及Training-Gain-Report-中文.md。
