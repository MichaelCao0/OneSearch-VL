# OneSearch-AV：首轮训练收益实验

本轮使用当前 212 个独立问题的 retained silver 数据，直接训练原生 Qwen2.5-Omni-7B，验证搜索轨迹监督能否提高来源留出的 SearchQA 表现。研究主线是 OneSearch-VL 范式的音视频搜索问答。

## 固定数据与对照

按自动录制家族与共享媒体字节的传递连接划分 172 个训练问题、40 个留出问题，另使用此前冻结的 16 dev cases。训练含 139 个 candidate families，留出含 32 个；train/eval 的自动家族和媒体字节交集均为 0。家族隔离是自动检查结果，尚未完成人工独立录制判定。

三个模型使用同一初始 checkpoint、匿名 AV、工具、原 OneSearch agent loop 和推理预算：

| 模型 | 新增训练 | 目的 |
|---|---|---|
| 原生 Omni base | 无 | 测量开箱搜索能力 |
| Answer-only LoRA | 同 172 题、真实 teacher 最终答案 | 测量答案监督收益 |
| Trajectory LoRA | 同 172 题、多轮实际工具轨迹及答案 | 测量轨迹监督收益 |

两种训练都调用原 `SFT/src/train.py`，使用 `qwen2_omni` 与真实视频音轨；仅更新 Thinker 的 language LoRA，rank 8、alpha 16、LR 1e-4、有效 batch 4、86 次更新、seed 42、约 2 epochs。所有 assistant action 和 answer 监督，user/tool/media mask。最长 trajectory 输入 18,223 token，未为降低长度截掉轨迹；最大 context 32,768。

一轮训练中的 trajectory 共 37,851 个 assistant label tokens，answer-only 为 9,725；实际输入平均长度分别为 7,892.6、4,831.4 token。因此本轮是相同问题与更新次数的 pilot，训练计算不同，需同时报告总 token 与 GPU 时间。

## 执行与评价

资源为 1 张 NVIDIA H20Z，1 个 kjob，任务 hard deadline 6 小时，无自动重试。提交 ID `slurm-profile-slurm-pctxb`。先训练两种格式并核查 optimizer、scheduler、RNG、有限梯度与 LoRA B 更新；评估加载每种 adapter 时逐项比较全部保存张量，确认实际使用训练权重。

评估共 56 个问题，每模型最多 8 turns、每 turn 512 output tokens，temperature 0，冻结搜索后端和相同 prompt。每条问题分别绑定原始 corpus，actor 不接收 reference、private cue 或 teacher 未来 observation。

主指标为相同留出问题上的完整答案语义正确率、字段正确率。另记录目标来源检索/读取、合法最终回答、工具调用数、失败类型和耗时。语义 judge 每题一条新调用，同时接收随机匿名排序的三份回答与公开参考；不包含模型名称。评分共至多 56 条调用（预先保留的上限 168 内），2 workers，不自动重发失败或不确定调用。相同 teacher 模型的另一次调用属于自动 silver 评估，不能替代人工 benchmark gold。

结果同时给出模型配对胜/负/平，以及按录制家族为采样单位的 paired bootstrap 95% 区间。40 题留出与 16 dev 分开报告，避免旧 dev 的元数据型问题掩盖来源留出的新问题表现。区间覆盖 0 时报告不确定；训练 loss 下降与工具调用增多不单独视为任务收益。

## 代码与输出

全部执行位于 `coder ssh 3` 的 `/home/jovyan/OneSearch-VL`。

- `research/av_train/learning_pilot.py`：录制家族/共享媒体分组、原 CPU 逐行验收绑定、Thinker adapter 加载与全部张量核查。
- `validation/test_av_learning_pilot.py`：5 个用例，包括实际小模型的一次 optimizer 更新、保存、重新加载和 logits 一致性。
- 私有 run：`research/runs/av-learning-pilot-20261013-v1`。
- `plan.json`、`data-freeze.json` 固定问题、split、预算与 hashes；原有每条 row 的 CPU audit 以内容不变且带 hash 的方式合并，没有声称重新运行解码审核。
- `gpu-v1` 保存两种训练 log/checkpoint、三组评估与 cluster receipts。
- `finish_after_job.py` 在 GPU job 完成后自动启动一次语义评分与中文报告；失败则保存状态，不自动重启。
- 预期结果：`pilot-results.private.json` 与 `Training-Gain-Report-中文.md`。

数据合成至 2K 的独立队列继续运行。本轮评估问题不参与这轮优化；后续训练需继续排除这份 frozen heldout 的录制家族，或者明确使用新的独立测试集。

## 当前事实与后续决策

本文件记录固定的实验设计与已提交任务，尚未报告训练收益。已有的 13 题 GPU 工程 pilot 只验证训练通路；本轮将在两种 SFT 均完成且三组评估评分产生后，才能判断当前数据是否提供正向信号。结果将决定后续 2K 训练、数据类型补充和更大规模/多 seed 比较。

## 2026-10-13 15:42：显存故障后的版本化修复

v1 的 `slurm-profile-slurm-pctxb` 于 14:39 因 CUDA OOM 停止。Trajectory 记录了 20 次更新，没有保存可恢复的 checkpoint；answer-only 和任务评估均未开始，不能报告收益。日志显示原配置 `flash_attn: disabled` 实际选用了 eager attention；大额 attention 张量在 backward 阶段耗尽 H20Z 显存。

v2 位于 `/home/jovyan/OneSearch-VL/research/runs/av-learning-pilot-20261013-v2`，使用原 trainer 支持的 `flash_attn: sdpa`，配置检查确认 Omni 的语言、音频及视觉均切换为 SDPA。所有 train/eval 问题、媒体、标签、seed、优化目标和 86 steps 保持相同，从原始模型初始化；旧输出保留。GPU job `slurm-profile-slurm-dqk7m` 先训练资源占用极端的三条原始 trajectory：最长 18,223 token、最多视频 patches、最多音轨。实际 forward/backward、LoRA 更新、loss/grad 和显存峰值通过后才能进入完整训练。

首任务已消耗 735 GPU 秒；替代任务 deadline 为 20,865 秒，两次合计最多 6 个单卡小时。没有自动重试；这是检查故障原因后的新版本。正式训练每 43 steps 保存完整 checkpoint 以保留中间状态。训练包装器仅调用原 `SFT/src/train.py` 并记录 CUDA 峰值，不改变原训练算法。

15:44 的 fresh audit 确认 v2 三条极端样本全部通过：真实 optimizer updates 为 3、LoRA B 非零更新、loss/grad 有限、trainer 日志确认 SDPA。最大 allocated 为 51.8937 GiB、reserved 为 62.5938 GiB。最长 18,223 token 与全部媒体/标签均保留；正式训练从原始 base 重新初始化，不使用 smoke 权重。当前已进入 172 题完整 trajectory 训练，尚无 held-out 收益数值。
