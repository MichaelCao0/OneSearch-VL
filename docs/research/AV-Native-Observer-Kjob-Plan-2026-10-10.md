# 原生 AV observer 的 Kjob 执行计划

用户于 2026-10-10 明确授权提交 Kjob 计算任务；开发继续在 `coder ssh 3` 的 `/home/jovyan/OneSearch-VL`，分支 `research/av-deepresearch`。本轮实现的是研究管线中的真实声画观察节点，后续方法训练及 Search for Generation 不在此轮范围。

## 固定范围与预算

- 个人 namespace `user-michaelcao0`、LocalQueue `user-queue-michaelcao0`，单节点单卡，8 CPU、64 GiB host RAM。
- preflight 的 Kubernetes `activeDeadlineSeconds=300`；模型 smoke `activeDeadlineSeconds=1800`，合计上限 35 GPU 分钟。脚本另外使用 `timeout`，`backoffLimit=0`，无自动 GPU 重试，不扩大卡数。
- CPU 阶段完成隔离 Pixi CUDA 环境及 immutable revision 下载；模型下载上限 30,000,000,000 bytes，校验 upstream LFS SHA-256。CPU default 的 manifest/lock 保持原样。
- 六个既有自然 AV 来源，各固定取原始 source-local `[0.5,2.5)` 秒，共 12 requested AV seconds；每次确定性生成最多 384 tokens。不根据模型结果挑选成功片段。
- 不上传原始媒体、权重、私有输入或候选 observation；公开代码、环境锁与审核后的汇总文档可沿已有授权推送。

## 执行顺序

1. read-only 检查个人队列、模板和挂载，使用 `kjobctl create slurm --dry-run client` 生成对象。
2. 在生成对象中明确单节点/单卡、Job deadline、无重试和 project labels，再进行 server dry-run，成功后创建。
3. preflight 验证 GPU inventory、共享代码/registry 与输出持久化。队列名称是 H200，已运行 worker 实际 inventory 是 `NVIDIA H20Z`；报告实际硬件。
4. 原模板 privileged 能看见整机 GPU。模型 job 将 privileged 关闭，使用 device-plugin 单个 GPU mask；若 PyTorch 仍看见多个设备，停止模型加载，不选择任意空闲卡。
5. 冻结六个实际 lossless AV clip 的 hash、时间映射和输入 hash；`make_observer_input` 及媒体准备纳入媒体工作计量。
6. 独立审核 backend、artifact manifest 与 CLI。实际 GPU Job 先记录 torch/CUDA、设备隔离并执行真实 CUDA matmul，再用本地 verified snapshot 生成。
7. 保存逐调用原始文本、结构化候选、解析失败、GPU allocator peak、tokens、时间和模型/代码/lock 身份。失败保留在固定分母；不把解析失败替代结构当成有效事件。
8. 读取 Job/Pod terminal status、持久文件、校验 hash 与观测 schema；通过后记录真实执行层状态。所有人审与八类语义 probe 保持 pending。

## 本轮成功标准与下一步

本轮目标是成功执行 GPU 模型和接通现有 private `SemanticObserver` 边界。`native_joint_av` 要求 helper、processor、generation 全部接收音视频，保留共同 clip 和原始时间映射；不将 ASR+frames 宣称为 native AV。

完成模型运行仍不等于 observer 的声源归因、时序偏移或跨源判别能力成立。下一步先填写自然片段参考与八类 probe 的人工复核，再判断 G0 是否可以通过；G0 未过不扩到 40 case、不开始 teacher trajectory 或训练。自动 schema/hash 检查只证明完整性，不能证明语义真值。

## CPU 预处理发现与处置

官方 helper 实际读取六个片段时，第一个 source 的 lossless clip 保留了 audio `output_start_time_s=1.012`，video 为零；原始媒体保存同步是正确的，但转成 ndarray 后 helper 不携带这个时间起点。当前 native backend 因而在 helper/model 之前拒绝该输入，不补零、不重选片段；固定六个输入的分母保持不变。2 ms 容差仅容纳 Matroska timestamp rounding。通过此门限不代表毫秒级事件定位或音画同步判断已成立；低 FPS sampling 与 array presentation 仍是模型输入近似。
