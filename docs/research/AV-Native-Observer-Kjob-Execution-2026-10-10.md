# 原生 AV observer 的 Kjob 执行报告（2026-10-10）

已在 `coder ssh 3` 完成一次真实 GPU smoke。模型为 `Qwen/Qwen2.5-Omni-7B`，本轮结果是 **五个片段完成原生音视频生成，零个片段通过严格结构化输出协议；语义能力验收仍未通过**。

## 结果与分母

固定六个既有自然来源，每个请求 source-local `[0.5,2.5)` 秒，共 12 requested AV seconds。没有根据模型结果换样例。下表为最终 v4 的固定分母；v3 与 v4 共保存 12 次 backend attempts，v3 六次均失败、零 generation，所有历史失败继续保留。

| 项目 | 观测结果 |
|---|---:|
| 固定输入 | 6 |
| backend attempts | 6 |
| 完成模型 generation | 5/6 |
| 生成前 clock gate 拒绝 | 1/6 |
| 严格 JSON / 结构协议通过 | 0/6（已生成输入为 0/5） |
| 原始输出解析问题 | 5 |
| 总 generated tokens | 939 |
| 达到 384 token 上限 | 1 |
| 人工语义验收 | 0；pending |
| `raw_av_observer_ready` | false |

第一片段 audio output origin 为 1.012 秒，video 为零。官方 helper 解码后的 ndarray 不保存这个起点；backend 在模型前显式拒绝，未补零、丢弃 offset 或重新选片段。另五个原始文本均出现字面量转义引号；还观察到空 uncertainty、缺失顶层字段或 token cap 截断等协议问题。严格 parser 保留原文并产生无有效事件的显式弃权，不做自动反转义、补字段或语义修复。公共摘要的 `cannot_determine_clip_count=5` 来自解析失败的占位弃权，不能解释为模型主动表达了五次不确定性。

最终 Job 为 `slurm-profile-slurm-gmfbk`，UTC 04:23:17 → 04:24:55，Pod exit 1，Kubernetes 状态 Failed / BackoffLimitExceeded。runner 的 exit 1 来自保留在固定分母中的 clock-gate failure；五次 generation 真实完成，模型加载和 CUDA 计算没有 OOM/初始化失败。不能把 Job 的终态写成 Complete，也不能把五次生成写成结构协议或论文实验成功。

## 成本和运行身份

实际 worker 为 **NVIDIA H20Z**。单节点单卡、8 CPU、64 GiB RAM；第四轮的 CPU request 与 limit 均设为 8，memory request 与 limit 均设为 64 GiB；此前模板只设置 8 CPU request、未设置 CPU hard limit。模型前完成单卡可见、分配 PCI 一致及实际 CUDA matmul 校验。

- 模型完整资产：22,379,297,323 bytes，20 files，immutable revision `ae9e1690543ffd5c0221dc27f79834d0294cba00`，全部文件重新校验。
- 资产 hash 验证：26.957 秒；模型加载：5.766 秒。
- 五次 generation 总计：24.692 秒；runner wall time：71.932 秒。
- Torch CUDA allocator peak allocated：18,230,961,152 bytes（16.979 GiB）；peak reserved：18,679,332,864 bytes。此口径不包括非 PyTorch 分配。
- 五个 Job 的保守累计时间：337 秒（约 5.62 GPU 分钟），低于原上限 2100 秒；全部已终止，无自动 Kubernetes backoff 重试。
- 无闭源 API、Serper 或联网检索请求；实际 provider requests 为 0。模型和媒体准备的 CPU 下载与处理独立记录，不计为 GPU generation。

| Job suffix | 结果 | Job.startTime → 终态秒数 | 模型执行 |
|---|---|---:|---|
| 24kgr | Complete | 14 | GPU inventory/shared-home preflight |
| 4zz4g | Failed | 44 | CUDA 初始化失败；零加载/生成 |
| h6jt7 | Failed | 21 | privilege 假设被证伪；零加载/生成 |
| 58x5k | Failed | 160 | CUDA matmul、模型加载；零生成 |
| gmfbk | Failed | 98 | 5 次 generation；1 次 clock 拒绝 |

## 修正与验证

1. 官方 helper 的实际 FPS 传给 processor，并关闭二次 frame sampling；requested FPS 不冒充实际 FPS。
2. runner 核对 live Python、六个核心包版本、distribution/import 路径、CUDA wheel build 与模块源文件 hash。
3. UUID CUDA mask 在本部署未能初始化；改为只读 UUID → PCI → CUDA ordinal 映射。新进程在任何 tensor 计算前验证单卡与分配 PCI。privileged worker 内的 CUDA mask 不构成 OS 级 GPU 访问隔离。
4. 修正 system `content` 字符串为 pinned Omni processor 要求的 list，采用官方 system 文本；保守观察与因果弃权规则继续位于 user 指令。
5. 新回归先在官方 guard 中复现 TypeError，再验证修正；实际五个固定片段在 CPU 上完成 backend messages → template → helper → processor 链路，真实 audio/video tensors 均存在。

本轮 fresh scoped checks：隔离 Omni runtime 的 backend/clock suite 27/27 通过；默认 CPU 同 suite 25 通过、2 个 pinned-only 跳过；既有 semantic observer 20/20 通过；compileall、shell syntax、diff checks 通过。此前已通过的 130 项 research regression 和 7 项 retrieval-switch checks 没有因本轮 scoped 改动重复运行。自动 observation schema/hash 审计与独立 read-only 审计结果单独保存；这些证明记录完整性，不证明内容为真。

## 可追溯性

本轮运行时 Git HEAD 为 `0eb2b6fbdee33ebeca551462caee778d5ea77316`，backend 包含随后尚未提交的消息格式修正。以下实际源文件 hash 确定了运行代码，不能只凭 HEAD 推断字节一致。

- backend code SHA-256：`3f657a13c2522465ee768498e75ba6507f4bffc23dcb1575fe30bb25b7c60025`
- Pixi lock SHA-256：`b124706cbceb5eff1ea6eacbe4fa93870bfc9fc10dcfb867a026b534c66bf3ce`
- model manifest SHA-256：`dc5264f22c6368d972f797430e9877a17b7af59cf513607c9c387eda0d0a9462`
- live runtime receipt SHA-256：`18128e7a8b258c5b836c839e5ae318d6bd446e222d0a8abf1f2fbdd7d47c513b`

全部 private evidence 保存于远端 `/home/jovyan/OneSearch-VL/research/runs/av-native-smoke-20261010-v1/`：包括每轮生成/提交对象、Job/Pod 状态、完整日志、模型 manifest、输入绑定、逐调用原文、解析问题、成本及 source freeze。运行目录、原始媒体、模型权重与 private gold 不进入 Git。本地 outputs 仅交付中文报告和审核后的公共汇总证据。

## 下一阶段的门限

1. **输出协议诊断**：固定已有输入与分母，先定义可检验的完整 JSON schema、非空 uncertainty 和有限事件数；比较清晰的 literal JSON 示例与适用的约束解码。每轮保留原文、协议通过率和额外成本。任何后处理抽取与原始模型遵从率分别报告。
2. **保留音视频时钟的输入适配**：为 stream origin、首帧/audio PTS、frame sampling 和输出区间设计 timestamp-aware processor 路径；正确处理延迟音频，不能通过改动 provenance 掩盖偏移。仍不得用本次低 FPS smoke 宣称精确时序能力。
3. **语义验收**：对原计划八类 audio/video 必要性、错配、偏移、重复实例和声源歧义等 probes 填写独立人工参考，检查跨源证据与弃权。先做自然片段 spot review，自动输出协议通过不替代人审。
4. **接入与扩展**：上述 G0 门限通过后，再把 verified observer 接入 `EpisodeTools.inspect_av`；随后扩大到 40 cases、做 matched-budget baseline、teacher trajectories 与训练。本轮尚未完成该接入、G0、训练或论文指标实验。

本轮剩余预算不自动转为新增模型调用；固定六输入 smoke 已结束。后续格式/时钟诊断使用新的 versioned run 和明确的独立分母。
