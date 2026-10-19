# OneSearch-AV：Full Thinker 验证与数据方法诊断

工作标题：**OneSearch-AV: Learning to Search from Audio-Visual Clues**。

本轮继续执行已确定的三条实验主线：学习与规模、声音与互补、数据方法。数据快照截至北京时间 **2026-10-16 21:59**；训练与诊断结果的完成时间分别记录。

## 1. 数据与执行状态

| 项目 | 实际进展 |
|---|---|
| CPU 合格的不同训练问题 | **1,652**，较上次 1,465 的汇报新增 187 |
| 按冻结质量视图 v4 排除 146 题后的可选训练池 | **1,506**；距 clean2000 还差 **494** |
| 当前生产 | v13 已自动接续，97 个来源已接受 79 个问题候选，正在 rollout/review/CPU 阶段；2 workers |
| 素材预取 | 两个 cohort 的缓存已准备，按缓冲上限等待消费 |
| Benchmark 候选 | **240 题**：其中 217 题 / 141 录制已冻结并形成质量视图，新增 23 题待合并复核 |
| Benchmark 自动内容审核 | **208 完成**：192 keep、2 revise、14 reject；9 未完成或未知 |
| 人工 benchmark gold | **0** |
| 下一 benchmark 采集目标 | **300 候选**，供最终 200 题 dev 的事实与模态人审筛选 |
| 当前可用磁盘 | 约 1,274 GiB |

合格数要求真实搜索轨迹、silver 保留、trajectory/answer-only 成对导出及原 trainer CPU 媒体和标签检查。Answer-only 不重复计题，出题候选不提前计入。v13 在接续时重新读取全局已付费来源，保留失败和未知回执。

完整交付目标保持 **20K train + 500 train-validation + 200 dev + 1K test**，2K 是下一训练节点。

## 2. Full Thinker 已完成真实更新与 checkpoint 回读

学生固定为 `Qwen/Qwen2.5-Omni-7B`，revision `ae9e1690543ffd5c0221dc27f79834d0294cba00`，沿原 `SFT/src/train.py`。本轮验证 full Thinker，训练语言 decoder 与 lm_head，冻结 AV 塔及 projector。

| 项目 | 验证结果 |
|---|---|
| 硬件 | **4 × NVIDIA H20Z**，每卡实际 139.8 GiB |
| 分布式策略 | DeepSpeed ZeRO-2，effective batch 4 |
| 输入 | clean500 中的 12 个完整极值样本；最长 **20,014 context tokens** |
| 真实更新 | **3 次 optimizer update**，loss 和 grad_norm 均有限且梯度范数为正 |
| 可训练参数 | **7,615,616,512**，339 个 tensor |
| 冻结参数 | **1,316,197,376**，1,007 个 tensor |
| 参数变化 | 四个 rank 均观察到语言参数抽样值变化，冻结参数抽样值不变 |
| 峰值 allocated | 各卡约 **69–78 GiB** |
| 峰值 reserved | 各卡约 **102–119 GiB** |
| Checkpoint | 权重抽样值、四个 optimizer 分片、scheduler 和四份 RNG 状态回读通过 |

新增独立 Pixi 环境 `runtime/sft-full`，固定 Torch 2.8.0+cu126、Transformers 4.57.6、Accelerate 1.11.0、DeepSpeed 0.16.9，并补齐 CUDA 12.6 compiler。原生产与 CPU audit 环境保持原路径。

GPU 作业 `slurm-profile-slurm-p897b` 完成了训练和保存；最后的验证器将 scheduler 误判为 DeepSpeed 内置状态，导致 Kubernetes 保留 Failed 终态。实际 scheduler 由 HF 保存于 `scheduler.pt`。修正验证器后，用 CPU 对已有 checkpoint 完整回读通过，没有重训，原失败记录和执行源码均保留。

这项结果证明 clean500 极值输入下的 full Thinker 训练路径可运行。正式学习收益由后续 500/2K 训练和同题评测给出；2K freeze 后还需要核对新集合的输入极值。

## 3. 完整 checkpoint 的原生评测路径

新增 `research/av_train/full_checkpoint.py`，将完整 SFT 权重严格加载到原 OmniRunner 的同一个 Thinker。它复用原生 AV 输入、工具和 agent loop，不引入新的神经网络结构。

CPU 测试覆盖完整权重加载、严格 state keys、错误架构和 shard 路径；实际保存的 checkpoint 为 **1,346 个 state tensor、4 个 shard**。

原生推理 smoke 固定两个原有评测问题，各做 AV 与 muted，限制每题 2 turns、每次 256 new tokens。第一批完成了两个 AV 和一个 muted 条件；最后一个条件在模型调用前遇到 FFV1→MP4 容器错误。

本轮将当前有效的静音处理整理为共享模块 `research/av_train/modality_inputs.py`：按源格式选择容器、复制视频编码包、移除全部音轨，并修正可跨进程读取的缓存收据。FFV1/MKV、H264/MP4 两类真实编解码测试通过，问题样本在去音轨前后的视频 packet hash 完全一致。补跑只执行缺失条件，前三个成功条件保留。

**21:53 补跑完成，两个问题 × AV/muted 的 4 个条件全部完成真实 generation。** 两次加载的完整 checkpoint 文件哈希一致；两个 AV 条件的实际输入 audio_count 为 1，两个 muted 条件为 0。最后一个静音条件生成 512 tokens（两轮各最多 256），GPU allocated 峰值 17.77 GiB。相应 GPU 作业已结束。

推理 smoke 用于验证权重加载、实际 generation 和模态干预；不作为任务成功率或训练收益结果。

三次有界 GPU 作业均已终止。按 Job start/end × 卡数计时，本轮资源验证与推理检查合计约 **0.57 GPU hours**（包含启动时间）。

## 4. Benchmark：完成内容分层，继续扩到 300 候选

217 个候选已对当时 **1,584 条 CPU 合格训练题**完成核查：录制、家族、媒体无交集，目标 URL 未出现在实际训练轨迹中。候选 membership SHA256 为 `a87cb10588109253b88dcd89aa77961c992beb02ed434efa5a2b697470651d2b`。

复用 70 份精确匹配的成功审核回执，对 142 个未审核问题新增原生 AV 内容预审：138 成功、4 失败。合并后覆盖 **208/217** 题，逐题输入、问题及 reference 身份均绑定，未重放原失败或未知调用。

208 题的自动模态标签为：joint 75、visual-primary 114、audio-primary 11、speech-text-primary 8。对应搜索需求为：external attribute 144、source identification 58、perception-only 6。这些统计包含全部已审题，尚未经过人工裁决。

192 个 keep 建议不足以直接得到 200 个人工有效 dev 题，因此已启动 v12 benchmark producer，候选目标提高到 **300**。它只消费训练前预留的新录制家族，最多新增 150 次 author 调用、1 worker、8 小时。21:59 已新增 23 题，累计 **240 候选**，等待下一批训练前预留来源。新增部分将在下一版 freeze 统一复核。正式 dev 仍需逐题完成事实、区间和模态角色人审。

## 5. 数据方法：同来源盲审给出的诊断

对 28 个相同来源、同为 Gemini 3.8 Flash 成功生成的案例，混合并隐藏生成方式标签，使用实际 AV 和 publisher 材料共同审核 **94 个问题**。每个来源只传输一次 AV，共 28 次新审核调用，全部完成。

“可用搜索题”要求：keep、片段与问题匹配、publisher 支持答案、不泄露关键身份线索、具有可链接的具体实例，并需要外部属性或来源检索。

| 生成方式 | 问题数 | 自动审核可用搜索题 | 比例 |
|---|---:|---:|---:|
| AV 条件生成 | 38 | **37** | **97.4%** |
| Publisher-text-only 生成 | 56 | **7** | **12.5%** |

按来源等权计算，可用比例分别为 96.4% 与 12.5%，配对差为 83.9 pp。Publisher-only 中 **46/56** 题被标记为泄露了关键身份线索。

例如，同一段双簧管演奏的 AV 问题是“视频中演奏双簧管的音乐家是谁？”，文本生成的问题则先提供了练习曲编号，再问演奏者。这使搜索可以从题干直接起步，减弱了理解片段的必要性。

这个诊断明确了出题管线需要检查的机制：**publisher 中有答案支持，还不足以保证问题需要利用 AV 线索。** 当前样本来自历史 AV 成功来源，两侧生成 prompt/schema 有差异，结果是自动诊断。下一轮使用新来源、共同 prompt/schema，以及更强的“隐藏可直接检索实体”规则，比较 AV+publisher 与 publisher-only；再用匹配题数与监督 tokens 的训练比较检验性能增长。

另有一题出现自动标签分歧：`avqa-expand-9df0617f5b6ac03d9fe0` 的旧视图标为 main_search，新盲审仍建议 keep，但把需求标为 perception-only。该题已记录到裁决队列；当前规模计数明确沿用冻结视图 v4，新诊断不被当作最终人工标签。

## 6. 接下来的实验顺序

1. 继续 v13 当前 79 个候选的真实轨迹生产，达到 2,000 个可选问题后自动冻结，并验证 clean500 严格嵌套。
2. 完整 checkpoint 的 AV/muted 推理检查已通过；正式训练复用此加载与静音路径，并按新数据极值做资源核对。
3. 固定新版 500/2K × trajectory/answer-only 四组 full Thinker 训练，完成后立即评测；同时补足有效 prompted-search 基线。
4. Benchmark 扩至 300 候选并开展事实、区间、搜索必要性与模态角色的人审，独立 test 按预留家族继续建设。
5. 将数据方法的正式消融升级为共同 prompt、实体隐藏、来源配对和预算匹配，分别报告数据质量与下游任务成功率。

本轮新代码在独立训练环境中合计通过 **18 项相关回归**，包含 2 项真实媒体静音回归。主开发继续位于 `coder ssh 3:/home/jovyan/OneSearch-VL`，branch `research/av-deepresearch`。本轮未 push。

## 7. 复现入口

- Full Thinker 资源测试：`/home/jovyan/OneSearch-VL/research/runs/av-full-thinker-smoke-20261016-v1/`
- 完整 checkpoint 推理：`/home/jovyan/OneSearch-VL/research/runs/av-full-checkpoint-infer-20261016-v1/` 与 `av-full-checkpoint-infer-20261016-v2/`
- Benchmark freeze/质量视图：`/home/jovyan/OneSearch-VL/research/runs/av-benchmark-freeze-217-20261016-v1/`
- 数据方法盲审：`/home/jovyan/OneSearch-VL/research/runs/av-construction-blind-quality-20261016-v1/`
- 新主代码：`research/av_train/full_thinker_smoke.py`、`full_checkpoint.py`、`grouped_content_audit.py`、`modality_inputs.py`
- 本机归档：`reports/continuation-2026-10-16-v4/`，只复制代码、配置和有界收据，不复制模型权重。
