## 17:14 实时状态更新（优先于下方旧快照）

两组 SFT 已完成，各 86 次 optimizer 更新；当前仍不能判断训练收益。17:13:51 核验：base 56/56，错误 0；trajectory 50/56，错误 42；answer-only 尚未开始。GPU 评估任务仍运行，已暂停其自动付费 judge，并保存 evaluation-validity.json，避免将解码失败当作能力比较。

CPU 检查确认模型输出维度 152064，tokenizer 注册 ID 共 151665，存在 399 个不可解码槽位（151665–152063）。单独解码其中一个 ID 可复现 TypeError: expected str instance, NoneType found。这个结果支持未定义输出 token 的解释，但尚未捕获真实失败序列的 token ID，不能断言全部错误已完成根因确认，也尚未修复 GPU 推理。

数据仍为 230 条独立保留问题，全部通过 CPU 检查，属于自动 silver，人工 benchmark gold 为 0。2K 生产队列在 400、503、400 的错误簇后停止；本次核验没有新增数据。8 路小样本出题吞吐 15.91 次/分钟，相比 3 路的 6.72 约 2.37 倍，尚非稳定端到端合格数据产率。

下一步先记录实际异常 token 并验证三组一致的词表处理，再复用已有 adapter 做评估；同时定位 API 400 的请求原因后恢复 8 路生产。旧冻结推理源码、失败轨迹和 checkpoint 均保留。

---

# OneSearch-AV 训练与造数据吞吐检查
检查时间：2026-10-13 16:46（北京时间）。本报告是实验过程与吞吐测量记录，尚无训练后的准确率增益结论。

## 当前训练与评估
- Student：Qwen2.5-Omni-7B，真实音视频输入，沿用 OneSearch 原始工具循环与 SFT 训练入口。
- 固定 172 条训练题、40 条按录制来源分组的 held-out 题及 16 条 dev 题；held-out 为自动 silver，尚非人工 gold。
- trajectory 和 answer-only 各完成 86 次真实 optimizer 更新。LoRA B 参数发生变化，有限 loss/梯度及 optimizer、scheduler、RNG 状态验证通过。
- trajectory 训练用时约 1,247 秒，峰值已分配显存 52.29 GiB；answer-only 约 969 秒，28.25 GiB。同题、同更新次数，但输入及监督 token 数不同，并非等计算预算。
- 首次评估的 168 个案例均在模型推理前被 SQLite 版本检查拒绝：CPU 环境为 3.53.4，GPU 环境不同。旧自动报告已另附 evaluation-validity.json 标为 invalid，不用于比较模型能力。
- GPU Pixi 环境固定 sqlite==3.53.4，11 个冻结检索库全部验证通过。复用已训练 checkpoint 的评估恢复任务 slurm-profile-slurm-wh9vf 已启动；两道真实 GPU smoke 通过，原始模型完整评估完成 6/56 题、基础设施错误 0。随后评估 trajectory 与 answer-only。
- 恢复任务为单 GPU、最长 2 小时，旧两次任务合计 3,175 GPU 秒；最坏总预算约 2.88 GPU 小时，保持在原 6 GPU 小时上限内。新任务不重训。

## API 并发实测
同一 gemini-3.8-flash、提示词及媒体处理规则，使用不同的未调用录制；不是配对性能实验。每条出题请求各消耗一个独立样本，没有重发失败或不确定请求。

| 出题设置 | 实际调用数 | 最大在途请求 | 吞吐 | 单调用中位耗时 | 本地媒体准备占比 |
|---|---:|---:|---:|---:|---:|
| 3 路 | 12 | 3 | 6.72 次/分钟 | 25.84 秒 | 0.0505% |
| 8 路 | 19 | 8 | 15.91 次/分钟 | 25.53 秒 | 0.0765% |

本轮吞吐比例约 2.37。8 路原计划 24 次，19 次开始后遇到一次 503，原队列停止规则中断剩余新样本；这是小规模测量，不能视作稳定长期吞吐或合格训练数据产率。

API 等待与响应解析占 99.9% 以上；读取、hash、base64、序列化合计不足 0.1%。这里的计时没有分开传输与服务端计算，因此不能仅据此断言全是模型计算耗时。

## 端到端瓶颈与修改
1. 出题、rollout、复核默认仅 3 路。新生产队列统一上限 8 路；同一轨迹内仍按工具因果顺序调用。
2. 原来每个 shard 连续处理 3 题：8 题 smoke 只生成 3 个 shard，用不满 8 个 worker。改成每题一个 shard，每题最多 8 次 API 调用；真实 rollout 计时确认最大在途为 8。
3. 单次 503、单个坏样本及单次网络错误曾停止整个队列。新版本隔离这些已调用失败，不重发请求，继续未触达样本；60 秒内累计 3 个相应失败才暂停。401/403/429 立即暂停。规则由进程间文件锁共享；旧规则默认保留。
4. 重建 corpus 存在大启动开销。上一次为 178 题构建 1,390 来源花了约 9 分 25 秒；新一轮复用既有冻结 corpus。全局的增量 corpus/materialization 缓存尚未实现。
5. rollout -> 复核 -> CPU 验证之间仍有阶段等待。8 题 smoke 实测分别约 183.6、20.4、50.0 秒，保留并完成 CPU 检查 5 条；端到端产率尚不能按出题吞吐比例外推。后续优先把 CPU 验证与下一批 API 工作重叠。

本轮新增 5 条，合格池从 218 到 223。性能测量中的新出题候选尚未计为合格训练数据。当前 2K 生产版本为 av-scale-2k-20261013-v6，正在处理未调用题目；后续 API 稳定性仍需观察。

## 实现与验证
代码均在 coder ssh 3 的 /home/jovyan/OneSearch-VL：
- research/av_train/teacher_telemetry.py：每次请求的准备、等待、字节数和安全状态计时；不记录提示词、答案、密钥和媒体。
- research/av_train/provider_resilience.py：新版本 opt-in 的跨进程失败阈值；不自动重试付费或不确定请求。
- research/av_train/scale_searchqa.py：按题分片、profiled rollout 子进程、各阶段耗时。
- runtime/sft-gpu/pixi.toml 与 pixi.lock：SQLite 精确版本锁定。
- 3 项计时/隐私/8 路并发测试、7 项失败隔离/共享状态/阈值测试及 2 项数据去重/分片回归通过。真实 API 并发及新合格数据亦已验证。

私有证据保存在研究 run 目录，不发布原始媒体、答案、凭据或权重：
- research/runs/av-scale-2k-20261013-v3/concurrency-benchmark.json
- research/runs/av-scale-2k-20261013-v5/cohort-00/eight-worker-smoke/chunk-0001/performance.json
- research/runs/av-scale-2k-20261013-v6/
- research/runs/av-learning-pilot-20261013-v2/gpu-v1/checkpoint-validation.json
- research/runs/av-learning-pilot-20261013-v2/evaluation-validity.json
- research/runs/av-learning-pilot-20261013-v3/

## 16:48 更新
合格数据池已到 230 条。v6 新增 7 条经过完整复核及 CPU 检查的样本；在后续批次遇到 60 秒内 3 个失败（400、503、400），触发共享暂停阈值。该版本停止记录已保留。原始模型评估完成 18/56 题，基础设施错误 0；尚无最终训练收益结果。
