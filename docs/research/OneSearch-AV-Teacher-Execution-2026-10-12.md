# OneSearch-AV 原生 teacher 数据执行记录

主项目：coder ssh 3，`/home/jovyan/OneSearch-VL`，`research/av-deepresearch`。完整研究与 SFT 方案见 `OneSearch-AV-Teacher-SFT-Plan-v1-2026-10-12.md`。

## 已完成的主链

Qwen2.5-Omni-7B student 原生接收音视频，在原 `run_agent_loop` 中调用搜索、网页读取与 `inspect_av`。强 teacher 通过原生 Gemini `generateContent` 接口接收完整 MP4 及其音轨，执行同一工具流程；成功轨迹进入原 `SFT/src/train.py` 的 LLaMA-Factory 路线。媒体裁剪与基本时间信息是工具输入，不由文字 observer 替代音视频。

本轮修复两个实际生产瓶颈。原自然片段最大 86.35 MB，超过 inline 请求预算；完整片段压缩后 20 例合计 28,151,970 bytes，最大 2,578,082 bytes，保持全部录制音轨，最大时长差 0.003193 秒。转码生成新 batch 与父媒体 lineage，不增加独立来源数。第二个瓶颈是 teacher 省略或写错输出标签；现用 JSON schema 选择单个动作，确定性转换至原 OneSearch 协议，保存原 provider 输出与双重 hash。转换不会补选动作、参数或观察。

## 真实候选与保留数据

| 数据阶段 | 实际执行 | 结果 | 数据性质 |
|---|---:|---|---|
| v5 完整片段压缩＋标签 prompt | 6 条，27 请求槽位 | 1 完成、5 失败 | 历史诊断保留 |
| v6 结构化输出初测 | 1 条，2 槽位＋1 诊断调用 | 参数未完整填入独立字段 | 失败保留，不进入训练 |
| v7 六参数 nullable schema | 20 条，71 槽位 | 15 完成、5 失败 | 完整轨迹校验通过 15 条 |
| v7 自动 silver review | 15 条，15 调用 | 13 保留、2 review 503 | 同一模型的独立调用，无人工签名 |
| paired SFT export | 13 个独立题目 | trajectory 13 行、answer-only 13 行 | 实际生成的答案与动作 |

当前 v7 计划共有 60 候选，其中 40 条未执行，没有自动重试。生成 queue 的 `status` 只统计生成阶段，固定 retained=0；实际保留数量以 review/export summary 为准。源码或媒体变更后建立新版本，不继续调用失效的冻结 plan。

v7 的 5 个失败为 3 条 HTTP 503、1 条达到轮数上限仍未给出最终回答、1 条多轮媒体累积超过 inline 大小预算；不能把 pipeline `status=ok` 直接当作有效训练数据。自动 review 检查答案字段、媒体是否匹配参考、query 是否来自可见可听线索及是否存在重大幻觉。它使用同一 Gemini 模型的另一次调用，因此有相关偏差；13 条是探索性 silver，benchmark gold=0。既有来源家族近重复隔离尚需补齐，不以这 20 题报告正式泛化准确率。

## 代码与验证

新增 `gemini_av_runner.py`、`teacher_actions.py`、`generation_receipts.py`、`teacher_rollouts.py` 和 `teacher_media.py`；扩展 `native_sft.py` 兼容强 teacher、候选/任务 lineage 与严格 action-observation 对应。原 trainer 的 Omni-only PTS sampler 保持普通视频与 VFR 的全片覆盖；LoRA target 排除音频/视觉 encoder 与 projector。

集成测试共 80 项：79 通过，1 个 pinned tokenizer 用例在默认环境跳过，但在 pinned trainer 环境单独通过。原 trainer CPU label、collator、media 与 sampler 测试 5 项全部通过。独立 reviewer 发现一个 structured receipt 的单独 exporter 导入问题，已用新子进程回归修复，14 项相关测试复核通过。真实 API 感知能力通过两组完整 MP4 的颜色＋语音对照；非语音声音计数仍需逐类测量。HTTP 200 与生成 receipt 本身不能证明感知准确率。

证据目录：

```text
research/runs/av-teacher-data-20261012-v1/integrated-tests-v5.log
research/runs/av-teacher-data-20261012-v1/original-trainer-cpu-tests-v2.log
research/runs/av-teacher-data-20261012-v5/normalized-batch/
research/runs/av-teacher-data-20261012-v7/candidate-plan/
research/runs/av-teacher-data-20261012-v7/silver-review-v2/
research/runs/av-teacher-data-20261012-v7/silver-sft-v1/
```

私有媒体、API 配置、silver verdict、运行日志和权重留在远端，不进入公共 Git 提交。代码、环境 lock、测试与中文文档按用户授权提交到 fork。

## 新的 GPU 训练结果

CPU 全数据 audit 通过：trajectory 与 answer-only 共 26 行，全部 AV、assistant labels 和 user/tool/media mask 通过；最长 trajectory 11,741 token、answer-only 5,834 token。Trajectory 共 46 个 assistant 回合、3,078 supervised token；answer-only 共 13 回合、993 supervised token。输入 token 总量不同，这次固定更新数的 pilot 不能代替计算量匹配的主实验。

`slurm-profile-slurm-dvt82` 已于 2026-10-12 完成，1 张 NVIDIA H20Z，实际 668 秒，无重试。两组各完成 16 个真实 optimizer updates，保存 `checkpoint-16`；optimizer step、scheduler epoch、Python/NumPy/CPU/CUDA RNG、非零有限梯度、LoRA B 更新与 392 个 adapter 张量逐项重载全部验证通过。没有执行训练恢复实验，也没有 held-out 性能增长结果。

证据：`research/runs/av-native-sft-20261012-v2/gpu-v1/checkpoint-validation.json`、`cluster/job-terminal.json` 和 `cluster/budget-after-job.json`。总 GPU 使用 4,154 秒，当前追加预算使用 2,112 / 14,400 秒。任务已 Complete，保留日志和权重；媒体和 checkpoints 不发布到 Git。

## 下一步学习实验

用户已重新允许有实质进展后提交 kjob。本轮已按 CPU 全数据 audit、冻结数据/源码、单 GPU 和既有预算执行，验证实际学习与 checkpoint。旧 2 行、8 updates 的 pilot 已验证训练保存重载，不重复宣称为新进展。

后续性能实验扩大到 1K–2K 保留轨迹，在同题、同初始权重和匹配工具预算下比较原生 Omni、answer-only SFT 与 trajectory SFT；用冻结 held-out QA 测量答案准确率、任务成功、音频贡献与训练计算量。15K–20K 是主数据规模目标，当前 13 条不足以验证研究假设。
