# OneSearch-AV-SFT：训练集、训练与性能验证

日期：2026-10-10。本文落实总体 proposal v2 第 7 节，不替代总体研究方案。

## 1. 要训练的 baseline

第一篇包含真正经过 AV research 轨迹训练的 OneSearch-AV-SFT。用发布 OneSearch-VL-8B 做初始化，通过 continued SFT 学习针对新的音视频工具和任务的研究策略。论文主要配对实验是同一起点的 OneSearch-AV-Prompted 与 OneSearch-AV-SFT；固定 observer、索引、工具与测试预算，测量新训练数据的收益。

首轮采用模块化架构：OneSearch-VL-8B controller + 固定 Qwen2.5-Omni-7B observer。LoRA 更新 controller 的 language policy，训练过程保持 perception/vision/projector 冻结。这一版的音频理解由 observer 提供，checkpoint 是模块化 AV agent 的 controller。随后可做 native Omni SFT，对应独立的 processor/backward 验证与模型对照。

训练先覆盖查询生成、来源选择、局部窗口选择、根据实际观察继续查询、跨源比较及有证据的最终回答。每条样本包含问题、真实 assistant 动作、工具实际 observation 和 final；observation 是条件输入，assistant action/final 是监督 target。

## 2. 训练数据怎样构建

先采集独立 train 专题/来源组，而不是把 16 个 dev cases 扩写成训练集。技术/机械、现场表演、动物/环境是首批采集域；每个组包含有比较关系的独立录制、背景文本与自然相似候选。跨源任务要能在这些材料中完成有内容的比较或事实补全。当前四份新素材仅为 authoring candidates，尚未构成正式训练来源组。

构建流程：

1. 登记原始媒体、许可、来源家族、时间轴；按录制/事件/近重复家族划分 train、train-validation、dev、test。
2. 从真实内容形成自然任务、原子必要事实与充分证据包。Benchmark 逐题人审；训练题采用自动核验和分层人审。
3. Teacher 只访问问题、初始线索和共享工具；通过实际搜索、AV inspect、文本读取完成多轮 rollout。Authoring reference 只给独立筛选器。
4. 筛选成功轨迹：答案正确、必要证据齐全、音视频引用落在实际读取区间、工具可独立回放，保留有效的不同路径。
5. 导出 ShareGPT conversations；审计真实 trainer token/labels、各轮 target 与序列长度，然后训练。

四类轨迹近似均衡：source discovery、event-instance localization、cross-source comparison、research composition。包含声画关系驱动的后续查询和选窗；单条“描述这个视频”的 QA 只作少量工具接通样本。

Teacher 优先采用已配置的强工具使用服务，或部署开放 teacher。当前尚未确认可用 Gemini endpoint/key，也尚无实际 teacher episodes。固定 observer 的描述是可错的观测候选；核验实际媒体支持关系，避免把其错误描述直接蒸馏进 student。

## 3. 训练规模与执行顺序

| 阶段 | 数据目标 | 训练与验证 |
|---|---:|---|
| Seed/connectivity | 64–256 条 retained trajectories，独立来源 | 16–32 个短 batch 的 backward/optimizer smoke；保存并 reload checkpoint |
| Pilot | 约 2K 条 | LoRA SFT，候选 1–2 epochs；0.5K/1K/2K 来源家族嵌套曲线 |
| Main | 10–20K 条 | 按 pilot 学习曲线确定规模与配置；正式训练 checkpoint 与 benchmark 主表 |

这些是目标规模，不是当前已保留数据量。首轮训练使用 BF16、gradient checkpointing、microbatch 1、有效 batch 16、LoRA rank 16/32；LR 候选 2e-5/1e-4，由 train-validation 选择。Context 由实际轨迹长度决定，不能静默截掉 final 或依赖观察。

数据 runtime 位于 `runtime/sft-data`，独立 Pixi lock，CPU torch 2.8.0、transformers 4.57.6、datasets 4.0.0、peft 0.18.1、trl 0.24.0。它验证 native trainer 预处理，不提供 CUDA 训练能力。真实训练另建 CUDA Pixi runtime 和有界 Kjob launcher，不能直接用 CPU lock 声称训练就绪。

## 4. 怎样证明性能提升

核心实验保留以下三行：

| 系统 | 新训练方式 | 要回答的问题 |
|---|---|---|
| OneSearch-AV-Prompted | 无新增 AV SFT | 原 checkpoint 搭工具的起点 |
| Answer-only SFT | 相同训练任务，仅答案监督 | 单纯学答案能够带来多少 |
| AV trajectory SFT | 多轮查询/窗口/组合/答案 | 工具策略监督是否有额外收益 |

固定检索/观察栈与推理预算；answer-only 对照报告监督 token 数，并增加 token-matched 子集。主指标包括 Answer Accuracy、Evidence-backed Success、完整证据包 source recall、正确实例与 moment recall。分任务族分析查询召回、窗口命中和跨源事实补全，结合学习曲线判断收益来自哪些能力。额外 reranking 在同一 SFT checkpoint 上比较。

16 个 dev cases 先用于逐题检查与闭环调试，不能支撑正式统计结论。正式性能比较使用扩展 dev 与独立 test，按来源家族配对统计。

## 5. 本轮实际验证与剩余工作

当前实际状态：16 个 dev 候选，10 AV + 6 text 来源；4 个 train authoring 候选，4 AV + 4 text 来源，约 90.70 秒新 AV。Train/dev 相同字节重叠为 0，但录制/事件家族审核未完成。Adjudicated reference、实际 teacher trajectories、retained SFT rows、AV SFT optimizer updates 均为 0。

已完成：

- 固定 native observer 的 episode adapter 与独立缓存回放接口，保存私有执行记录、向 actor 返回精简 observation。模型执行与语义核验分开记录。
- 搜索返回 scoped media type/duration，controller 可据此选择有效窗口；episode 聚合实际准备/observer 成本，缺失 tokens 用 unknown observation 计数表示。
- 实际 LLaMA-Factory converter/template/processor 检查通过：fixture 观察 tokens 全部 mask，所有 assistant actions/final 保留，无静默截断。使用发布 Instruct controller 对应的 `qwen3_vl_nothink`，`train_on_prompt=false`、`mask_history=false`。
- 独立 reviewer A/B semantic packet 与 importer。空表保持 pending；分歧进入 adjudication。
- CPU fixture episode 实际运行 search/AV preparation；2 次工具调用，50 prepared video frames、43,562 audio samples。Observer 与 controller 为明确的 contract fixture，没有模型 inference，没有可保留的训练样本。
- Research regression 188/188，native trainer preprocessing 4/4 通过。本轮未提交新 GPU job。

后续按顺序完成来源组与任务证据核验、配置真实 teacher/observer rollout 入口、合成并核验 64–256 条 seed、对保留数据运行 native label audit、CUDA optimizer/checkpoint smoke、2K pilot 与 Prompted 对比。Python `ToolRuntime(observer=...)` 已支持 native adapter；现有 teacher CLI 的默认 observer 仍未配置，不能直接执行 CLI 就声称语义 rollout 已接通。

本轮证据保存在私有 `research/runs/av-data-pipeline-20261010-v1`：native-runner-green.log、research-regression.log、native-label-tests-v2.log、native-label-fixture、native-contract-episode、semantic reviewer packets。原媒体、private references 和 tokenization records 不进入 GitHub。


## 6. 可复跑的验证命令

从 repo 根目录执行：

```bash
pixi run --locked python -m unittest discover -s validation/research -p 'test_*.py' -v
pixi run --locked --manifest-path runtime/sft-data/pixi.toml python -m unittest discover -s validation/sft_data -p 'test_*.py' -v
```

Native trainer tests 必须使用第二条独立 runtime；根目录 inference 环境不包含 peft/trainer 依赖。复核时曾误用根环境，得到 `ModuleNotFoundError: No module named 'peft'`，换回上述已锁定数据环境后 4/4 通过，无须混装依赖。
