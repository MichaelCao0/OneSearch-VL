# OneSearch-AV 训练数据扩量：执行与使用说明（2026-10-12）

## 目标与当前产出

本轮先扩大音视频来源、不同题目和成功搜索轨迹，继续使用 OneSearch-VL 搜索问答范式。Student 是 Qwen2.5-Omni-7B；teacher 使用已验证的原生 Gemini AV 输入。Actor 实际看到音视频，再通过原版 run_agent_loop 调用 inspect_av、text_search 和 visit_webpage，最后回答问题。当前搜索使用冻结的真实发布者网页语料。

本轮新采集目标为 300 份 AV 文件，下载上限 20 GB、700 次尝试；加上原有 100 份素材，构成约 400 份来源的候选池。同一录制的转码、裁剪、同一道题的多次采样分别记录，不当作新独立来源或题目。15K–20K 成功轨迹仍是后续训练目标。

截至 **2026-10-12T06:33:22.678782+00:00** 的执行快照：

| 阶段 | 当前实际结果 |
|---|---:|
| 本轮新增、带可解码音视频的素材 | 225 / 300 |
| 已有池中未用素材的 AV 出题调用 | 69 次，53 完成、16 失败 |
| 该组出题候选 | 54 题、45 份素材 |
| 第一组新增搜索轨迹计划 | 54 道不同题目，每题 1 条候选 |
| 该组已完成搜索轨迹 | 47 |
| 新增 silver 已审核并导出 | 44 |
| 加上此前 13 条后的累计已导出 | 57 |
| 首批 160 份新素材经家族分组后选出的新 AV 输入 | 106 份，出题进行中 |
| Human benchmark gold | 0 |
| 本轮新提交 GPU Job | 0 |

这些是探索性 silver 训练数据。独立调用同一个 Gemini 模型做审核，存在模型偏差相关性；不是人审 benchmark gold，也不代表已测得 student 性能增长。素材文件数与自动候选家族数不能直接证明独立录制数。

## 已执行的扩量流程

1. **继续采集公开素材和原始网页。** 保留原始字节、发布者信息、网页快照、音轨与视频探测结果、下载失败与字节预算。现有采集器串行节流，支持保存进度和冷却后人工显式续跑。
2. **扩大题目来源。** 原来 100 份素材中，排除已用及已知 dev 家族后，选取 69 个未使用的候选录制家族。每个输入保留完整的选定片段及真实音轨，传给原生 AV teacher，每份素材最多生成两个要求不同事实的问题。
3. **冻结新增来源批次。** 首批 160 份新增素材与旧池、dev 素材共同做 CPU 指纹和候选家族分组；270 条资产记录对应 267 个不同文件哈希、205 个候选家族。除已用/保留家族后，106 份新素材进入出题。剩余采集结束后再处理余下素材，排除这些已尝试家族。
4. **逐批生成真实搜索轨迹。** 已有池的新增题目分成 24 / 24 / 6 题，三路独立子进程执行原始 agent loop。每题本轮仅采样 1 条，最多 8 轮；失败和不确定请求不自动重发。新增素材出题后按 25 份来源一批进入同一流程。
5. **独立调用审核并导出。** 审核实际 AV、问题、最终回答、真实工具返回及私有发布者答案候选。成功结果绑定 producer 完成时的轨迹哈希，再交给已有 native_sft 导出器生成 trajectory / answer_only 配对数据。训练输入不含私有答案、来源路径或 teacher 的观察说明。
6. **原始训练器 CPU 验证。** 对新增已导出批次运行原版 LLaMA-Factory 的媒体与标签审计，检查原生音视频保留、完整片段采样和 assistant-only supervision。扩数据阶段继续保留分片，不重复提交十几条数据的 GPU pilot。

## 代码与接口

公共实现均在 `/home/jovyan/OneSearch-VL`，使用 coder ssh 3 和 Pixi：

- `research/av_data/source_inventory.py`：验证媒体哈希，取真实帧指纹，与保留的 dev/test 素材一起做确定性的候选家族分组。
- `research/av_train/question_authoring.py`：prepare / run / status / materialize；原生 AV 出题、调用记录、逐题过滤、匿名 query 和私有 publisher references 构造。
- `research/av_train/teacher_silver_review.py`：独立 AV silver 审核；review 锁覆盖 pending 选择，每条 paid attempt 使用 create-only reservation，不覆盖或重发已保存的尝试。
- `research/av_train/teacher_rollouts.py`：复用已有冻结 teacher 计划、分片与原版 agent loop。
- `research/av_train/native_sft.py`：复用已有成功轨迹验证、配对导出及原版训练器 audit。
- `research/av_data/clue_batch.py`：30 秒裁剪边界允许 1 微秒浮点误差，仍拒绝实际超限。

新出题提示将具体视听描述保留在 private cue 字段，要求给 actor 的问题简短，避免用 teacher 的长文字观察替代学生自己的 AV 观察。已生成的早期题目保留原样和版本，新批次使用修订后的提示。

```bash
# 1. 对 acquisition manifest 和保留的 dev/test registry 建候选家族。
pixi run --locked python -m research.av_data.source_inventory \
  --manifest /absolute/path/acquisition-manifest.json \
  --reserved-registry /absolute/path/dev-sources.private.jsonl \
  --out-dir /absolute/path/new-inventory

# 2. 准备并冻结 AV 出题计划。used-drafts 的每条记录需含 acquisition_source_id。
pixi run --locked python -m research.av_train.question_authoring prepare \
  --manifest /absolute/path/acquisition-manifest.json \
  --inventory /absolute/path/new-inventory/inventory.private.json \
  --reserved-registry /absolute/path/dev-sources.private.jsonl \
  --used-drafts /absolute/path/already-attempted.private.json \
  --teacher-config research/configs/gemini_teacher.example.json \
  --out-dir /absolute/path/new-author-plan --limit 200

# 3. 在显式加载远端私有环境配置的进程中运行，CLI 不加载或打印密钥。
pixi run --locked python -m research.av_train.question_authoring run \
  --plan-dir /absolute/path/new-author-plan --workers 3 --max-sources 200

# 4. 冻结实际完成的题目/媒体，再由 teacher_rollouts.py 创建、执行搜索计划。
pixi run --locked python -m research.av_train.question_authoring materialize \
  --plan-dir /absolute/path/new-author-plan \
  --manifest /absolute/path/acquisition-manifest.json \
  --inventory /absolute/path/new-inventory/inventory.private.json \
  --reserved-registry /absolute/path/dev-sources.private.jsonl \
  --out-dir /absolute/path/new-question-batch

# 5. 对已完成的搜索轨迹做 silver 审核，复用 native_sft.py export。
pixi run --locked python -m research.av_train.teacher_silver_review \
  --plan-dir /absolute/path/rollout-plan \
  --out-dir /absolute/path/new-silver-review --workers 3 --max-candidates 400
```

## 本轮验证与数据位置

新增测试覆盖真实 publisher quote、中文短答案泄漏、别名重复计题、dev 家族传播、provider 输出和 AV 哈希、HTTP 失败续跑不重发、producer 完成哈希、审核 attempt 不覆盖，以及真实本地 HTTP fixture 到匿名题目 batch 的链路。集成 **44 个测试全部通过**，另有 `git diff --check`。首个新增分片 21 条轨迹及其 answer-only 对照（共 42 行）已通过原版训练器 CPU 标签/媒体审计，完整片段采样覆盖通过；另外两个已导出分片正在同样审计。

数据与运行证据只保存在远端，不提交原始媒体、私有 references、审核 verdict 或凭据：

```text
research/runs/av-scale-20261012-v4/
  acquisition-v1/                    # 300 份素材采集目标，进度与下载证据
  authoring-unused-v1/                # 69 份原有未用素材的 AV 出题
  production-v3/production-chunk-*/  # 54 道题的搜索、silver 和 SFT 分片
  new-sources-v1/                    # 首批 160 份新素材、106 份选择及后续生成
  remaining-sources-v1/              # 排除前述已尝试家族后的剩余素材处理
  progress.py                       # 只读汇总，不读凭据、不调用模型
```

```bash
pixi run --locked python research/runs/av-scale-20261012-v4/progress.py
```

下一阶段以这轮实际出题率、轨迹完成率和 silver 保留率估算采集需求，继续增加来源与事实类型，逐步形成 1K–2K retained pool，再进行具有学习曲线意义的 SFT 与 held-out 比较。音频消融、视觉消融与搜索预算匹配实验用于检验学生是否真正学到 AV 搜索能力。
