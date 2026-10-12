# 原生 AV Teacher 数据生产实现计划

**Goal:** 在原 OneSearch agent loop 中接入原生音视频 Gemini teacher，生成可分片、可恢复、可验收的真实工具轨迹，支持后续 15K–20K SFT。

**Architecture:** Student 保持 Qwen2.5-Omni-7B；teacher 使用独立 Gemini REST runner，原始 AV 作为 inlineData 发送，沿用 inspect_av、冻结搜索与网页读取。分片工作队列记录候选 ID、原问题 ID、配置、输入媒体和源码身份；候选生成与成功验收独立。

**Tech Stack:** coder ssh 3 / /home/jovyan/OneSearch-VL，Pixi Python 3.11、requests、FFmpeg、原 OneSearch 与 LLaMA-Factory。

**Spec:** 用户 2026-10-12 确认的 15K–20K teacher trajectory SFT 方案，以及 docs/research/OneSearch-AV-Native-Omni-Architecture-v4-2026-10-12.md。

## 全局约束

- 最新用户追加授权：有实质实现进展后可以提交 kjob。先完成代码与真实数据，再按项目 preflight/budget 开展有明确实验目标的 GPU 工作。
- Teacher 原生接收视频及其音轨，不以文字观察替代 AV；真实工具返回必须参与 history。
- Actor 只拿问题、匿名 AV 和工具；答案与验收只供独立 review。
- 配置仅保存环境变量名称；不得保存或输出 API key。
- 网络/模型响应使用本地 HTTP fixture 验证；fixture 输出不作为训练数据。
- 训练数据只来自 train，按录制家族隔离 dev/test；队列完成不等于验收成功。
- 保留历史原生 Omni receipts，新增 provider-neutral receipts 不允许无生成记录的文本进入 SFT。
- 新 run 版本化；崩溃候选保留且不自动重复发送付费请求；不改写既有轨迹。

## Task 1：inspect_av 参数类型回归

Files: onesearch_vl/onesearch_infer/tools.py；validation/test_native_av_tools.py。

- [x] 写回归：video=['video_1']、{'id':'video_1'}、123 返回 Tool execution error 和空媒体。
- [x] 运行测试并确认旧代码触发 TypeError。
- [x] 在字典查找前验证字符串，再运行真实裁剪与原 pipeline tests。

## Task 2：Gemini 原生 AV runner 和生成记录

Files: onesearch_vl/onesearch_infer/gemini_av_runner.py；research/av_train/generation_receipts.py；validation/test_gemini_av_teacher.py。

Interfaces: GeminiAVRunner(settings).infer(contents, system_instruction, cfg) 返回原 pipeline 的 Gemini-shaped response；generation_identity(response, snapshot) 返回 provider/model identity。

- [x] 写本地 HTTP integration tests：实际视频 bytes、两轮历史、裁剪视频、错误/截断/超限、输出不带 key 或 AV base64。
- [x] 红测后实现请求序列化、native generateContent 传输、错误清理和媒体/输出 hash 记录。
- [x] 验证不丢音视频，不让文本/静帧 provider 冒充原生 AV。

## Task 3：SFT 导出兼容 teacher 与候选 ID

Files: research/av_train/native_sft.py；validation/test_native_omni_sft.py。

- [x] 写失败测试：真实格式的外部 teacher receipt 被当前硬编码拒绝；多个 candidate 指向同一 train task。
- [x] 以 generation_identity 检查每轮同一个 provider/model，核对输出和输入媒体 hash；保留旧 Omni receipt 路径。
- [x] 导出 lineage 分别记录 candidate_id/task_id，保持原生 AV、assistant targets 与 paired answer-only。
- [x] 运行原 exporter/gate 和 original trainer CPU tests。

## Task 4：分片候选生产队列

Files: research/av_train/teacher_rollouts.py；research/configs/gemini_teacher.example.json；validation/test_teacher_rollouts.py。

Interfaces: prepare_plan(batch_dir, out_dir, teacher_config, prompt_file, candidates_per_task, shard_size)；run_shard(plan_dir, shard_id, max_candidates, max_api_calls)；summarize_plan(plan_dir)。

- [x] 写失败测试：候选分片不重复、train-only、已完成不重复请求、crash 保留、call budget、配置/输入变更拒绝、没有答案进入 actor。
- [x] 实现确定性 manifest、独占 shard lock、原 process_native_av_case 调用、原子状态和独立 review 状态。
- [x] 显式 plan / run / status CLI；无后台 launcher、无 kjob 提交。
- [x] 用本地 HTTP fixture 跑真实 AV + frozen search 两轮流程；准备 20题 × 3候选队列；真实 API 先做媒体能力 probe，能力未确认不开始正式候选。

## Task 5：交付

- [x] 中文说明 15K–20K 总体规模、teacher 与 student 分工、现有 20题/97来源和 API 媒体能力边界。
- [x] 运行相关回归、diff --check、检查不包含私有数据/权重/配置。
- [x] 将公共代码、测试、环境和文档 commit/push 到既有 research/av-deepresearch；校验 GitHub SHA。

## 2026-10-12 实际执行补充

- inspect_av、teacher runner、provider-neutral exporter、shard queue 的红绿测试已完成；严格 action/observation 校验修复独立 review 的 P1。
- OpenAI 兼容 adapter 已接入并经过本地 HTTP AV 测试；当前实际 gateway 的 video_url 格式不满足感知验证，因此不用于首批生产。
- 当前 gateway 还提供原生 Gemini 接口；完整 MP4 的两组画面/语音对照已通过，非语音计数仍不稳定，需要按类评价。
- CPU：20/20 自然 AV processor 输入通过，80 项集成测试（1 pinned tokenizer 默认环境跳过，已单独通过），5 项 original trainer CPU tests 通过。
- 20题 × 3候选已 versioned plan；真实 teacher 候选是否启动与完成数以 execution 文档和对应状态文件为准。
- Task 5 的公共文件 commit/push 待本轮完成后检查；此 checkbox 只表示交付准备，不能据此宣称 push 已完成。

- 实际首轮 6 候选失败，8 请求槽位中包含编码前拦截；原始片段请求超限与多个/未完成动作是已观察原因。新增 full-duration teacher_media 压缩（新 batch、父媒体 lineage，VFR 与真实音轨解码回归通过），并版本化强化逐轮动作 prompt；旧失败 run 保留。

## 后续结构化协议与真实采集

- teacher action JSON schema 选择单个动作，runner 确定性转为原 OneSearch tags；原 provider 文本与双 hash 校验保留。
- 参数 schema 要求六个 nullable 字段，未使用字段为空，不把 start_s/end_s 塞进 handle。
- v7 完成 20 条首候选：15 通过轨迹校验、5 失败、71 请求槽位；40 候选未执行。
- 15 条独立自动审核保留 13 条 silver，另 2 条 review 503；paired 导出每组 13 行，13 个独立任务。没有人工 gold。
- 80 项集成 tests（79 pass、1 pinned tokenizer skip，后者独立 pass）；原 trainer CPU 5 项通过。
- 单独 exporter 的 structured receipt import 已修复，新子进程红绿回归和 reviewer 14 tests 复核通过。
- GPU 新任务以 CPU 全数据 audit 和冻结后的预算为准，状态见 teacher execution 文档。

- 公共代码 d574188 已推送并回读 GitHub SHA；工作区干净。
- 新 kjob slurm-profile-slurm-dvt82 Complete：1 张 H20Z、668 秒，两组各 13 行、16 updates，optimizer/scheduler/RNG 与 392 adapter tensors 重载核对通过。无 held-out 性能主张。
