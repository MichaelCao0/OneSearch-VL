# OneSearch-AV：2K 训练数据积累执行记录

2026-10-13，用户授权继续积累到 2K。阶段目标为 **2,000 个不同问题的、审核保留且通过原版 SFT CPU 媒体与标签检查的实际搜索轨迹**。后续 15K–20K SFT 数据目标保持为研究计划。

## 当前已验证基数

2026-10-13 11:21（中国时间）完成首轮服务恢复验收：

- 原有 188 条：13 条早期数据与 175 条扩展数据，均有成功轨迹和原版训练器 CPU 检查记录。
- 新教师服务小批 4 条：4/4 完成实际工具调用，4/4 经同模型独立调用的自动 silver 审核保留，4/4 通过 CPU 检查。
- 共 **192 条不同题目的合格轨迹**。对应 answer-only 训练对照有 192 行，不能合计成 384 道题。
- 本次未提交新的 GPU 任务；数据通过 CPU 检查不代表新模型训练或性能实验完成。自动 silver 不计入人审 benchmark gold。

持续变化的生产数量以远端只读 progress 命令为准，不把下载来源、出题候选、完成轨迹、审核保留和 CPU 合格数量混用。

## 教师服务恢复

此前配置的 `gemini-3.1-pro` 原生接口返回 HTTP 404。`gemini-3.5-flash` 虽返回 HTTP 200，内容是停用提示且缺少正常 usage，不能用于数据生成。

`gemini-3.8-flash` 在现有服务的原生 `generateContent` 路径上通过了：

1. 文本控制：返回指定口令、STOP 和有效 usage。
2. 两个不同 MP4 音视频控制：分别正确识别黄/绿画面并逐字转录两个不同的语音口令。
3. 四道实际搜索问答：实际执行 OneSearch 的工具循环，并完成审核与原版 SFT CPU 检查。

2K 队列使用独立 `ONESEARCH_2K_TEACHER_MODEL` 环境变量固定教师模型。教师配置 JSON 只存环境变量名称；凭据从远端私有 0600 文件读取，不进入 Git、日志或公开文档。旧付费调用及其原始响应保留，不覆盖成新模型结果。

## 连续生产流程

队列在 `coder ssh 3` 的 `/home/jovyan/OneSearch-VL` 运行，采用 Pixi。

1. 已有首批来源中，只处理尚无任何付费尝试记录的题目。失败、完成或中断的旧候选都排除；三条中断记录已做无请求的状态对账。
2. 继续处理现有 1,000 个新音视频来源，纳入历史来源和开发集保留来源一起计算候选录制家族，避免重新选择已尝试家族。
3. 每个新录制进行一次原生 AV 出题调用，最多两个答案字段不同的问题。新批先取最多六个录制验证出题，再继续未尝试录制。
4. 构建冻结检索库和匿名 AV 题目；每个出题批次只构建一次库。题目分片复用同一个检索库，最多 25 题/片、3 个 API worker。
5. 每题一次候选生成，最多 8 轮，每轮最多 4,096 输出 token；复用 `onesearch_infer.pipeline.run_agent_loop`，执行真实 `text_search`、`visit_webpage`、`inspect_av` 工具与观察。
6. 同模型另一次调用检查答案字段、媒体匹配、查询依据和实质幻觉；只有通过的实际答案与轨迹进入导出。
7. 运行原版 LLaMA-Factory 的 converter、Omni plugin、supervised processor 和 collator，验证真实音频、视频采样、全部 assistant 标签与 user/tool/media mask；检查通过后计入 2K。
8. 不足 2K 时，用扩展查询池继续分批采集。达到 2,000 个不同问题的 CPU 合格轨迹后停止，并保留全部分片及来源索引。

跨导出目录按 task ID 与录制/问题组合去重，并核对行、轨迹和导出摘要的哈希。同题的多条候选与 answer-only 对照不能增加问题计数。录制家族仍是自动候选分组，正式 source-disjoint benchmark 需另做人审。

## 实现与本次修复

- `research/av_train/scale_searchqa.py`：共享冻结检索库的题目切片、去重后合格数量核验、实际生成/审核/导出/CPU 检查循环。
- `research/av_train/provider_guard.py`：HTTP、传输或缺失正常生成 receipt 时发布共享停止信号。已经在途的请求记录完成；后续未尝试任务不发送请求。没有自动付费重试。
- `question_authoring.py`、`teacher_rollouts.py`、`teacher_silver_review.py`：三个阶段接入停止保护。输出、失败和中断记录保持版本化。
- `research/av_data/source_inventory.py`：复用经过验证的帧指纹，仍重新验证媒体哈希；减少重复解码。预先规范化元数据并减少不必要的标题相似度计算。80 个真实来源的分组结果与优化前完全相同。
- `question_authoring.materialize`：隔离无效时长，检索库的父视频时长使用原生媒体时钟，与裁剪保持一致。
- `native_sft.verify_assistant_target_text`：用实际 tokenizer normalizer 核对训练标签，允许原生 Unicode 规范化，仍拒绝被改写的工具动作或查询。

## 队列位置与预算

远端运行目录：`/home/jovyan/OneSearch-VL/research/runs/av-scale-2k-20261013-v1/`。

- `plan.json`：2K 目标、模型和数据路径、固定预算。
- `pipeline.py` / `pipeline.pid` / `pipeline.log`：连续阶段控制与进程记录。
- `state.json` / `progress.py`：阶段状态与只读实际计数。
- `cohort-*/`：每批的采集、录制家族、出题、冻结检索库与训练导出。
- `source-freeze.json` / `frozen-source/`：源代码哈希与副本。
- `provider-stop.json`：服务异常停止记录；需要检查后采用新版本继续，不能删除它以重放失败付费调用。

预算是上限，不是实际用量：现有新来源 1,000 个，最多另采集 5 批 × 1,000 个；每批最多下载 2,500 次、接收 60 GB，每个资产最多 150 MB。新增出题调用最多 6,000 次；每个录制最多两个题目，最多 12,000 个候选轨迹、每个候选最多 8 次生成调用。达到合格问题目标即停止，实际调用从记录计数。最长阶段预算 72 小时，耗尽或服务失败时保存明确状态。

367 个未来采集查询覆盖鸟类、动物、音乐、机械与现场/环境场景；每批排除历史 page ID。下载素材不自动成为训练数据。

## 只读进度命令

在 `coder ssh 3` 上执行：

```bash
cd /home/jovyan/OneSearch-VL
pixi run --locked python research/runs/av-scale-2k-20261013-v1/progress.py
```

查看当前首批题目生产阶段：

```bash
cat /home/jovyan/OneSearch-VL/research/runs/av-scale-2k-20261013-v1/first200-unattempted-v1/state.json
```

本次公开交付为代码、验证与执行文档。私有来源、媒体、答案、teacher 原始记录、API 配置、权重及运行目录不公开。

## 代码验证

本轮相关回归共 62 个测试通过，无失败、错误或跳过，覆盖原生教师、出题、旧候选保持、服务停止、自动审核、实际轨迹导出、分片/去重、缓存与检索开关。另在 pinned 原版 SFT runtime 上通过真实 Omni tokenizer 的 Bengali Unicode 规范化测试。4 条新模型真实轨迹与对应 4 条 answer-only 对照均通过原版训练器 CPU 媒体/标签检查。`git diff --check` 通过。
