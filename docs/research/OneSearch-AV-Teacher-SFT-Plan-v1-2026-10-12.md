# OneSearch-AV：15K–20K 强 teacher 轨迹 SFT 方案

日期：2026-10-12。主代码：coder ssh 3 的 `/home/jovyan/OneSearch-VL`，分支 `research/av-deepresearch`。

## 1. 研究目标和主张

让原生 Omni 模型学会利用音视频线索进行搜索问答：接收真实音视频与问题，识别有价值的线索，必要时裁剪检查，构造搜索 query，读取网页，最后回答。这篇论文沿用 OneSearch-VL 的任务范式，来源数量由问题决定。Search for Generation 留给后续论文。

核心假设是：**强多模态 teacher 的成功工具轨迹，可以让 Qwen2.5-Omni-7B 学到比直接答案 SFT 更有效的音视频搜索策略。** 首先用 SFT 得到明确的性能增长与数据规模曲线，后续再考虑 RL。主要贡献围绕原生 AV 搜索能力、规模化可训练轨迹以及识别音频贡献的 benchmark 组织。

Student 固定为 Qwen2.5-Omni-7B。它原生接收视频及真实音轨，接着自主执行工具。Teacher 独立使用 `gemini-3.1-pro`，接受同一匿名 AV、问题和真实工具返回；不把 observer 的文字描述当作媒体输入。主链复用原 `onesearch_infer.pipeline.run_agent_loop` 与原 `SFT/src/train.py`，保留官方许可与基线。

15K–20K 指最终保留的成功轨迹数，不能由少量问题重复采样凑数。当前已有数据和目标规模见第 7 节。

## 2. 数据规模与覆盖

| 阶段 | 独立问题目标 | 生成候选目标 | 保留轨迹目标 | 目的 |
|---|---:|---:|---:|---|
| 接口与流程试跑 | 20 | 60，每题 3 个 | 按事实验收 | 检查 AV、真实工具、多轮历史和可训练导出 |
| 首轮学习实验 | 1K–1.5K | 2K–4.5K | 1K–2K | 验证 trajectory SFT 是否改善 held-out search QA |
| 主训练集 | 10K–12K | 24K–36K | 15K–20K | 训练主模型并建立 1K/5K/10K/15K/20K 规模曲线 |
| 开发与测试 | dev 约 300，test 约 1K | 独立生成与核实 | 经审核 benchmark | 评价答案、搜索成功和音频贡献 |

这些是规划值；保留率和 API 预算以首轮实际统计修订。主训练集优先取得 5K–6K 个不同录制家族，每条媒体通常产生 1–3 个不同问题，同一问题通常最多保留 2 条行为有差异的成功轨迹。记录四个不同数量：媒体/录制家族、独立问题、候选轨迹、保留轨迹。数据规模曲线按独立问题抽样，不把同题不同候选拆到不同规模或 split。

覆盖六类可听与可看的线索：机械/技术演示、动物与环境、音乐与表演、现场活动、说话与公共演讲、交通与地点。问题应需要从线索进入检索，答案包含可核实的外部事实。语音、非语音声音、视觉实体以及联合消歧都要有足够样本。少量纯感知 QA 用于保持 AV 能力，检索型问题是主训练内容。

初始输入可使用 15–120 秒的自然片段，检查工具再裁剪 3–20 秒；实际时长由有意义的事件决定。现有 20 题最多 30 秒，只是首批输入测试。长片段进入生产前要实测序列长度、请求体积和媒体信息覆盖；不能把所有长视频统一截为首几秒。现有 inline 请求总大小上限为 20 MB，多轮历史也计入；超限显式失败。当前已实现版本化的完整片段压缩：宽度上限 640，H.264 CRF 28 / VBV 600k，AAC 64k，保持 VFR、完整时长及录制音轨；单文件目标 3 MB。原数据保留，生成新 batch 与父媒体/变换/输出 hash 记录。更长视频后续实现上传协议。

## 3. 数据生产管线

1. **取得媒体与公开参考。** 收集可研究使用的真实音视频、许可、原始 URL、录制家族、媒体 hash 和公开网页。冻结检索语料与网页正文，媒体文件和私有事实记录留在远端。保留录制时间、裁剪区间与基本音画同步。
2. **构造题目与答案候选。** Author 可以看原始媒体和公开参考来写自然问题、答案候选、答案出处以及线索区间。生成题目时检查是否真有可感知线索，是否需要检索、是否只是重复标题或元数据。题目数据合成与 teacher rollout 是不同阶段；当前 queue 实现的是后者，尚未实现大规模自动题目合成。
3. **按录制家族拆分。** 同一录制、裁剪、转码、转载与近重复都落在同一个 split。先建立 family ID，再生成训练问题。内容 hash 去重只是第一层，不替代录制家族判断。Dev/test 在合成训练题之前冻结。
4. **生成匿名 actor 输入。** Actor 只收到问题、匿名真实 AV 和工具，初始输入不含标题、答案、私有 source 路径或未来 observation。搜索后获取公开网页中的事实是正常任务行为。
5. **强 teacher 执行真实轨迹。** 每题生成 2–3 个候选，沿用 `inspect_av`、`text_search`、`image_search`、`visit_webpage`。实际可用工具由冻结 backend 和 prompt 定义，不伪造网页或动作。完整输入与所有裁剪媒体保留在下一轮 history。
6. **轨迹与语义筛选。** 自动检查完成状态、唯一合法动作、真实工具返回、历史一致性、模型/媒体/输出身份；然后检查答案是否被公开证据支持，query 是否使用实际线索，是否存在错误来源或声音幻觉。Teacher 使用结构化 JSON 选择单个动作，runner 仅确定性转为原 OneSearch 标签；原 provider 输出、hash 与转换结果一起保存，不改写历史失败。失败候选保留原因，可明确生成新的版本；不会自动重发不确定付费请求。大规模训练采用独立语义判定加分层人工抽查；目前 silver 审核是同一 Gemini 模型的另一次调用，存在相关偏差，不能作为人工 benchmark gold。
7. **导出原生 AV SFT。** 成功 verdict 绑定具体 trajectory hash 和 train split。导出 ShareGPT、原生视频列表与真实录制音频 sidecar。Assistant 的可见计划、工具调用和最终回答进入 label；user、tool observation 和音视频输入不作为监督目标。不保存 API 的隐藏 reasoning 字段。

## 4. SFT、baseline 与规模实验

第一轮训练对 Omni 的语言决策部分做 LoRA，固定音频/视觉 encoder 与多模态 projector；先学习如何用已有 AV 感知能力做搜索。是否进一步训练 encoder 由后续消融决定。原 LLaMA-Factory 使用 `qwen2_omni`、`use_audio_in_video: true`、`train_on_prompt: false`、`mask_history: false`。Loss 是所有保留 assistant token 上的 cross entropy。

最小主表包含：

| Baseline | 学习/输入 | 回答的问题 |
|---|---|---|
| 原始 OneSearch-VL | 原视觉模型与原工具范式 | AV 扩展相对原方法有什么收益 |
| 原生 Omni，未训练搜索 | 相同 AV、相同工具和预算 | 开箱 Omni 能做到什么 |
| Omni answer-only SFT | 同题 teacher 最终答案，初始 AV | 只学答案能否获得同样提升 |
| Omni trajectory SFT | 完整成功工具轨迹和原生 AV | 搜索行为监督是否有额外收益 |
| 视频 + ASR 文字控制器 | 模块化感知结果驱动搜索 | 原生联合 AV 相对常见拼接方案的价值 |
| 强 teacher | 相同匿名 AV 和冻结检索 | 提供能力参照，记录不同模型与计算成本 |

主比较使用相同 train 问题、source split、student 初始权重与推理工具预算。同时报告训练样本数、assistant label token、包含媒体的实际训练 token、updates、GPU 时间、推理调用和累计媒体处理量。Answer-only 与 trajectory 长度不同，安排相同问题规模及相近训练计算量的两组比较，不只匹配 epoch 数。

数据规模取 1K/5K/10K/15K/20K 的嵌套、按问题和录制家族分组抽样集合。主看 held-out answer accuracy 和需要检索问题上的 task success；比较无裁剪、无音频、ASR 替代真实音轨以及 1/2/3 个候选的消融。正式报告采用至少 3 个 seed 和置信区间，学习实验先确定合理的超参数范围。

## 5. Benchmark 与 analysis

Benchmark 使用 AV 输入加可核实事实问答，来源数量由题目决定。每题记录答案、可接受别名、参考网页、关键 AV 区间、录制家族及是否需要音频/视觉/检索。设计约 1K 个正式测试问题，并保持按类和难度的覆盖；当前 16 dev 和 20 train 候选不等于正式 benchmark。

主要评价答案准确率和检索型 task success，补充首个有效 query、有效来源命中、回答证据支持率、工具调用数与时间/费用。声音识别、视觉实体识别、query 构造、网页选择和答案综合分别分析。工具调用更多本身不视为能力提升。

音频贡献通过对照测量：原 AV、视频静音、仅音频、视频加 ASR、保持画面但替换声音。替换声音的样本需保持问题仍有明确正确答案，或明确标为干预分析；不能把不成立的问题混入主准确率。特别报告语音线索和非语音声音两组。增加标题/元数据线索排查，避免模型通过文件名或问题模板定位来源。

## 6. 代码入口与运行约定

- `onesearch_vl/onesearch_infer/omni_runner.py`：student 原生 AV 推理与 processor 输入。
- `onesearch_vl/onesearch_infer/av_io.py`、`tools.py`：真实媒体 handle、基本时间和 `inspect_av` 裁剪。
- `onesearch_vl/onesearch_infer/pipeline.py`：原 OneSearch agent loop；记录真实多轮消息与工具返回。
- `onesearch_vl/onesearch_infer/gemini_av_runner.py`：完整 AV `inlineData`、原生 Gemini REST、请求预算与安全生成记录。
- `onesearch_vl/onesearch_infer/openai_av_runner.py`：实验性的 `video_url` gateway adapter；具体 endpoint 必须实测。当前服务的该格式未通过媒体能力探针，不能用于生产。
- `research/av_train/teacher_media.py`：保留完整事件与录制音轨的 CPU 压缩，建立新 batch、真实时间和父媒体 lineage。
- `research/av_train/teacher_rollouts.py`：`plan / run / status` 分片队列、候选 ID、源码/媒体绑定、恢复与 API 请求槽位预算。
- `research/av_train/generation_receipts.py`：local Omni 与远端 teacher 的生成身份和媒体/输出完整性校验。
- `onesearch_vl/onesearch_infer/teacher_actions.py`：teacher JSON schema、工具参数校验与确定性协议转换；保持原 student 的标签协议和真实工具执行。
- `research/av_train/native_sft.py`：实际轨迹验收、paired trajectory/answer-only 导出、原 trainer 的 label/media audit 与配置准备。
- `SFT/src/llamafactory/data/mm_plugin.py`：Omni 路径的实际 PTS 视频采样；原版其他模型路径保持原行为。

配置 JSON 只记录环境变量名称。私有凭据文件是 `/home/jovyan/.config/onesearch-av/teacher.env`，权限 0600，不进入 Git。运行时显式加载环境；`status` 不会自行读取私有文件。

```bash
cd /home/jovyan/OneSearch-VL
set -a
source /home/jovyan/.config/onesearch-av/teacher.env
set +a
pixi run --locked python -m research.av_train.teacher_rollouts status \
  --plan-dir research/runs/av-teacher-data-20261012-v7/candidate-plan
```

生产运行需要显式传 `--max-candidates` 与 `--max-api-calls`。后者计入传输前预留的请求槽位，包括失败和超时，不能当作服务的最终账单。失败/中断候选不自动 retry。修改被绑定的源码或媒体后建立新 plan，不覆盖旧 run。

## 7. 当前执行证据

- 20 个自然 train AV 输入已完成 CPU processor 解码与 tokenization；每例同时含视频和真实音轨，最长 30 秒，最大初始 prompt 3,695 token，未加载模型。
- 当前冻结语料包含 97 个 AV 来源与对应捕获网页。已排除 3 个与 dev 重叠的来源；录制家族的完整近重复隔离仍待完成。
- 20/20 输入完成完整时长压缩：最大单文件从 86.35 MB 降到 2.58 MB，最大时长差 3.2 ms，全部保留录制音轨。v7 计划 20题 × 3候选 = 60；实际执行每题首个候选，共 20 条、71 请求槽位，15 条通过轨迹校验，5 条失败，40 条候选未执行。结构通过率从 v5 的 1/6 提升到 v7 首六的 4/6；这是采集有效率，不是模型任务准确率或性能增长。正式 15K–20K 数据尚未生成。
- v7 的 15 条完成轨迹经过另一次自动审核；13 条保留为 silver、2 条审核遇到 HTTP 503，后者暂不保留。原 JSON code fence 中的完整审核 verdict 在新的 review-v2 中确定性解析，原回答和 hash 保留，没有额外 API 请求。已导出 13 个独立问题的 paired trajectory/answer-only，尚无人工 gold。
- 原生 Gemini 完整 MP4 输入通过两组独立语音/画面对照：黄色/绿色画面与 Copper Falcon/Violet Harbor 转写均正确。隔离的蜂鸣计数、短音频访问表现不稳定，非语音感知需要后续逐类测量。OpenAI `video_url` 路径会返回错误感知，已排除出首批生产；当前使用同一 gateway 的 `/api/v1beta` 原生接口。合成控制媒体只用于接口验证，不进入训练集。
- 相关集成测试 80 项：79 通过，1 项在默认 CPU 环境跳过；该 tokenizer 用例在 pinned 原 trainer 环境单独通过。原 trainer 的 5 项 CPU label/media/sampler tests 通过。独立 review 发现并修复 structured receipt 的单独 exporter 导入问题，14 项相关测试复核通过。
- 之前 trajectory 与 answer-only 各做过 2 行、8 个真实 optimizer updates 的 GPU 工程 pilot，保存与重载已验证；它证明训练通路，尚无 held-out 性能增长结果。
- 本轮新 kjob `slurm-profile-slurm-dvt82` 已 Complete：1 张 NVIDIA H20Z，13 个独立题目，每种格式 13 行，trajectory/answer-only 各 16 次真实 optimizer updates；保存 optimizer/scheduler/RNG，LoRA B 更新，392 个 adapter 张量 GPU 重载逐项相等。耗时 668 秒（0.186 单卡小时），没有自动重试。此轮验证真实 teacher 数据训练通路；尚未验证 held-out 性能增长。
