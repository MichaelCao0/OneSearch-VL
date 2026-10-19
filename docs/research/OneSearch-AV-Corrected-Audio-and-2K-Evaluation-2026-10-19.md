# OneSearch-AV：音频修复后的重评与 500→2K 训练对照

日期：北京时间 2026-10-19。以下成绩均来自修复后重新生成的 rollout、五模型十条件联合匿名评分；历史失效音频消融不参与比较。

**本轮已完成 5 模型 × 2 模态 × 56 题 = 560 条 rollout；联合盲评流程结束，有效共同题 52/56，unknown 4。**

当前最明确的结果是：**真实搜索轨迹 SFT 大幅改善了任务完成；500→2K 的答案成功率继续上升，提升主要出现在 heldout，而音频收益尚未稳定。** 联合评分中，2K 原声为 36/52，500 原声为 33/52；2K 静音为 34/52，500 静音为 28/52。两组规模差值的录制家族 bootstrap 区间均跨 0。

这轮也给出了下一步方法设计的直接依据：2K 减少了重复调用，但仍会在已检索到目标后反复搜索；一些答案字段正确的案例仍绑定了错误视频。下一版应把改进放在 AV 锚点到实体的绑定、根据结果调整查询、及时完成答案，以及真正需要声音的组合问题上。

## 同题主结果

答案成功要求输出协议完整、所有参考字段正确且不与参考冲突。“支持充分”（原指标 supported_success）再排除重大、无参考证据支持的附加断言。Judge 读取题目、参考字段、冻结支持引文和最终答案，不观看音视频，也不审核完整搜索轨迹；该指标不等于逐步引用或全链条证据验证。评分为自动 silver，未作人工 gold 裁决。

所有十组只使用相同的 52 道有效题；4 道未通过评分响应校验或服务请求失败的题统一保留为 unknown，不按答错处理，也没有重放收费请求。unknown task IDs：avqa-dev-011, avqa-expand-1b413ce4f34900ce7c31, avqa-expand-4d45525ea16d2847aa15, avqa-expand-b579a9a2f7a8b3770373。

| 训练方式 | 原声：答案成功 | 静音：答案成功 | 原声：支持充分 | 静音：支持充分 |
|---|---:|---:|---:|---:|
| Qwen2.5-Omni-7B 基座 | 0/52（0.0%） | 1/52（1.9%） | 0/52（0.0%） | 1/52（1.9%） |
| 500 answer-only | 2/52（3.8%） | 1/52（1.9%） | 2/52（3.8%） | 1/52（1.9%） |
| 500 trajectory SFT | 33/52（63.5%） | 28/52（53.8%） | 30/52（57.7%） | 25/52（48.1%） |
| 2K answer-only | 4/52（7.7%） | 5/52（9.6%） | 3/52（5.8%） | 3/52（5.8%） |
| 2K trajectory SFT | 36/52（69.2%） | 34/52（65.4%） | 32/52（61.5%） | 31/52（59.6%） |

## 原声与静音的配对差异

以下差值均为原声减静音。按同一道题配对，并以自动录制家族为单位进行 10,000 次 bootstrap（seed=42）。置信区间反映这批题目的采样不确定性，不覆盖多训练 seed 变异。

- 500 trajectory SFT，答案成功：+9.6 pp；95% CI [-1.8, +20.4] pp；胜/负/平=7/2/43。
- 500 trajectory SFT，支持充分：+9.6 pp；95% CI [-2.0, +21.6] pp；胜/负/平=8/3/41。
- 2K trajectory SFT，答案成功：+3.8 pp；95% CI [-7.7, +16.0] pp；胜/负/平=5/3/44。
- 2K trajectory SFT，支持充分：+1.9 pp；95% CI [-8.2, +11.8] pp；胜/负/平=4/3/45。

若将 unknown 题的未知配对差值全部取最有利/最不利情况，2K 原声减静音在原计划 56 题上的完成区间为 [-3.6, +10.7] pp；这与 bootstrap 采样区间含义不同。

## 500→2K 的配对差异

以下差值为 2K 减 500。500 道训练题是 2K 的严格子集，两组独立从相同 pinned base 开始训练。

- trajectory SFT，原声，答案成功：+5.8 pp；95% CI [-8.0, +19.2] pp；胜/负/平=8/5/39。
- trajectory SFT，原声，支持充分：+3.8 pp；95% CI [-10.4, +17.9] pp；胜/负/平=8/6/38。

原声答案成功在计划 56 题上的 unknown 极端完成区间：[-1.8, +12.5] pp。

- trajectory SFT，静音，答案成功：+11.5 pp；95% CI [-3.6, +26.8] pp；胜/负/平=10/4/38。
- trajectory SFT，静音，支持充分：+11.5 pp；95% CI [-3.8, +27.5] pp；胜/负/平=11/5/36。

静音答案成功在计划 56 题上的 unknown 极端完成区间：[+3.6, +17.9] pp。

## 评分稳定性与真实案例

在相同的 52 道有效题、完全相同的 2K 最终答案上，中期双候选评分得到原声 33/52、静音 34/52；十候选联合评分得到原声 36/52、静音 34/52。3 个原声答案的答案成功标签发生变化，涉及 Little Corella 的别名表述、作曲家正确但作品名错误、拉丁学名正确但中文物种名冲突。联合评分是本轮预先指定的最终口径，原始两套回执均保留；这一敏感性进一步说明，当前不能以 2K 原声净多 2 题来宣称可靠的音频增益。

- **2K 的改善案例：节奏健身活动。** `avqa-dev-012` 中，500 模型把画面概括为“公园里用棍子和大球玩耍”，连续 8 次使用同一查询，未完成答案。2K 模型更换查询、访问实际返回的页面，给出 JTV Studios 与 2015-07-21。这里可以直接观察到查询调整和任务完成的改善。
- **2K 的退步案例：动画创作软件。** `avqa-expand-3fd4eaffd07ded099aaf` 中，2K 已检索到 Hungry Wolf/Flipnote 信息，却反复搜索直到 8 轮结束；500 则搜索、访问页面并完成 Flipnote Studio 答案。说明扩大数据尚未解决有证据时的停止决策。
- **音频相关但绑定未完成的案例：德语织布片。** `avqa-expand-4812e0d53990a4c54357` 与 `avqa-expand-9ece4b69b967c77e91b4` 属于同一录制家族。原声模型生成德语句子查询并访问《Bäuerliche Leinenweberei》第 4 集，参考对应第 3 集；拍摄者和地点相同，使答案字段可以得分。此例能展示查询发生变化，但不能证明正确视频绑定或完整证据链。

案例观察来自既有真实工具轨迹，没有新增模型请求，也没有据此回改主表标签。完整原声/静音翻转、500/2K 翻转、三个评分分歧案例另存于[运行目录中的22例附录](../../research/runs/av-corrected-audio-evaluation-20261019-v1/analysis-v1/关键案例与评分差异-中文.md)。

## 对后续实验的决定

保留 2K 模型作为当前 warm start 和后续数据方法的基线。下一批优先按已提出的组合构题方案建设 500 道新任务，先验证 64 道的可解性与锚点绑定；用实际音频线索约束检索、区分相似实体，再依赖外部关系或运算得到答案。Benchmark 同时标明音频贡献和必要证据依赖，增加对错误实体绑定的审核。强 Omni teacher 的价值应通过这类任务上的实际成功率来检验。

下一次训练对照应保持基座、训练配方、总 token/更新预算和固定评测题一致，分别检验加入组合题和加入经核验的音频线索题的收益。在线 DeepSeek Search/Fetch 另作同题检索环境对照。本轮未额外启动这些数据调用、训练或在线重评。

## heldout 与 development 分开报告

| 数据划分 | 模型 | 原声答案成功 | 静音答案成功 | 原声支持充分 | 静音支持充分 |
|---|---|---:|---:|---:|---:|
| heldout_silver | Qwen2.5-Omni-7B 基座 | 0/37 | 1/37 | 0/37 | 1/37 |
| heldout_silver | 500 answer-only | 2/37 | 1/37 | 2/37 | 1/37 |
| heldout_silver | 500 trajectory SFT | 22/37 | 17/37 | 19/37 | 15/37 |
| heldout_silver | 2K answer-only | 4/37 | 5/37 | 3/37 | 3/37 |
| heldout_silver | 2K trajectory SFT | 26/37 | 23/37 | 22/37 | 20/37 |
| dev16_publisher_candidates | Qwen2.5-Omni-7B 基座 | 0/15 | 0/15 | 0/15 | 0/15 |
| dev16_publisher_candidates | 500 answer-only | 0/15 | 0/15 | 0/15 | 0/15 |
| dev16_publisher_candidates | 500 trajectory SFT | 11/15 | 11/15 | 11/15 | 10/15 |
| dev16_publisher_candidates | 2K answer-only | 0/15 | 0/15 | 0/15 | 0/15 |
| dev16_publisher_candidates | 2K trajectory SFT | 10/15 | 11/15 | 10/15 | 11/15 |

## 搜索行为

| 模型 / 输入 | 完成最终答案 | 目标URL被返回 | 平均工具次数 | 重复相同工具调用总数 |
|---|---:|---:|---:|---:|
| Qwen2.5-Omni-7B 基座 / native_av | 6/56 | 0/56 | 0.07 | 0 |
| Qwen2.5-Omni-7B 基座 / muted_video | 3/56 | 1/56 | 0.07 | 0 |
| 500 answer-only / native_av | 56/56 | 0/56 | 0.00 | 0 |
| 500 answer-only / muted_video | 56/56 | 0/56 | 0.00 | 0 |
| 500 trajectory SFT / native_av | 40/56 | 34/56 | 3.82 | 84 |
| 500 trajectory SFT / muted_video | 37/56 | 32/56 | 4.21 | 95 |
| 2K answer-only / native_av | 56/56 | 0/56 | 0.00 | 0 |
| 2K answer-only / muted_video | 56/56 | 0/56 | 0.00 | 0 |
| 2K trajectory SFT / native_av | 41/56 | 38/56 | 3.80 | 57 |
| 2K trajectory SFT / muted_video | 38/56 | 39/56 | 4.04 | 54 |

目标URL匹配是精确来源标识诊断，镜像或等价页面可能使该计数偏低；它不是语义正确率。最终答案数、调用数同样不能替代任务成功率。

## 已核验的执行与训练条件

- 推理修复：统一通过 videos_kwargs 传递 use_audio_in_video、fps、do_sample_frames；生成前检查 audio token 数与编码音频特征长度匹配。
- 全部实际生成轮次均通过音频 token 绑定与 GPU tensor 交付检查；原声为正数，静音为 0。初始视频帧数、形状、采样 FPS 一致，去音轨后的 encoded video packets 与原件一致。
- 四组真实训练 token cache 的每一行都有正数音频 token；媒体位置的 label 均为 -100。训练路径与本次诊断出的旧推理 kwargs 问题不同。
- 基座：Qwen/Qwen2.5-Omni-7B，revision ae9e1690543ffd5c0221dc27f79834d0294cba00。
- 原 OneSearch agent loop / full_eval.py 与 SFT/src/train.py；Full Thinker，语言部分 7,615,616,512 参数可训，AV towers/projectors 1,316,197,376 参数冻结；无 LoRA。
- 两组均为 LR=1e-5、2 epochs、effective batch=4、seed/data_seed=42、bf16/SDPA、4GPU ZeRO-2。500 为 250 更新，2K 为 1,000 更新。比较结论对应当前同 2 epoch 配方的规模收益；更新数增加到 4 倍，token 量约为 4 倍，未做等计算量控制。
- 轨迹数据每 epoch：500 为 4,014,576 context / 110,243 supervised tokens；2K 为 16,317,511 / 429,064。
- 评测：同 40 heldout silver + 16 development publisher candidates；8 turns、512 new tokens/turn、temperature=0；冻结本地语料检索。DeepSeek 在线检索未参与本轮。
- 评分：Gemini-3.8-Flash 对同题十个匿名候选联合评判。中期双候选评分仅用于进度，最终表格统一使用联合评分，不混用口径。

5 个单GPU评测 kjob 均 Complete；按 job start/completion 计算、包含启动开销的 wall GPUh 合计约 1.836。本轮没有新增训练。281 项 live source/config hash 核验一致；分析模块 8 项测试通过。

## 证据与复跑

- 主运行目录：`/home/jovyan/OneSearch-VL/research/runs/av-corrected-audio-evaluation-20261019-v1/`。
- 配对分析：`analysis-v1/paired-results.private.json`；全体翻转题与评分：`analysis-v1/flip-reviews.private.json`。
- 逐题模型输出和执行轨迹：各模型 `gpu-v1/eval/`；评分回执：`judge-all/semantic-review-v1/records/`。
- 训练输入核验：`training-cache-token-audit.json`；Job UID、最终状态及归档哈希保存在本地 `reports/corrected-reevaluation-2026-10-19/final/`。
- 分析代码：`research/av_train/corrected_evaluation_analysis.py`；测试：`validation/test_corrected_evaluation_analysis.py`。

```bash
cd /home/jovyan/OneSearch-VL
pixi run --locked python -m research.av_train.corrected_evaluation_analysis \
  --suite research/runs/av-corrected-audio-evaluation-20261019-v1 \
  --output research/runs/av-corrected-audio-evaluation-20261019-v1/analysis-new-version
```

输出目录必须是新版本。该命令只读现有结果并分析，不重新调用模型或评分API。
