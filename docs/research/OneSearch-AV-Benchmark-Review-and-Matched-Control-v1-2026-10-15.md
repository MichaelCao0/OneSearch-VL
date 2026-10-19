# OneSearch-AV：新增 benchmark 审核与同题训练对照

日期：2026-10-15。执行位置：`coder ssh 3:/home/jovyan/OneSearch-VL`。

## 本轮交付

新增 benchmark 的 24 个独立录制案例已生成四种审核媒体：原声 AV、实际移除音轨的视频、audio-only 和来源上下文。它们来自已预留的 benchmark cohort-00，未按 student 成绩选择。题目、答案与声音角色保持人工待审；不计入训练问题。

审核入口：[24 题 benchmark review v3](http://127.0.0.1:8781/OneSearch-AV-Benchmark-Review-v3-2026-10-15/benchmark-modality-review/index.html)。优先检查第 1 题 MV 导演、第 4 题低音管乐团作品作曲家、第 19 题木管独奏作曲家；第 6 题缆车广播和第 10 题音乐歌词用于对照语音检索。建议先只看片段尝试辨认并搜索，再展开发布者材料。

## 质量与可播放性

- 新原生 AV 自动内容审核 24 次：23 completed / 1 failed；不重放失败请求。23 个完成例的规则化建议均为 keep，仍需人工核实事实和实际可辨认性。
- 模型建议的角色为 visual 16、speech/text 3、audio 1、joint 3。这是自动意见，不能据此推断人类可解率或证明音频收益。
- 这批来自既有素材预留，不代表下一阶段音乐 30% / 赛事 30% / 现场活动 20% / 其他 20% 的最终分布。后续新采集继续按已批准优先级运行。
- 120 个媒体/静帧文件哈希核对；96 次媒体 Range 请求全部 HTTP 206；静音视频包和时间戳保持一致，audio-only 不含画面。
- IAB 实际播放第 1 题四种模式：时间均推进，readyState=4，无媒体错误；已看到原声视频画面。只听声音时视频和静帧隐藏；结束时恢复原声并暂停。人审记录保持 0/24、答案核实 0。

远端媒体包：`research/runs/av-benchmark-modality-review-20261015-v3/benchmark-modality-review/`。
自动核查：`research/runs/av-benchmark-content-audit-24-20261015-v2/`。

## 新训练数据进展

截至北京时间 17:31，`verified_pool()` 给出 **979 retained / 938 CPU-qualified / 41 待 CPU 审核**。训练与 benchmark worker 的 host、PID、start ticks、boot ID 均匹配；后台继续朝 2K 合格训练问题推进。新增 benchmark 仍为 43 个 dev 候选，正式 human gold 为 0。

## 500 题 matched answer-only 对照

新 run：`research/runs/av-learning-500-controls-20261015-v1/`。完全复用 clean500 冻结成员与初始 AV，从相同 pinned Qwen2.5-Omni-7B 初始化；answer-only 中的 `<response>` 内容与实际 trajectory final 逐题相同，去掉工具历史和 final 前的 think。全体 1,000 个 paired rows 的哈希、原有媒体/标签审核及源码身份重新核对，不重复解码同一冻结素材。

训练配置复用已完成 trajectory LoRA：rank8/alpha16、LR1e-4、batch4、2epochs、250updates、seed42。配置与原版只差 dataset/output_dir，原 trainer dataclass 字段与数据注册检查通过。现阶段是数据与 recipe 冻结；GPU 训练尚未执行。

| 每 epoch | Trajectory | Answer-only |
|---|---:|---:|
| 问题数 | 500 | 500 |
| Context tokens | 4,053,859 | 2,442,485 |
| Supervised tokens | 112,947 | 29,999 |
| 最大序列 | 20,014 | 5,834 |

主比较固定问题、epochs、updates 和推理工具/预算，同时报告实际 tokens 与 GPU 时间；等计算量对照另列。推理阶段对 base、answer-only、trajectory 使用同一冻结 prompt 与工具，no-search 另作分析。

下一步是完成 strong prompted base 的 dev-only 提示选择，执行该 answer-only 训练并立即接同题评测；新增生产数据达到 2K 后冻结 paired 数据并开展 full Thinker smoke/训练。本轮没有新增 GPU 任务。

## 记录

本地媒体迁移/Range/浏览器证明见审核包根目录的 transfer-receipt.json、http-range-validation.json 和 browser-playback-receipt.json。远端现场快照、内容审核与配对控制核查摘要保存在 `docs/research/records/2026-10-15-benchmark-review-v3/`。旧计划和结果均保留。
