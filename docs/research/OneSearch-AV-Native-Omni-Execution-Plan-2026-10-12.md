# OneSearch-AV 原生 Omni 执行计划

日期：2026-10-12。当前请求是审阅下一步计划；本文列出待执行的代码、数据、实验任务。主模型架构以原生 Omni 方案 v4 为准。本次没有启动新的训练或修改主推理行为。

## 目标

在用户的 OneSearch-VL fork 上，把 Qwen2.5-Omni-7B 接成原生音视频搜索 agent。初始线索和后续检查返回的真实音视频进入同一模型，模型自主产生查询、工具调用和答案。用搜索轨迹 SFT 建立性能基线，再研究声画线索驱动的查询改写与主动片段选择。

工作地点固定 coder ssh 3，主仓库 /home/jovyan/OneSearch-VL，分支 research/av-deepresearch，环境用 Pixi。推理主循环与正式 SFT 复用原仓库；research/ 用于数据与实验入口。

## 本次核对的起点

- 主仓库 HEAD 为 96ee973；已有未提交的 AGENTS.md 架构范围更新和 v4 文档。
- 新原生 Omni 主控制器尚未实现；LocalOmniController 仍继承文字输入调用。
- 原版 select_timespan 会裁出实际 MP4，但来源是预先采样并重新合成的无声 grid，参数是采样索引。原始 AV 裁剪要在此工具机制上扩展。
- 最新 75 窗口批次在预处理阶段全部失败；不继续用这批严格时钟路径作为主线。
- 新素材采集已完成 14 条独立候选，约 442 MB，总时长约 811 秒；它们是可解码候选，还不是训练题或 benchmark gold。80 条采集目标未达成。
- GPU 扩展授权为 4 单卡小时，本轮已消耗 115 秒，扩展额度余约 3 小时 58 分钟；按实际 Job 时长持续记账。

## 一、原生 Omni runner 与原版搜索循环

### 拟修改或新增的代码

| 路径（相对 /home/jovyan/OneSearch-VL） | 具体工作 |
|---|---|
| onesearch_vl/onesearch_infer/runners.py | 在 build_runner 注册 Omni，并沿用 BaseRunner 接口 |
| onesearch_vl/onesearch_infer/omni_runner.py（拟新增） | Qwen2.5-Omni 加载、真实 AV processor、Thinker 生成、推理资源统计 |
| onesearch_vl/onesearch_infer/messages.py | 多轮消息转换保留初始 AV、搜索文本、后续 AV 片段 |
| onesearch_vl/onesearch_infer/config.py | Omni 模型注册、媒体时长/采样/像素参数 |
| onesearch_vl/onesearch_infer/pipeline.py | 原始 AV bootstrap 接入 run_agent_loop，工具媒体持续进入 history |
| onesearch_vl/onesearch_infer/prompts/ | 沿用原工具调用格式，添加 AV 检查工具定义和提示 |

实现原则：一条多轮主循环，不另写第二个模型 agent。关闭 Talker 输出；先用已有已下载权重及已锁定的 Omni 环境。其他 runner 的默认行为保留。

### 验证和交付

先做 CPU 输入检查，确认音频特征、视频张量、媒体 token 实际存在，搜索返回后的 history 仍保留 AV。然后 GPU 跑机械、音乐、鸟鸣/环境三个实际例子；目标是模型自主生成工具调用，读到真实检索结果并回答。用户可看原始片段、查询、工具返回与最终答案。

首批输入以实际 10–30 秒片段为主；已有有用的短片保留。暂不恢复旧的两秒窗口 observer 限制。长音视频后续按检查工具分段。

## 二、在原版视频工具上扩展真实 AV 检查

| 路径 | 具体工作 |
|---|---|
| onesearch_vl/onesearch_infer/tools.py | 注册 inspect_av，返回 clip handle、实际媒体与基本时间信息 |
| onesearch_vl/onesearch_infer/video_io.py 或 av_io.py（拟新增） | 按原视频秒数裁剪，保留音频、正常播放节奏与源片段映射 |
| onesearch_vl/onesearch_infer/pipeline.py | 将工具返回的真实 AV 加入下一轮输入 |

工具参数建议为 `{video, start_s, end_s}`；handle 私下绑定本地素材，不将私有答案或素材文件名送入模型。返回最低限度 metadata：source_start_s、source_end_s、clip_duration_s、has_audio。

保持基本同步，支持常规 VFR 与非零时间偏移；抽帧与重采样使用官方 Omni helper。像素 min/max 显式设置且前后一致。验证真实有声片段、连续片段、可变帧率输入和缺失音频的行为。旧严格时钟 utilities 与失败产物保留作历史，不再作为普通搜索 QA 的输入门槛。

## 三、接入可替换搜索后端，跑 16 个 development cases

当前没有 Serper，先把冻结语料接入原版 text_search / visit_webpage 协议。内部复用 research/av_search 的索引与正文读取，对主模型保持原工具名称和多轮返回方式。稍后可切换外部服务。

拟新增 search_backends.py 或等价小型接口：后端可选、搜索与阅读返回统一、现有默认搜索行为继续可用。冻结语料和后续联网结果分别记录。

运行现有 16 个 development candidates 的原生 Omni Search Prompted。逐题保存：初始 AV、查询、命中来源、阅读正文、检查片段、答案、资源成本。16 题用于定位实际失败和配对开发；正式 benchmark 扩容使用另行核实的答案与独立来源家族。

交付逐题网页及表格，重点呈现模型是否利用声音/画面、是否搜到目标、是否真正阅读、是否答对；审核页面默认提供可播放、有声、有上下文的片段。

## 四、扩大独立来源与自然搜索题

数据目标分三步，均为目标而非已有量：

| 阶段 | 来源与题目目标 | 用途 |
|---|---|---|
| 第一批扩大 | 100–200 条独立录制、500–1,000 个训练题候选 | 从目前少量素材扩大覆盖 |
| 训练 pilot | 2K 条经过筛选的有效搜索轨迹 | 首轮数据规模与策略训练比较 |
| Benchmark v1 | 300–500 个核实答案的测试题 | 正式主表与音频/视频贡献分析 |

正式训练与 benchmark 按原始录制、事件、近重复家族划分。现有 16 个题不因反复开发而成为正式测试集。候选数、成功 rollout 数、保留训练轨迹数单独计数；2K 指可用训练轨迹，重采样失败尝试不充数。

题目优先：声音和画面共同识别动物/机器/音乐或活动；音视频线索找到实体后询问外部事实；语音或文字线索辅助地点/事件识别。仅在自然题目需要时附时间定位。发布账号、许可证、上传日期等字段题保留管线诊断，不作为主体。

采集扩大同时进行问句与参考答案候选整理。真实 Omni 自主生成搜索轨迹，脚本只执行与筛选；训练 silver 可通过可靠来源支持、答案检查和抽样复核形成，benchmark gold 单独核实。避免每条训练数据都先等待人工审核。

## 五、原生 AV 轨迹 SFT

使用现有 SFT/src/llamafactory 中 qwen2_omni template/plugin 和 Thinker loader。新增 Omni dataset registration 与实验配置，并直接使用原 trainer。正式配置先检查依赖兼容，保持独立 Pixi 环境，避免混入 Qwen3-VL 推理依赖。

训练样本包括：原始 AV + 问题、模型查询/检查动作、真实工具观察、最终答案。媒体文件映射与多轮 tensor 保留；只监督 assistant 工具动作和答案，用户输入及工具输出 mask 掉。答案与完整轨迹两个配置使用同一 retained question/trajectory pool、同一初始权重和可比预算。

第一轮冻结音频/视觉 encoder，LoRA 训练 Thinker 的语言决策模块。先完成真实 AV forward/backward、标签检查与 checkpoint 载入，再进入数百条 SFT pilot；收集到 2K 合格轨迹后扩大训练。LoRA 是训练参数选择，原生音视频输入持续保留。

## 六、性能比较与方法迭代

| 系统 | 比较目的 |
|---|---|
| Omni Direct | 音视频直接回答基线 |
| Omni Search Prompted | 搜索工具带来的收益 |
| Omni Search Answer-only SFT | 答案监督收益 |
| Omni Search Trajectory SFT | 轨迹监督收益，核心训练比较 |
| 同一 Omni 的 A-only / V-only / AV | 量化声音、画面及联合贡献 |
| 原 OneSearch-VL-8B / 模块化组合 | 附加系统对照 |

主指标 Answer Accuracy。辅助分析检索命中、查询改写成功率、实际阅读、片段选择、工具次数、音视频秒数、token、计算量。比较使用相同来源划分与推理预算；训练计算与监督 token 实际记录。

方法首先探索“声画线索驱动的查询改写与主动片段选择”：初始声画线索发起查询，搜索失败或实体含混时，模型选择重新检查相关片段后改写查询。用相同 Omni 的固定查询/固定检查 baseline 做消融。先建立 trajectory SFT 基线和实测收益，再根据失败类型决定引入偏好优化或 RL。

## 下一轮直接交付

1. 原版主干上的 Omni runner、AV 工具和可替换搜索后端代码 diff。
2. 三个真实 AV 搜索示例及 16 个 development cases 的完整轨迹页。
3. 扩大来源/问题清单、可用训练轨迹计数和样本展示。
4. 原 trainer 的 Omni SFT 运行配置、实际 checkpoint 与首轮配对结果。

先完成 1–2，再把有效轨迹用于 3–4。来源采集可与实现交错推进。继续使用既有 GPU 授权，优先花在原生多轮验证、轨迹采集、SFT 和评估上。2K 训练与正式 benchmark 的进一步算时根据实际吞吐估算，不承诺在剩余四小时额度内完成所有规模。


## 执行进度入口（2026-10-12 更新）

本文件保留制定时的架构/执行计划。原生 Omni、原 trainer SFT 工程验证与强 teacher 生产代码已推进；最新规模、输入契约与执行证据见 `docs/research/OneSearch-AV-Teacher-SFT-Plan-v1-2026-10-12.md` 和对应 teacher execution 文档。不能把旧计划的“尚未实现”字段当作当前代码状态。
