# OneSearch-AV：数据、训练与配对实验进展

日期：2026-10-12。主代码和计算均在 coder ssh 3，分支 research/av-deepresearch。

## 本轮交付

第一篇继续采用 OneSearch-VL 式音视频搜索问答：用户给问题和可选音视频线索，agent 检查、搜索、读取并直接回答。Search for Generation 留到后续工作。

本轮完成了 6 条真实模型最终回答的 silver 数据、一次 6 样本 / 8-step LoRA 训练，以及两种配对比较。训练、保存、重载和实验链路已实际执行；本轮没有观察到性能增长。

| 项目 | 实际结果 |
|---|---|
| 新训练候选 | 6 题，3 个初始音视频来源：机械、街头音乐、降雨 |
| publisher-field silver | 6 条；工具路径由脚本引导，最终回答由 OneSearch-VL-8B 实际生成 |
| 严格排除诊断来源的子集 | 2 条；另 4 条因检索结果包含诊断来源，只作为训练探索 |
| 最新训练 | 从原始 8B 新初始化，6 条样本，8 次 optimizer update |
| 指定真实工具历史后的回答比较 | base 3/3，旧 4-step LoRA 3/3 请求字段匹配 |
| 模型自主工具调用比较 | base 0/2，新 8-step LoRA 0/2 请求字段匹配，均未完成有效回答 |
| 正式 benchmark gold | 0；16 个 development candidates 仍预留 |
| 验证 | research 242 个、SFT 6 个测试通过 |

## 数据如何产生

对三个已有录制，各提出两个直接问题：机器型号与地点、音乐地点与发布信息、雨景地点与标题日期。实际执行 inspect_av → 标题辅助 search_sources → read_text，最后由冻结的公开 OneSearch-VL-8B 生成紧凑 JSON 回答。

AV observation 来自此前已经真实执行的 Omni 输出，按原素材哈希和精确时间窗回放；本轮没有重新执行音视频模型。脚本选择工具路径，并使用公开标题辅助查找，因此这些是 assisted curriculum，不能当作自主搜索 teacher 的成功轨迹。

6 个回答的请求字段都在真实读到的 publisher 文本和最终答案中出现。筛选没有验证全部陈述、AV 语义或引用正确性；训练数据标为 silver，允许先推进训练探索。原 1 条 seed 保留为历史版本，本次没有重复加入；其中机械问题与旧 seed 接近，不能按新增独立录制计数。

初始录制不同仍不足以保证测试来源完全未被训练看见。旧 adapter 的训练检索结果里已出现 Bates 拖拉机，机械诊断题因此有事实暴露。新 6 条数据的检索结果也包含音乐、降雨诊断来源。这两组比较统一作为内部诊断，不能称为正式 held-out accuracy。

## 训练事实

- 模型：OneSearch-VL/OneSearch-VL-8B，revision 2623603767b9fef7bab05ad0c3cd4d464301e8df。
- 方式：语言层 q_proj/v_proj LoRA，rank 8、alpha 16；视觉参数冻结。
- 数据：6 条多轮 silver，全部 observation tokens 被 mask，assistant action/final 参与监督；本次 8 次更新合计处理 1960 个 supervised tokens，包含重复样本处理，不代表不同 token 的数量。
- 8 步依次轮换 6 个样本，loss 是不同样本的值，不能用第一步与最后一步直接推断统一 loss 下降。
- 梯度有限且非零；adapter 参数发生变化。保存后重载摘要为 23e1e336c4ef7963e863b6372cbe85bef53e6c809b726a1611c002af95193174，与训练结束摘要一致。
- optimizer、scheduler、PyTorch/CUDA RNG 与 global_step 已保存。
- checkpoint：/home/jovyan/OneSearch-VL/research/runs/av-searchqa-20261012-v1/expanded-sft-v1/training/checkpoint。

这是使用仓库 LLaMA-Factory converter/template/labels 的自定义 PyTorch 训练，不是官方完整训练 recipe。新 8-step checkpoint 与之前 4-step checkpoint 分别从原始 base 初始化。

## 比较结果和失败分析

第一组把相同的真实工具结果交给 base 与旧 adapter，三个请求字段题都完成了匹配，两者回答一致。它证明当前模型在正确 publisher 文本已经给出的条件下能完成字段问答，不能证明搜索能力。该条件下题目饱和，没有观察到训练收益。

第二组不给指定工具路径，由 base 和新 adapter 自己决策。每题预算为 3 次工具调用、4 秒请求 AV、每次生成最多 128 tokens；两道题分别是咖啡馆音乐和站台降雨。

base 先用默认 0–2 秒调用 inspect_av，但这两段录制的可用精确窗口分别为 22–24 和 0.504–2.504 秒。收到真实错误与合法窗口后，它把错误 observation 改写为不合法的 action 字符串，导致解析失败。

新 adapter 第一轮输出了扁平 action 字符串格式，并连续写多个尚未执行的动作，最后在 128-token 上限截断。这些输出没有被修补成成功动作。两者都没有执行有效搜索或拿到有效 AV observation，因此这次结果的结论是：当前微量训练尚未解决工具格式和窗口选择。

已进一步比较全部 6 条数据的 24 个 assistant 前缀：canonical runtime 的 token 前缀与实际 LLaMA-Factory 训练前缀逐 token 相等。原始失败输出和完整工具记录保留，不把结果归咎于未经检查的 template 差异。

## 下一阶段：围绕搜索性能推进

1. 先将 16 道 development candidates 改为直接搜索问答，冻结来源家族与答案；训练候选可继续用 silver 扩充，不等待全部 benchmark gold。
2. 输入片段使用独立 clue ID，检索目标保留 corpus source ID；移除二者直接关联及输入来源的 publisher 标题，避免按同一 ID 直接读取答案。构造同类机器、相似音乐和相似雨景作为检索干扰项。
3. 第一批扩大到 64 条有效训练轨迹，覆盖 AV 线索生成查询、查询改写、合法窗口、工具错误恢复、证据足够时结束回答；把 action 格式正确与单步调用作为基础训练验收。
4. 用可执行的 teacher 动作采集替代只给标题的 curriculum；混合容易与困难案例。探索融合声音事件、可见实体和语音文本的查询，以及根据候选差异选择下一段检查区间。
5. 在同一训练量与推理预算下比较 Prompted baseline、Answer-only SFT、Trajectory SFT，再做去音频、去视觉、去查询改写与去恢复样本消融。记录答案正确、目标来源 Recall@k、完成率、有效工具调用、AV 请求秒数与生成 tokens。

论文主线是音视频线索如何提高搜索问答性能。后续性能归因需要更大来源集合与固定 dev；当前两个极小诊断不用于论文效果声明。

## 计算与保存

本轮三个 Job 分别用时 101、71、75 秒；累计 1969 / 2100 Job 秒，剩余 131 秒。时间包含启动、加载和退出开销。Job 均已结束，当前活动项目 Job 为 0，没有自动重试。

原数据、模型、checkpoint、private terms 与完整运行文件均保留在远端；GitHub 仅提交代码、launcher、测试和研究文档。

可视化入口：http://127.0.0.1:8781/OneSearch-AV-SearchQA-Results-2026-10-12/index.html
