# OneSearch-AV 搜索问答：执行进展

日期：2026-10-11。执行地点：coder ssh 3，/home/jovyan/OneSearch-VL。

## 本轮结果

按用户确认的范围，把第一篇收敛为 OneSearch-VL 式音视频搜索问答。来源数量由问题决定，不强制跨来源比较或报告。新 proposal 为 OneSearch-AV-Research-Proposal-v3-SearchQA-2026-10-11.md。原 proposal、题目和运行记录保留。

三组既有素材改为 12 道直接搜索问答，11 段音视频、11 份发布者文本页面。首批问题主要询问机器身份/年份、表演录制地点、雨景地点等发布者属性；适合连接性 pilot，尚不足以代表正式开放互联网搜索难度。另有 16 道 development candidates 预留，正式 benchmark gold 为 0。

真实运行了 OneSearch-VL-8B 与 Qwen2.5-Omni-7B 自主 controller。没有得到可保留的自主成功轨迹。原 8B 的失败包括格式不匹配、重复工具调用、错误查询语言和未结束回答。Omni 的自主试跑直接报告信息不足，没有实际调用工具。它们都是 pilot 执行结果，未经 gold 评分，不能当正式 accuracy。

随后执行 3 条引导式 curriculum：脚本编写 inspect_av → 标题辅助 search_sources → read_text 的路径，工具实际执行，AV 内容来自精确绑定的已执行 Omni 输出回放，最终答案由真实 Omni 生成。动作监督来自 authoring recipe，最终答案来自模型；这与自主 teacher 轨迹分开记录。

3 条候选中保留 **1 条 publisher-fact silver**：Rochet-Schneider、1919、Citroën B14 三项请求字段同时出现在实际读取的发布者文本和模型答案中。音乐题输出解析失败；雨景题答案未包含必要地点字段 California，均拒绝。这是字段匹配筛选，不是完整答案或 AV 语义的人审认证。来源家族审核仍待完成；该 seed 仅用于工程训练验证。

## 已执行的训练

最新验证使用 guided-silver-v3：学生 system prompt 与 canonical JSON action/final targets 对齐。前一版 prompt 不一致的 smoke 保留为历史诊断。

- 初始化：OneSearch-VL/OneSearch-VL-8B，revision 2623603767b9fef7bab05ad0c3cd4d464301e8df。
- 训练方式：语言层 q_proj/v_proj LoRA，rank 8、alpha 16；视觉层冻结。
- 数据：1 条 assisted silver，多轮真实工具 observation 作为上下文；所有 observation tokens mask，assistant 动作及 final 共 232 supervised tokens。
- 实际 optimizer updates：**4**；各步 loss：[1.1744, 1.1583, 1.1216, 1.0774]，梯度有限且非零。
- adapter 参数确实变化；保存后重载的参数 SHA-256 与训练结束时相同。
- optimizer、scheduler、PyTorch/CUDA RNG、global_step 已保存。
- checkpoint：/home/jovyan/OneSearch-VL/research/runs/av-searchqa-20261011-v1/optimizer-smoke-v2/training/checkpoint。

这是使用真实 LLaMA-Factory converter/template/labels 的自定义 PyTorch LoRA smoke，不是原论文完整训练 recipe。单样本训练 loss 下降说明反向传播和保存链路可执行；尚无 held-out 性能提升证据。

## 已解决的执行问题

1. 生成到第一个完整动作就返回工具，防止模型继续编造尚未执行的 observation。
2. native tool-call 与内部 canonical action/final 之间显式转换；导出学生 prompt 与目标格式对齐。
3. 每题进度写入独立 create-only 文件，避免第二题覆盖已有回执。
4. 搜索索引 CPU/GPU SQLite 版本不同。分别构建索引；公开 documents hash 和 source IDs 一致，检索 recipe/version 与 corpus hash 分别记录。
5. 非法文本区间返回合法范围；不存在的 AV 回放区间返回该源可用窗口，允许模型恢复。
6. 修复 source collection 把工具签名里的 source_id 类型字符串误当真实来源引用的问题。历史原始 trace 未改写；派生 trace 只重算引用集合，记录原 trace hash。
7. 新增独立 runtime/sft-gpu Pixi 环境，与 root CPU、AV observer 分离。

最终 research regression **235/235**，SFT native labels 与 optimizer tests **6/6**。最后一次 git diff --check 通过。

## 计算与接下来的工作

累计 Job 时间 **1722 / 2100 秒**，剩余 378 秒；当前活动 Job：[]。Job 时间包括准备、加载和退出开销，不等于训练 kernel 时间。两个 smoke 从同一原始 checkpoint 分别开始，不能合称一次连续 8-step 训练。

下一阶段应先扩大 cold-start 数据并改进 teacher 的工具适配，目标 64–256 条合格 seed；随后做更有意义的 LoRA pilot。首要能力是根据实际线索改写查询、选择合法音视频窗口、从工具错误恢复、在证据足够时回答。保留 Answer-only SFT、Prompted 和 trajectory SFT 对照。

正式 benchmark 扩展到真实外部信息需求、难度更高的识别与属性问答，同时完成录制家族划分和答案审核。当前 Serper/强工具 teacher 服务未配置，冻结语料 pilot 与开放 web 结果保持区分。现阶段不存在正式 benchmark 分数或 SFT 性能比较。

## 审核入口

- 题目与采样帧：http://127.0.0.1:8781/OneSearch-AV-SearchQA-Pilot-2026-10-11/index.html
- 真实模型轨迹：http://127.0.0.1:8781/OneSearch-AV-SearchQA-Pilot-2026-10-11/trajectories.html

页面保存原模型回答及各工具实际结果，成功、失败、silver 与人工 gold 分开标记。原媒体和 checkpoint 保留在远端，未公开发布。
