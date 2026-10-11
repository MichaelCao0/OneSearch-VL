# OneSearch-AV 匿名音视频搜索问答：下一批数据与训练实验

本批目标是得到真实成功搜索轨迹，并在同一组 development 问题上比较训练前后表现。首篇沿用 OneSearch-VL 的直接搜索问答范式。

## 已有数据与审核入口

- development：16 道题、8 个录制来源，实际输入约 9–12 秒；审核页提供较长上下文与完整视频。
- 扩展训练候选：21 道题、9 个录制来源，实际输入约 5–12 秒。其中机械 8 题、现场音乐 4 题、降雨环境 9 题。
- 21 题的 38 个候选答案字段均有冻结发布者原文支持。原始字节与来源 URL 和 reserved development 无交集；录制/事件近重复家族仍待人工复核。
- 训练审核入口：http://127.0.0.1:8781/OneSearch-AV-Train21-Review-2026-10-12/review/index.html 。审核浏览器播放尚未复验，视频文件及 HTTP 资源会单独验证。
- 审核内容：问题是否自然；是否需要外部检索；声音对辨认是否必要或有帮助；发布者文本是否真正支持答案。更长的审核上下文与模型输入区间分别标注。
- 当前 benchmark gold 为 0；21 题是作者候选，不是已完成的 SFT 轨迹。

## 模型、预算与比较

Controller 起点为发布的 OneSearch-VL-8B，revision `2623603767b9fef7bab05ad0c3cd4d464301e8df`，加载前检查权重 hash。Omni observer 为 Qwen2.5-Omni-7B，revision `ae9e1690543ffd5c0221dc27f79834d0294cba00`。

对每个输入检查头部、中部和尾部各 2 秒。训练 9 个实际输入共 27 个窗口；development 当前 16 个独立容器文件共 48 个窗口，总计 75 次新的 observer 调用。多个训练问题共享同一实际输入的执行观察。窗口间未检查的内容不能由观察结果推定。

采集最多 64 次真实随机生成尝试，记录逐次 seed、temperature=0.7、top_p=0.9。每题最多 6 个模型决定的工具调用。三个预设输入检查作为上下文，不加入 assistant 训练目标；搜索、读取和最终回答由真实模型生成。只保留成功检索、实际读取到支持字段且答案包含所问字段的 silver 轨迹；这项自动筛选仍不能替代语义判断。去重后保留多少条由运行结果决定。

比较三种配置：

| 配置 | 训练目标 | 评测 |
|---|---|---|
| OneSearch-AV-Prompted | 发布 8B 起点，无本批更新 | 16 道 development 问题 |
| Answer-only SFT | 只监督最终答案，屏蔽早先 assistant 动作 | 同一题、同一观察与检索索引 |
| AV trajectory SFT | 同时监督模型生成的工具动作与答案 | 同一题、同一观察与检索索引 |

两种 LoRA 从同一 base 开始，rank=8，alpha=16，q_proj/v_proj，视觉参数冻结；每种实际 32 个 optimizer updates，学习率 1e-4，输入行顺序与 seed 相同。使用真实 LLaMA-Factory converter/template/labels 的有界自定义训练循环。不同监督目标对应不同 supervised token 总数，分别记录；此轮不能称为复现官方完整 SFT recipe。

训练门槛为至少 6 道不同问题、3 个录制来源的 retained silver。未达到时保留真实失败结果并改进 teacher/query，不把失败或作者动作当训练答案。超时已完成的逐题产物可恢复汇总，不重新生成或覆盖历史产物。

## GPU 执行请求

原已授权总额度 2100 Job seconds，已实际使用 2042，剩余 58。本文件只准备下一批，未追加授权、未提交新 Job。拟申请额外 3600 个单 GPU Job seconds，四段上限如下；每段 1 GPU、8 CPU、64 GiB，backoff=0。

| 阶段 | Job 秒数上限 | 入口 | 验收 |
|---|---:|---|---|
| observe | 900 | av_gpu_multiwindow.slurm.sh observe | 实际执行 75 个 AV 输入，保存逐窗输出 |
| teacher | 1200 | av_gpu_multiwindow.slurm.sh teacher | 最多 64 次尝试及逐题保留/拒绝原因 |
| train | 600 | av_gpu_multiwindow.slurm.sh train | 两种各 32 次更新，权重变化及重载校验 |
| evaluate | 900 | av_gpu_multiwindow.slurm.sh evaluate | 三种模型逐题对照与配对诊断 |

提交时设置对应 Kubernetes Job activeDeadlineSeconds，将启动时间计入各段额度。前一段失败或训练样本不够时不自动扩大计算。运行前先完成全部 CPU 预处理、冻结 code/lock/input identity，并匹配实际分配的 GPU PCI 和 CUDA matmul。

CPU preflight（全部窗口准备完之后，在 sft-gpu Pixi 环境执行）:

```bash
pixi run --locked --manifest-path runtime/sft-gpu/pixi.toml python -m research.av_train.multiwindow_preflight --run-dir research/runs/av-multiwindow-20261012-v1 --output-name gpu-inputs-v2
```

GPU 脚本、preflight 和采集/训练/评测 CLI 已准备；CPU 检查不等于它们已在 GPU 完整运行。

## 评测维度与后续扩展

当前先报告每题候选字段命中、目标来源召回、实际读到目标文本、工具调用、观察秒数与失败类型。正式主指标 Answer Accuracy 需要人工确认答案语义，字段字符串匹配不能代替它。开发集用于定位问题，不作为独立测试集。

本批对照先确认轨迹训练是否带来收益。后续扩展 benchmark 的音频线索、动作/时序线索、声画联合识别及外部知识问答，增加来源家族独立的 test；训练扩展到 128–256 条、2K pilot。进一步比较无搜索、无音频、固定窗口与动态检查。冻结语料结果与开放互联网检索结果分别报告。


## 采集、训练和评测的输入对齐修正

三个初始窗口仍通过 runtime 实际检查，并计入工具和观察预算；执行记录保存在 trace 中。Controller 首次生成前收到一个用户上下文，其中同时包含任务与三个公共观察。SFT 导出使用相同函数生成这段上下文，评测也执行同一路径；预设检查动作不进入 assistant 训练目标。新多窗口运行不加入导出中不存在的额外终止提示。

已用真实发布 8B tokenizer、LLaMA-Factory converter/template/processor 完成 CPU fixture 检查：搜索、读取和最终回答三个生成前缀均与 native 训练 token 前缀完全一致；fixture 的 trajectory targets 为 99 token，answer-only 为 27 token，观察保持屏蔽。数字仅描述该测试 fixture，不是训练数据量、真实 teacher 执行或 optimizer update。

对照汇总同时要求任务、媒体绑定、observer 目录、采样设置、工具预算、生成上限、上下文协议和 base checkpoint 一致；adapter 路径按模型配置分别记录。条件不同的输出不能进入配对比较。

原 `gpu-inputs-v1` 保留其旧源码状态；修正后的源码在新 `gpu-inputs-v2` 中重新做 preflight，复用同一批 75 个已准备媒体窗口。GPU 追加额度请求仍待用户答复，这些 CPU 修正不使用新 GPU Job 秒数。
