# 在 OneSearch-VL fork 上继续开发 AV-DeepResearch

范围：第一篇做多源音视频检索、取证和研究报告；Search for Generation 后续独立。
开发位置：coder workspace 2，`/home/jovyan/OneSearch-VL`。

当前权威方案：`AV-DeepResearch-Research-Plan-v2-2026-10-09.md`；第一阶段执行清单：`../superpowers/plans/2026-10-09-av-deepresearch-phase1.md`。旧 audit 中“需全量官方 benchmark 后才研究”的建议已被后续用户确定的本地 8B / AV 研究路线替代。

## 基地判断

可以复用这个 codebase，但它目前是视觉 agent。先保留 published baseline 与官方可运行入口；新增能力在研究模块中实现，实验明确记录 adapter。

可复用：controller/tool-call 协议、图像与视频局部工具、轨迹记录、Qwen3VLRunner、SFT 数据格式、rLLM/verl/Megatron 训练骨架。
不直接复用为 AV 证据真值：VGEG/EVGR、最终文本 evaluator、未保留原始时间映射的 frame-grid 和页面 summary。

## 开发顺序

1. **已完成：本地 8B 最小复现。** Pixi lock、pinned 8B 权重、真实视频/PNG subset、local tools 和关闭检索的 native agent 已通过 bounded smoke；不代表全量 benchmark 或 audio 能力。
2. **原始 AV 输入与工具。** 增加保持音轨的原始媒体登记、source-local 时间映射、原生 AV observer；Qwen3-VL controller 接收其可审核观察，而不伪装成原生 audio model。新 tools 必须真正读到指定音频/视频区间。
3. **证据账本。** source/version/hash、local interval、observed proposition、同事件/同实体/同录制等关系和未知状态；原始证据与预测结果分开，转载谱系可追溯。
4. **同前缀分支 runner。** 独立 snapshot/continuation API；保存相同观察前缀与候选，冻结 continuation identity，配对 seed 与总成本。当前 resume 只能跳过整案，不能替代该能力。
5. **方法与 scorer。** 逐项后果、同标签 Q、强规则/VoI；GCR 与重大无依据断言 scorer 独立于原 final-answer evaluator。通过 10/40 案门槛后才训练 selector。
6. **训练接入。** 推理侧验证后再对齐 RL 的第二套 messages/tools；先 SFT 小量可审核数据与真实 optimizer smoke，GPU preflight 通过后才扩大训练。

## 必须保持的实验边界

- 当前 video benchmark 是 silent video，不能验证 audio 优势；以后新采自然 AV evidence packages。
- 同一录制的多URL不算独立来源；grid index不直接等于原始AV秒数。
- summary与检索snippet不是引用原文；保存原始内容和定位信息。
- 默认官方prompt字节保持一致，检索关闭仅显式flag启用。
- 预训练模型、观察器、writer、候选池与监督标签配平，不能把更强服务换入造成的收益说成方法收益。
- 发布SFT LR=1e-5、RL launcher=RLOO、train-only入口与论文recipe存在差异，完整训练复现需另记协议。

详见同目录 Research Plan 和固定SHA codebase audits。
