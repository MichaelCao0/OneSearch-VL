# OneSearch-VL 远端复现 Implementation Plan

> 工作位置更新（2026-10-10）：用户最新指定 `coder ssh 3`，仍使用 `/home/jovyan/OneSearch-VL`。本文的 workspace 2 及当时能力状态是历史计划/运行快照；最新实况见 [研究入口](../../research/README.md) 和 Phase 1 验收报告。研究方法与 gate 仍按 v2 执行。


> For agentic workers: 使用 executing-plans 执行下面的检查点；用户已授权继续执行，无需再次选择执行方式。

**Goal:** 在 coder workspace 2 中，以用户 fork 为长期开发入口，用 Pixi 验证公开代码、数据和 8B 权重，并保留官方 baseline。
**Architecture:** 先建立隔离的推理环境和版本锁；保存官方入口真实运行证据。CPU smoke、在线工具、正式 benchmark 和训练复现分层记录，AV 新方法在基线验证后单独添加。
**Tech Stack:** Pixi、Python 3.11、PyTorch CPU、Transformers、ffmpeg；后续 CUDA 环境独立。
**Spec:** 本 chat 2026-10-09 用户指令及 AV-DeepResearch Research Plan v1。

## Global Constraints

- 主开发目录 /home/jovyan/OneSearch-VL；coder ssh 2；origin 为 MichaelCao0/OneSearch-VL。
- baseline SHA 0b7b01ed4872aaccd9af0f6f3d23a81814cf55bd；开发 branch research/av-deepresearch。
- 保留 upstream LICENSE / notices；不将外部代码宣称为原创。
- 不输出密钥；不替换系统环境；不终止无关项目进程。
- 环境、模型和媒体版本均固定；训练/工具 smoke 不代表论文指标复现。

## Task 1：环境与资产

- [x] 连接 workspace 2、检查当前设备/存储、clone fork 并建立开发 branch。
- [x] 添加 pixi.toml 和忽略运行产物，pixi install 生成 pixi.lock。
- [x] 固定模型/数据 HF SHA；下载公开 8B 权重和 1-video/128-frame subset，完整 hash / 官方 verification 通过。
- [x] 模型 class / CPU runtime / 原始 MP4+PNG subset 已验证；真实 8B load/generate 见 minimal-20261009-091814。

## Task 2：官方入口分层验证

- [x] 官方 CLI help / dry-run 已通过；真实 select_frame/select_timespan/crop 已执行；官方 subset preparation/verify 通过。
- [x] native Qwen3VLRunner 在 CPU BF16 真实 load/generate；闭合图像/视频 response、工具轨迹、token 与时间已记录。
- [x] 用户确认本轮仅本地 8B；关闭检索，最小图像/视频 agent 无工具错误通过；不依赖 Gemini。
- [ ] 后续独立阶段：小批和完整307案官方视频 benchmark；本轮用户选定本地最小复现，尚未运行，服务/硬件/协议需先验证。

## Task 3：代码基地审查与交付

- [x] inference / SFT / RL 接口、recipe 差异、audio 丢弃与 source-clock 缺口审查完成；完整数据合成 pipeline 未在公开发布中找到。
- [x] remote AGENTS.md、中文复现报告、AV roadmap 与源码 audit 完成；研究方法尚未实现。
- [x] git diff / Pixi lock / fresh 回归与语法 / 轨迹和 hash 检查通过；没有 commit/push/PR。
