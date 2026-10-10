# AV observer 输出协议和时钟适配实现计划

> **For agentic workers:** Use superpowers:subagent-driven-development to implement and independently review the bounded tasks below.

**Goal:** 对上一轮固定六个 AV 输入建立严格、可追溯的结构化输出，并验证音轨起点适配；原始媒体、input hash 和 source-local mapping 均不改变。
**Architecture:** 先仅修正 prompt，并用原始 strict clock gate 跑格式诊断；随后以显式选择的 timestamp-aware audio-array adapter 跑独立诊断。两轮结果分别报告。保持 raw 文本和严格 parser，所有转换必须具有可审计 receipt。
**Tech Stack:** coder ssh 3；隔离 Pixi CUDA runtime；pinned Qwen2.5-Omni-7B；PyAV/FFmpeg；单 NVIDIA GPU Kjob。
**Spec:** 已由用户确认的 docs/research/AV-Native-Observer-Kjob-Execution-2026-10-10.md 下一阶段门限，以及本轮“继续修改然后执行”。

## Global Constraints

- 仓库 /home/jovyan/OneSearch-VL，research/av-deepresearch；复用原六个固定 input，单次384 tokens；不扩大语义 benchmark。
- 每个 Job 单卡、8 CPU request/limit、64 GiB request/limit；backoffLimit=0。
- 累计保守 Job 时间仍受2100秒上限约束，前轮已用337秒；新两轮各最多700秒，合计保留控制器退出余量。
- 新运行目录 research/runs/av-observer-repair-20261010-v1；不覆盖旧日志/输出，不公开原始媒体、权重或私有参考。
- 明确区分真实generation、未处理原文协议通过率、任何模型输入变换、自动schema/hash检查、语义人审和G0。

## Task 1: 输出协议修正

- [x] agent只修改 QwenOmniBackend._messages 和独立prompt测试；修正 start/end边界措辞，提供完整literal JSON模板，最多两个简洁事件，明确非空uncertainty及全部顶层字段。
- [x] 先验证旧prompt不能满足新协议合同；新prompt通过测试及真实pinned template/processor调用。任何示例必须空events，避免注入自然场景事件。
- [x] strict clock policy下对六个固定输入跑 protocol-v1；第一片段仍计clock rejection；保存raw、strict parse率和token cap。

## Task 2: Timestamp-aware输入适配

- [x] 新模块负责从实际audio frame PTS构建16kHz公共clip轴数组，并记录实际coverage与无媒体区间；原素材和binding不变。
- [x] 输入缺失/非单调/非有限PTS或不能证明的video时钟仍fail closed。不得把合成填零区域称为录得的静音。
- [x] synthetic delayed-pulse/时间断点probe证明PTS位置、coverage和原始文件hash；真实六输入CPU probe确认01/02/06官方processor消费两个modality tensor，03/04/05显式拒绝。
- [x] root以显式clock-policy参数整合，默认仍保持旧strict policy。记录adapter module/receipt hashes，生成结果不能在无audio支持的区域接受audio/joint事件。
- [x] 六个输入跑 clock-v1，独立记录与protocol-v1的输入/策略差异；不作为一般VFR支持或精确语义时间能力证明。

## Task 3: 独立审计与交付

- [x] CPU相关测试、compile/shell syntax和diff检查；冻结源码、锁、模型manifest和固定输入hash。
- [x] 逐调用raw/hash/schema、tokens、clock receipt与Job终态独立审计，统一预算口径。
- 中文执行报告已经形成；沿既有授权commit/push研究分支，最终SHA和clean读回由独立Push-Receipt记录。
- [x] 人工语义review、八类probes、G0、inspect_av接入及训练仍独立验收，不因格式修正自动通过。

## 依据 protocol-v1 结果的受限调整

protocol-v1 五次 generation 中四次在 JSON 后追加句点；pinned tokenizer 将 `}.` 编为单一 token 7810，普通 stopping 无法在同一个 token 内截断。保持严格 parser 与原文，只在实际问题后增加明确输出终止指令；新增 protocol-v2 在相同 strict_origin 上独立验证，再以同一 prompt 执行 clock-v1。该调整不保证模型遵从。

protocol-v1 Job 时间82秒；与前轮337秒合计419秒，剩余1681秒。后续两 Job 各700秒deadline/620秒脚本timeout，仍在累计2100秒上限内，零自动重试。新的clock路径只允许已证明可适配的01、02、06进入generation，03/04/05继续显式计为失败，固定分母6不改变。调整的预计剩余generation上限8次、tokens384/次。所有预算版本保留。

## protocol-v2 后的生成端修正

明确prompt后严格结构仅2/6通过，3/5真实generation仍有尾随句点。新增显式 `output_policy=json_object_v1`：通过真实tokenizer定位并禁用单token `}.`，并在生成的完整文本首次成为合法顶层JSON对象时停止。不修改生成后的raw，不做去句点/抽取，不保证schema或语义。普通 `unconstrained` 仍为默认策略。先protocol-v3验证，再同策略clock-v1；两Job仍各700秒deadline，累计预算已用498秒、剩余1602秒。严格格式统计与窄生成约束率单独报告。

## 独立审查 P1：视频模型时间网格修正

clock-v1三次真实generation通过结构和audio coverage检查，但官方helper端点linspace取帧仍造成约0.3秒temporal patch标记偏差。原始CFR证明不能替代模型时间网格证明。新增显式 `av_pts_grid_v1`，以 j / sample_fps 为目标取最近真实decoded视频帧，记录实际PTS、最大量化偏差、processor scalar网格与数组hash；只支持已证明CFR和偶数frame count，不允许一般VFR、额外复制末帧或无证据时间重写。测试同步pulse、容器tick和非整数预算拒绝。

新avgrid-v1最多一个单卡Job、700秒deadline/620秒timeout、固定6输入和3次预期generation、384tokens/次；提交前累计652秒，剩余1448秒。所有旧源码、结果和限制保留。帧量化误差及processor temporal patch pooling仍不能作为语义时间准确度证明。

## 执行结束读回

原计划任务、两次prompt诊断、生成端窄约束及独立视频grid修正均完成并有分阶段证据。最终71项CPU validation通过，五个新Job全部终态，累计727/2100秒。执行结果见 `AV-Observer-Repair-Execution-2026-10-10.md`，公共数值见 `AV-Observer-Repair-Evidence-2026-10-10.json`。人工语义/G0/inspect_av/训练仍pending，不因本计划checkbox改变状态。
