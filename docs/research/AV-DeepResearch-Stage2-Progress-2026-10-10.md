# Stage 2 工程与资料验收

日期：2026-10-10。工作位置 `coder ssh 3`，`/home/jovyan/OneSearch-VL`，branch `research/av-deepresearch`；沿用 Pixi CPU lock。

本轮已完成可用 CPU 范围内的三案准备、混合来源与精确文本读取、媒体工作量插桩、observer 输入/输出契约和最新技术回放。自然任务均待审核；真实语义 backend、人工 gold、G0/G1 与方法实验尚未完成。

## 三案与采集

canonical 私有包为 `research/runs/next-stage-20261010-v1/cases-v3`：3 案 × 4 来源 × 4 目标项，统一 12-source exploration corpus。包含 6 条已有 AV 原件和 6 个带版本的 Commons API HTML fragments；共 100,321,615 bytes、190.849221 AV 秒，无新增媒体下载。网页 raw bytes/hash、body_text/hash、提取 recipe、capture time/version 均保存，body 与 FTS quote 坐标一致。网页 rights_basis 仍为 unknown。

机械案比较机械循环/阶段与陈述；表演案核对击打/配乐及发布归属；环境案检查共现、画外声与归因边界。环境案的瀑布/corella 是跨情景对照，尚不能证明必要的多源路径；审核不接受时需替换或补源。没有同事件跨录制或独立观察裁决。发布者页不算第二个独立观察。

一次有界 Vimeo 请求得到 HTTP 200、8,061 bytes，但正文为 security challenge；已排除，尝试和早期包/修订收据保留。9 个原采集 exclusion 未覆盖。3 份 A、3 份 B、3 份裁决表均为空白，human gold=0。

## 代码与成本

新增 `web_sources.py` 与 `prepare_cases.py`。dispatcher 接受可信 media/web registry；`read_text` 返回冻结正文 1–4000 字符的 exact quote，拒绝额外字段、foreign source 和越界；网页不能进入 `inspect_av`。

每个 episode 创建并核验私有 corpus snapshot，所有搜索/文本工具执行于该 snapshot；原 corpus 持续替换仍被拒绝。独立审查发现并复现过 check-to-use race，已用 failing ABA regression test 修复；独立复验四类工具均读取初始版本，没有新文本带旧 hash。

`actual_cost.media_work` 记录实际 `run_media` 进程、wall time、失败/partial unknown、FFprobe frame/sample scans、FFmpeg input decoder summaries 和 AVIO reported counters。episode totals 包含初始化工作，preflight 单列。prepared clip counts 与输入解码分别保留。文件大小、AVIO counters 与 physical I/O 不等价；decoder discarded work、hash/copy/physical I/O、CPU/memory/GPU 未完整计量，观测量是 lower bounds，`full_decode_work_accounting_ready=false`。`max_av_seconds` 仍是逻辑请求预算。

## Observer

`semantic_observer.py` 提供私有 source/derived hash、原始区间、shared clock mapping、modalities 和 question 输入契约；未来 backend 必须 pin model revision/config，保存 raw output/hash、structured candidates、错误和实际调用成本。没有 backend 时明确 unavailable；contract fixture 与 model execution 分开，service failure 的 provider work 可为 unknown。观察 ID 已改 UUID；异常 Unicode 回复能记录 failed 并保留已知 provider 请求数。

当前 torch 2.8.0+cpu，CUDA false/0，MPS false，没有配置可用 backend。Qwen2.5-Omni-7B 仅为未部署候选。八类 probe 只是待绑定/人工参考模板：planned=8、bound semantic probes=0、实际 model executions=0、human semantic passes=0。raw_av_observer_ready=false；模块尚未接入 inspect_av 作为语义提供者。

## 验证

- `pixi run validate-research`：130/130 passed。
- `pixi run validate-retrieval`：7/7 passed；compileall、locked Pixi 和 diff check 通过。
- cases-v3 的当前回放 cases-replay-v2：21 calls、6 exact quotes、6 AV clips、12 requested seconds、0 failures/0 providers。全 header/receipt identity、actor/trusted equality、clip/raw hashes 与 cost totals readback 通过。
- 原 10 案当前回放 natural-replay-v4：40 calls、10 AV clips、20 requested seconds、0 failures/0 providers；10 headers 与 40 receipts 全部 pin 相同 manifest identity，源代码 snapshot readback 无漂移。历史 v2/v3 与中间回放保留。

这些是确定性基础设施回放，不是 agent 检索准确率、语义成功率或方法收益。阶段证据和机器相关审核包保存在 Git-ignore 的 run/private 区；公共分支只包含代码、测试和研究文档。

## 下一步与 gate

1. 先审核三案的实际声画对应、来源角色和需要额外检索的证据路径；两名真实审核者独立提交，随后裁决，不能以两个 LLM agent 冒充人类审核。
2. 获得可用 GPU 或指定可访问的原生 AV service 后，在独立环境 pin backend，执行八类真实 probes并做人工语义检查。
3. 补齐/替换 10 案，至少 8 案成为可裁决包、关键引用全部可回放后才通过 G0。音频必要性和多源语义路径仍须验证。
4. G0、observer 与成本口径验收后再扩 40/G1；随后才是 80×4×2=640suffix/G2。主 comparator 仍为 structured consequences vs scalar Q + 同辅助标签、相同 encoder/可见信息/标签预算。没有 teacher/method experiment/training 结果。
