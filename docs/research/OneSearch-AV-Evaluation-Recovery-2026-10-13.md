# OneSearch-AV 首轮评测修复与重跑

更新时间：2026-10-13 17:35（北京时间）。

## 已验证的根因与修复

原评测已经执行完三组各 56 题。base 无执行错误；trajectory 45/56 题执行错误；answer-only 无执行错误。轨迹模型的异常样本在真实 GPU 重放时生成未定义 token ID 151856，随后在原生 OneSearch runner 的 action stopping criterion 内触发 tokenizer.decode 的 NoneType 异常。

Qwen2.5-Omni-7B 的 Thinker 输出层有 152064 个槽位，但 tokenizer 只注册 151665 个 ID。修复在 generation 时统一 suppress 399 个没有 tokenizer 定义的输出槽位，保留模型原有 suppress 配置；所有已注册普通 token 和特殊 token 都保留。原始模型与两组 SFT 使用完全相同的处理。实际同题 GPU 重放从第一轮解码错误恢复到 8 轮执行无解码错误；这不意味着该题回答正确。

实现位于 coder ssh 3 的 /home/jovyan/OneSearch-VL/onesearch_vl/onesearch_infer/omni_runner.py。新增 regression 先复现失败，再验证最高 logit 为未定义槽位时仍生成可解码的正确动作。15 项 runner 检查在固定 transformers 4.57.6 环境全部通过。原评测完成后才修改源码；旧脚本、原 runner 内容和失败轨迹保留。

## 新一轮真实评测

- Job：onesearch-av-eval-vocab-20261013-v4，2026-10-13 17:28:46 启动。
- 复用原 trajectory 和 answer-only adapter，不重新训练。
- 先对每个模型运行相同的两题 smoke，再执行每组 56 题：40 source-family heldout silver + 16 dev candidates。
- 原始 agent loop、8 轮、每轮最多 512 tokens、temperature=0、同一 prompt 和冻结 corpus。
- 将任何执行错误设为自动评分的阻断条件，避免基础设施失败混入语义能力统计。
- 完成后 4 路匿名语义 judge，一题一次请求同时评分三个随机匿名答案，最多 56 次请求；不自动重发失败请求。
- 报告语义正确率、字段正确率、工具调用、目标来源检索/读取、配对差值及按录制家族 bootstrap 区间；训练数据与参考仍为自动 silver。
- 历史已用 5774 GPU 秒；新 job 最长 7200 秒，总计最坏 12974 秒（约 3.60 GPU 小时），低于 6 GPU 小时上限。

截至本记录，完整三组语义正确率尚未生成。自动完成监控已启动；仅在完整无执行错误的评测后调用 judge。

## 证据

远端 /home/jovyan/OneSearch-VL 下：

- research/runs/av-learning-pilot-20261013-v3/decode-diagnostic-v1/observed-decode-errors.json
- research/runs/av-learning-pilot-20261013-v3/decode-diagnostic-v1/report.json
- research/runs/av-learning-pilot-20261013-v3/vocab-test-red.log
- research/runs/av-learning-pilot-20261013-v3/vocab-test-pinned.log
- research/runs/av-learning-pilot-20261013-v4/gpu-v1/cluster/budget-preflight.json
- research/runs/av-learning-pilot-20261013-v4/gpu-v1/cluster/source-freeze.json
- research/runs/av-learning-pilot-20261013-v4/completion-state.json
