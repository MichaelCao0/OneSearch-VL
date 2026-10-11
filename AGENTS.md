# OneSearch-VL / AV-DeepResearch 开发约定

## 工作位置与环境

用户明确要求：后续开发、环境和实验都在用户当前指定的 Coder workspace；2026-10-10 最新指定为 `coder ssh 3`（替代此前的 `coder ssh 2`）；主项目 `/home/jovyan/OneSearch-VL`，环境由 Pixi 管理。不要在 macOS 或其他 workspace 开发第二份主代码。

- origin：`https://github.com/MichaelCao0/OneSearch-VL.git`。
- 官方基线 Git SHA：`0b7b01ed4872aaccd9af0f6f3d23a81814cf55bd`。
- 当前开发 branch：`research/av-deepresearch`。保留官方 baseline 与署名、LICENSE / THIRD_PARTY_NOTICES。
- 用 `pixi install --locked` 和 `pixi run`；不要将推理、SFT 与 RL 依赖混装进系统 Python。
- 当前 default 是 CPU 推理环境（torch 2.8.0+cpu、transformers 5.2.0）；将来恢复 GPU 后另建 CUDA 环境，不将 CPU 导入测试叫 GPU 验证。

## 当前用户授权与实验协议

- 用户没有 Serper，已授权本轮关闭联网检索，先最小复现：推理子进程设 `ONESEARCH_DISABLE_RETRIEVAL=1`。
- unset/0 保持官方 prompt 和默认工具；只给本轮 inference 传 flag，不将它全局写进 RL launcher。
- 关闭检索指去掉本实验的 text/image search；不要停止其他项目后台服务。
- Gemini / judge 是独立模型服务，不在此开关内；配置从用户指定的远端文件读取，禁止输出或写入仓库密钥。
- 不将无检索／短 token／降采样 smoke 报告为论文完整 benchmark 复现。
- 模型和数据按 pinned revision 下载；权重必须匹配 HF LFS SHA-256 后才能加载。
- 每次 run 使用新输出目录，记录源码 diff、模型/数据 revision、环境 lock hash、开关、budget、实际输入与错误。

## 开发范围

2026-10-11 用户明确收敛范围：第一篇沿用 OneSearch-VL 的搜索问答范式，扩展音视频线索与检查能力；来源数量由问题决定，不强制跨来源比较或研究报告。Search for Generation 留作下一篇。

当前任务范围以 `docs/research/OneSearch-AV-Research-Proposal-v3-SearchQA-2026-10-11.md` 为准；此前 proposal 和 codebase audit 保留历史依据。Benchmark gold 严格人审，但不阻止明确标为探索的真实 rollout 与 silver 数据候选采集。
原始代码会丢弃音频，grid clock 不能直接当 source clock；原始时间映射、冻结 metadata corpus、严格任务隔离与 media-only receipts 已实现；语义 AV observer、可修订证据账本、同前缀分支 runner 和 GCR scorer 仍待实现。

不得把 gold graph、答案、来源路径或未来 observation 提供给 actor。预测后果不能直接成为引用。转载不等于独立证据。

## 验证与交付

- 检索开关回归：`pixi run python -m unittest discover -s validation -p 'test_offline_retrieval.py' -v`。
- 官方 CLI：`pixi run infer --modality video --dry-run ...`；真实 load/generate 与工具返回另存日志。
- ordinary evaluator 仅检查最终答案，不能替代 evidence correctness / GCR。
- 发布 launcher 的 RLOO 和 SFT LR 与论文有差异，训练成功不自动等于原论文 recipe 复现。
- 2026-10-10 用户明确授权将本轮代码、环境 lock、测试和研究文档 commit/push 到自己的 fork 的 `research/av-deepresearch` 分支。后续公共动作按当次用户授权执行；PR、merge、原始数据/权重和私有 gold 发布仍须明确授权。
