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

任务定义沿用 `docs/research/OneSearch-AV-Research-Proposal-v3-SearchQA-2026-10-11.md`；2026-10-12 用户修正后的主模型架构以 `docs/research/OneSearch-AV-Native-Omni-Architecture-v4-2026-10-12.md` 为准：Omni 原生接收音视频并执行搜索，替代固定 observer 文字转交 OneSearch 8B 的主线；复用原版搜索框架，8B 与模块化路线保留对照。此前方案保留历史依据。Benchmark gold 严格人审，但不阻止明确标为探索的真实 rollout 与 silver 数据候选采集。
原版视觉路径不原生保留音频，grid clock 不能直接当 source clock。已有冻结语料、媒体预处理和历史 Omni observer。当前主线已在原版推理主干接入原生 Omni runner、真实多轮 AV 输入和原 LLaMA-Factory Omni 轨迹 SFT；LocalOmniController 文字路线仅作历史对照；时间处理保留基本同步并支持普通裁剪/VFR，不把旧严格时钟门槛或证据账本作为搜索 QA 主线。源码准备、CPU 验证、GPU 实际运行与性能增长分别报告。

不得把 gold graph、答案、来源路径或未来 observation 提供给 actor。预测后果不能直接成为引用。转载不等于独立证据。

## 2026-10-12 最新执行约束与 teacher 主线

- 用户最新追加授权（2026-10-12）：有实质实现进展后可以提交 kjob。先完成 teacher AV 输入确认、真实成功数据与训练导出，再按已验证的项目 launcher/preflight/budget 提交有明确实验目标的 GPU Job；当前不能把未验证数据直接当正式 SFT 数据。
- Student 是 Qwen2.5-Omni-7B 原生 AV；15K–20K 是保留成功 teacher 轨迹的训练目标，不代表已有数据。
- Teacher API 凭据仅保存在远端 `/home/jovyan/.config/onesearch-av/teacher.env`（0600），配置 JSON 只写环境变量名称，日志与 Git 禁止凭据和媒体 base64。
- 原生 Gemini 与 OpenAI 兼容 gateway 分别使用对应 runner；HTTP 200、模型列表或生成 receipt 不证明实际音视频理解。正式生成先验证具体媒体格式能传递可辨认的画面和声音。
- 首批 20题 × 3候选是队列配置；完整时长的 teacher 传输媒体用 `research/av_train/teacher_media.py` 生成版本化新 batch，保留父媒体 lineage，不把同一录制的转码当独立来源。完成 rollout、语义成功、训练保留和模型性能分别计数。

## 验证与交付

- 检索开关回归：`pixi run python -m unittest discover -s validation -p 'test_offline_retrieval.py' -v`。
- 官方 CLI：`pixi run infer --modality video --dry-run ...`；真实 load/generate 与工具返回另存日志。
- ordinary evaluator 仅检查最终答案，不能替代 evidence correctness / GCR。
- 发布 launcher 的 RLOO 和 SFT LR 与论文有差异，训练成功不自动等于原论文 recipe 复现。
- 2026-10-10 用户明确授权将本轮代码、环境 lock、测试和研究文档 commit/push 到自己的 fork 的 `research/av-deepresearch` 分支。后续公共动作按当次用户授权执行；PR、merge、原始数据/权重和私有 gold 发布仍须明确授权。
