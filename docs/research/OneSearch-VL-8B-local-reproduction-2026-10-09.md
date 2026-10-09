# OneSearch-VL 8B 本地最小复现与 codebase 开源边界

日期：2026-10-09。用户指定 fork 为 MichaelCao0/OneSearch-VL，后续主开发环境为 coder ssh 2 + Pixi；本轮先关闭在线检索，完成发布 8B 的本地最小复现，不依赖 Gemini。

## 结论

公开 8B 权重已在 workspace 2 的 CPU BF16 上真实加载、生成，并通过 native single-image/video agent 的完整回答检查。视频任务真实执行 select_frame 后将官方原分辨率 PNG 返回模型；最终文本与独立查看的原始帧一致。最后一轮没有工具错误。

这是发布 checkpoint 和本地工具链的 integration smoke，使用自定义问题与短预算。没有运行完整官方 benchmark、在线搜索、judge、SFT 或 RL；不能据此声称论文指标已复现、模型具有原生音频能力，或新研究方法已有收益。

公开 codebase 适合作为后续开发基础。完整 VGEG 数据合成管线在本次检查的公开主分支和 HF 发布中未找到；公开的是成品数据、准备/注册脚本、训练和推理/评测代码。

## 1. 版本与运行位置

| 项目 | 固定值 |
|---|---|
| 主机 | coder-workspace-2，用户 jovyan |
| 主项目 | /home/jovyan/OneSearch-VL |
| origin | https://github.com/MichaelCao0/OneSearch-VL.git |
| upstream | https://github.com/appletea233/OneSearch-VL.git |
| branch | research/av-deepresearch |
| Git baseline | 0b7b01ed4872aaccd9af0f6f3d23a81814cf55bd |
| baseline tag | baseline/onesearch-vl-20261009 |
| 模型 | OneSearch-VL/OneSearch-VL-8B |
| 模型 revision | 2623603767b9fef7bab05ad0c3cd4d464301e8df |
| 视频数据 | OneSearch-VL/OneSearch-Video-Bench |
| 视频数据 revision | 8694a570d777066e053fc258623a429cc13a2294 |
| 最后通过的输出目录 | /home/jovyan/OneSearch-VL/repro/results/minimal-20261009-091814 |

所有代码改动目前保留在远端工作树；没有 commit、push 或 PR。macOS 上只保留分析资料与用户交付文件，没有建立另一份主开发环境。

## 2. Pixi 与设备

Pixi 0.63.2；Linux x86_64；Python 3.11；torch 2.8.0+cpu；torchvision 0.23.0+cpu；transformers 5.2.0；qwen-vl-utils 0.0.14；accelerate 1.15.0；huggingface-hub 1.33.0。完整依赖由 pixi.lock 固定。

pixi.lock SHA-256：61086af140150b93060cd9677823a781877c32e5a355a295c32b2d73e4b4fac6。

本次 live 检查没有 /dev/nvidia*，torch.cuda.is_available() 为 false，不能访问 GPU。推理实际采用 CPU、torch.bfloat16、SDPA、16 个 torch threads；checkpoint 配置的 text_config.use_cache=false 未改动。CPU smoke 通过不代表 CUDA/SFT/RL 环境已经验证。

在远端项目目录可执行：

```bash
cd /home/jovyan/OneSearch-VL
pixi install --locked
pixi run validate-retrieval
pixi run minimal
```

minimal 会设置 ONESEARCH_DISABLE_RETRIEVAL=1、HF_HUB_OFFLINE=1、TRANSFORMERS_OFFLINE=1；启动自己拥有的 loopback 媒体服务，使用随机空闲端口，运行结束后关闭。模型与数据须事先下载并具有匹配的完整性 receipt。

## 3. 发布资产与完整性

8B 的 model.safetensors 为 17,534,340,584 bytes。下载后完整 SHA-256 与 HF LFS 声明一致，并完成写入持久盘后的 readback hash：

f73186b544f7ff267d1150cc5ef3340edbf196d21b46794e4a623734304c8c5b。

权重在 /home/jovyan/OneSearch-VL/repro/assets/model-8b；download-receipt.json 保存 revision、字节数、预期及实际 hash。

视频子集保留原 QA ID 0 对应的 media/0039.mp4 和 128 个官方原分辨率 PNG，共 129 个媒体文件。通过 ZIP range 下载及 CRC 校验取得，保存逐文件 hash；未用 MP4 解码帧替换官方 PNG。官方发布 prepare.py --verify-only fresh 返回：

```json
{"samples":1,"videos":1,"frames":128,"verified":true}
```

资产位于 /home/jovyan/OneSearch-VL/repro/assets/video-subset。子集是用于媒体和工具链验证的 1 个案例，不是全部 307 QA。最后 inference 使用自定义 closing-screen 题，而没有评测该原 QA 的答案。

## 4. 最后通过的实际模型与工具运行

native build_runner('8b') 加载下载的发布权重，而不是 mock 或另一个基础模型。原始 runner 日志标签仍叫 Qwen3-VL-8B-Instruct；实际 checkpoint 路径和 hash 指向上述 OneSearch 发布模型。

| 验证 | 实际结果 | 轮数 / 生成 token | 耗时 |
|---|---|---|---|
| 合成图像 | `<response>red square</response>` | 1 轮 / 60 tokens | 35.06 s |
| 发布视频 | `<response>THANKS FOR WATCHING</response>` | 2 轮 / 91+68 tokens | 110.99 s |
| 模型 load | CPU BF16，真实参数加载 | — | 4.62 s |

视频第一轮模型按任务要求输出 select_frame(video_1, frame_index=127)；dispatcher 成功返回 img_1，下一轮输入包含真实 frame_127.png，随后生成最终回答。独立查看该 PNG 可见黑底绿色文字 THANKS / FOR / WATCHING。该测试检查模型—工具—新媒体输入—最终回答的闭环，不测自主规划能力。 本轮工具 JSON 有两个同值 name 键，Python parser 接受并执行成功；这是保留在原始轨迹中的格式瑕疵，不能称严格 JSON 输出质量已全面验证。

每轮最大 256 生成 token，最多 3 轮；初始视频读取最多 4 帧，max_pixels=50176。工具 grid 仍保留 128 个官方帧槽位，因此 select_frame(127) 可以取得最末官方帧。这些 budget 与问题协议均偏离完整官方评测。

另由脚本手工执行并检查了三种真实本地操作：select_frame、select_timespan（ffmpeg 生成 clip）、crop。三者都产生新媒体 handles；它们是独立工具验证，不能称模型自主选择了全部三种工具。模型轨迹中实际调用的是 select_frame。

完成检查要求最终 response 标签闭合、无 inference error、无 Tool execution error，并检查实际 response_text 中的工具调用。不能只依赖 pipeline 的 status='ok'：它也可能在 token/turn 耗尽时返回该状态。

## 5. 关闭检索的实现与 fresh 检查

生产代码只修改 onesearch_vl/onesearch_infer/prompts/__init__.py 和 tools.py：

- ONESEARCH_DISABLE_RETRIEVAL=1 隐藏 search/visit 工具和必须检索的指令；dispatcher 在 provider、图片解析或上传之前阻止相关 canonical/legacy 别名。
- unset/0 保持发布 prompt 字节一致；prompt/tool-name cache 按 modality 和开关状态区分。
- 开关仅传入本轮 inference 子进程，没有关闭其他项目服务，也没有将它全局写入 RL。
- 这是 retrieval 开关，不会自动禁用 OCR、模型或 judge 服务。最后 smoke 在自定义问题中明确要求只用本地 select_frame，没有调用这些外部服务。

fresh 验证：pixi install --locked 成功；检索开关 unittest 7/7 passed；git diff --check 成功；改动的 Python 文件语法检查成功；官方 subset 验证成功。回归测试核验开关行为，不代表 Serper 等真实外部服务已验证。

## 6. 已保留的失败与协议调整

没有删除失败目录，也没有把它们统计为通过：

| 尝试 | 观察 | 处理 |
|---|---|---|
| minimal-20261009-084409 | 当前 decord wheel 无法读取 loopback HTTP MP4；torchvision fallback 零帧 | 初始视频改用原发布 MP4 的本地路径；PNG 和工具发布仍经 loopback HTTP |
| minimal-20261009-085530 | 96-token 视频输出被截断在 think 中 | 记录 incomplete，增加合理 token 上限 |
| minimal-20261009-085814 | 有闭合正确回答，但 layout_parsing 未配置，产生工具错误后直接识图恢复 | 保留为错误恢复案例；不称无错误通过 |
| minimal-20261009-091215 | 图像 96-token 首轮工具 JSON 被截断，后续虽修复答对，严格检查仍拒绝 | 保存当时脚本；明确直接读图，增加图像 cap 至 256 |
| minimal-20261009-091814 | 图像/视频均闭合、无工具错误 | 最后通过的本地 integration smoke |

最终采用本地 MP4 的 transport 偏离官方可达 HTTP 输入协议，已经写入 run-manifest。没有在生产 decoder 中绕过错误或替换原始媒体。

早期大 HTTP ranges 曾产生同长度但错误的权重内容，完整 hash 检查拒绝了它们；之后采用 64 MiB 严格 ranges、完整 hash 与持久盘 readback 验证。只有最终匹配 receipt 的文件才被加载。失败文件留在 ignored assets 中，不参与模型加载。

## 7. 数据合成管线是否开源

截至本次 fresh 检查，官方 main 与用户 fork 的 HEAD 均为 0b7b01e；官方 recursive tree truncated=false。公开主分支没有找到从视觉源筛选到 VGEG、QA、专家轨迹的完整生成入口、配置和配套 prompt。

| 内容 | 发布情况与实际作用 |
|---|---|
| SFT-110K 成品 | HF 有单图、多图、视频 LLaMA-Factory JSON 与媒体 ZIP；本轮未下载/逐条核数整个训练集 |
| RL-10K 成品 | HF card 声明 10,059 train tasks，data/、evgr_gt/、train.jsonl 与媒体 ZIP |
| SFT/scripts/relocate_media.py | 校验已下载媒体、将相对路径解析为本地路径，另写训练 JSON；不会生成新 QA |
| RL 数据注册与 reward_preflight | 注册现成 tasks、校验 task/GT 配对和媒体；不构图、不生成语义标注 |
| README 数据引擎图 | 介绍方法流程与阶段，不是可执行 data engine |
| 完整合成 pipeline / prompt templates | 本次公开发布中未找到；论文 Appendix A 明确写“prompt templates are omitted” |

缺失的可执行部分主要是：视觉源筛选/事件与对象锚点生成；实体核验与 web evidence graph 扩展；从图生成 QA/VGEG；视觉指代改写与质量过滤；专家轨迹的生成、答案/过程筛选和 orchestration。

公开 actor prompts 是推理/rollout 使用的模板，不能代替这些数据合成阶段的 prompts。公开 evgr_gt 中有 claim_facts、boxes 或 grounding annotations，也不能等同完整源图与合成中间产物。

RL HF card 还链接了旧仓库 appletea233/onesearch-vl-submit；本次匿名 API 查询返回 404，不能据此断言它是私有或已删除。本轮可检查的公开代码来自 appletea233/OneSearch-VL。

主要来源：

- [官方公开 codebase](https://github.com/appletea233/OneSearch-VL/tree/0b7b01ed4872aaccd9af0f6f3d23a81814cf55bd)
- [SFT 数据树](https://huggingface.co/datasets/OneSearch-VL/OneSearch-VL-SFT-110K/tree/64c9da7f2af3362f26f497ad28f13ca834797b37)
- [RL 数据卡](https://huggingface.co/datasets/OneSearch-VL/OneSearch-VL-RL-10K/blob/1e0c8334a1604d8cc8e0edabf34c15c94d2899e2/README.md)
- [SFT 路径准备脚本](https://github.com/appletea233/OneSearch-VL/blob/0b7b01ed4872aaccd9af0f6f3d23a81814cf55bd/SFT/scripts/relocate_media.py)
- [论文 Appendix A](https://arxiv.org/pdf/2610.12419)

## 8. 作为我们的 AV codebase：可以复用与需要新增

可以直接复用 controller/tool-call 协议、native 8B runner、视觉工具、agent loop、trajectory 记录；SFT 和 rLLM/verl/Megatron 接口可作为后续训练基础，但本轮没有验证训练。

现有视频 preprocessing/clip 路径使用 ffmpeg -an，消息格式没有 audio payload。现有 runtime grid 还会丢弃原始 source timestamp 映射；grid 索引不能直接当报告中的原始 AV 时间。inference/RL 有两套工具实现，新增契约需要对齐。

建议下一阶段在当前 fork 增加独立 research 模块，顺序为：

1. 原始 AV source registry：保存音轨、source hash/version、原始时间、派生片段映射及来源关系。
2. 固定 AV observer：同步读取原始音频/画面并返回可引用 evidence records，先与 controller 分离。
3. 冻结多源检索 corpus 与数据合成：构建图、QA/report targets、未知/冲突/转载 cases，保存生成和过滤 provenance；从小批人工审核任务开始。
4. 同前缀状态快照与 continuation branch runner，用相同输入、observer 和预算比较真实候选行动后果。
5. AV research scorer：claim support、声画/跨源绑定、来源独立性、unknown 和成本；保持官方 answer-accuracy/EVGR 评测独立。

第一篇仍聚焦多源 Audio-Video search、证据核验和研究报告；Search for Generation 留到后期。音频/多源方法模块目前尚未实现，不将此次基础设施通过算作方法实验结果。

代码审查还发现发布 recipe 与论文存在差异：SFT YAML LR=1e-5 而论文表为 2e-5；发布 RL launcher 使用 RLOO，而论文描述 GRPO；训练入口未启用 held-out validation。后续若复现训练，必须明确采用发布 recipe 还是 paper-aligned 变体，并另做独立评测。

## 9. 可核对的记录

远端最后通过目录含 run-manifest.json、completion-review.json、single-image/video trajectory 和 local-tools.json；日志为 repro/logs/minimal-inference-final-20261009.log。源文件与锁的 postrun hash、媒体和权重 receipt、闭合响应/工具错误检查、token usage 和失败尝试列表汇总在 docs/research/OneSearch-VL-8B-local-repro-evidence-2026-10-09.json。

hash 记录用于核对文件对应关系，不替代独立语义评测。原始发布全量 benchmark、在线检索和 judge 尚未运行；没有 accuracy 或 training-efficacy 数字。
