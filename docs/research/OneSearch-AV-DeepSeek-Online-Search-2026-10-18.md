# OneSearch-AV：DeepSeek 在线检索接入与实验迁移

记录日期：2026-10-18。最近读取：2026-10-18T14:12:45.738528+00:00（UTC；北京时间加 8 小时）。主代码与运行均位于 `coder ssh 3:/home/jovyan/OneSearch-VL`，Pixi 环境。本次没有 commit/push，没有新提交 GPU Job。

## 结论与目标

DeepSeek Search/Fetch 已接入原 OneSearch 工具分发器。新的研究运行可以使用开放网页搜索与正文抓取，研究目标为：Omni 从真实声音与画面中形成可检索线索，调用外部搜索、读取证据并回答问题。搜索服务直接返回结构化网页结果，无需再提供 Serper 或额外摘要模型服务。

后续开放网络实验使用新入口；已经冻结的 2K 数据和正在运行的 500/2K Full Thinker 对照继续按原离线协议完成，保留其控制变量价值。新在线条件、旧离线条件分别报告。

## 接口映射

| 原工具 | 新执行路径 | 模型角色 |
| --- | --- | --- |
| `text_search(q, top_k)` | `POST https://api.deepseek.com/v1/tools/search` | Omni 自己根据 AV 线索构造 query |
| `visit_webpage(url)` | `POST https://api.deepseek.com/v1/tools/fetch` | Omni 读取真实网页 Markdown 并推理 |
| `inspect_av(video,start_s,end_s)` | 原 `av_io.crop_native_av` | 原生返回真实片段，Omni 继续看与听 |

DeepSeek 这里提供网页文字搜索与抓取。音频指纹匹配、视频反向搜索、图像反向搜索需要另外的服务；当前 audio_video 工具集合本就为上述三个工具，无需改变该工具集合。Omni 基座与原 `process_native_av_case → run_agent_loop → execute_tool` 路径继续使用。

官方资料：[使用指南](https://api-docs.deepseek.com/zh-cn/guides/search/)、[Search schema](https://api-docs.deepseek.com/zh-cn/api/search/)、[Fetch schema](https://api-docs.deepseek.com/zh-cn/api/fetch/)。官方单价是 Search / Fetch 各 ¥5/千次成功且有内容的请求，账号级限速各 5 QPS。原始 API 与特定 Harness 的限时计费政策分开解释。

## 已实现代码

- `/home/jovyan/OneSearch-VL/research/av_search/deepseek_backend.py`：与原 backend 相同的 `text_search`、`visit_webpage` 接口，保留真实 URL、网页摘要和原文。使用 API 排序；不根据答案筛选检索结果，不生成额外摘要。
- `/home/jovyan/OneSearch-VL/research/av_search/online_searchqa.py`：新的在线入口，支持原始 Omni、完整 Thinker checkpoint、native AV 和全程静音，支持不加载模型的 `--prepare-only`。
- `/home/jovyan/OneSearch-VL/validation/test_deepseek_backend.py`：14 项后端与原 native AV loop 集成测试。

新后端对每次请求先记录 attempt，再传输；成功响应保存 body/hash/时间和请求参数，相同请求可从本 run 缓存读取。失败、超时和未完成 attempt 均保留，不自动重放。没有离线 fallback。Search 和 Fetch 通过项目共享文件锁分别间隔至少 0.30 秒；每个 run 有明确总请求槽上限。其他应用若共用该账号，其流量仍占官方账号额度。

`q` 保持原样，明确验证 1–128 个 Unicode 字符，不截断关键词；超过 128 字符的原问题不送入可选 `ori_question` 字段。`lang` 保留接口兼容，语言约束由 query 中的 `lang:` 显式指定。正文给模型最多 20K 字符；完整响应保存在本次运行的请求记录中。每条新返回结果标明文字类型和检索时间，网页内容保持“外部证据”角色。

凭据只存远端私有文件 `/home/jovyan/.config/onesearch-av/deepseek-search.env`，权限 `0o600`。源码和发布文件中不保存 key。新增源码扫描已通过。

## 已完成真实验证

1. 三个公共查询实际成功，各返回 5 条网页结果。耗时分别约 0.413、0.393、0.313 秒。这是小样本端到端延迟，不能作为吞吐基准。
2. 抓取 DeepSeek 官方文档成功，返回 6,402 字符，约 0.896 秒。
3. 经原 `execute_tool` 重放缓存搜索，返回 8,615 字符；随后真实抓取搜索命中的 OneSearch-VL GitHub 页面，返回 14,649 字符，均无工具执行错误。一次搜索命中缓存，没有新增 HTTP 请求。
4. 合计 **5 次真实 Search/Fetch 请求，5 次成功且有内容，估算 ¥0.025**。估算按公开单价，未读取账单；Gemini teacher 的费用另计。
5. 14 项新增测试通过；加上 native AV、多轮静音和检索开关回归，合计 **26 项测试通过**。模拟 actor 的集成测试确实经过原 agent loop，并验证网页观察进入下一轮、原生音视频仍在历史中。
6. 已准备同一批 16 个 development 问题的原声与静音在线输入，共 32 个条件；两组 prepare-only 检查通过，**尚未执行 Omni 在线推理与语义评分**。

## Teacher 单题试跑与当前边界

尝试记录均保留在 `/home/jovyan/OneSearch-VL/research/runs/av-deepseek-integration-20261018-v1/`。

- v1：dev-005 原始输入为 30MB MKV，在本地媒体校验时退出，尚未发 HTTP；一个 runner 调用槽不等于一个实际模型 HTTP 请求。
- v2：将该输入全时长转为 1,063,640 字节 MP4；12 秒完整时长和音轨均保留。误选旧 OpenAI 路由得到 404，尚未触发 DeepSeek，不重放该失败请求。
- v3：改用已配置的原生 Gemini 路由，选择未尝试的 dev-007。返回未满足完整文本动作验收，没有触发 DeepSeek；保留失败，不计正确答案或训练数据。
- v4：选择未尝试的 dev-009，使用现有生产的 `json_v1` teacher 协议、4096 token 上限与生产搜索提示词，全时长规范媒体。状态 `completed`，模型调用槽 6/6，工具调用 `['text_search', 'visit_webpage', 'text_search', 'text_search', 'text_search', 'text_search']`，DeepSeek 请求 6 次；后台身份核查存活=False。

v4 逐轮读回确认：6 次模型生成均有完整 receipt，模型自主执行 **5 次 text_search + 1 次 visit_webpage**。5 次搜索成功返回内容；目标 Commons 文件页 Fetch 返回 HTTP 404。6 轮结束仍是 tool_call，**没有合法最终 response**，不能计作任务成功或训练保留样本。`completed` 仅表示该次有界执行结束。逐题任务是否可由开放网络完成、检索召回和证据支持仍需固定集评测。

全轮合计 **11 次 DeepSeek HTTP 请求，10 次成功且有内容、1 次 Fetch 404**；按公开成功请求单价估算 **¥0.05**，不含 Gemini 模型费用。基础 API 验证的 5/5 与模型 pilot 的 5/6 分开保留。

因此 Search/Fetch 与原工具分发器的真实接入已成立；模型自主使用在线检索的收益，仍需读回完整轨迹并独立评分。运行成功、答案正确、证据支持和音频收益分别验收。以上 pilot 不加入训练集，不作为 benchmark 结果。

## 后续实验与数据生产

先完成当前离线四组 Full Thinker 对照，同时在新入口做固定 16 题在线开发评测；模型包括 base、trajectory500、trajectory2000，并补 answer-only 对照。各版本使用相同搜索服务、turn/token/请求预算及原声/全程静音条件。先检查 task success、supported success、工具调用、URL 可访问性和时间/费用，再扩到正式 dev/test。

下一批新 teacher 轨迹使用在线后端：actor 只收到匿名真实 AV 与问题，从自身观察生成 query；写题者参考页不作为固定检索答案池。完整保存 Search/Fetch 响应、时间、检索参数和轨迹，后续用成功缓存进行逐查询重放。新旧语料标明 backend 类型；同一问题的 answer-only 是训练对照，不能增加独立问题计数。现有 2K 是离线 warm start，后续在线轨迹单独统计。

假设每个保留样本需要 6 次成功 Search/Fetch，2K 样本检索费约 ¥60、20K 约 ¥600；若保留率为 p，则约除以 p。该式仅用于预算，实际调用次数、失败、缓存和保留率尚需实测，且不含 teacher 模型费用。完整论文目标继续是 20K train、500 validation、200 dev 和 1K test。

API 服务本身是实验基础设施。论文贡献继续围绕 AV 线索驱动搜索、训练数据构建、benchmark 和模型任务性能；在线服务让开放网络实验更接近 OneSearch-VL 的搜索问答设定。

## 冻结与证据

当前正式 suite 的 279 项源文件与配置 hash 全部一致：`True`。没有修改原 SFT 主干、agent loop 或旧 FrozenSearchBackend。

本地证据：`reports/deepseek-online-2026-10-18/evidence/manifest.json`，共归档 37 个小文件，逐项 SHA-256 核验通过。未复制视频、权重或凭据。新源码 SHA-256：

```json
{
  "research/av_search/deepseek_backend.py": "ba3478f6e623338e75e6a2b7420868b4fe41d6114e8dc328bc474676708d22e5",
  "research/av_search/online_searchqa.py": "29d4759e29ed7a9e3b05eb563dd45b887ca2a64f3d671d0c5c6468ce3ed982f6",
  "validation/test_deepseek_backend.py": "25d82291db9b6be537d6a07b1bd55f1a70b6b055642cf6ba83a8115ee763e05a"
}
```

CPU 输入准备命令示例（本次已执行同等检查；不加载模型、不发搜索请求）：

```bash
cd /home/jovyan/OneSearch-VL
pixi run --locked python -m research.av_search.online_searchqa run \
  --cases-json research/runs/av-deepseek-integration-20261018-v1/online-dev16-cases.private.json \
  --model-manifest research/runs/av-native-smoke-20261010-v1/runtime-setup/model-manifest.json \
  --out-dir research/runs/av-online-dev16-NEW-VERSION \
  --max-search-requests 128 --max-turns 8 --max-new-tokens 512 \
  --modality native_av --prepare-only
```

静音组使用 `--modality muted_video`；训练模型增加 `--full-checkpoint /absolute/path/to/verified/checkpoint`。每次使用新输出目录。实际 GPU 推理去掉 `--prepare-only`，沿已验证的集群 launcher/preflight 使用其 GPU Python 环境；仓库根目录默认 Pixi 是 CPU 环境。以上不是新 GPU Job 的提交记录。
