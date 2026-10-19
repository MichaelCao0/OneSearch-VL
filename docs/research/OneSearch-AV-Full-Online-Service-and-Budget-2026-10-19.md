# OneSearch-AV：完整在线服务与调用预算

日期：2026-10-19。按用户确认的 ¥5/1000 次，即 ¥0.005/次计算。以下只含 DeepSeek Search/Fetch，不含 Gemini、GPU、人工审核和存储传输。

## 完整服务准备状态

新在线入口默认启用 Search need_content=True，并把正文片段实际放进工具 observation；visit_webpage 使用 Fetch。结果保留真实 URL，正文限长并标明截断，完整原始响应保存在本 run 回执中。--no-need-content 可保留 summary-only 模式；旧冻结离线数据和成绩不改写。

新增 --prompt-file 支持原样使用离线冻结提示词。17 项后端和原生 loop 测试通过，基座/500/2K × 原声/静音共96条输入完成CPU prepare-only。没有新增收费请求，没有提交GPU任务；这不是线上性能结果。

## 眼前一轮在线对照

16题 × 3模型 × 2模态 = 96条rollout。按每条3–6次Search/Fetch，预计288–576次，费用¥1.44–2.88。每组16题最多128次Search+Fetch，六组合计768次；所有请求均计费的绝对上界为¥3.84。

现有2K离线轨迹的联网类工具调用为：原声169/56=3.02次/题，静音185/56=3.30次/题，合计354/112=3.16次/题。联网可能需要更多次，3–6是预算假设。

Search取top_k=10仍是一次HTTP请求；随后Fetch一个URL另算一次。Search附带正文不额外产生一次Fetch。inspect_av不调用DeepSeek。成功缓存不再发相同HTTP请求，估算暂不扣除该节省。

## 新数据生产预算

新多跳/组合数据还需构题时的事实拓展、teacher实际搜索解题和必要核实。先假设每候选题总计8–15次Search/Fetch、最终不同问题保留率50%；这是工程预算，须用首批64候选实测校准。

请求数 ≈ 目标保留问题数 ÷ 最终保留率 × 每候选题Search/Fetch次数。

| 目标 | 调用次数估计 | DeepSeek费用估计 |
|---|---:|---:|
| 新500条训练 | 8,000–15,000 | ¥40–75 |
| 新2K条训练 | 32,000–60,000 | ¥160–300 |
| 新20K条训练 | 320,000–600,000 | ¥1,600–3,000 |
| 完整数据：20K train+500 validation+1200 bench | 347,200–651,000 | ¥1,736–3,255 |

首批64候选约512–960次，即¥2.56–4.80；它们不等于64条保留成功数据。

## 完整论文工作量的中心估计

20K train、500 validation、200 dev和1000 test，共21700条最终不同问题。每候选12次、保留率50%，构建约520800次，¥2604。

给1200题做3模型×2模态的一轮评测，共7200条rollout，按4.5次/rollout约32400次，¥162。两项合计553200次、¥2766；加20%余量为663840次、约¥3319。未包含额外baseline、重复seed或后续多次重评。

如果保留率降到30%，同样21700条数据、每候选12次，仅构建即约868000次、¥4340。主要不确定量是最终保留率和每题证据深度。

## 证据与执行状态

- 主代码：/home/jovyan/OneSearch-VL/research/av_search/deepseek_backend.py、research/av_search/online_searchqa.py。
- 运行配置：research/runs/av-full-online-preparation-20261019-v1/plan.json，status=cpu_prepared_not_submitted。
- 同目录保留source-before、tests-red.log、tests-green.log、validation.json及六组CPU准备收据。
- 本地预算：reports/deepseek-budget-2026-10-19/budget.json。
- 接口说明：[DeepSeek官方指南](https://api-docs.deepseek.com/zh-cn/guides/search/)。
