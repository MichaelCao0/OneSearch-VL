# OneSearch-AV：Full Thinker 500题结果与在线检索进度

核查时间：北京时间 **2026-10-18 23:43:34**。远端主仓库：`coder ssh 3:/home/jovyan/OneSearch-VL`。本次为读取实际作业、checkpoint 验证和逐题评分记录，未发模型 API 请求，未新增 GPU Job。

## 当前完成情况

500 trajectory 与同题 answer-only Full Thinker 均完成250步，四rank一致。两组 checkpoint 的权重样本、optimizer分片、scheduler和RNG记录校验通过。基座和两组训练模型均完成56题×原声/全程静音评测，合计336个模型条件，执行错误0。使用同一个 Qwen2.5-Omni-7B pinned base、原 SFT/src/train.py、4×H20Z/ZeRO2；约7.616B语言参数可训练，AV towers/projectors冻结，没有LoRA。

2K trajectory 当前四rank均为 **246/1000实际optimizer updates**，预处理已结束。近期每步约4–5秒，剩余优化估计50–65分钟，另加保存与校验；该估计随吞吐变化。随后自动评测，2K answer-only 接续训练。目前2K没有最终checkpoint或任务成功率。后台orchestrator身份核实存活，279项冻结源文件/配置均一致；可用磁盘约753.5GiB。

## 500题统一问题集合的自动语义结果

| 模型与输入 | heldout共同38题 | development16题 |
| --- | ---: | ---: |
| 基座，原声 | 0/38（0.0%） | 0/16（0.0%） |
| 500 answer-only，原声 | 2/38（5.3%） | 0/16（0.0%） |
| 500 trajectory，原声 | 18/38（47.4%） | 11/16（68.8%） |
| 500 trajectory，全程静音 | 18/38（47.4%） | 12/16（75.0%） |

原heldout计划40题。trajectory与answer-only分别有1条自动评分未通过完整性校验，而且是不同问题；基座40条均有评分。上表统一取三组共同有效的38题，不把评分失败当作错误答案，也不重放这些请求。每组独立自动盲评可用55/56或56/56，最终所有模型同一次联合盲评仍待2K结果完成。

共同38题上，trajectory相对answer-only提高 **42.1个百分点**；按自动recording family聚类bootstrap的95%区间约 **[23.8,61.8]个百分点**。这些是基于当前自动silver参考与评分的探索性统计，尚非人工gold benchmark或开放网络结果。原声trajectory的18个成功答案均通过当前评分的supported_success条件。

当前证据支持：搜索轨迹SFT已让Omni在此离线任务上形成有效检索问答行为。音频独立收益尚未建立：heldout原声与静音同为18/38；dev静音12/16、原声11/16。后续用音频能改变搜索路径的问题与真实开放网络评测检验这一点。

训练题数与optimizer updates匹配，context/supervised token总量不同。该对照显示轨迹配方的实用收益，尚未隔离更多训练token与动作监督各自的贡献。基座执行工具很少，dev上的更强prompt/search baseline仍需补齐。

## DeepSeek在线检索

Search/Fetch后端及新native AV入口已接通；26项CPU/集成回归通过。基础API验证与真实teacher链路合计11次DeepSeek请求，10次有内容、1次Commons目标页Fetch 404，按公开成功请求单价估算¥0.05，Gemini另计。

真实teacher在原生AV输入下自主执行5次搜索和1次网页访问；6轮结束无最终response，不计成功任务或训练样本。**Omni的16题在线评测目前仅完成输入准备，尚未提交/执行**；上表均属原冻结离线检索条件。私有配置仍存在，权限0600，未读取或输出凭据。

下一轮在线对照优先使用已验证的500 trajectory checkpoint与同基座，固定16题、相同turn/token/Search/Fetch预算和原声/全程静音条件，报告实际最终答题、证据支持、网页访问失败、时间与费用。在线请求快照与离线实验分开保存。现有2K Full Thinker串行训练和自动评测继续运行。

## 实际证据

远端 suite：`/home/jovyan/OneSearch-VL/research/runs/av-full-thinker-learning-20261018-v2/`。

- `trajectory500/validation.json`与`answer500/validation.json`：训练/保存完整性验证。
- `judge-base/`、`judge-trajectory500/`、`judge-answer500/`：逐题自动评分记录。
- 本地`suite-latest.json`：当前作业身份、每rank更新数、输入/输出和源文件冻结核查。
- 本地`common-question-results.json`：相同问题集合上的计算，保留task IDs和失败题号。
- 本地`online.json`：在线后端与teacher试跑状态读取。

本地证据目录：`reports/status-2026-10-18-after-deepseek/`。本次没有commit/push。
