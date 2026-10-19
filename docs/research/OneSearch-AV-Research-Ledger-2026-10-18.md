# OneSearch-AV Research Ledger 2026-10-18

# OneSearch-AV 执行进度（2026-10-18 20:26，北京时间）

目前共有 **2,092 个不同训练问题**完成成对导出、媒体与监督标签 CPU 检查，比昨日的 1,974 条增加 **118 条**。扣除冻结质量视图的 146 个排除项，当前 **1,946 条 eligible silver**，距离正式冻结 2K 还差 **54 条**。eligible silver 包含尚未独立内容审核的问题，不代表人工 gold。

## 数据生产

将既有 collector 的单次查询返回上限由 20 提高到 50 后，已实际采集三批共 **750 段候选音视频**。前两批进入 authoring / rollout / review，第三批 250 段已下载保留，未进入本轮训练计数。

上一轮 `av-data-production-20261017-v1` 于 **2026-10-17 19:56** 因 `provider_transport` 触发保护退出，freezer 如实记录 `producer_finished_below_target`。旧 producer、prefetch 和 freezer 均已核对进程身份，确认不在运行。付费失败或不确定请求保留原记录，没有重放。

已完成逐任务恢复检查：停下的 chunk 有 25 条 producer 记录完整的已完成轨迹、8 条失败轨迹和 15 道未建立 attempt state 的 pending 题；同一 authoring batch 还有 33 道从未进入 rollout chunk 的题。已校验原 plan、轨迹完成记录和源码哈希。

当前已启动新的 CPU 后台恢复：

- producer：`/home/jovyan/OneSearch-VL/research/runs/av-data-production-20261018-v1`
- freezer：`/home/jovyan/OneSearch-VL/research/runs/av-learning-2k-preparation-20261018-v1`
- 首先为 25 条已完成轨迹做首次独立自动审核，随后只对 15+33=48 道未执行题目做新 rollout。
- 8 条旧失败轨迹明确排除。新恢复预算为 6 小时、最多 25 次补审和 48 个新 rollout slots，沿用每题最多 8 次 actor 请求；自动收费失败重试为 0。
- 审核并发 1，后续 rollout 并发 2；不重新采集素材，也不提交数据生产 kjob。

2026-10-18 20:26 回读，producer 与 freezer 身份均存活；11 项恢复冻结文件与 5 项 freezer 冻结文件一致。新补审已有 **3 条 completed，2 条 success**，另 1 条 running，未出现新的 provider-stop。这些结果尚未完成导出和原训练器 CPU 检查，**没有提前计入 2,092 / 1,946**。

## Benchmark 与训练

Benchmark 本轮重新按实际任务与来源记录核对，仍为 **304 道候选题，198 个不同录制**。其中此前冻结的 217 题已有 208 条自动审核：192 keep、2 revise、14 reject，9 条 unknown/unreviewed；新增 87 题尚未纳入同一完整审核。人工 gold 为 0，后续从候选池中核实并筛出 200 道 dev。

**新的 Full Thinker 500 / 2K 正式训练尚未启动，没有新的性能分数。** 已完成的资源与接口验证包括：4×H20Z、ZeRO2、原 `SFT/src/train.py` 实际 3 个 optimizer updates，以及完整 checkpoint 的 native AV / muted 推理接口验证。该 smoke 不能作为正式学习收益。

历史 500 题 LoRA 在共同 38 道 silver / frozen-corpus 评测题上的 answer success：base 0/38、answer-only 3/38、trajectory 24/38、muted trajectory 22/38。历史数据集与新 clean500、Full Thinker route 不同，后续正式规模对照使用冻结的嵌套 500/2000，并报告训练 tokens 和 GPU 成本。

最新集群审计没有 active GPU Job，昨天之后没有新增 GPU Job。远端剩余磁盘约 **1,185 GiB**。

## 下一项交付

优先完成最后 54 条可选数据、产出 clean2000 冻结与嵌套成员校验，再按实际长度和媒体极值准备正式 Full Thinker 训练与即时任务成功率评测。Freezer 负责冻结数据，不会自动提交 GPU 训练。

本轮恢复采用原生产函数和训练器，主代码在 `coder ssh 3:/home/jovyan/OneSearch-VL`；没有改动活跃的冻结源码，没有 commit/push。实际回读与启动前检查保存在本地 `reports/progress-2026-10-18/`。


# OneSearch-AV：2K 冻结与正式训练启动

执行快照：**2026-10-18 21:40，北京时间**。主项目仍为 `coder ssh 3:/home/jovyan/OneSearch-VL`。

## 已完成的 2K 数据里程碑

本轮已达到 **2,146 个 CPU 合格不同问题**，扣除冻结质量规则中的 146 个排除项，正式冻结 **2,000 条训练问题**，来自 **1,475 个录制**。clean500 的全部成员都包含在 clean2000 中。

- 冻结时间：2026-10-18 **21:01**。
- 2K 成员 SHA-256：`b84bc04c17bc474350d90a5bd6648ce6a2e6bbfdc0c97e16df7503364084c54d`。
- 数据目录：`/home/jovyan/OneSearch-VL/research/runs/av-learning-2k-preparation-20261018-v1/clean2000`。
- Teacher 构成：Gemini 3.8 Flash 1,903 条、Gemini 3.1 Pro 97 条。
- 最长 trajectory 上下文 **23,860 tokens**；每轮 16,317,511 个 context tokens、429,064 个 supervised tokens。
- 全部样本已有保留成功的自动轨迹审核及绑定的原训练器 CPU 媒体/标签检查。另有 384 条通过额外的题目内容审核；1,616 条尚未经过该额外审核。全部属于 silver，不标成人工 gold。
- 冻结选择按 recording/question 去重，排除已知 reject/revise/control/unknown；答案监督版本和 trajectory 版本是同题对照，不重复计数。原 CPU 解码收据按未改变的行精确组合复用，没有把组合收据说成再次完整解码。

本轮恢复先为 25 条此前已完成轨迹做首次独立审核，再只运行未执行题目，旧失败和不确定付费请求没有重放。旧数据与失败记录均保留。

## 新极值样本的实际 GPU 验证

新资源作业 `slurm-profile-slurm-n8h2b` 已为 **Complete**，验证结果为 **passed**。

从 clean2000 中选取 12 个未改变的极值样本，覆盖最长上下文、最多监督 token 与媒体数量极值。4×H20Z、ZeRO2、原 `SFT/src/train.py` 实际完成 3 个 optimizer updates，峰值约 **85.2 GiB allocated / 127.5 GiB reserved**。训练参数为 7,615,616,512，冻结 AV 参数为 1,316,197,376。

已回读保存权重的训练样本值、4 份 optimizer 状态、scheduler 与 4 份 RNG；scheduler 明确支持独立 HF `scheduler.pt`。该结果验证资源与训练保存流程，性能收益由后续正式评测给出。

## 正式实验配方与运行状态

当前执行目录：`/home/jovyan/OneSearch-VL/research/runs/av-full-thinker-learning-20261018-v2`。

| 实验 | 问题数 | 更新数 | 每轮 context tokens | 每轮 supervised tokens |
| --- | ---: | ---: | ---: | ---: |
| trajectory500 | 500 | 250 | 4,014,576 | 110,243 |
| answer500 | 500 | 250 | 2,458,745 | 30,097 |
| trajectory2000 | 2,000 | 1,000 | 16,317,511 | 429,064 |
| answer2000 | 2,000 | 1,000 | 10,151,803 | 116,441 |

四组均从同一 pinned `Qwen/Qwen2.5-Omni-7B` 基座开始，revision `ae9e1690543ffd5c0221dc27f79834d0294cba00`；沿用原 `SFT/src/train.py`，Full Thinker 全参数训练语言部分，冻结 AV towers/projectors；LR `1e-5`、2 epochs、有效 batch 4、bf16、SDPA、gradient checkpointing、ZeRO2。使用 4 张训练卡时 gradient accumulation 为 1。无新增神经适配器，无 LoRA。

比较固定了问题、初始化、epochs 和同规模更新数；trajectory 与 answer-only 的 context / supervised token 成本不同，表中单独报告，不能称为等计算预算实验。

2026-10-18 21:40 回读：

- `trajectory500` 正式训练 Job **`slurm-profile-slurm-mtt2m`** 已启动，正在对 500 条样本做原训练器预处理；尚无已记录的 optimizer update，不能把该阶段写成训练完成。
- 基座评测 Job **`slurm-profile-slurm-j5l5g`** 已启动，已完成原声 **21/56** 与静音 **21/56** 题，执行错误为 0。语义评分尚未产生。
- 后台 orchestrator 身份核实存活，279 项冻结文件均一致。其余三组正式训练及四组对应评测已完成 manifest/server dry-run 与数据绑定，尚未提交。
- 四组训练串行；最多并行一个 4 GPU 训练和一个 1 GPU 评测，总计最多 5 GPU。

## 训练完成后的接续流程

每组训练完成后先验证完整 checkpoint 结构、模型权重样本、optimizer/scheduler/RNG，再提交该组固定 **56 题 × native AV / muted** 的单卡评测，同时允许下一组训练开始。该评测集是 40 heldout silver + 16 dev，不是新的人工 benchmark。

所有版本使用相同 prompt、最多 8 turns、每轮最多 512 new tokens。静音在初始输入及每次工具返回时物理去除音轨，模型输入检查要求 audio_count 为 0；原声条件检查实际音频输入。共享 `modality_inputs.py` 处理 FFV1/MKV 与 MP4，并验证编码视频包保持。

每个版本完成后启动 CPU 独立语义评分，报告 answer success、supported success、目标来源检索/访问和 unknown；所有版本完成后再做共同盲审对比。评分模型看不到模型名和模态标签。最多 6×56=336 个新评分请求槽，单阶段并发 2，无收费失败重放。

训练预算上限：500 题作业每组 4 GPU×6h；2K 每组 4 GPU×12h；四组训练最多 144 GPUh。5 个单卡评测最多 15 GPUh，资源验证上限 6 GPUh。以上均为硬上限，不是实际用量估计。每个 Job 的 backoffLimit 为 0；排队和运行有界，失败保存记录并停止后续接续。当前未给出训练完成时间估计，实际吞吐需要正式更新记录。

## 启动修复与证据

初版 suite v1 的基座评测因缺少 `AV_SFT_RUN` 在模型加载前退出，保留 Failed Job `slurm-profile-slurm-8d86s`，没有生成评测轨迹，也没有提交正式训练。v2 补齐环境变量及 `eval-runtime/cluster` 目录绑定，五个评测 launcher 均通过 CPU 环境契约检查；实际基座评测已执行 42 个条件，验证修复生效。资源验证直接复用已通过的结果，没有重跑。

新增 `full_thinker_training.py` 将原 3 步验证扩展为显式的 250/1,000 步训练；`full_eval.py` 复用原 native agent loop 并加载完整 Thinker 权重；`full_task_success.py` 复用盲评规则。原训练主干没有改动。训练/保存/输入相关 27 项 CPU 回归和 4 项评分 schema 测试已通过；9 个后续 Job 的 server dry-run 已通过。

本地已归档 **86 个配置、源码和收据文件**，逐项传输 SHA-256 检查通过，没有复制 checkpoint tensors。归档：`reports/continue-2026-10-18/evidence/`；实时检查脚本：`work/audit_full_suite_live_20261018.py`。Benchmark 仍为 304 个候选 / 198 个录制，人工 gold 0。本次没有 commit/push。


## 21:46 运行更新

四个 rank 均已记录 **23/250 个实际 optimizer updates**，原训练器已完成预处理并进入优化。对应基座原声与静音评测均完成 **38/56** 题，合计 76 个条件，执行错误 0。后续解析当前日志验证全部已记录 loss 和 grad_norm 有限、梯度为正；训练 loss 不作为任务成功率。正式训练的最高已观测显存约 73.0 GiB allocated / 121.5 GiB reserved。


## DeepSeek 在线检索接入

新增 DeepSeek Search/Fetch 后端及在线 native AV 入口，5 次真实请求均有结果、估算 ¥0.025；26 项回归通过，16题×两模态准备完成。模型自主在线检索仍需完成轨迹和语义验收，teacher 试跑逐版本保存。已有正式对照 279 项冻结文件保持一致。详细证据与接续计划见 [接入文档](OneSearch-AV-DeepSeek-Online-Search-2026-10-18.md)。没有新提交 GPU Job 或 push。

### 在线模型链路读回

Gemini 原生 AV + 生产 json_v1 协议，6 轮真实生成全部有 receipt，自主执行 5 次在线搜索和1次网页访问。搜索5/5有内容，Commons 文件页 Fetch 404；用满6轮仍无最终 response，不计任务成功或训练数据。基础验证与模型pilot合计11次DeepSeek请求，10次有内容、1次404，按官方单价估算 ¥0.05，Gemini另计。固定16题在线原声/静音输入已准备，Omni在线评测尚未运行。


## 23:43 Full Thinker 500题结果与2K进度

两组500训练均250步且保存校验通过；基座和两组训练模型完成336个原声/静音评测条件，错误0。统一三组共同38题：基座0/38，answer-only2/38，trajectory原声18/38、静音18/38；dev16为0、0、11、12。两个不同题评分不完整，保留unknown。离线silver结果显示轨迹收益；尚未建立音频独立收益。2K trajectory四rank均246/1000实际更新；后台存活、279冻结文件一致。DeepSeek入口已通，Omni线上16题尚未执行。详细证据见[OneSearch-AV-FullThinker500-Results-and-Online-Status-2026-10-18.md](OneSearch-AV-FullThinker500-Results-and-Online-Status-2026-10-18.md)。
