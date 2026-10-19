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
