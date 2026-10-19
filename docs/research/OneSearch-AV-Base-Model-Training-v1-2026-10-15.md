# OneSearch-AV 基座训练与强基线方案 v1

日期：2026-10-15。对应用户最新要求：调一个基座模型，主要 follow OneSearch-VL。主执行位置 coder ssh 3，/home/jovyan/OneSearch-VL；本文件区分已有运行与下一步实验。

## 1. 固定一个主基座

选用 Qwen2.5-Omni-7B，revision ae9e1690543ffd5c0221dc27f79834d0294cba00。初始片段和 inspect_av 返回的片段直接作为原生视频/音轨进入同一个 Omni Thinker；Thinker 学习线索识别、查询、网页读取、片段检查和最终回答。搜索与工具使用 OneSearch-VL 原 agent loop，训练入口为原 SFT/src/train.py。训练目标是 assistant 的实际行动与答案，user、工具 observations、媒体占位对应 label 被 mask。

原路径：onesearch_vl/onesearch_infer/pipeline.py；新增原生模型 runner：onesearch_vl/onesearch_infer/omni_runner.py；配对数据与原 trainer 核查：research/av_train/native_sft.py。复用的不是另一个文字 controller；LoRA adapter 是更新主 Omni Thinker 参数的方式。

## 2. 与官方 recipe 的一致与差别

本轮直接检查 SFT/examples/onesearch/qwen3_vl_8b_joint_110k.yaml：官方发布 recipe 使用 Qwen3-VL-8B-Instruct，full fine-tuning，冻结 vision tower 与 multimodal projector，110K joint image/video 轨迹、global batch128、LR1e-5、8epochs、cutoff32768、2FPS。它是发布配置，不自动等于论文所有设置。

我们的原生 AV 主模型改为 Qwen2.5-Omni-7B。当前500 pilot 用 Thinker LoRA rank8/alpha16、LR1e-4、batch4、2epochs、250updates；因此属于 AV 定制训练实验，不能叫官方训练 recipe 原样复现。原生媒体、搜索轨迹监督、原 trainer 与后续语义成功率评测，才是当前 follow 的主线。

为提高性能，训练路线覆盖 full Thinker SFT：沿官方“冻结感知塔、训练语言推理部分”思路，保留音频/视觉输入，冻结 visual、audio_tower 与 projector，训练 Thinker language model；不训练生成语音 Talker。原 SFT 的 composite model 注册已将 qwen2_5_omni_thinker 的 audio_tower 计入冻结塔。先验证实际 trainable 名单、有限梯度、保存重载和显存，再正式运行；准备配置不等于 full SFT 已完成。

## 3. 最小且有说服力的比较

| 条件 | 是否训练 | 输入与监督 | 要回答的问题 |
|---|---|---|---|
| Omni direct QA | 否 | 原生 AV + 问题，无工具 | 模型内部知识能做到什么 |
| Omni prompted search | 否 | 原生 AV + 同一搜索环境；dev 选择 prompt | 充分提示以后，训练还有多少收益 |
| Omni answer-only SFT | 是 | 与轨迹 SFT 同题的初始 AV + 最终答案 | 收益是否只是学会输出答案 |
| OneSearch-AV trajectory SFT | 是 | 同题真实多轮 AV/工具/答案 | 学会搜索行动带来的增量 |
| Trajectory SFT，静音/video+ASR/audio-only | 相同权重 | 真实重跑、相同搜索预算 | 声音及联合模态起什么作用 |
| Official OneSearch-VL-8B | 官方权重 | 原视觉工具能力 | 与已有视觉搜索模型的关系 |

训练与评测分开：answer-only 主对照在测试时也可获得相同工具及 prompted-search 协议；另列 no-search answer-only，避免把工具可用性差异当监督形式收益。轨迹通常有更多 supervised/context tokens，必须报告真实 token 与 GPU 成本；主表固定题目与推理预算，补做近似训练 token/compute 匹配分析。

## 4. 执行顺序与调参预算

1. 先完成869池的人工抽查、2K CPU合格数据扩容和首批独立 benchmark 人审。当前 benchmark gold 为0，不阻止标为silver的探索评测。
2. 在现有同一500题冻结集上补 answer-only；原 clean500 trajectory 权重可复用，不为比较重新挑数据。强 prompted base 在独立dev上选提示，test不参与选择。
3. 2K trajectory/answer-only 从同一 pinned base 分别初始化；默认先复用已验证 LoRA recipe，不能把500 adapter继续训练后当作从零2K结果。小规模搜索 LR={5e-5,1e-4}、rank={8,32}，先单seed，依据dev task success选择，最多4配置；不要按最低训练loss挑选。
4. 在同一2K冻结数据上做 full Thinker SFT 性能验证，初始候选 LR={1e-5,2e-5}、2epochs、global batch16、BF16、ZeRO2，2–4张GPU按现场容量与长样本smoke确定。此为候选预算，实际选用后写新freeze；不把它叫与LoRA等计算量。训练正常保存后立即同题评测。
5. 数据到5K/10K/20K后扩大已获dev收益的配置；主结果与核心 answer-only 对照以3seed为目标。SFT后若有稳定的query/来源选择瓶颈，再沿OneSearch-VL增加RL；RL不是当前数据阶段的前置条件。

Prompt候选为现有长协议、精简中文协议、从独立train构造的两例完整工具交互协议；工具集合、输出语法、token/turn预算、媒体预处理和合法动作归一化完全一致。示例不引用dev/test答案与来源。先选协议完整率，再按dev语义成功选择，最终冻结并对base与SFT采用同一规则。若额外做grammar-constrained动作输出，对所有模型对称报告。

## 5. 目前看到的瓶颈与结果边界

已有500轨迹pilot在heldout的39道可判题上成功23题（59.0%）；静音21题。旧dev的15道可判题上原声8题、静音13题。现有54道可判题均为silver，不能据此作正式声音优势结论。

本轮检查base真实trace：56道native AV全部执行完成，但只有4题有完整final协议，工具仅出现3次inspect_av，没有text_search或visit_webpage；示例常在几十token后结束于think或未完成tool_call，而非512token长预算被耗尽。SFT首先教会了协议与行动；基座的0成功不能替代对充分提示、完整示例及相同解码规则的比较。该诊断需要后续实际prompt实验验证，不能只归因于模型感知差。

## 6. 论文记录

每次训练记录dataset成员/媒体hash、初始化权重、可训练参数、recipe、实际updates/tokens、GPU小时；每次评测保存同题最终答案和真实工具轨迹、逐题语义成功、unsupported断言、费用及unknown。正式benchmark测试集来源在训练前预留，不能按student成败筛选。

补充验证：已执行原trainer full-tuning函数的CPU meta参数检查，7,615,616,512语言参数可训练，1,316,197,376感知/投影参数冻结，Talker未包含。没有加载权重或真实梯度；GPU运行尚未提交。
