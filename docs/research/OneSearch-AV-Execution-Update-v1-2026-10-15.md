# OneSearch-AV 数据、Benchmark 与基座训练执行记录 v1

核查时间：2026-10-15T07:41:23.777690+00:00。主执行 coder ssh 3，/home/jovyan/OneSearch-VL。本文追加最新状态；v5主计划和初始snapshot中的855+14、生产已停止等是此前核查时的历史状态。

## 已完成与正在运行

- 待核14条已全部通过原trainer CPU媒体/标签检查，原合格池由855升至869。
- 当前 verified_pool：**904 CPU-qualified / 904 retained**，差额是待完成CPU核验的silver；不是人工gold。数据计数以自动配对去重和CPUaudit为准。
- 直接CPU后台生产v5当前live=True，阶段 acquire_new_recordings；身份核对host、boot ID和start ticks。47次新AV出题尝试，得到44道实际工具rollout候选；8 workers；目标2K。付费失败/未知不自动replay。后续每新cohort先预留200个benchmark候选来源家族，再生产训练数据。
- 首批benchmark从33个自动隔离且未付费尝试的录制预留，30个完成媒体准备，23个出题调用完成、7个失败不重试；产出19题/14录制，全部split=dev。人工gold仍为0。19题没有按student成败筛选。
- benchmark actor任务不含答案字段和目标来源ID，所有目标录制进入训练排除registry；自动来源家族还须人工查近重复，不能宣称正式独立benchmark已完成。
- 展示10个不同训练录制案例，以及14个不同benchmark录制案例（19题中每录制展示1题）。模型输入是真实原声片段，另提供附近60–90秒上下文；原片短于这个长度时给全长，不用循环补长。

## 审核与数据质量发现

采集query类别不是实际内容标签：训练样例中bird对应民歌合唱、animal对应政治演讲、music对应课堂。保留这些例子与旧版本；新版页面将其标为“采集查询类别，真实内容待核实”。五类采集覆盖不等于五类真实任务分布，正式数据卡按核实后的实际场景重算。

Benchmark也包含鸟种/犬种/人物识别等可能仅靠感知或内部知识回答的候选。它们尚未被自动升级成需要搜索的正式问题；审核需判断问题自然性、外部信息必要性、真实声音作用、字段答案支持和关键区间。属于纯感知的候选可作控制或改写，不能充当主搜索集合的证据。网页发布者说明是事实候选依据，不是独立真值证明。

## 主基座与论文路线

固定Qwen2.5-Omni-7B，原生AV直接进入Thinker；沿OneSearch-VL的真实工具轨迹SFT和原SFT/src/train.py。当前500为LoRA pilot；官方发布recipe是full fine-tuning，二者区别已记录。新增同题answer-only、强prompted base与2K full Thinker SFT计划，尚未执行这些新训练。

本轮从原始base结果核查：56道native AV只有4道完整final，3次inspect_av，无文本搜索或网页读取。现有SFT收益含协议学习，正式主表须补充分提示的base和matched-data answer-only。已有clean500 heldout23/39、muted21/39为silver结果；dev8/15、muted13/15，声音稳定增益未建立。

## 实际交付

- OneSearch-AV-Base-Model-Training-v1-2026-10-15.md：主基座、官方recipe差别、full Thinker路线、baseline和调参预算。
- records/2026-10-15-base-model-and-data/：实验spec、base协议诊断、执行snapshot和媒体验证；spec明确planned_not_launched。
- research/av_train/question_authoring.py增加split参数，默认train；开发评测用dev，16项相关回归通过。运行中的生产源码freeze保持未变。
- research/av_data/training_sample_review.py：从实际任务与绑定媒体生成可播放训练/benchmark审核包；source、输入、上下文的hash与区间收据齐全。
- 远端review run：research/runs/av-data-review-20261015-v1/，最终train-review-v5、benchmark-review-v1；旧版本保留。

本轮未提交新GPU任务。API出题、真实工具轨迹、silver自动审核、CPU训练兼容、人工语义gold与训练收益分别计数。

## 下一步

先把当前2K生产完成并核查case；独立来源继续积累100题benchmark pilot、再到200dev/1Ktest。预留来源尚未自动全部出题，不把reservation当完成题量。将人审反馈用于修正场景/模态/搜索必要性分层与下一版出题要求；冻结数据后依序补500 answer-only、prompted base、2K paired训练与full Thinker性能实验，每训完立即评测。最终训练数据目标仍为20K不同成功问题，15K–20K来自10/12原目标；paired answer-only不重复计作新问题。

## Full Thinker CPU meta 预检

2026-10-15T07:43:47.506032+00:00，在原trainer的_setup_full_tuning与真实pinned模型config上构建meta tensors：语言部分7,615,616,512参数可训练，visual/audio tower与projector共1,316,197,376参数冻结，模型中无Talker。CPU断言通过；没有加载真实权重、没有CUDA或optimizer update，实际梯度、保存重载和显存还须GPU smoke。证据与可重跑脚本见records/2026-10-15-base-model-and-data/full-thinker-meta-preflight.json、preflight_full_thinker.py。
