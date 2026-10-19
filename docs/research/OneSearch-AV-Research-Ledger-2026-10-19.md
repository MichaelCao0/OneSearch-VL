
# 2026-10-19 数据格式与Teacher监督核查

正式2K绑定原ShareGPT轨迹，真实video+同源16kHz audio、全assistant轮监督；1903 Flash/97 Pro，lineage均gemini_native。带AV复审主线为same-model separate-call。2K中884题inspect、608题visit，audio_required冻结字段全pending，尚无系统音频依赖标签。已固定训练集64题/64录制供强teacher同预算对照，未调用新模型、未新增训练题。实际原行、teacher receipt、mask代码与新方案见[OneSearch-AV-Training-Format-and-Senior-Omni-Supervision-2026-10-19.md](OneSearch-AV-Training-Format-and-Senior-Omni-Supervision-2026-10-19.md)。

## 与OneSearch-VL组合构题对齐审查

复用agent loop/SFT不等于复现VGEG构题。实际2K：1479单字段、1557单次搜索、1917单support文档。当前要求答案逐字出现在publisher_text，限制派生答案与多事实组合；固定30秒切片也未按多锚点选择。已形成500新题/拟80%组合的修正方案，新组合数据尚未生成，未改冻结作业。详细依据见[OneSearch-AV-Data-Engine-Alignment-and-Revision-2026-10-19.md](OneSearch-AV-Data-Engine-Alignment-and-Revision-2026-10-19.md)。


## Coder 4 接管、audio token 修复与旧 checkpoint 清理

记录时间：2026-10-19 02:45:24 CST。

四组 full Thinker 训练均完成并通过检查点验证：trajectory500 / answer500 各 250 更新，trajectory2000 / answer2000 各 1,000 更新。用户指定操作环境为 coder ssh 4，主仓库仍为 /home/jovyan/OneSearch-VL；GPU 使用 kjob，数据生产使用 CPU。

旧 OmniRunner 混用 flat use_audio_in_video/fps/do_sample_frames 与 videos_kwargs，pinned ProcessorMixin 因此忽略 flat video 选项。真实 12 秒原声案例解出音频特征但仅有 0 个 audio tokens；修复后为 300 个，视频均为 1,440 tokens。已增加 token/feature 长度一致性检查，22 项测试通过。旧音频消融保留原始产物并标记失效，不能用于音频因果解释；异 FPS 多轮另有显式展开，不能概括为所有历史原声均没有音频。

修复后的 GPU smoke 使用原 2K trajectory checkpoint，在 4 道 dev 题执行原声/静音共 8 条 rollout。kjob slurm-profile-slurm-n2bkr 已 Complete，每轮 token 绑定均通过，峰值 allocated 显存 21.53 GiB。这验证了实际执行，性能增长仍须正式评测。

正式修复重评目录：research/runs/av-corrected-audio-evaluation-20261019-v1。五模型 base / trajectory500 / answer500 / trajectory2000 / answer2000 复用已完成检查点，同 56 题、8 轮/512 new tokens、原声和全程去音轨，先 trajectory2000，每次最多 1 GPU。最多 5 个 2 小时 kjob（10 GPUh），无新增训练。2K 阶段与五模型联合盲评各 56 个新评分槽位，新增上限 112；连同旧 224 不超过原 336；失败和 unknown 不重放。

按用户要求删除两份旧 smoke：av-full-thinker-smoke-20261016-v1/checkpoint、av-full-thinker-2k-resource-20261018-v1/checkpoint，共 275.008 GiB 文件载荷。删除后这两份 smoke 原训练状态不可恢复；配置、状态小文件、清单和验证记录归档在 research/runs/av-checkpoint-cleanup-20261019-v1。四组正式推理模型及续训状态、基座和数据保留。共享盘目录 du 与逐文件统计不同，空间变化可能异步，删除载荷与全盘读数分别报告。


## 修复后的五模型重评完成与配对结果

记录时间：2026-10-19T03:41:42.262311+08:00。

五个单GPU kjob 全部 Complete（实际 UID 已复核），560 条真实 rollout 执行错误为 0。原 trajectory2000 job 按 UID 接续；其余四模型在原五GPU/10GPUh总预算内并行，未新增训练；job wall GPUh 合计约 1.836（含启动）。281 项 source/config hash 一致，完整逐轮 token/feature/GPU 交付核查通过，静音音轨去除且视频 packets 保持一致。分析模块 8 项测试通过。

联合盲评 56 题：52 completed、4 unknown（1 HTTP503、3 HTTP200输出校验未通过），十条件共同使用相同52题，不重放收费失败调用。联合口径：500 trajectory 原声33/52、静音28/52；2K trajectory 原声36/52、静音34/52；500 answer-only 为2/52与1/52，2K answer-only为4/52与5/52；base为0/52与1/52。

录制家族配对bootstrap：原声2K−500 +5.8 pp，95% CI[-8.0,+19.2]；静音 +11.5 pp，CI[-3.6,+26.8]；500原声−静音 +9.6 pp，CI[-1.8,+20.4]；2K +3.8 pp，CI[-7.7,+16.0]。500严格包含于2K，但同2epoch分别250/1000更新，非等计算量数据因果对照。

相同52题和答案的2K评分稳定性：双候选原声33、静音34；联合原声36、静音34，有3个原声答案成功标签变化。另发现同系列错误视频绑定而答案字段相同的案例，主分不作事后调整，22案例和原始回执完整保留。当前清楚的收益是trajectory SFT的工具任务学习；规模与音频收益仍需更大、按音频贡献分层的评测与更稳裁决。下一版优先已规划的500组合/音频线索任务，先验证64题；本轮没有追加API数据生产、训练或在线重评。

完整结果：[音频修复后重评与2K对照](OneSearch-AV-Corrected-Audio-and-2K-Evaluation-2026-10-19.md)；计数、CI与执行收据摘要见 records/2026-10-19-corrected-audio-evaluation-v1/evidence-summary.json。原始私有回执、逐题轨迹、训练模型均保留，无 Git push。


## 倒挂解释与当前500／2K评测泄露复核

2026-10-19T10:55:42.519419+08:00。

对实际SFT行/lineage/训练成员SHA复核：56题相对当前500和2K的task ID、规范化题面、登记家族、媒体SHA、目标/支持URL及完整页标题重叠均0。录制隔离不等于实体隔离；例如白鲸学名和贝多芬可跨录制出现。未经人审的语义近重复、预训练污染仍未知。40题heldout来自旧silver候选的32家族；16题dev来自8家族，dev检索库仅10公开URL。56题均audio_required pending。应将本轮成绩定位为开发/诊断套件，而非封存正式test。

同52题、同答案的2K双候选评分33原声/34静音，联合评分36/34；3个原声标签变化可改变正负。规模与模态交互(2K_AV−500_AV)−(2K_V−500_V)=−5.8pp，家族bootstrap95%[-19.2,+7.8]pp，不能据此归因于语音。语音/音乐/环境声尚未分层。

新增只读CPU审计及说明：[倒挂解释、声音收益与评测泄露审计](OneSearch-AV-Audio-Inversion-and-Evaluation-Leakage-Audit-2026-10-19.md)。完整逐题记录research/runs/av-evaluation-leakage-audit-20261019-v1/audit-v2.private.json，保留v1与词边界修正记录。没有新模型/API请求、GPU任务或训练数据修改，未push。


## Full Search/Fetch preparation and request budget

Search need_content=True now reaches the actor observation; Fetch remains enabled. Added exact prompt-file support. 17 tests passed; 96 conditions CPU-prepared, no GPU submitted, no new API requests. Six arms have 128 combined Search/Fetch slots each: 768 total, at most CNY3.84 at 5/1000. Data production budget assumes 8-15 requests/candidate and 50% unique-question retention, to be measured on the initial candidates. Full 21700-question construction plus one 3-model/2-modality benchmark round: central estimate 553200 requests/CNY2766, with 20% margin CNY3319. Gemini/GPU/human/storage costs excluded. See [full service and budget](OneSearch-AV-Full-Online-Service-and-Budget-2026-10-19.md).

## 在线构题实现和内容错配诊断

新增五阶段在线数据引擎、组合图检查及审核/导出代码；Search/Fetch 已实际执行。兼容 video_url 输入出现实际视频与描述错配，运行已隔离，新增合格组合训练样本为 0。详细见 [实现与诊断](OneSearch-AV-Online-Compositional-Engine-and-Transport-Diagnostic-2026-10-19.md)。本次用户明确授权 push 代码到自己的 fork。
