# AV seed 独立语义审核与标注说明

日期：2026-10-09。适用于 Phase 1 的 10 个 draft_unreviewed seed。所有表单都是私有标注工作材料，不进入 actor 的 corpus 或 TaskPublic。

当前完成的是公开原始文件、metadata、描述页版本、hash、ffprobe 与非静音信号检查。它们证明文件与处理身份，不证明自然性、声音内容、同期声、音频必要性、声源或事件关系。当前没有人工签名、gold 或已确认独立事件簇。A/B 表由两名真实审核者分别填写；生成器不得填写人的身份、结论或审核时间。

## 1. 独立流程

1. 管理者分别交付相同 task、冻结 corpus/registry、原始媒体和本说明。A/B 在第一次提交前不能看对方答案或自动 teacher 的建议答案。
2. 审核者先做模态控制的独立记录，再看完整 raw AV 校验。若已看过完整 AV，不能把之后的遮蔽回答称为盲态必要性验证；勾选 prior_exposure 并转交没有暴露的审核者，或只保留为非盲诊断。
3. 按 metadata-only → ASR-only → ASR+frames → raw AV 顺序记录每个目标项的可裁决性、证据与缺口。逐级保留原记录，不回填早先的判断。没有真实 ASR/frames 时记 unavailable，不能写“已评估”。
4. 完整回放原文件及必要区间，检查自然性、编辑/音乐风险、可见/可听事实、时钟和 provenance。
5. 对问题中可能预设的声源、实体、事件或因果逐项检查。题干预设无依据时改写或拒绝；候选问题本身不是事实。
6. 提交 A/B 私有表。第三方裁决者再比较分歧、重看原证据并填写裁决表。A/B 一致不是自动真值；关键证据仍必须可核验。
7. 管理者验证版本、区间、quote、AND/OR evidence package 和权限隔离，再决定接受、修题后重标、只保留 exploration 或拒绝。未完成记录不导出 ClaimGold。

## 2. 模态控制和音频必要性

| 条件 | 允许的内容 | 必须隐藏的内容 |
|---|---|---|
| metadata-only | 所有真实公开标题/描述、网页原文与出处 | 原始音轨、视频帧、ASR、OCR、声学摘要、指纹/embedding 返回 |
| ASR-only | 上述 metadata + 真实词级转录及其文字检索 | 音轨、帧、非语音声学标签/摘要，例如 [applause]、声学指纹/embedding |
| ASR+frames | 上述 ASR + 所有相关视觉帧/视觉内容，视觉预算与 raw AV 对照公平 | 原始音轨及所有非语言音频派生信息 |
| raw AV | 原始同步声画及共同工具返回 | 私有 gold、未来观察、对方表单与 teacher 答案 |

“ASR-only”这里指没有音轨和视觉的转录条件，同时保留真实公开 metadata；不能通过删除公开标题制造 audio 优势。每种条件都记录所用输入 manifest/hash。ASR 本身可能错，需记录版本和不确定片段。条件缺少输入时填 unavailable，不猜测替代。

每个目标项最终可多选 speech-sufficient、non-speech-acoustic、joint-AV-binding、visual/text-sufficient-control，并解释为何充分或不足。raw AV 模型答对／ASR 模型答错只是系统结果，不能定义任务本质必要性。非静音 RMS 只证明有信号，也可能是噪声、配乐或配音。若完整 AV 不足以确定声源，仍保留 unknown。

## 3. 可裁决目标、观察与绑定

先写清楚 proposition 的对象、量词、时间范围和限定词。区分“同 URL”“同 uploader”“同一录制”“同一次事件”“同实体”“同类操作”“独立观察”。

Direct observation 只写可直接听/看的事实及原始区间。例如“3.1–3.5 s 有一次短促撞击，画面同段出现接触动作”。“撞击由该对象造成”是独立 binding；“接触导致故障”是物理因果命题，需要额外证据。相邻掌声不证明观众动机，相似音色不证明具体物种或身份，画中对象不自动等于声源。

Source assertion 保存“来源声称什么”，不自动变成所声称事实。External fact 需要网页原文与版本。来源标题/作者描述不能替代原始媒体语义审核。

status 只对明确 proposition 填 supported / contradicted / undetermined。expected_decision 可以表达同次事件、不同事件、同类操作或 unknown。两字段分别填写，不用“答案不确定”混写。

## 4. 引用、时钟和证据包

- 媒体引用：source_id、asset_sha256、原始 source-local [start_s,end_s)，0 ≤ start < end ≤ duration。用原始公共播放轴与 timestamp_origin；禁止用 OneSearch grid index 当原始秒数。
- 网页引用：source_id、版本 hash、exact quote 与 start_char/end_char，必须匹配冻结 body_text。模型摘要不能当网页原文。版本变化需新 source/version，不覆写旧证据。
- 同一个可接受证据包里所有必要证据是 AND；不同有效包是 OR。例：包一需要片段 A 和出处 B，包二可用另一组 C 和 D。若仅列 OR 单片但结论需要跨源组合，则证据不充分。
- 正确答案配错误媒体、错误区间或错误 quote 不合格。允许不同来源构成等价证据包，由裁决者记录并对所有系统一致接受。
- unknown / undetermined 必须限定 within_snapshot，引用已经检查的相容/冲突材料，说明缺失的声源身份、对应线索或原始出处，以及什么新增证据可区分。没搜到不等于不存在。

## 5. 来源谱系与独立性

原始录制与其 Commons 描述页/API 是同资产的文件和 publisher description，不是两次独立观察。多个 URL、不同 uploader 或不同文件 hash 也不自动独立。共享音轨只能提示音轨关系，不能单独证明同视频、同事件或造假。

relation_status 允许 unknown；有可审核 basis_refs 才决定 same / different / independent / dependent。记录级复制与事件级相关是不同关系。独立性尚未确认时，不把它算成额外独立支持。两个不同录制可以观察同一事件；两条不相同视频可以复用音轨。

## 6. 当前 pilot 的专门风险

- 三类 seed 数量为技术 4、现场/表演 4、环境/动物 2，不能视为均衡分层或现实任务比例。技术类有三个 steam-engine 标题，相关性与多样性待审核；不按文件数宣称独立事件。
- Cardio Drumming 来源是 JTV Studios/Vimeo。应优先核对剪辑、配乐、同期声和真实可核验的动作；“有 drumming 标题和 audio stream”不证明原始打击声。
- 一项 Holiday Lights/frog jumping sequence 候选已因公开 metadata concern 从 canonical 10 seed 排除，原资产保留。这个筛查不是人类语义结论。
- 动物标题可能使 species/identity 从 metadata 可得；raw AV 不应被要求重证已公开标题，也不能把标题当实际声源。
- 原始文件加来源描述页只够做单 seed/出处审核。跨源深研究 gold 还需要可核验的相关录制、自然干扰、原始出处和争议关系；当前不保证已具备这些。

## 7. 验收与停门槛

每项记录真实耗时、可访问性、问题是否自然、目标项可裁决性及不确定性。首批至少 8/10 案形成公共证据可裁决包，100% 关键证据有 hash+合法区间/quote，才能讨论 G0。没有达到就报告实际产率，先改问题/来源；不能用 teacher 合成答案补足。

若重大分歧超过约20%，先修定义并重标；它是 pilot QC 门槛，不是正式统计可靠性。缺 ASR/frames 模态控制、缺原始语义审查或缺独立性证据时分别标记 pending。Phase 1 seeds、通过格式验证的记录、A/B 已提交记录、裁决接受 gold 和独立事件簇必须分别计数。
