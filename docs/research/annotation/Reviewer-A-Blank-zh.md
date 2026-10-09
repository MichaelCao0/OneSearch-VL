# 审核者 A：独立空白表

状态：blank_unreviewed；私有标注材料。不可读取审核者 B 结果，不填写为已完成 gold。

- task_id / task revision：____
- corpus_id / registry manifest/hash：____
- 真实审核者姓名或内部 ID：____
- 开始/提交时间及耗时：____
- prior_exposure（是否之前看过完整 AV/teacher 答案）：____

## Seed 审核

| source_id / hash | 原文件可回放？ | 自然录制/编辑/配乐/旁白/未知及证据 | 同期声可确认？ | 原始时钟/offset 问题 | 仅技术 receipt 已验证的项目 |
|---|---|---|---|---|---|
| ____ | ____ | ____ | ____ | ____ | ____ |

公开标题/描述有哪些 assertion，哪些不能视为观察：____

## 每个目标项重复填写

- claim_id / 原 target question：____
- proposition、对象、量词、时间范围、qualifiers：____
- 题干是否预设未验证事实；保留/改写/拒绝理由：____

| 条件 | 输入 manifest/hash/版本 | available/unavailable | 本条件判断/置信与限制 | 充分证据？ | 引用或仍缺什么 |
|---|---|---|---|---|---|
| metadata-only | ____ | ____ | ____ | ____ | ____ |
| ASR-only | ____ | ____ | ____ | ____ | ____ |
| ASR+frames | ____ | ____ | ____ | ____ | ____ |
| raw AV | ____ | ____ | ____ | ____ | ____ |

- speech-sufficient / non-speech-acoustic / joint-AV-binding / visual-text-control：____；必要性论证：____
- Direct observations（只写观察；每条附 source/hash/[start,end)）：____
- Source assertions / External facts（附 exact quote / char bounds /版本）：____
- expected_decision：____；status（supported/contradicted/undetermined）：____
- scope：within_snapshot；缺失证据及新增何种证据能区分：____
- 有效证据包 P1（必要 refs 全部 AND）：____
- 替代证据包 P2（与 P1 OR）：____
- 必要 bindings、状态、basis_refs、观察与声源/因果之间的缺口：____

## 来源关系

| 两个/多个 source_id | relation type（recording/event/independence/其他） | same/different/independent/dependent/unknown | basis_refs | 可撤销/待补证据 |
|---|---|---|---|---|
| ____ | ____ | ____ | ____ | ____ |

录制与其描述页不计为独立观察。未证实 independent 保留 unknown。

## 提交检查

- [ ] 每条关键引用匹配冻结 hash 与原始 clock/quote。
- [ ] 未以标题、RMS、模型回答或 co-occurrence 代替语义/声源/因果真值。
- [ ] 音频遮蔽同时隐藏非语音摘要、标签、指纹/embeddings。
- [ ] 未读取 B 的表或 teacher 答案；若暴露，已注明。
- [ ] blank/pending/unavailable 未伪装为已评估。
- [ ] 建议接受/改写重标/只留 exploration/拒绝：____；原因：____。

真实审核者提交确认：____（由真实审核者填写）
