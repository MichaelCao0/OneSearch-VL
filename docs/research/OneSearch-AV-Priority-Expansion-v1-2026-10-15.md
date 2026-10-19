# OneSearch-AV：用户审核后的优先扩产（2026-10-15）

用户已确认音乐、赛事和现场活动题型适合作为扩产重点。当前任务沿用 OneSearch-VL 的音视频搜索问答范式：从真实音视频辨认对象或事件，再搜索外部属性。用户确认的是题型方向；各题事实、答案和区间仍需核实。

## 执行改动

1. 候选采样目标：音乐 30%、赛事 30%、现场活动 20%、其他 20%。69 个新查询使用相应顺序；出题队列也按相同比例调度。下载查询和元数据只用于采样，不作为已核实内容标签；供应不足时自然回退，实际比例另记。
2. 已完成的 295 段素材直接复用。旧 v5 在采集阶段安全移交，没有中断付费请求，源媒体和失败记录保留。后续采集每批上限 250 段，不再等待整批 1,000 段才开始出题。
3. 首批最多预留 60 个新的 benchmark 录制家族，后续每批最多预留 50 个。先排除历史出题付费尝试、训练轨迹实际观察到的 URL 和媒体 hash，再预留家族，之后才进入训练出题。benchmark 多批次 worker 总出题请求上限 160，不会对每个 cohort 重复给 160 的额度。
4. author prompt v3 强化正向任务模式：片段中的实体确实可见/可听，问题不泄露身份线索，外部属性具有可检索连接；音乐作品、创作者、赛事身份等优先。prompt 已更改不代表质量收益已验证。
5. 原生 AV 出题、真实搜索轨迹、silver 审核和原 trainer CPU 媒体/标签检查继续执行。训练数据目标仍先到 2K，不把下载、出题或导出数量算作合格训练数量。

## 本轮修复和验证

初次 v6 在付费调用之前因 33 行稀疏 reservation 缺失 public_metadata 停止。来源库存函数要求完整发布者元数据。v7 按 source_id 和媒体 sha256 精确回连已有源记录补全，不编造字段；64 行原有元数据保持原值。后续新增 reservation 同时保存源元数据。v6 的失败证据保留，没有重放付费请求。

21 项相关测试通过：优先调度 5、benchmark 家族/实际训练观察隔离和多批次选择 9、native authoring 与失败不重放 4、metadata 精确回连 3。真实生产另行检查 PID、启动 ticks、boot ID、冻结源码和阶段产物。

## 计数和路径

启动时 904 条不同训练问题完成 silver 保留和 CPU 检查，2K 是阶段目标。之前 19 条 benchmark dev 问题候选来自 14 个录制；本次预留和新生成的候选另记。human gold 仍为 0。当前没有新增 GPU 训练结果。

- 主代码：coder ssh 3:/home/jovyan/OneSearch-VL
- 生产：research/runs/av-data-production-20261015-v7
- benchmark：research/runs/av-reserved-benchmark-20261015-v4
- 调度：research/av_train/source_priority.py
- 预留元数据回连：research/av_train/reservation_metadata.py
- benchmark 多批次消费：新 benchmark run 的 pipeline.py；基础选择和出题复用 research/av_train/reserved_benchmark.py

训练主线保持 Qwen2.5-Omni-7B、原 agent loop 和 SFT/src/train.py。新数据冻结后做同题 prompted base、answer-only、搜索轨迹 SFT 的匹配预算评测，并以任务成功率验证收益；native/muted 对照用于量化声音带来的增益。

## 16:50 最新核验

2,699/2,699 条素材库存分组已完成；从复用的 295 段素材中，60 个新 benchmark 录制家族已实际完成预留。训练与 benchmark worker 均核验存活，已进入原生 teacher 媒体准备；冻结源码无变化。此时新增出题请求仍为 0，合格训练问题仍为 904。
