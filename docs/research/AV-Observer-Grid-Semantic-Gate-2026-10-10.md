# AV observer：时间网格与语义验收门限

本文件针对2026-10-10固定六片段诊断。第一篇仍聚焦多源audio-video search、取证与研究报告；不把用于generation的素材搜索并入本轮实验。

## 当前门限含义

- 模型加载、真实generation与严格JSON协议是基础设施检查。
- `audio_pts_v1`证明actual decoded/native-resampler PTS的数组放置与recorded coverage；无音频填零不能支持audio/joint事件。
- 独立审查发现官方video helper端点linspace取帧与processor scalar FPS网格不一致，需单独 `av_pts_grid_v1` 版本处理。
- nearest-PTS视频取帧只能把网格误差约束至原视频帧量化尺度，processor还会每两个frame合成temporal patch。任何精确语义事件边界均需要额外评价，不能由receipt替代。
- parse通过的内容仍是candidate observations。原始音视频、private reference、人类语义判断和source身份核验尚未形成G0通过证据。

## 时间网格验收

1. 固定六输入和原binding/media hashes保持不变；unsupported输入保留在分母。
2. 独立记actual audio sample coverage、decoded video PTS、选取indices、目标j/fps、实际PTS、frame/patch量化差及传入processor的fps。
3. synthetic同步audio/visual pulse验证1秒位置；25fps验证半帧量化误差披露；丢帧/VFR/noninteger sample count显式拒绝。
4. CPU真实processor只证明数据/网格契约；单卡GPU三次自然输入generation另记，不能等同pulse语义定位。

## 下一阶段八类probe

以相同基础source group构造探针，不向模型提供gold或类别提示：

| 类别 | 需要验证的行为 | 主要失效模式 |
|---|---|---|
| 音频独有线索 | 听到画面无法确定的事件 | 仅看画面猜测 |
| 视觉独有线索 | 识别无对应声音的视觉变化 | 把缺音频当不存在 |
| AV同步 | 区分音画对应关系 | 共现即因果 |
| AV错位 | 对人工错位保留不确定性 | 强制匹配声源 |
| 音频缺失 | filler不支持silence/event claims | 将填零当录到的静音 |
| 重复事件 | 找到正确实例与区间 | 返回任意同类片段 |
| 多源冲突 | 保留来源差异、时间与未定项 | 合并矛盾事实 |
| 信息不足 | 证据不足时abstain | 为填写报告编造 |

先做每类2个development probe（共16个），按source family分组固定split，另设8个冻结holdout；这是一项后续计划，不是已生成数据。人工reference记录可观察内容、时间容差、允许替代解释和不可确定项；不要求人审判断不能由素材支持的唯一声源或因果。合成干预和自然观察分别评价。

## 基线与预算

在完全相同工具次数、请求AV秒数、帧像素和输出token上限下比较：text/metadata-only、audio-only、video-only、naive joint AV、显式coverage/grid joint AV。先回答观察能力是否依赖真正AV证据，再接入OneSearch的inspect_av；不能以修好JSON宣称搜索方法贡献。

若16个development probe的协议、时钟和人工reference仍不稳定，停止扩大数据、做SFT/RL或宣称论文指标，定位最小失败路径。只有reference、来源、重复实例和多源证据核验通过G0，才推进首个多源search闭环；paper1的主要评价为evidence sufficiency、event-instance binding、conflict handling和budget-matched report support。生成论文继续保留为后续工作。
