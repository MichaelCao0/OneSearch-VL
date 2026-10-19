# OneSearch-AV：100录制内容抽查与模态审核包 v2

更新：2026-10-15T08:26:45.136180+00:00。本轮实际完成独立100录制原生AV自动抽查，并生成训练14例和benchmark14例的模态审核页。实际执行coder ssh 3；human gold仍为0。

## 1. 固定抽样与实际结果

从当时904个CPU合格训练问题中，排除已付费抽查的24个录制及其媒体别名，得到884个可选题；按animal/bird/event/machine/music采集query各取20个不同录制，使用4worker，最多100请求。没有按student表现或本轮judge结果选择这100例。91例完成、9例失败；失败不重试。

统一规则的逐题建议为 **66 keep / 15 perception-control / 4 revise / 6 reject / 9 unknown**。这是采集query分层样本的自动silver意见，不等于整个904池的人工精度；原训练池没有自动删除或改写。

| 采集query | 完成 | keep | control | revise | reject | unknown |
|---|---:|---:|---:|---:|---:|---:|
| animal | 17/20 | 12 | 3 | 1 | 1 | 3 |
| bird | 17/20 | 8 | 9 | 0 | 0 | 3 |
| event | 19/20 | 14 | 1 | 1 | 3 | 1 |
| machine | 19/20 | 15 | 1 | 2 | 1 | 1 |
| music | 19/20 | 17 | 1 | 0 | 1 | 1 |

真实内容建议为动物/鸟23、音乐/表演21、机械/交通21、环境/事件9、语音/讲座8、其他9。它与采集query有明显差异；正式data card应按实际内容重新标注。自动模态角色为visual_primary62、joint18、audio_primary6、speech_text_primary5；这是模型判断，须用人审与真正模态消融验证。

## 2. 对数据构造的具体反馈

- **查询线索泄露**：如把机车编号06-018，或事件年份/联赛背景写入题干；题干应简短，具体可辨别细节由模型从AV提取。
- **短片遗漏所问对象**：7秒列车片段中后部机车没有出现，不能仅凭上传者全文分类标签出题。当前authoring固定从录制约20%处取最多30秒；该策略真实存在，后续应让出题确认所有所问实体出现在输入，必要时选更合适区间。
- **缺乏可辨别路径的隐藏地点**：普通羊群或瀑布被问具体国家/郡，须验证从片段线索到来源的可行路径。难度高本身不构成删除理由；reject是待人审建议，不以teacher熟悉程度直接决定正式benchmark。
- **感知识别与直接播报答案**：熟悉鸟种、机车型号、旁白已说出答案应单列诊断。识别题可通过搜索解决；是否属于主搜索集合应结合题意和人工判断，不能把模型能回忆名称一概认定为无价值数据。
- **值得扩产的候选**：音乐作品/背景配乐、赛事现场解说、采访项目、声画联合技术活动，已有具体原生AV案例。下一cohort优先补这些题型，并保持视觉/动物等覆盖。

现阶段不更改运行中的v5源码和数据，不把自动控制/拒绝建议替换人审。下一版构造需比较修正前后同预算的可用率、模态分布及下游搜索成功率。

## 3. 可以直接审核的14例训练样例

这是有目的选择的检查集：4个joint、3个audio-primary、3个visual-primary的keep建议，加2个reject、1个revise、1个control。它用于检查正负案例和规则边界，不用于估计总体比例。

| 页面题号 | 实际问题 | 自动实际类别 | 自动模态 | 主搜索建议 |
|---|---|---|---|---|
| 1 | 视频中报道的这场高中男子冰球加时赛发生在哪一年？ | mixed_other | joint | keep |
| 2 | 视频中担任钢琴独奏的年轻演奏家是谁？ | music_performance | joint | keep |
| 3 | 这段特技表演视频记录的是哪一场摩托车展会活动？ | machine_transport | joint | keep |
| 4 | 视频中解析出的这种从银河系中心延伸出的天文结构称为什么？它是基于哪台空间望远镜的数据被发现的？ | speech_lecture | joint | keep |
| 5 | 视频中演奏者吹奏的是哪首练习曲作品？ | music_performance | audio_primary | keep |
| 6 | 视频中使用的背景音乐曲名是什么？其创作者是谁？ | speech_lecture | audio_primary | keep |
| 7 | 视频中演奏的这首乐曲所属的乐队名称及曲目名称分别是什么？ | music_performance | audio_primary | keep |
| 8 | 视频中展示的这台早期蒸汽机车复制品是以哪台历史名车为原型？其目前陈列于哪座交通博物馆？ | machine_transport | visual_primary | keep |
| 9 | 视频中正在接受采访并发表观点的学者是谁？ | speech_lecture | visual_primary | keep |
| 10 | 这段记录鸟类发声机制的X射线研究最初发表于哪本学术期刊？ | animal_bird | visual_primary | keep |
| 11 | 这段视频中的羊群拍摄于哪个国家？ | animal_bird | visual_primary | reject |
| 12 | 视频中展现的这一瀑布景观位于肯尼亚的哪个郡？ | environment_event | visual_primary | reject |
| 13 | 视频中记录了2015年伊朗足球联赛中大量球迷涌入球场内草坪的混乱场面，该场比赛是在哪座体育场进行的？ | environment_event | visual_primary | revise |
| 14 | 视频中出现的这只鸟属于什么物种？请写出其英文常用名及拉丁学名。 | animal_bird | visual_primary | control |

Benchmark另保留之前14个录制dev候选，增加相同模态入口；题目、参考和人审draft schema保持不变，候选不自动升级为gold。

## 4. 模态入口和实际验证

每题提供原声画面、仅画面（文件已移除音轨）、只听声音（原音轨提取为16kHz mono WAV）、60–90秒来源上下文（短录制按实际长度）。只听时隐藏视频和静帧，切换播放会停止其他播放器。自动意见默认折叠，便于先独立判断。

28题的56个派生媒体已验证：静音文件无音轨、所有原视频压缩数据及逐包PTS/DTS/duration完全一致、音频文件无画面、原始输入hash未变。一个原片音轨27.6875秒、视频27.239111秒；移除音轨后的文件保留全部27.239111秒画面，页面记录各自实际时长，不补帧或截短原画面。最初版本的容器总时长断言不适用于原声画尾部不齐，失败版本和诊断保留。

远端包：research/runs/av-modality-quality-review-20261015-v2/train-quality-review、benchmark-modality-review。源100例与原始API结果仍在research/runs/av-train-content-audit-100-20261015-v1/。当前receipt验证的是媒体和时间保持；浏览器实际播放另记。

## 5. 下一阶段与训练

维持一个Qwen2.5-Omni-7B基座，原agent loop、原SFT/src/train.py。持续完成2K合格数据与独立benchmark候选，人审检查代表案例；然后冻结数据，补strong prompted base、matched answer-only及full Thinker训练并立即评测。20K train、200dev/1Ktest仍是完整交付目标。本轮没有训练新权重或提交GPU任务。

配套records/2026-10-15-quality-audit-100/保存汇总、实际token使用（仅provider有返回的部分）、模态媒体验证。费用不从未知费率估造。

## 6. 浏览器交付验证与下一版配置

28题的4类媒体共112个HTTP Range请求通过206检查；实际浏览器原声播放到18.68秒、静音到24.04秒、仅音频完整到30秒，无媒体错误。benchmark第3题去音轨后27.239秒视频完整播放结束，逐包视频时间保持。只听模式确认隐藏视频与静帧，未修改任何人工标签。记录见browser-playback-validation.json与http-range-validation.json。

下一版出题prompt已准备在prompts/av-question-authoring-v3-20261015/：沿用原closed JSON，重点防止描述线索泄露、所问对象未在当前区间、缺乏可辨别路径的隐藏元数据；增加音乐/赛事/技术演示外部属性问题的正面模式。尚未用它调用API或改写v5；效果待对照。

最新执行snapshot：2026-10-15T08:34:00.274282+00:00，CPU合格904，新采可用录制285；生产live=True，benchmark等待live=True。当前先整批采集1000、再预留来源/出题，导致付费生成在采集期间空闲；后续缩小cohort到250并按比例预留50可缩短首批等待，需在新版本应用，避免改变正在运行的输入freeze。
