# 原生 AV 数据质量抽查 v1

时间：2026-10-15T07:59:09.957644+00:00。24个已展示录制，一次实际AV审查/案例，4并发，21完成、3失败不重试。自动silver，不是human gold。

## 分层结论

主搜索集自动建议：{'keep': 16, 'unknown': 3, 'control': 5}。训练/benchmark分别：{'train': {'keep': 9, 'unknown': 1}, 'dev': {'keep': 7, 'unknown': 2, 'control': 5}}。

原审核模型20次keep、1次control，但其中5次search_requirement=perception_only。主搜索资格规则优先于泛化keep：纯感知题进入control；音画不匹配/实例链接无支持进入reject；答案支持不足/题面泄露线索进入revise。不能把21完成误报为21合格搜索题。

完成项的实际内容建议分布：music_performance5、machine_transport8、animal_bird5、speech_lecture3。模态角色建议visual_primary12、joint7、audio_primary2。这些是teacher看原声后的推断，尚无真实静音/ASR遮蔽测试，不证明音频必要性。

## 对后续数据管线的修改

1. 采集类别只用作采集provenance；实际内容分类采用原生AV审核sidecar，人审确认后用于正式覆盖统计。
2. 纯物种/犬种/人物识别保留为感知控制，主搜索题应要求可从素材定位的外部属性或具体来源实例；避免随意隐藏地点日期。
3. 为声音/声画联合任务补充有区分力的自然声音来源，单列语音文本、非语音声音与视觉控制；不强行把所有背景音描述成必要线索。
4. 新reserved benchmark生产使用上述更明确的主搜索出题约束。运行中v5训练生产保留源码freeze；抽查sidecar不覆盖已有轨迹或标签。
5. 同一教师另一次调用仍可能过于宽松，正式benchmark使用独立人审及模态遮蔽对照。

## 逐题意见

| split | 题目 | 采集类别 | 实际内容建议 | 主搜索建议 | 依据 |
|---|---|---|---|---|---|
| train | 视频中展示传统民俗舞蹈的舞者属于印度的哪一部族？ | event | music_performance | keep | 服饰与仪式特征明确，结合视觉线索可检索对应部族，材料支持完备。 |
| train | 视频中展示了哪艘船的动力系统？该船主要位于德国的哪个州？ | machine | machine_transport | keep | 视频中展示了船只机械与船名标识，外查所在联邦州线索明确，发布者信息支持完备。 |
| train | 视频中在草地与路面间觅食的鸟类学名是什么？其在日语中被称作什么？ | bird | animal_bird | keep | 视频中鸟类特征清晰，问题考查物种学名及日文名称，信息对应准确且来源可考。 |
| dev | 视频中鸣笛驶离站台的这趟列车是从哪座城市的车站发出的？ | machine | machine_transport | keep | 可通过机车编号ЭП1М-716及画面检索确认列车出发地伊热夫斯克。 |
| train | 视频中摩托车队穿行于高山雪壁间的这段路线属于哪条公路？位于哪个国家？ | machine | machine_transport | keep | 视频线索充足，结合视觉画面可检索定位原片及公路背景，发布者信息支持完备。 |
| dev | 视频拍摄前一天发生的哪种恶劣天气现象导致当地铁路接触网严重结冰？ | machine | unknown | unknown | ValueError |
| dev | 视频中出镜接受采访的美国著名演员是谁？ | bird | speech_lecture | control | 视频中人物特征清晰，问答匹配且来源信息支持名人身份识别。 |
| train | 视频中发表演讲的政治人物是谁？该演讲发生于美国的哪一个州？ | animal | speech_lecture | keep | 音视频线索清晰，结合人脸与演讲内容可检索定位人物身份及所属州。 |
| dev | 视频中演奏的乐曲是由哪位作曲家创作的？ | music | music_performance | keep | 音画匹配，需通过听辨琴曲旋律检索对应作曲家，信息与来源一致。 |
| dev | 视频中在车内不停嚎叫的宠物犬属于哪一犬种？ | animal | animal_bird | control | 犬种特征清晰，仅凭视听感知即可识别，适合作为对照题。 |
| train | 视频中海狮聚集的这处知名洞穴景点位于美国的哪个州？ | animal | animal_bird | keep | 音画清晰呈现海狮洞场景，设问地理归属明确且来源有据。 |
| dev | 视频中迎面鸣笛驶来的双节货运列车是由哪一型号的电力机车牵引的？ | machine | machine_transport | control | 机车型号主要依赖视觉外观特征识别，声音起辅助验证作用，整体符合视听理解要求。 |
| train | 视频中演奏的小提琴曲曲名是什么？演奏者是谁？ | music | music_performance | keep | 音画质清晰且契合提问，曲名与演奏者均有元数据支持。 |
| train | 视频中在讲台前授课的教授是谁？该门课程的名称是什么？ | music | speech_lecture | keep | 音画线索明确，可有效检索定位该耶鲁公开课的授课教授与课程名称，参考信息完整匹配。 |
| dev | 视频中在枝桠间鸣叫的鸟属于哪一物种（写出其英文名或学名）？ | animal | animal_bird | control | 视频与鸣声特征清晰对应黑斑蝗莺，问题明确且来源匹配良好。 |
| train | 视频中围坐在餐桌旁的老人合唱的民歌叫什么？演唱者来自哪个村庄？ | bird | music_performance | keep | 音视频与问答内容匹配，曲名与村庄信息均有发布者元数据支持。 |
| dev | 视频中列车鸣笛驶过的这座铁路高架桥通常称为什么？ | machine | machine_transport | keep | 声画契合，视听线索明确，所问高架桥可通过地标检索验证且发布者信息支持。 |
| dev | 视频中出现的这种鸟类的学名是什么？ | bird | animal_bird | control | 视频清晰展示鸟类外形特征，结合视觉即可识别物种学名，信息与发布者元数据一致。 |
| train | 视频中运转的人偶机械装置所属的钟楼叫什么？它面向哪座广场？ | event | unknown | unknown | ValueError |
| dev | 视频中鸣响标志性音乐喇叭的红色列车所途经的地下车站位于日本哪个县？ | machine | machine_transport | keep | 音视频线索清晰，结合车型与音乐喇叭可检索定位车站所在县，材料支持完整。 |
| dev | 视频中边走边拉奏提琴的演奏者是谁？ | music | music_performance | keep | 音画内容与提问匹配，发布者元数据支持答案，检索路径明确。 |
| dev | 视频中这列机车驶离的火车站是什么车站？ | machine | unknown | unknown | ValueError |
| dev | 视频中鸣笛驶出的客运列车由哪种型号的机车牵引？ | machine | machine_transport | keep | 音画契合，可通过机车外观特征检索定位具体型号，来源支持充分。 |
| dev | 视频中出现的这台机车的具体型号及编号是什么？ | machine | machine_transport | keep | 视频中机车外观特征明确，需结合图像检索特定机车编号，设问合理。 |

可播放审核：train-review-v6与benchmark-review-v2，自动意见默认折叠，不修改人审选择。旧页面和原始API结果保留。
