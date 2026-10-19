# OneSearch-AV：实际训练格式、Teacher监督与升级建议

核查日期：北京时间2026-10-19。主仓库 `coder ssh 3:/home/jovyan/OneSearch-VL`。本次读取冻结2K实际训练行、formal train.yaml、原LLaMA-Factory label代码、teacher生成receipt与silver review实现；没有新模型API调用、GPU提交或训练数据修改。

## 实际训练数据

正式训练器绑定的数据：
`/home/jovyan/OneSearch-VL/research/runs/av-learning-2k-preparation-20261018-v1/clean2000/train-paired/trajectory.json`

2,000行，每行一个问题的完整真实执行轨迹；同题answer_only.json也有2,000行，后者只保留初始真实AV+最终答案，作为训练对照，不增加独立问题数。原ShareGPT格式字段为system/tools/conversations/images/videos/audios。

以下从第0行缩写而来：长提示词和工具正文省略、媒体路径改为显示别名，不能作为原行逐字复现。完整原行另存为actual-trajectory-example-search.json，逐项远端/本地SHA256核对通过。

```json
{
  "system": "音视频搜索规则与工具定义（长提示词省略）",
  "tools": "",
  "conversations": [
    {
      "from": "human",
      "value": "<video><audio>\nvideo_1: 30.000 seconds; audio track present. inspect_av uses seconds relative to the selected handle.\nQuestion: 这段执法记录仪记录的警员与少年打篮球互动发生在哪个公园？"
    },
    {
      "from": "gpt",
      "value": "<think>Search for PoliceActivity video of police officers playing basketball with kids.</think>\n<tool_call>{\"name\":\"text_search\",\"arguments\":{\"q\":\"PoliceActivity police officers playing basketball with kids park\",\"top_k\":5}}</tool_call>"
    },
    {
      "from": "human",
      "value": "<tool_response><observation>搜索实际返回的网页标题、URL、正文摘要（此处省略长正文）</observation></tool_response>"
    },
    {
      "from": "gpt",
      "value": "<think>Confirm the park name from the search results.</think>\n<response>这段执法记录仪记录的警员与少年打篮球互动发生在位于美国佛罗里达州可可市（Cocoa, Florida）的**乔·李·史密斯公园**（Joe Lee Smith Park）。</response>"
    }
  ],
  "images": [],
  "videos": [
    "input-video.mp4（显示用路径别名）"
  ],
  "audios": [
    "same-recording-audio.wav（显示用路径别名）"
  ]
}
```

human/gpt是ShareGPT角色标签；gpt不表示数据由GPT模型生成。工具真实返回通过human中的tool_response/observation写入历史。工具schema位于system提示词，所以此数据tools列为空字符串。

videos是真实本地视频路径；audios是同一视频真实音轨解码为单声道16kHz PCM的文件。<video><audio>占位符由qwen2_omni模板读取为真实模态特征。实际formal配置use_audio_in_video=true，原mm_plugin.py会将同一视频/音频配对构造时序输入。发生inspect_av时，工具返回新clip的视频与同源音轨，额外占位符写在该轮observation里，媒体按历史出现顺序追加至数组，并非仅把裁剪结果描述成文字。

第1行的真实样本包含：29.992秒列车AV → inspect_av(video_1,6,12) → inspect_av(clip_1,2,4.5) → 再检查其他区间 → 检索EF210 → 访问页面 → 答铁路线路/车次。原行actual-trajectory-example-inspect.json含5个视频与5个同源音轨。该例主要体现视觉识别与外部事实查找，不能据此证明音频必要。

## 监督哪些token

正式trainer配置核对为：template=qwen2_omni，train_on_prompt=false，mask_history=false。原SFT/src/llamafactory/data/processor/supervised.py将每轮source上下文label置IGNORE_INDEX，每轮assistant target保留label。因此模型学习每一轮助手的可见简短计划、工具名称/参数、继续或停止决策以及最终答案；用户问题、system、网页观察和媒体占位等作为输入上下文，不要求模型生成工具结果。

当前是teacher轨迹的序列级SFT。gold/reference、teacher身份、媒体hash、retention verdict等保存在private sidecar；dataset_info的训练列不包含这些字段，不把参考答案预先塞进actor初始输入。音视频tower/projector当前冻结，Full Thinker语言参数可训练。

## 当前已有的Teacher监督

按2K成员与lineage核对：

| Teacher | 保留轨迹数 |
| --- | ---: |
| gemini-3.8-flash | 1,903 |
| gemini-3.1-pro | 97 |
| 合计 | 2,000 |

2K lineage的provider全部为gemini_native。抽查generation receipt包含实际video/mp4媒体hash/bytes、模型版本、输入输出hash与生成token数；actor每轮接收实际AV及截至当时的真实工具返回。

主生产管线teacher_silver_review.py还有独立调用：同一teacher型号重新接收原始AV、问题、实际动作/观察和publisher参考，判断answer_fields_supported、media_matches_reference、queries_grounded、no_material_hallucination四项。它是带原AV的same-model separate-call自动silver复审，不是独立更强模型验证，也不是人工gold。

## 从实际2K看到的改进空间

| 实际字段/行为 | 问题数 |
| --- | ---: |
| 含真实音轨 | 2,000 |
| 调用过inspect_av | 884 |
| 实际包含额外视频输入 | 882 |
| 调用过text_search | 1,989 |
| 调用过visit_webpage | 608 |
| 无工具调用 | 8 |
| 冻结reference的audio_required仍为pending_content_review | 2,000 |

保留轨迹共2,717次text_search、1,394次inspect_av和624次visit_webpage。884是调用问题数，882是实际额外视频输入的问题数，不能混写为884条裁剪均成功。

全部带音轨不等于全部依赖声音。现有384条额外内容审核也保存在独立view，不能把audio_required字段未回填说成所有样本都未审核；准确缺口是尚无覆盖2K的、逐题可核验的音频必要性/证据时段标签。现有整体boolean复审也不能测出移除声音后是否仍能完成相同任务。

500 Full Thinker共同38题原声/静音同为18/38，与当前尚未建立音频独立收益的结论一致。更强teacher的任务是提升实际感知、搜索线索选择、工具决策与证据支撑，不能仅换一个模型名后假定产生音频收益。

## 建议的senior teacher实验

已从训练集固定选出64题、64个不同录制；按music/event/sports/other各16题抽样，分层是原query scenario hint，不是经核实的内容标签。对应senior-teacher64-selection.private.json。该文件仅用于准备，**新teacher尚未选择，实验尚未执行，当前未增加独立训练题数**。

1. 用当前Flash和候选强AV teacher接收相同匿名AV+问题，使用相同DeepSeek工具、turn/token/请求预算完成轨迹；actor不看参考答案或目标URL。按原声与物理去声条件分别执行；保持视频与问题不变。
2. 对每条轨迹增加可核验的监督：听到/看到的线索及输入相对区间、线索如何转成query、实际网页是否支持答案。先把这些作为审核sidecar，再仅将核实后的简短线索计划纳入assistant监督，避免把未验证声音猜测写成确定事实。
3. 另用独立模型带原AV盲审：线索是否存在、区间是否对、查询是否来自线索、答案是否确由返回证据支持。复审可以看reference；actor不能看。标记audio-primary/joint/visual-primary，保留视觉控制题，不强行把每题都标成需要声音。
4. 选出同任务上更好的teacher后，制作500题的current-teacher/strong-teacher同题训练对照，控制来源划分、训练更新并报告supervised/context tokens；训练后立即固定题评测，再扩到2K和20K。

候选模型需要先确认在现有endpoint可用、真实接收AV，并在相同问题/工具预算上更好。官方当前将Gemini 4 Argon定位为旗舰reasoning模型，并明确支持Video/Audio输入、function calling及structured outputs，可作为闭源候选；该gateway可用性本次未测试。[Gemini 4 Argon官方规格](https://ai.google.dev/gemini-api/docs/models/gemini-4-argon)

独立开源AV复核可考虑Qwen3-Omni-30B-A3B-Thinking，官方明确其支持音视频输入和文本输出；作为候选仍需在本项目任务中验证，而不是按总参数量直接判定强于现有teacher。[Qwen官方仓库](https://github.com/QwenLM/Qwen3-Omni)

## 代码与证据

- 导出：`/home/jovyan/OneSearch-VL/research/av_train/native_sft.py`
- 生成：`/home/jovyan/OneSearch-VL/research/av_train/teacher_rollouts.py`
- 带AV的自动复审：`/home/jovyan/OneSearch-VL/research/av_train/teacher_silver_review.py`
- 监督mask：`/home/jovyan/OneSearch-VL/SFT/src/llamafactory/data/processor/supervised.py`
- 正式训练配置：`/home/jovyan/OneSearch-VL/research/runs/av-full-thinker-learning-20261018-v2/trajectory2000/train.yaml`
- 远端审核产物：`/home/jovyan/OneSearch-VL/research/runs/av-training-format-audit-20261019-v1/`
- 本地证据：`reports/data-format-teacher-audit-2026-10-19/`，完整原行与64题选择在examples子目录；4项文件传输hash均一致。

保留现有2K数据与正在执行的训练，新的teacher标签/重生成使用新版本，不改写历史实验。
