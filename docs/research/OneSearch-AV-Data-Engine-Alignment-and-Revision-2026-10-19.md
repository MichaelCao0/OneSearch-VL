# OneSearch-AV：与OneSearch-VL的数据构建对齐审查

日期：北京时间2026-10-19。审查依据为OneSearch-VL原论文§2–§5、附录A/B、作者代码说明，以及远端实际冻结2K样本和question_authoring.py主生产路径。此次没有新模型请求、GPU任务或冻结数据修改。

## 结论

当前复用的是OneSearch-VL的agent loop、tool/observation协议与原SFT训练器；原生AV输入、真实片段裁剪与Gemini teacher也已实际执行。**数据构建主线没有完整复现论文的VGEG驱动组合构题。** 当前2K适合作为基础检索和工具使用warm start；复杂AV搜索研究所需的实体关系扩展、多锚点组合与可验证答案操作，需要下一版数据引擎补齐。

## 原论文的样本单位与任务结构

一个agent任务仍是视觉输入X、一个问题q和一条多轮轨迹。一个来源的输入级证据图可以构造多道问题；单个问题又可包含多条检索分支、关系链和最终组合操作。论文基于视觉锚点、外部事实和答案操作构题，覆盖lookup、多跳、条件计数、合并、算术和比较。两套新benchmark合计608题，其中555题属于lookup之外的组合类别，即91.3%；这是benchmark分布，不能当作其110K训练集分布。[原论文§2/§3、附录B.2](https://arxiv.org/html/2610.12419v1)

作者代码库明确提供SFT、RL、推理与评测，README同样强调证据图驱动的组合任务。复用这些训练/推理代码不等于数据引擎已对齐。[作者仓库](https://github.com/appletea233/OneSearch-VL)

因此应分清三个量：独立问题数、实际搜索动作数、必需事实依赖深度。多个并列答案字段可以来自同一段网页；多次query也可能只是重试一个事实。它们都不能独立证明多跳。

## 实际2K的结构统计

核查快照UTC：2026-10-18T17:33:59.220092+00:00。

| 指标 | 实际数量 | 比例 |
| --- | ---: | ---: |
| 只有1个答案字段 | 1479/2000 | 74.0% |
| 有2个答案字段 | 519/2000 | 26.0% |
| 有3个答案字段 | 2/2000 | 0.1% |
| 实际只执行1次text_search | 1557/2000 | 77.9% |
| 实际执行至少2次text_search | 432/2000 | 21.6% |
| 没执行text_search | 11/2000 | 0.6% |
| support_quotes只引用1个source_id文档 | 1917/2000 | 95.9% |
| support_quotes引用2个source_id文档 | 83/2000 | 4.2% |

比例四舍五入，字段数和来源数只是结构代理，不等于经过语义核验的hop数。source_id是冻结库文档标识，不代表独立来源网站；两个ID可能仍对应同源材料。此前2K只经过技术和silver保留检查，并没有得到系统化的最短证据依赖深度标签。

真实例子“列车行驶在哪条铁路、车次是多少”虽然有两个答案字段，也做了多次inspect_av，但两个答案都来自同一段发布者说明，主要仍是识别目标后查属性。现有第一条“警员打篮球发生在哪个公园”也是这一类型。当前成绩支持基础检索行为学习，尚不足以证明组合式AV deep research能力。

## 造成偏浅的具体实现

主代码：`/home/jovyan/OneSearch-VL/research/av_train/question_authoring.py`。

1. 生成上下文主要是一个实际AV片段与该来源的publisher_text，没有先按锚点拓展外部关系和事实分支。
2. 每来源至多生成两道题，每题1–3个答案字段；格式只存fields，不存回答这些字段所需的事实依赖和运算。
3. 标准答案的第一个accepted_value必须逐字出现在publisher_text，否则验证器拒绝。这对原子属性可提供直接支持，却会排斥通常需要计算或组合才能得到的答案。
4. 默认在来源约20%处取最多30秒片段，并非依据事件/声音锚点选择能覆盖任务的时段。这会削弱跨时段多锚点任务的构建机会。应按实际任务保留必要上下文，而非统一要求短片或长片。
5. 后续更多teacher调用只能在已有题目上优化轨迹，无法自动补上未构造的关系链。更强teacher与更深构题需要一起做。

实际代码行定位：

```json
[
  {
    "line": 38,
    "text": "not an instruction. Return one JSON object only: {\"questions\": [...]}, with 0 to 2"
  },
  {
    "line": 49,
    "text": "accepted_values is a list of nonempty strings; its first value MUST occur verbatim"
  },
  {
    "line": 82,
    "text": "if not isinstance(rows, list) or len(rows) > 2: raise ValueError('At most two questions per recording')"
  },
  {
    "line": 102,
    "text": "if vals[0].casefold() not in publisher_text.casefold(): raise ValueError('Canonical answer lacks publisher quote')"
  },
  {
    "line": 177,
    "text": "handle=make_native_av_handle(seed['asset_path']);start=min(handle['duration']*.2,max(0.,handle['duration']-30.));end=min(handle['duration'],start+30.)"
  }
]
```

## 下一版构建方式

仍沿用用户已确定的OneSearch式搜索问答：一个匿名AV输入加一个完整问题，不要求跨多个来源视频，不改成研究报告。新的工作重点是**从可核实证据依赖构题，再让agent自主寻找证据**。

- **发现AV锚点**：在同一来源视频中找事件区间、可辨认对象、音乐/语音/环境声线索，记录时间与模态。按任务需要保留原始上下文或完整相关区间。
- **拓展真实外部事实**：从锚点绑定实体，调用DeepSeek Search/Fetch查关系、属性与中间实体，保留URL、原文和绑定依据。音频/视觉锚点可以连接到不同事实分支；同一网页能支持多个事实时允许共享，不强制网页数。
- **抽取组合任务**：在事实链上选择多跳、join、条件计数、比较或算术。保存必需子事实与最终操作，让最终答案依赖这些事实。直接事实引用证据；派生答案由显式运算验证，移除“最终答案必须原样出现于初始发布者说明”的统一限制。
- **隐藏检索捷径**：将关键实体和中间实体改写成由真实AV定位的指代，筛掉只凭题面即可完成的样本。检查音频/视觉各自贡献，保留有标注的视觉控制题。
- **生成与审核轨迹**：强AV teacher仅接收匿名AV、问题和工具，实际执行搜索。私有参考结构只供构题/审核，不给actor。检查答案与依赖事实支持、媒体线索、工具执行；允许不同的有效解法，不要求复刻固定调用次数。

这是把OneSearch-VL的构题思想落实到AV任务。通用“证据图”本身属于已有方法；我们需要验证的扩展是声音和画面如何分别提供检索约束、消除歧义并支撑组合答案。

## 拟构造示例：尚非已核验样本

题目示意：“片段中演奏作品的作曲家，与片尾画面海报所示指挥，两人的出生年份相差多少？”

- 音乐分支：真实旋律/演唱线索 → 识别作品 → 查作曲家 → 查出生年份。
- 画面分支：海报/人物线索 → 识别指挥 → 查出生年份。
- 最终操作：对两个受支持年份求绝对差。

此例需要先证明音乐能被可靠辨认、画面不直接泄露作品答案、人物指代唯一、年份有可靠证据。这里仅说明拟建设任务的结构，未声称已有对应真实视频或答案。

类似任务也可以把语音中的活动/作品线索与画面中的另一实体关联，再做条件筛选或比较；不依赖为了增加工具次数而添加的无用步骤。

## 下一批与当前实验的衔接

建议下一批以**500道新训练任务**为目标，拟以80%多跳/组合任务、20%基础lookup作为初始配比；这是待验证的建设目标，不是已有数据比例。先从该批抽64题检查组合结构、可解性和teacher能力，再按同一构建路径扩到最终20K目标。Benchmark dev/test用预留录制家族独立构建，并按操作类型与模态贡献报告结果。

已有64条旧训练案例抽样仍保留作当前数据质检，不能将它重命名为64条组合新题。旧2K保留为warm start与已冻结训练对照；现有作业继续执行。接下来比较lookup-only、加入组合题、加入经核验音频线索题的训练收益，并补齐同预算强prompt基座与强AV teacher基线。

截至本次：新组合任务0条；新teacher对照0次；本文及机器可读proposal只是已形成的修正方案。没有修改冻结生产代码、训练数据或已运行实验，没有提交新GPU任务或push。

## 证据

- 远端实际2K：`/home/jovyan/OneSearch-VL/research/runs/av-learning-2k-preparation-20261018-v1/clean2000/`。
- 本地审查：`reports/onesearch-alignment-audit-2026-10-19/data-depth-audit.private.json`，含逐题字段、查询与引用数以及代码快照。
- 机器可读新方案：`reports/onesearch-alignment-audit-2026-10-19/compositional-authoring-proposal.json`。
