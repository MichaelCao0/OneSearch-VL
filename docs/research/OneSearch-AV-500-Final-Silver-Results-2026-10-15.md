# OneSearch-AV 500 条训练与同题评测

UTC：2026-10-15T04:44:04.330969+00:00；500 条训练题，原 SFT/src/train.py，Thinker LoRA，250 次实际更新。

匿名 silver judge 完成 54/56 题，每题同时评分三模型 × 两种输入。40 heldout 与 16 dev 分开报告。

主要指标要求完整最终回答、所有请求字段正确且无矛盾；supported success 还要求无实质无依据附加断言。缺失 judge 为 unknown，完成上下界不是置信区间。

## heldout_silver

| 模型与输入 | 成功 | 任务成功率 | Supported success | Unknown |
|---|---:|---:|---:|---:|
| base__native_av | 0/39 | 0.0% | 0.0% | 1 |
| base__muted_video | 1/39 | 2.6% | 2.6% | 1 |
| old172__native_av | 15/39 | 38.5% | 38.5% | 1 |
| old172__muted_video | 15/39 | 38.5% | 38.5% | 1 |
| clean500__native_av | 23/39 | 59.0% | 59.0% | 1 |
| clean500__muted_video | 21/39 | 53.8% | 53.8% | 1 |

- clean500__native_av minus base__native_av：59.0 pp，录制家族 bootstrap 95% [43.9, 75.0]，wins/losses=23/0。
- clean500__native_av minus old172__native_av：20.5 pp，录制家族 bootstrap 95% [2.5, 38.5]，wins/losses=11/3。
- clean500__native_av minus clean500__muted_video：5.1 pp，录制家族 bootstrap 95% [-8.1, 19.4]，wins/losses=5/3。
- base__native_av minus base__muted_video：-2.6 pp，录制家族 bootstrap 95% [-8.3, 0.0]，wins/losses=0/1。
- old172__native_av minus old172__muted_video：0.0 pp，录制家族 bootstrap 95% [-14.3, 14.3]，wins/losses=4/4。

## dev16_publisher_candidates

| 模型与输入 | 成功 | 任务成功率 | Supported success | Unknown |
|---|---:|---:|---:|---:|
| base__native_av | 0/15 | 0.0% | 0.0% | 1 |
| base__muted_video | 0/15 | 0.0% | 0.0% | 1 |
| old172__native_av | 9/15 | 60.0% | 53.3% | 1 |
| old172__muted_video | 11/15 | 73.3% | 73.3% | 1 |
| clean500__native_av | 8/15 | 53.3% | 53.3% | 1 |
| clean500__muted_video | 13/15 | 86.7% | 86.7% | 1 |

- clean500__native_av minus base__native_av：53.3 pp，录制家族 bootstrap 95% [38.5, 68.8]，wins/losses=8/0。
- clean500__native_av minus old172__native_av：-6.7 pp，录制家族 bootstrap 95% [-33.3, 21.4]，wins/losses=2/3。
- clean500__native_av minus clean500__muted_video：-33.3 pp，录制家族 bootstrap 95% [-46.7, -14.3]，wins/losses=0/5。
- base__native_av minus base__muted_video：0.0 pp，录制家族 bootstrap 95% [0.0, 0.0]，wins/losses=0/0。
- old172__native_av minus old172__muted_video：-13.3 pp，录制家族 bootstrap 95% [-35.7, 12.5]，wins/losses=1/3。

## all

| 模型与输入 | 成功 | 任务成功率 | Supported success | Unknown |
|---|---:|---:|---:|---:|
| base__native_av | 0/54 | 0.0% | 0.0% | 2 |
| base__muted_video | 1/54 | 1.9% | 1.9% | 2 |
| old172__native_av | 24/54 | 44.4% | 42.6% | 2 |
| old172__muted_video | 26/54 | 48.1% | 48.1% | 2 |
| clean500__native_av | 31/54 | 57.4% | 57.4% | 2 |
| clean500__muted_video | 34/54 | 63.0% | 63.0% | 2 |

- clean500__native_av minus base__native_av：57.4 pp，录制家族 bootstrap 95% [45.5, 69.4]，wins/losses=31/0。
- clean500__native_av minus old172__native_av：13.0 pp，录制家族 bootstrap 95% [-1.9, 29.1]，wins/losses=13/6。
- clean500__native_av minus clean500__muted_video：-5.6 pp，录制家族 bootstrap 95% [-17.9, 7.8]，wins/losses=5/8。
- base__native_av minus base__muted_video：-1.9 pp，录制家族 bootstrap 95% [-6.0, 0.0]，wins/losses=0/1。
- old172__native_av minus old172__muted_video：-3.7 pp，录制家族 bootstrap 95% [-16.4, 8.9]，wins/losses=5/7。

旧 172 条模型存在历史页面暴露，用作历史参考。新 500 条训练已排除冻结评测参考页面、已知候选录制和媒体字节；模型预训练来源未知。参考答案和 judge 为自动 silver，尚未转为人工 benchmark gold。

静音对照复制相同编码视频包、去掉所有音轨；初始和工具返回的视频均经同一转换，模型实际输入音频数必须为 0。该配对实验衡量音频可用性的效果，不能替代人工核实题目必须依赖声音。
