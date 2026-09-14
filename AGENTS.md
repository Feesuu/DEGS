# DEGS project rules

- 唯一正式方法是 `DEGS 0.77.41 Stable R1`。不得在正式运行路径中拼接或调用历史版本。
- 一切修改从第一性原理出发：目标是提高经验图质量、检索质量及其对下游任务的真实帮助。
- 图质量审计必须保存并分析，但不能成为阻止 retrieval、Agent 或 evaluation 的 gate。
- train 仅为 SpreadsheetBench `[0,200)`；development 固定 `[200,400)`、分母 200。禁止 development outcome、gold、verifier result 或 Agent trace 进入构图与检索。
- source 只接纳 original-success，或最终有效 patch 加其对应 replay-success trace；不保存失败 replay history。
- 动态图按 25 个连续 batch、每批 8 个 train index 发布；已形成的 Canonical 组单调保留，不翻旧账拆组。
- Canonical identity 只比较节点本身的可迁移微观操作，不以前驱、后继或 workflow 上下文作为 SAME 的必要条件。
- online retrieval 固定 query-only V2 NeedGraph、workflow top-8、Need→Canonical top-8、beam32、input-only cited clarification、完整候选级 workbook-role late fusion、正常路径 deterministic C0、fallback Selector。
- cache 只能加速。空 cache 必须能调用在线 LLM/embedding 服务完成全部正式步骤；不得把旧 bundle 或旧 cache 作为输入依赖。
- 代码保持简洁。不要加入父版本 pin、accepted-bundle 链、源码归档 hash、secret policy、运行时源码复制或与算法无关的安全协议。
- 保留真正影响实验正确性的检查：split、分母、模型、关键超参数、输入人口、bundle 行数、evaluator 与结果完整性。
- 长实验保存 run dir、命令、阶段时间、token/request 使用、日志、结果与 evaluator 输出；单项 LLM 失败记录后继续，系统服务不可用才停止整轮。
- 修改 prompt 后必须重跑由该 prompt 生成的全部下游工件；不得用旧 prompt cache 冒充新结果。
