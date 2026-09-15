# DEGS project rules

- 唯一正式方法是 `DEGS 0.78.0 Evidence-Bounded EIR Dynamic`。`0.77.41 Stable R1` 只作为 Git 历史基线，不得进入 0.78.0 正式运行路径。
- 第一性原理目标：学习能够被当前任务证据正确约束、能够真实帮助后续 Agent 的程序性经验；图质量审计必须保存，但不能成为 retrieval、Agent 或 evaluation 的 gate。
- 一个 train batch 固定包含 8 个逻辑 task；批内所有 task 只读同一个冻结 `G(k-1)`，rollout/verifier/repair/reflection 可并发，图变更按 train index 顺序原子提交为 `G(k)`。
- 每个 train task 只运行一次 Agent。失败时最多保留最终有效 patch 与它对应的一次 fresh replay success；失败 patch 和早期失败 replay 不进入正经验。
- Reflection 是唯一学习调用：同时 reconcile 已检索经验、提取未被已有经验解释的成功微操作、恢复成功 episode procedure。未解决失败和基础设施失败不得写入正节点或正边。
- `SUPPORT` 不改文本、不制造同义节点；`QUALIFY/CORRECT` 只能由完整的 failure → final patch → fresh replay success → verifier success 因果证据授权。
- ExperienceNode 表示有 observable guard、current-task parameter binding、局部可执行 operation 与预期 state transition 的因果微操作。不得把任务特有常量无依据提升成通用规则。
- Canonical identity 稳定、成员只增不拆；语义修改创建历史可追溯的新 version。SAME 只比较节点的 guard-binding-operation-effect，不把前驱、后继或 workflow 上下文当作必要条件。
- 正式检索固定为：query + dataset observable context 召回 active Canonical top-5；每个 anchor 最多带 2 个一跳邻居作结构上下文；一次 binding LLM 对每个 anchor 判定 `SATISFIED/UNKNOWN/CONFLICT` 并绑定当前任务参数。只有验证后的 guidance 注入 Agent。
- SpreadsheetBench train `[0,200)` 从空 `G0` 运行 25×8 动态学习；development `[200,400)` 固定 200 分母。Skill2Bench 固定 seed-42 train-100/test-200，每个 Step 独立检索但完整 task 只运行一次 Agent。WikiTQ/HiTab 只读同模型 SpreadsheetBench 冻结图。
- 数据集与模型的 state、cache、bundle、轨迹和评测目录严格隔离。禁止 development/test outcome、gold、verifier 或 Agent trace 进入 train 学习。
- cache 只用于加速；空 cache 必须可执行全部在线 LLM/embedding 调用。已完成且身份完全一致的阶段可以续跑；不得因 Reflection 等后续失败而重复已完成 Agent/Verifier/Replay。
- 不得静默截断 query、trajectory、action、observation、patch 或 final response。单 item 的 completion/schema/context/runtime 错误记录并继续；确认是系统服务整体不可用或状态损坏时才停止。
- 固定 SpreadsheetBench Agent/replay 8 并发，binding/reflection/Canonical producer 16 并发，30 turns，32,000 completion tokens，100,000 server context，temperature 0，thinking false；除非形成显式新版本，不得静默改变。
- 每轮保存批耗时、LLM 请求与 token、失败分类、LearningDelta action、图版本/连通性和 evaluator 输出。prompt 改变后必须从空状态重跑它生成的全部下游工件。
