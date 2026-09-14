# Experience-SimGRAG retrieval

DEGS 对每个 query 在线生成 NeedGraph，并从 ExperienceGraph 中选出一条可迁移的经验子图。缓存仅复用完全相同的请求；空缓存时所有 producer 和 embedding 都直接调用配置的服务。

## 1. NeedGraph

NeedGraph 将任务表达为因果微观操作。节点保留比较对象、参考状态、触发条件、状态更新时间、操作范围和停止条件；不能通过拆分改变题意。边表示一个 Need 的输出或建立的状态被另一个 Need 使用。

输入工作簿只生成 value-masked 结构角色证据。clarification producer 可以把既有 NeedNode 绑定到这些观察，但不能新增 Need、生成公式或预先给出完整解法。

## 2. Local recall

NeedNode 与 Canonical 仅按 operation 文本召回，无阈值保留 top-8：

\[
s_o(n,c)=\cos(e(n_\text{operation}),e(c_\text{operation})),
\qquad d_o(n,c)=1-s_o(n,c).
\]

完整的 `applicability`、`inputs` 和 `outputs` 不参与第一层过滤，但保留到候选解释和 Agent 经验中。raw query 同时召回 top-8 source workflows，为候选的来源上下文和 fallback 提供证据。

## 3. Graph-guided search

每个 NeedNode 从其 top-8 Canonical 候选中选择映射，允许多个 Need 使用同一模板。搜索使用 beam width 32，不枚举笛卡尔积。partial cost 为平均 operation distance 与已确定 NeedEdge 的未满足比例之和。

完整 mapping 形成后，系统只在真实 ExperienceGraph 边上恢复 connector path；不存在路径时记录 `unsatisfied_need_edges`，但不拒绝候选、不伪造边。

## 4. Workbook-role late fusion

目标和来源 input workbook 使用同一 value-masked role signature。该相似度只在完整候选形成后参与排序：

\[
L_\text{semantic}=\frac{L_\text{operation}+L_\text{workflow-context}}{2},
\qquad
L_\text{final}=L_\text{semantic}+L_\text{structure}.
\]

workflow context 不改变 operation top-8，不设阈值，也不 veto 候选。

## 5. Selection and rendering

正常路径使用图排序第一的 C0。NeedGraph 失败或图搜索为空时，LLM Selector 从 query top-8 workflows 中选择一个 source-grounded 候选；Selector 失败时使用确定性 C0 fallback。

Agent 最终只接收一个候选，包括 Need→Canonical mapping、Canonical operation/contracts、真实 source occurrence、连续或组合 witness，以及未被图支持的 NeedEdge。它不接收 raw workbook observations、完整训练轨迹、gold、verifier 或 development outcome。

每个 query 的 audit 保存召回、beam、mapping、connector、occurrence evidence、候选排序和 fallback 状态。审计用于分析，不阻止检索。
