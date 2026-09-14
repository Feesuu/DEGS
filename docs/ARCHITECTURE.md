# DEGS 0.77.41 Stable R1 architecture

## Offline graph construction

1. `source_replay_executor.py`：对 train `[0,200)` 执行 Agent、verifier、失败 patch 与 fresh replay。
2. `source_rebuild.py`：从 original-success 或最终 patch + replay-success 提取因果微观操作，并用 reviewer 统一节点粒度；LLM producer 全局 async16。
3. `section_graph.py`：每条轨迹形成 ExperienceNodes 与有向 occurrence edges。节点字段为 `operation`、`applicability`、`inputs`、`outputs`。
4. `incremental_graph.py`：将 200 个 index 确定性切成 25×8 顺序插入。新节点先做 node-only top-16 召回，再由 SAME_TEMPLATE 判断融合；历史 Canonical 组不拆分。
5. `graph_quality.py`：报告 source 覆盖、Canonical size、singleton、跨 workflow 融合、WCC/SCC、最大分量覆盖、候选未融合与 edge provenance。审计不阻断后续流程。

## Online retrieval

1. Query 生成 V2 NeedGraph。
2. 若 NeedNode 的值表示存在输入歧义，clarification producer 只能引用输入工作簿证据来解释既有 NeedNode，不能新增步骤或解题结论。
3. Query 对 source workflow 取 top-8；每个 NeedNode 对 Canonical operation 取 top-8。
4. Experience-SimGRAG 在真实图边与 occurrence evidence 上执行 beam32 搜索。
5. 只有完整候选形成后，才将目标/来源工作簿的 value-masked role similarity 与 operation similarity 做 late fusion。
6. 正常路径注入排序第一的 C0；NeedGraph 或搜索无候选时，使用 workflow top-8 的 LLM Selector，Selector 失败则回退 C0。

## Dataset adapters

`bundle.py` 实现 development `[200,400)`；`population_bundle.py` 是 Soft/Hard、WikiTQ 与 HiTab 的共享在线检索核心。各 adapter 只负责把其输入人口投影为统一 `_Task`，不改变图、检索、Selector 或 Agent 方法。

## Cache semantics

- embedding cache：`normalized_text_sha256 → embedding`；
- retrieval cache：prompt、schema、producer protocol 与 payload 的联合 identity；
- cache miss 调用在线服务；cache hit 只复用完全相同的请求结果；
- cache 可以为空，也可以删除，不影响方法定义。
