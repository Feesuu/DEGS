# DEGS method boundary

唯一正式方法是 **DEGS 0.77.41 Stable R1**。

## Source evidence

只允许两类 train 证据进入图：

- verifier 成功的 original trajectory；
- 最终有效 patch 的 `instructions/checks` 与使用该 patch 后成功的那一条 fresh replay trajectory。

更早失败 replay、旧 patch、development outcome、gold、target verifier 和 target Agent trace 均不能进入 source extraction。

ExperienceNode 是一个可独立迁移、局部可执行、会改变任务状态或产生任务特定判别证据的微观操作。schema 只有 `operation`、`applicability[]`、`inputs[]`、`outputs[]`。宏观多操作必须拆分；import、普通加载、工具调用、循环迭代、纯保存和完成确认不单独成节点。

边只表示 target 消费 source 的产物，或 source 建立 target 的适用状态。非法边逐条删除，不因一条边丢掉整条 workflow。

## Graph construction

train `[0,200)` 以 25×8 动态进入。Canonical 仅根据节点本身在参数替换后是否为同一种可迁移操作进行融合，不使用前驱、后继、轨迹位置或 workflow 支持次数，也不制定 ontology。已提交的组单调保留；新 batch 可以加入或合并组，但不能翻旧账拆组。

ExperienceGraph 的边全部来自真实 source occurrence edge 投影。graph-quality audit 分析 source 粒度、Canonical size/singleton、跨 workflow 融合、WCC/SCC、最大分量和 retrieval usage，但不阻止下游运行。

## Online retrieval

每个 target 从原始 query 在线生成 NeedGraph；Need→Canonical top-8、workflow top-8、beam32。input workbook 的 value-masked role context 只在完整候选形成后做 soft late fusion。正常路径选择 C0；NeedGraph 或搜索失败时使用 source-workflow Selector fallback。cache 不是输入依赖，空 cache 必须可以生成完整 bundle。

## Fixed experiment protocol

SpreadsheetBench development 固定 `[200,400)`、分母 200。Agent/replay 并发 8；source、Canonical 和 retrieval producer 并发 16；temperature 0；thinking false；Agent 30 turns；单次 completion 32,000 tokens；服务 context 100,000 tokens。9B 与 27B 的模型生成工件完全分开。

正式结果必须使用同一任务 population 和 LibreOffice evaluator，并报告异常完成数量。不得以 smoke、局部 slice 或图连通性替代完整 benchmark。
