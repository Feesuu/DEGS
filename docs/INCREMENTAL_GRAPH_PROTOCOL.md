# Incremental ExperienceGraph protocol

## Input stream

SpreadsheetBench train `[0,200)` 按 index 顺序切成 25 个连续 batch，每批 8 个 index。一个 batch 中可以包含 original-success、replay-success 或被排除的失败任务；batch size 表示到达的任务数，不表示成功 workflow 数。

source extraction 和 review 对 train 全体以 async16 完成，再按固定 index 发布 batch 输入。source prompt 改变后必须从空 state 重新提取并重建，不能导入旧 prompt 的 graph。

## Canonical update

1. 新 source node 以 singleton 进入当前 batch。
2. 为新节点生成 node-only View；embedding 按 normalized text SHA256 缓存。
3. 每个 frontier group 按 alias cosine 召回 distinct Canonical top-16，并保留 exact-text 候选。
4. LLM 只比较两个节点组所表达的可迁移微观操作，输出 `SAME_TEMPLATE`、`DIFFERENT_TEMPLATE` 或 `UNCERTAIN`。
5. `SAME_TEMPLATE` 将两个完整组做单调 union；已提交组不能拆分。
6. 新 union 继续进入 frontier，失效父候选被丢弃；一次 `DIFFERENT_TEMPLATE` 不形成永久 cannot-link。
7. source occurrence edges 投影到当前 Canonical heads。自环、环和单 workflow 支持边均合法。

Canonical identity 不使用 predecessor、successor、轨迹位置、workflow 支持次数或固定 ontology。

## Persistence and failure handling

SQLite 保存 source workflows、occurrence edges、Canonical members/heads、Views、merge jobs/events、content-addressed embeddings、retrieval jobs 和 graph-quality audit。LLM 请求不持有长事务；一个 batch 的图状态与 snapshot 在完成后一起提交。

单个 View/MERGE 的 completion、schema 或 transport failure 会记录并跳过该 item，其他 item 继续。只有整个依赖服务不可用或 state/artifact 无法解析时，当前 stage 才失败。重复执行同一已提交 batch 返回同一 snapshot。

graph-quality audit 始终保存，但不是 retrieval 或 evaluation gate。
