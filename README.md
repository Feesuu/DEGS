# DEGS

DEGS（Dynamic Experience Graph for Skills）把成功轨迹与“失败后被有效 patch 修复”的轨迹提取为经验图，再针对新任务在线检索可迁移的经验。仓库中只有一个正式方法：

> **DEGS 0.77.41 Stable R1**

当前正式版本的独立身份、固定方法、数据集路线和写作引用信息统一见
[VERSION.md](VERSION.md)。后续开发只在本目录进行。

该方法在固定 SpreadsheetBench development `[200,400)`、固定 200 分母、Qwen3.5-9B-AWQ 下的已审计结果是 `88/200 = 44.0%`。这是当前已有实验结果，不代表 WikiTQ、HiTab 或其他数据集的成绩。

## 方法流程

```text
train[0,200) rollout + verifier
  ├─ original success ─────────────────────┐
  └─ failure → patch → fresh replay success ┤
                                             ↓
                           causal micro-operation extraction + review
                                             ↓
                              25 batches × 8 incremental graph
                                             ↓
                   monotonic node-only Canonical + occurrence edges
                                             ↓
query → NeedGraph → top-k graph search → deterministic C0 / fallback Selector
                                             ↓
                              retrieved experience → Agent → evaluator
```

缓存不是输入，也不是前置条件。SQLite 中的逐文本 embedding 与 LLM response cache 只用于续跑加速；空缓存时，source extraction、Canonical、NeedGraph、clarification、Selector 和 embedding 都会调用配置的在线服务并产生同一种正式工件。

## 安装

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e '.[test]'
```

## 获取 SpreadsheetBench

```bash
.venv/bin/python scripts/fetch_spreadsheetbench.py \
  --checkout /path/to/Trace2Skill
```

脚本固定 Trace2Skill commit `3d0b52a140f002a512930252b613c49048f7d5ac`，并核验：

- verified-400：train `[0,200)`，development `[200,400)`；
- Full：912 tasks；去掉固定 train cases 后为 2,529 cases。

## 从头运行 SpreadsheetBench

先设置服务凭据：

```bash
export DEGS_API_KEY='...'
export DEGS_EMBEDDING_API_KEY='...'
```

再运行单一入口：

```bash
.venv/bin/python scripts/run_full_campaign.py \
  --profile 9b \
  --run-root /path/to/runs/degs-9b \
  --trace2skill-checkout /path/to/Trace2Skill \
  --generation-base-url http://host:port/v1 \
  --embedding-base-url http://host:port/v1
```

`--profile 27b` 使用相同流程与超参数，仅切换模型为 Qwen3.5-27B-AWQ。9B 与 27B 必须使用各自从头生成的 train 轨迹、replay、source graph、state、bundle 和 Agent outputs。

## Skill2Bench

Skill2Bench 使用本地 baseline 的固定 seed-42 划分：train 100、test
200。先取得官方 evaluator/data 快照并复现划分：

```bash
git clone https://github.com/Gen-Verse/Skill-Entropy-RL.git /path/to/Skill-Entropy-RL
git -C /path/to/Skill-Entropy-RL checkout 813a07fb1e4ea629d86196b36022187f13e6c3dd

.venv/bin/python scripts/prepare_skill2bench_data.py \
  --source-dir /path/to/Skill-Entropy-RL/skill2_bench \
  --baseline-root /path/to/Trace2Skill_Skill2Bench \
  --output-dir /path/to/skill2bench-split
```

运行完整的 9B train rollout、Step repair/replay、13 个动态图 batch、逐
Step 检索和 test-200：

```bash
.venv/bin/python scripts/run_skill2bench.py \
  --profile 9b \
  --train-path /path/to/skill2bench-split/train_100.jsonl \
  --test-path /path/to/skill2bench-split/test_200.jsonl \
  --baseline-root /path/to/Trace2Skill_Skill2Bench \
  --official-evaluator-root /path/to/Skill-Entropy-RL \
  --run-root /path/to/runs/skill2bench-9b \
  --generation-base-url http://host:port/v1 \
  --generation-api-key-file /path/to/generation.key \
  --embedding-base-url http://embedding-host:port/v1 \
  --embedding-api-key-file /path/to/embedding.key
```

`--profile 27b` 运行完全相同的流程但使用独立 27B 工件。正式实验先跑
9B。完整协议见
[docs/SKILL2BENCH_PROTOCOL.md](docs/SKILL2BENCH_PROTOCOL.md)。

## WikiTQ / HiTab

这两个数据集使用相同的 DEGS 图与同一个在线检索实现，不再经过任何父版本 bundle：

```bash
.venv/bin/python scripts/run_tableqa_ood.py \
  --profile 9b \
  --source-dataset-path /path/to/verified-400/dataset.json \
  --snapshot-manifest-path /path/to/final/snapshot_manifest.json \
  --state-db /path/to/incremental_state.sqlite3 \
  --wikitq-source-repo /path/to/WikiTableQuestions \
  --hitab-source-repo /path/to/HiTab \
  --run-root /path/to/runs/degs-ood \
  --generation-base-url http://host:port/v1 \
  --embedding-base-url http://host:port/v1
```

这里的 graph/state 只读；WikiTQ 与 HiTab 分别写自己的 retrieval cache、
bundle、Agent output 和 evaluation。`--profile 27b` 必须配套传入从 27B
SpreadsheetBench train 独立构建的冻结图。

## 代码地图

- `source_rebuild.py`：统一 source extraction 与 review；
- `incremental_graph.py`：25×8 动态构图与 Canonical；
- `bundle.py`：development 在线检索；
- `population_bundle.py`：Soft/Hard、WikiTQ、HiTab 共用的在线检索核心；
- `degs_skill2bench/`：Step 级 source、逐 Step query、完整 task Agent/evaluator；
- `benchmark.py` / `soft_hard_benchmark.py` / `ood_benchmark.py`：Agent 执行；
- `evaluate.py` / `soft_hard_evaluate.py` / `ood_evaluate.py`：评测。

完整协议见 [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md)，架构见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 测试

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
git diff --check
```
