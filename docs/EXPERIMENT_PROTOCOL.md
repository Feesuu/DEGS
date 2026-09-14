# Experiment protocol

| Item | Fixed value |
| --- | --- |
| Method | DEGS 0.77.41 Stable R1 |
| SpreadsheetBench train | `[0,200)` |
| SpreadsheetBench development | `[200,400)`, denominator 200 |
| Full population | 912 tasks / 2,529 cases after fixed train-case exclusion |
| Train graph arrival | 25 contiguous batches × 8 |
| Agent / replay concurrency | 8 |
| Extraction / Canonical / retrieval producer concurrency | 16 |
| Embedding model | Qwen3-Embedding-8B |
| Agent model | Qwen3.5-9B-AWQ or Qwen3.5-27B-AWQ |
| Temperature / thinking | 0 / false |
| Agent turns | 30 |
| SpreadsheetBench Agent / extraction / Canonical / retrieval completion | 32,000 tokens per request |
| Replay patch generation and replay Agent completion | 16,384 tokens per request |
| Server context | 100,000 tokens |
| Explicit seed | none; use the existing VRF policy |
| Development evaluator | SpreadsheetBench comparator after LibreOffice recalculation |
| Graph-quality audit | always saved, never a run gate |

## Source admission

- Original Agent trace with verifier success; or
- one failed trace’s final validated patch plus the single fresh replay trace that succeeds.

Failure history before the successful replay is not graph input. A source/review/Canonical item-level completion or JSON failure is recorded and the remaining population continues.

## Information boundary

Graph construction reads only train evidence. Retrieval reads the target instruction, declared input workbook, answer position and train ExperienceGraph. It does not read target output, gold workbook, verifier result, Agent trace or target outcome. Evaluation reads gold only after Agent generation finishes.

## Cache and resume

Cache is optional. A new state/cache must invoke the online producers. Resume may reuse only an identical request. 9B and 27B never share generated trajectories, replay evidence, source graph, state database, bundle or Agent output.

## Result reporting

Every benchmark report must state dataset/split, model, denominator, completed and abnormal runs, fixed hyperparameters, evaluator version, exact run directory, accuracy and failure decomposition. Smoke tests and partial slices are not benchmark scores.

## Multi-dataset profiles

- SpreadsheetBench：9B/27B 都从各自 train `[0,200)` 独立 rollout、replay 和构图；development 固定 `[200,400)`，Soft/Hard 使用固定 912-task population。
- Skill2Bench：固定 seed-42 train-100/test-200；Agent/replay async8，producer async16，Agent 不显式发送 `max_tokens`，30 turns，thinking false；同一个 profile model 评 open-ended Step。所有有分数且未满分的 closed/open Step 都进入最多三次 patch + fresh full-task replay。
- WikiTQ/HiTab：不在 OOD 数据上学习。只读相同模型的冻结 SpreadsheetBench 图，并分别运行在线 NeedGraph、检索、Agent 与官方 evaluator。

三种流程都支持 Qwen3.5-9B-AWQ 与 Qwen3.5-27B-AWQ，且模型生成工件、图、cache、bundle 和输出不能跨 profile 共用。第一轮正式跨数据集实验使用 9B。
