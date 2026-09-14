# Full-chain implementation

The executable path is:

```text
official SpreadsheetBench checkout
→ train[0,200) rollout
→ LibreOffice verifier
→ failure patch + fresh replay
→ source extraction/review
→ 25×8 incremental ExperienceGraph
→ one online DEGS retrieval bundle
→ development[200,400)
→ Soft/Hard 912-task / 2,529-case run
→ LibreOffice evaluation
```

`scripts/run_full_campaign.py` executes this path from the current checkout. Each stage saves command, timing, log and declared output. Producer and Agent token usage are written by the existing runtime usage logs and summarized by `scripts/summarize_campaign.py`.

9B and 27B use identical dataset and evaluator protocols but separate run roots and all model-generated artifacts.
