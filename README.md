# AML mule-account detection, validated like a bank would

![tests](../../actions/workflows/tests.yml/badge.svg)

Most AML portfolio projects stop at "the model got a high score." This one is
about the part banks actually spend their time on: **independent validation**.
It covers conceptual soundness, outcomes against a rules benchmark, calibration,
stability monitoring and a stress test, in the structure of a Fed SR 11-7
model risk review.

**→ [Read the validation report](reports/validation_report.md)**

## The problem

Transaction-monitoring rules flag far more accounts than investigators can
review. In this data, firing on any of six standard rules gives 65,889 alerts
at 1.2% precision. The job here is **alert prioritization**: given a fixed
review budget, put the laundering-involved accounts at the top of the queue.

## Headline result

HI-Small test window (Sept 5-8, never seen in training), 253,113 accounts,
2,747 laundering-involved (1.09%):

| Model | PR-AUC | Precision @ 500 alerts | Recall @ 500 alerts |
|---|---|---|---|
| Rules, hand-set thresholds (6 typologies) | 0.016 | 7% | 1.3% |
| Rules, thresholds tuned on the train window | 0.019 | 10% | 1.9% |
| Logistic regression | 0.085 | 18% | 3.3% |
| **LightGBM** | **0.334** | **99%** | **18.0%** (max possible: 18.2%) |

LightGBM gets **10x the recall of the tuned rules** at the same alert budget
(13x against the hand-set ones).

**Final holdout** (Sept 7-10, scored once with frozen models, hashes recorded):
PR-AUC 0.322 and 93% precision at 500. Counting only launderers whose labels no
design decision had seen, PR-AUC is **0.196**, recall at 500 is 12.5%, and the lift
is still 8x the tuned rules. That's the conservative number.

The validation also found things a single score would hide:

- **Scores aren't probabilities in a new period.** The base rate doubled between
  windows and the model under-predicts, so use it as a ranking.
- **The score distribution drifted on test (PSI 0.42) while no single top feature
  did.** On the holdout it was 0.03, which points to weekday mix rather than
  model decay. Monitoring has to track both scores and features.
- **No memorization.** Accounts first seen in the test window rank as well as
  familiar ones (PR-AUC 0.356 vs 0.328).
- **About 40% of the model's gain comes from payment-format mix**, a habit of the
  simulator. Without those features PR-AUC is 0.273, still 8x the tuned rules' recall.
- **Stress test (train HI → score LI):** precision at 500 falls to 36%, but a
  model trained on LI itself does no better, so the ranking transfers and the
  population is simply harder.
- **The originally planned "predict the next 48h" label fails** (PR-AUC 0.031).
  The data says why: 27% of next-window launderers have no prior activity at all.

![Outcomes](reports/figures/outcomes.png)

## How it's built

| Step | Script | What it does |
|---|---|---|
| 1 | `src/download.py`, `src/ingest.py` | Fetch the two CSVs, write typed Parquet, sanity-check (timestamps, amounts, daily volume) |
| 2 | `src/features.py` | 21 account behavior features in DuckDB: fan-in/out, pass-through within 24h/48h, time from receive to send, burstiness, channel mix, structuring band |
| 3 | `src/split.py` | Time-based windows: train Sept 1-4, test Sept 5-8, no overlap |
| 4 | `src/train.py` | Rules baseline, logistic regression, LightGBM; run logs to `reports/runs/` |
| 5 | `src/validate.py` | Every figure and number, and renders the report from a template |
| 5b | `src/holdout.py` | Final holdout, scored once with frozen models; refuses to rerun |
| 6 | `tests/` | Hand-computed feature tests, leakage tests, metric tests |

Things deliberately *not* done: random row splits, rebalanced test sets,
accuracy as a metric, hyperparameter tuning on the test window. The reasoning
for each design choice, with the alternatives considered, is in
[DECISIONS.md](DECISIONS.md).

## Run it

Needs Python 3.11 and ~3 GB of free disk. On macOS, LightGBM also needs
`brew install libomp`.

```bash
make setup      # venv + pinned requirements
make data       # download HI-Small / LI-Small (~1 GB) and ingest to Parquet
make all        # split -> train -> validate; rebuilds reports/ (~2 min)
make test       # 26 tests, no data needed
```

Kaggle downloads of this public dataset worked without an API token when this
was written. If yours asks for one, see the
[kagglehub docs](https://github.com/Kaggle/kagglehub#authenticate).

## Data

Altman, E., Blanuša, J., von Niederhäusern, L., Egressy, B., Anghel, A., &
Atasu, K. (2023). *Realistic Synthetic Financial Transactions for
Anti-Money Laundering Models.* NeurIPS 2023 Datasets and Benchmarks Track.
Data hosted on Kaggle at
[`ealtman2019/ibm-transactions-for-anti-money-laundering-aml`](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml).

## Limitations

The data is synthetic, has 10 usable days and a single backtest, and its labels
are perfect. Real AML labels come from SARs and are incomplete, delayed and shaped
by the rules that generated the alerts. The numbers here show the method, not
real-bank performance. Section 10 of the report has the full list.

## License

MIT
