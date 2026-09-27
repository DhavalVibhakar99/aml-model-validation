# Decisions

One entry per design choice: decision / alternatives considered / why this one /
what would change our mind.

## 1. Ingest keeps every row; the sparse tail is trimmed later, in split.py
- **Decision:** ingest writes all rows as-is. Only Sept 1-10 is used for
  features/labels; Sept 11+ gets dropped at the split step.
- **Alternatives:** trim inside ingest; keep the tail and let the label window
  run into it.
- **Why:** from Sept 11 on there are only a few hundred txns/day and ~60% of them are
  laundering (0.1-0.2% before that), in both HI and LI. It looks like the
  generator finishing laundering patterns it had already started, not real
  activity. A label window there would inflate recall/precision. Trimming in
  split.py keeps the Parquet a faithful copy of the source, so the trim shows
  up in one place.
- **What would change our mind:** the IBM paper describing the tail as
  intentional behavior we're supposed to model.

## 2. Sept 1 is a warm-up day, flagged for features.py
- **Observation:** Sept 1 has ~2x a normal weekday's volume, and ~45% of its rows
  are self-transfers (src = dst), versus ~2% on other days. It looks like
  balance seeding at simulation start.
- **Open question for step 2:** whether self-transfers and/or day 1 should
  count toward behavior features. Not decided yet.

## 3. Download single files via kagglehub, not the full bundle
- **Decision:** `kagglehub.dataset_download(..., path="HI-Small_Trans.csv")`
  per file. The full dataset is several GB and we only need the two Small sets.
- **What would change our mind:** needing the Medium/Large sets for a
  scale test.
