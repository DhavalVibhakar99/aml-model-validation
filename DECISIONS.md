# Decisions

One entry per design choice: decision / alternatives considered / why this one /
what would change our mind.

## 1. Ingest keeps every row; the sparse tail is trimmed later, in split.py
- **Decision:** ingest writes all rows as-is. Only Sept 1-10 is used for
  features/labels; Sept 11+ gets dropped at the split step.
- **Alternatives:** trim inside ingest; keep the tail and let the label window
  run into it.
- **Why:** from Sept 11 on there are only a few hundred txns/day and ~60% of them are
  laundering, vs 0.03-0.21% per day before that (HI; LI is 0.02-0.11%). 86% of
  HI tail laundering txns (81% LI) touch an account that was already laundering
  before Sept 11, so this is the generator finishing patterns it had already
  started, not new activity. A label window there would inflate recall/precision. Trimming in
  split.py keeps the Parquet a faithful copy of the source, so the trim shows
  up in one place.
- **What would change our mind:** the IBM paper describing the tail as
  intentional behavior we're supposed to model.

## 2. Sept 1 is a warm-up day, flagged for features.py
- **Observation:** Sept 1 has ~2x a normal weekday's volume, and ~45% of its rows
  are self-transfers (src = dst), versus ~2% on other days. It looks like
  balance seeding at simulation start.
- **Decision (Dhaval):** exclude self-transfers from the behavior features
  (counterparties, in/out amounts, pass-through, timing, format mix), but keep
  them as their own flag feature: self-transfer count in the feature window,
  **ignoring Sept 1**. Day 1 itself stays in.
- **Alternatives:** drop self-transfers entirely; drop day 1 entirely; leave
  both alone.
- **Why:** counting a self-transfer as a counterparty pads fan-in/fan-out and
  pushes out/in toward 1 for ordinary accounts. But dropping them loses signal:
  accounts with a self-transfer after Sept 1 are 13.9% laundering-involved (HI;
  LI 4.1%) vs ~0.5-1.1% otherwise. The Sept 1 seeding transfers are left out of
  the flag because ~70% of accounts have one, and only the train feature window
  (days 1-4) contains Sept 1, so counting them would create a train/test shift.
  With self-transfers removed, Sept 1 is ~1.3x the next Thursday rather than
  2.3x, so keeping day 1 is fine.
- **What would change our mind:** the flag coming out unstable (high PSI)
  between windows, or the 13.9% not holding up once labels come from a later
  window than the features.

## 3. Download single files via kagglehub, not the full bundle
- **Decision:** `kagglehub.dataset_download(..., path="HI-Small_Trans.csv")`
  per file. The full bundle also holds the Medium/Large sets, which we don't
  need, and the disk had ~2 GB free.
- **What would change our mind:** needing the Medium/Large sets for a
  scale test.

## 4. Feature definitions
- **Decision:** 22 account-level features over non-self transfers in
  [start, cutoff), all amounts in USD. Pass-through = USD sent out within
  24h/48h of the account's most recent incoming payment, divided by USD
  received (capped at 10). Flow balance is `out / (in + out)`, which stays
  bounded. Burstiness is Goh & Barabasi's (sd - mean)/(sd + mean) of
  inter-transaction gaps. Plus a count of transfers in the $8k-$10k
  structuring band.
- **Alternatives:** raw out/in ratio (infinite for accounts that only send);
  FIFO matching of each dollar in to a dollar out (more exact, much harder to
  explain and test); summing native-currency amounts (meaningless: 1 BTC and
  1 Yen would add up to 2).
- **Why:** each feature has a one-sentence meaning an investigator would
  recognise, and each has a hand-computed unit test. FX rates are medians of
  implied rates from cross-currency transfers in the *training* feature window;
  the simulator's rates are fixed, so this is exact to ~4 significant figures.
- **What would change our mind:** real data with moving FX rates (you'd use a
  daily rate table), or the capped pass-through saturating for a big share of
  positive accounts.
