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
- **Decision:** 21 account-level features over non-self transfers in
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


## 5. Windows: train = Sept 1-4, test = Sept 5-8, forecast labels on the next 2 days
- **Decision:** two 4-day feature windows that don't overlap at all: train
  Sept 1-4 and test Sept 5-8. Each also carries a forecast label for the 2 days
  after it (Sept 5-6 and Sept 9-10, see #6).
- **Alternatives:** the first proposal (train 1-4/5-6, test 3-8/9-10). Its test
  feature window is 6 days vs 4, so a count like `in_count` would mean
  something different in train and test, and the train and test windows overlap.
- **Why:** same lengths, fully disjoint, and every day from Sept 1-10 gets used.
  The weekday mix differs (train is Thu-Sun, test is Mon-Thu), and the PSI
  checks show what that does.
- **What would change our mind:** a longer dataset, which would allow a proper
  rolling backtest over many cutoffs instead of one.

## 6. Primary label = detection (same window), forecast kept as a secondary analysis
- **Decision:** label = account sent/received >= 1 laundering txn *within* the
  feature window. The CLAUDE.md default (laundering in the *next* window) is
  still built as `label_next` and reported as a secondary result.
- **Evidence (first run, HI):** under forecasting, LightGBM test PR-AUC was 0.031
  (in-sample 0.354, so badly overfit), recall@1000 was 4.9%, and logistic
  regression reached only 0.019 even in-sample. Of 1,397 forecast positives,
  only 229 (16%) were laundering in their own feature window, and 379 (27%)
  had no history at all. So even an oracle that knew past labels would top
  out around 16% recall. Under detection, the same features give test PR-AUC 0.334 vs
  0.371 in-sample, so it generalizes.
- **Why this is the right framing, not just the better number:** a TM system
  reviews an account's recent activity and asks whether *that* activity was
  suspicious. The SAR is filed on activity that already happened. "Who will
  launder in the next 48h" is a harder, different product. Leakage is still
  controlled: features never read `is_laundering` (test_leakage.py), and train
  and test periods don't overlap.
- **What would change our mind:** a use case that is explicitly early-warning
  (e.g. freezing accounts before funds move), or much longer data where
  laundering persists across windows.

## 7. Model settings: fixed hyperparameters, no class weights, rules scored by count
- **Decision:** LightGBM and LR use fixed, conservative settings with no
  tuning (LightGBM: 400 trees, lr 0.03, min_child_samples 200). No class
  weights or resampling. The rules baseline scores = number of the 6 typology
  rules hit, with ties broken by dollar flow.
- **Alternatives:** tuning on a validation window; `is_unbalance` /
  SMOTE; an ACH-specific rule.
- **Why:** with 10 days of data the only window left for tuning is the test
  window, so tuning would bias the test result upward. Class weights barely
  change ranking but inflate predicted probabilities, and calibration is one of
  our metrics. The rules follow standard typologies (rapid movement, fan-in/out,
  structuring, velocity, high-risk channel) and not the dataset's quirks. An
  "ACH rule" would be tuning the baseline to the answer.
- **What would change our mind:** a longer dataset with a clean validation
  window (then tune), or the unweighted LightGBM failing to rank positives.

## 8. The report is rendered from a template by validate.py
- **Decision:** `reports/validation_report.md` is generated from
  `reports/templates/validation_report.md`, with every number a placeholder
  filled by `src/validate.py`. The formatted values also go to
  `reports/report_values.json`.
- **Alternatives:** hand-write the report and paste numbers in; a notebook.
- **Why:** "every number is produced by a script" is only true if a number
  can't be typed by hand. A rerun rebuilds the whole document, so the prose
  can't drift from the code. The prose that interprets numbers is still
  hand-written, so it needs rereading after any change that moves the results.
- **What would change our mind:** a report long enough that templating gets
  in the way of editing (then use a proper tool, e.g. Quarto).

## 9. Rules baseline thresholds tuned on the train window; lift measured against the tuned set
- **Decision:** coordinate descent over a grid of plausible thresholds per rule
  (with "off" as an option), maximizing mean precision@{100,500,1000} on HI
  train. Converged in 4 passes and kept 2 of the 6 rules: fan-in >= 7 and
  velocity >= 2. All lift numbers now use the tuned set; the hand-set set is
  still reported.
- **Result:** test precision@500 for the rules went 7% -> 10%, so LightGBM's
  recall lift at 500 dropped from 13x to 10x, and at 1,000 from 14x to 7x.
  Train objective doubled (0.085 -> 0.165) but test gained less, so some of the
  tuning was specific to the training window.
- **Alternatives:** keep hand-set thresholds only (the comparison would favor
  the model); tune per rule independently (ignores how the rules combine in
  the ranking); fit rule weights (at that point it's a model, not rules).
- **Why:** a challenger should be measured against the strongest baseline you
  could reasonably build from the same data. The tuned set is a good *ranker*
  but not a deployable alert set (velocity >= 2 fires on 75% of accounts), and
  the report says so.
- **What would change our mind:** a constraint on total alert volume per
  rule, which is how real rule tuning works (above/below-the-line testing).
  That would give a more realistic, and probably weaker, tuned baseline.

## 10. Seen vs new accounts: no memorization signal
- **Decision:** report test metrics split by whether the account was active in
  the train window, keeping the global ranking and budget.
- **Result:** 94% of test accounts were seen in train. LightGBM PR-AUC is 0.328
  on seen accounts and 0.356 on new ones. New accounts are 2.6x more likely to be
  laundering. Only 9% of test positives were also train positives.
- **What would change our mind:** a gap favoring seen accounts, which would
  suggest the model is recognizing familiar behavior rather than typologies.

## 11. Final holdout = Sept 7-10, scored once with frozen artifacts
- **Decision (made before scoring):** Sept 9-10 are usable (HI: 862k txns, 956
  laundering, 0.08-0.21%/day, no tail effect). The models expect 4 days of
  history, so the holdout window is Sept 7-10 with the same detection label.
  It's scored once by `src/holdout.py` using the saved LightGBM/LR files and
  rule thresholds. The model file hashes and the git sha are recorded in the
  output. No model, feature, threshold or report claim was changed after
  seeing it.
- **The catch:** Sept 7-8 overlap the test window, which informed decisions
  (the label framing, #6). So the holdout also reports **fresh positives**:
  accounts whose laundering falls *only* on Sept 9-10, i.e. labels no decision
  has ever seen. The recall on those is the uncontaminated number.
- **Alternatives:** a 2-day Sept 9-10 window (every count feature would be
  about half its training value, so we'd be measuring a window-length mismatch,
  not generalization); no holdout (then the test window is doing double duty as
  development and final evaluation).
- **What would change our mind:** more data. A proper holdout would be a full
  later period that doesn't overlap anything.
- **Result (scored once, commit 7d02a45):** LightGBM PR-AUC 0.322 (test: 0.334),
  93% precision at 500, 8x the tuned rules' recall. Fresh labels only: PR-AUC
  0.196, recall@500 12.5%, still 8x the rules. Score PSI vs train 0.030 (test was
  0.423), which supports the weekday-mix explanation of the test drift. A
  loading bug was found and fixed *before* scoring: the pickled LR pointed at
  `__main__.log_heavy`. The fix was checked to reproduce every test score
  exactly (max diff 0.0) before the holdout ran.
