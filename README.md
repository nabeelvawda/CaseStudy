# Crypto Trading Signal Pipeline

**GitHub repo:** [Add link here]
**Interactive interface (Streamlit):** https://casestudy-gpgyxslxnoe6vmcoflr5f8.streamlit.app/
---

## Executive Summary

This project explores `bot_logs`, a table of trading signals from 27 bots on BTC/USDT, and builds a pipeline to predict 24-hour price direction. Part 1 surfaces several real data issues — an unsorted raw file, two separate outage gaps, a bot roster that roughly doubles partway through the dataset, a schema change mid-stream in one bot, and a handful of always-Hold bots whose decisions carry no signal. Part 2 builds a bronze/silver/gold pipeline on top of this, then trains a logistic regression to predict price direction from `BasicMovingAverageBot` and `ExponentialMovingAverageBot` signals, chosen to match the brief's own single-scalar interface example.

The honest result: the trained model performs at or below chance in both the pre- and post-rollout eras (ROC-AUC 0.499 and 0.296 respectively), and in the post-rollout period a much simpler baseline — just following the bot's own Buy/Sell call — clearly outperforms it (64.9% vs 44.0% accuracy). This is reported as a genuine, diagnosed finding rather than hidden: the post-rollout collapse was tested against a class-imbalance explanation and ruled out, pointing instead to too small a training window to generalise. Part 3 builds a Streamlit interface that shows both the model's prediction and the bot's own logic side by side, so this result is visible in the demo itself rather than only in the README. Part 4 covers the limitations, what a longer timeline would change, and what would need to change to run this with real money.

---

## Part 1 — Exploration & Understanding

Each row in the `bot_logs` table represents a decision made by the various bots at a given time. The logs span from 2025/02/03 to 2026/09/13 (the day I extracted the data), with a gap from 2026/05/08 to 2026/09/03. However, since the brief specified that the data should end in May 2026, any data beyond 2026/05/08 was discarded. There are 40040 rows in total, and three symbols are present (ETH/USDT, XRP/USDT, and BTC/USDT). There are 27 bots in this dataset. Bots differ in the scope of metrics or parameters required to make a decision. Some bots, such as the BasicMovingAverageBot and RelativeStrengthIndexBot, require only one input. Others, such as RegimeCompositeBot and BayesianCompositeBot, require more than 12 parameters to make a decision. The four bots present before 2026/04/03 fire once an hour, while the 23 other bots added after 2026/04/03 fire more frequently, roughly every 35–40 minutes.

Each time a bot action is taken, the time, the bot name, the current price of the symbol, the symbol, the scoped metric and the ID of that transaction are given.

### Data issues

| Issue | Impact | What I did |
|---|---|---|
| A handful of early entries contain the symbols ETH/USDT and XRP/USDT rather than BTC/USDT. | Would mix non-BTC prices into what should be a single-symbol dataset. | The other symbols were discarded to have all the logs have a single symbol. |
| There is a gap in the data between 2026/04/12 and 2026/04/29. | 82 out of 13159 signals had a 24-hour target time falling inside the gap. | I excluded these rows from the labelled dataset by capping the as-of lookup to a maximum tolerance of 3 hours — any signal without a valid price observation within that tolerance is dropped rather than mislabeled. Given the small number of affected rows relative to the full dataset, this had negligible impact on sample size. |
| AnomalyDetectionBot, BayesianCompositeBot, and StatsForecastBot never emit Buy or Sell, only Hold. | Would skew the agreement percentage, since they always count as a non-Buy vote regardless of what the rest of the ensemble signalled. | These bots were omitted from bot-agreement calculations. |
| The MacdEmaBot gains an `ema` key from 2026-04-04, splitting its history into two shapes. | Would introduce inconsistent columns if parsed naively. | This was not a problem, as this bot was omitted from modelling since I narrowed the scope to the BasicMovingAverageBot and ExponentialMovingAverageBot. |

---

## Part 2

### 2a. Data Modelling

The data modelling process followed the medallion architecture, as the raw data has real schema inconsistency across bots (and even within one bot, over time), so a staged validation process was needed before anything could be built for modelling.

**Bronze:** the files remained as they were. The bronze layer is structured this way to preserve history.

**Silver:** this is where the data was filtered and cleaned, making it ready for querying and engineering. The silver layer contained two tables, `silver_signals` and `silver_metrics`.
- `silver_signals` largely kept the data the same but flagged AnomalyDetectionBot, BayesianCompositeBot, and StatsForecastBot as non-informative. More flags were added to filter the scoped symbols, the time before the 23 other bots were added, and the outage window between 2026-05-08 and 2026-09-03.
- `silver_metrics` expanded `scoped_metric` into rows. This was done instead of columns because the data schema would not have to change constantly if a new bot were added with different parameters. The ID, bot name, and timestamps remained the same to create a link between the two tables.

**Gold:** four tables:
- `gold_price_timeline` — combines `bot_logs.current_price` with `price_log` to create a lookup price table.
- `gold_ma_ema_base` — rows contain `current_price`, `moving_average`, the engineered `pct_diff_from_ma` (percentage difference from moving average), a `bot_type` flag (Basic vs Exponential), the bot's own decision (kept for the baseline comparison), and the `is_v2_era` flag (after 2026/04/03). This is the feature layer, built once, independent of whatever label or horizon gets attached later.
- `gold_bot_agreement` — a separate table rather than a column bolted onto the base table, because it's conditionally meaningful: it only exists for signals from 2026-04-03 onward (before that, only 4 bots exist at all, so "agreement" is undefined). Keeping it separate and left-joining it in means pre-rollout rows cleanly get NULL rather than a fabricated value.
- `gold_ma_ema_features_labels` — the final table actually used for training: base features + agreement (where it exists) + the label, joined via the as-of price lookup. This is the one and only place the label gets attached, kept separate from feature engineering so that changing the label (different horizon, different definition) never requires re-deriving the features themselves.

**General principle:** price timeline (reusable) → features (per bot family, label-independent) → agreement (conditional, separate) → labelled table (where everything meets, built last). Keeping these apart means each piece can be reused, debugged, or changed independently.

#### Defining a "successful" signal

A signal is successful if its direction matches the subsequent price movement: a Buy is successful when price is higher 24h later (`label_up = 1`), and a Sell is successful when price is lower 24h later (`label_up = 0`).

**Justification:** the brief's interface spec asks for an up/down prediction with a confidence score, hence a binary framing matches that and keeps the modelling and demo consistent. A 24-hour horizon was selected because a shorter horizon (~1 hour) can reflect market friction rather than genuine price trends. 24 hours provides enough time for actual market trends to outweigh short-term noise while remaining short enough to maintain clean data density.

#### Preventing look-ahead leakage

Two separate mechanisms, worth stating both explicitly since they solve different problems:

1. **The label lookup only ever looks forward, never backward, and is capped.** For each signal, the label is built from an as-of join that finds the earliest price observation at or after `signaled_at + 24h`, with a maximum tolerance of 3 hours. If no valid observation exists within that window (e.g. the signal falls right before the April 12–29 gap), the row is dropped rather than mislabeled — this is what caught the 82 rows that would otherwise have been labelled using prices up to 17 days later.
2. **Train/test splits are strictly chronological, never randomly shuffled.** Because 24-hour label windows for signals fired close together (bots fire roughly hourly) overlap heavily, they're serially correlated — a random split could put a training row and a test row whose forward-looking windows overlap almost entirely, which would leak test-period outcome information into training. Splitting by date (train = earlier period, test = later period) avoids this entirely.

#### Feature engineering

Mapped to the brief's four categories:

- **Price:** raw BTC price isn't fed into the model directly; instead, it is transformed into a relative ratio (`pct_diff_from_ma`) so the model can generalise across different price levels.
- **Indicator values:** the raw `moving_average` is transformed into `pct_diff_from_ma = (current_price - moving_average) / moving_average`, rather than being used raw, because a raw MA value isn't comparable across BTC's full price range over 19 months (a $97,500 MA means something very different at $60k BTC vs $120k BTC) — the percentage difference is scale-free.
- **Signal timing:** this was considered but not implemented into the trained model given the time restriction of the case study.
- **Bot agreement:** `pct_buy_agreement` and `active_bot_count`, computed via an as-of join against every other active bot's most recent decision at or before the MA/EMA signal's timestamp — explicitly excluding the three always-Hold bots, since they'd mechanically skew the percentage regardless of what the rest of the ensemble was doing. Only populated from 2026-04-03 onward, used only in the post-rollout model variant.

### 2b. Prediction

For the prediction model, a logistic regression model was used. Logistic regression is fast, easy to implement, and creates a clear formula that connects the input data to the result. The prediction model was built to be like the BasicMovingAverageBot and ExponentialMovingAverageBot, as the brief's worked example ties `indicator_value` to a single scalar on the price scale.

The model was trained using an 80/20 split on the available data, using the earliest 80% of dates for training and the remaining 20% for testing. Chronological ordering was chosen over a random shuffle because the 24-hour label windows overlap for signals fired close together (bots fire roughly hourly). A random shuffle would let a training row "see" almost the same future price window as a test row, leaking information across the split.

The model was compared against three baselines: always-predict-up, majority class, and "follow the bot" (taking the bot's own historical Buy/Sell call as the prediction). The first two test whether the model beats trivial guessing, and the third tests whether modelling the underlying data adds anything over trusting the bot's built-in logic.

To measure the output of the model, precision, recall, F1 and ROC-AUC were used, rather than accuracy alone. This was to ensure that the model did not appear accurate by simply predicting the majority class. The results are shown below:

```
=== PRE 2026-04-03 (legacy, 4-bot universe) (n=11678) ===
[baseline: always up ]     accuracy=0.447  precision=0.447  recall=1.000  f1=0.618
[baseline: majority class] accuracy=0.447  precision=0.447  recall=1.000  f1=0.618
[baseline: follow bot]     accuracy=0.495  precision=0.442  recall=0.496  f1=0.468
[logistic regression ]     accuracy=0.451  precision=0.445  recall=0.926  f1=0.602  roc_auc=0.499

=== POST 2026-04-03 (full ensemble) (n=839) ===
[baseline: always up ]     accuracy=0.440  precision=0.440  recall=1.000  f1=0.612
[baseline: majority class] accuracy=0.440  precision=0.440  recall=1.000  f1=0.612
[baseline: follow bot]     accuracy=0.649  precision=0.577  recall=0.757  f1=0.655
[logistic regression ]     accuracy=0.440  precision=0.440  recall=1.000  f1=0.612  roc_auc=0.296
```

ROC-AUC mattered most since it's threshold-independent and tells you whether the model's probability ranking means anything at all. The results show that, pre-rollout of the other bots, the model's ranking ability is as good as a coin flip (ROC-AUC = 0.499). Post-rollout, ROC-AUC = 0.296 (below chance, meaning its rankings were, if anything, inverted), and the model collapsed to predicting "up" for every test row.

The engineered features didn't produce a model that reliably beat chance in either era; the "follow the bot" baseline outperformed the trained model in the post-rollout period (64.9% vs 44.0% accuracy); `pct_diff_from_ma`'s effect was small and inconsistent across eras.

---

## Part 3 — Interactive Interface

The interface, hosted on Streamlit, displays both the model's prediction and the baseline to ensure transparent evaluation, given the model's near-chance accuracy. The pre-rollout model was chosen for deployment because it is larger and more stable, while the post-rollout model was discarded after collapsing during training. The Explore tab uses a three-part sequence to walk reviewers through the analytical process:

- **Signal History Chart:** plots `current_price` against `moving_average` over time to visually demonstrate why `pct_diff_from_ma` was selected as a key feature.
- **Feature Distribution Chart:** displays a histogram of `pct_diff_from_ma` to illustrate data spread and show whether signals provided genuine class separation.
- **Evaluation Summary Table:** surfaces baseline vs. model performance metrics directly in the app to show near-chance results and contextualise live predictions.

**Logical flow:** the layout is arranged sequentially (relationship, distribution, then performance) to mirror the natural path of model investigation.

---

## Part 4 — Written Reflection

### Limitations
- Trained model performs at/below chance in both eras (ROC-AUC = 0.499 pre-rollout, = 0.296 post-rollout) — the "follow the bot" baseline beat it outright post-rollout.
- Post-rollout training set is tiny (~1 month) — too small and too regime-specific to generalise; confirmed via the `class_weight="balanced"` test, which ruled out imbalance as the cause.
- Only two bot families modelled (MA, EMA) — RSI and MacdEma were explored but not built out, scoped down to match the brief's single-scalar interface example.
- "Signal timing" — one of the brief's four named feature categories — was never actually engineered into the model.
- Bot agreement is undefined pre-2026-04-03 (only 4 bots existed), so it's only usable in the post-rollout model variant — an inconsistency between the two era models rather than one unified approach.
- The 24-hour label horizon was chosen from reasoned trade-offs, not tested empirically against alternatives (1h, multi-day).
- The as-of price lookup with a 3-hour tolerance drops a small number of rows (82) near the April gap.
- Correlated-subquery bot agreement calculation doesn't scale — fine at this dataset's size, not production-viable as-is.
- Single-symbol scope (BTC/USDT only) — no evidence this generalises to other assets.

### If I had more time
- Build the RSI and MacdEma models, and add a bot-family selector to the Streamlit interface.
- Engineer and test signal-timing features (hour of day, time since the bot's last signal).
- Systematically test multiple horizons (1h, 24h, multi-day) rather than picking one and reasoning about it in the abstract.
- Try heavier models (tree-based/ensemble) once there's enough post-rollout history to support them without overfitting.

### Production concerns (running with real money)
- A model at/below chance should never size real capital — this result alone would block production use without major rework.
- No transaction costs, slippage, or spread modelled anywhere in the evaluation. Accuracy numbers say nothing about whether any edge survives real trading costs.
- Having access to the latest data would be necessary to keep the model current. Data-quality monitoring would also be important, so production gets alerted when there's a problem with the data rather than relying on human review.
- No model retraining/versioning strategy: given bot cadence, and that the roster has already changed once (the April rollout), the model needs a plan for when/how it gets refreshed.
- Needs a proper backtest with realistic order execution before any live capital, not just a held-out accuracy check.
- Have a capital limit in place to ensure that if the bot glitches or gets caught in a loop, not all of the capital gets spent.

### Data wishlist
- Order book / bid-ask spread data — needed to judge whether any model edge survives real execution costs.
- More post-rollout history — 5 weeks before the outage isn't enough to train or validate the ensemble-era model properly.
- Volume data — absent from all three tables, and a natural feature for MA-style signals.
- Additional symbols beyond BTC/USDT, to test whether anything here generalises.
- More rows in `agent_decisions` — only 19 rows across 2 days; more history would make it a genuine downstream evaluation set rather than just qualitative context.
- Ground-truth outcomes at multiple horizons, to properly test horizon choice rather than justify one from first principles.

### Productionisation (Databricks-specific)
- **Orchestration:** Databricks Workflows/Jobs to run the bronze → silver → gold pipeline on a schedule matching bot cadence (roughly hourly).
- **Incremental loads:** Delta Lake MERGE/upsert keyed on signal `id`, with a timestamp watermark, instead of full reprocessing on every run.
- **Data quality checks as pipeline code, not manual analysis:** Delta Live Tables expectations (or equivalent) to catch schema drift (like the MacdEmaBot `ema` key) and gaps automatically, rather than relying on someone re-running the exploration by hand.
- **Cost:** job clusters that spin down when idle, or serverless SQL warehouses for the gold-layer queries, instead of an always-on cluster.
- **Monitoring:** Lakehouse Monitoring or an equivalent dashboard tracking data freshness, schema drift, class balance drift, and model performance drift over time.
- **Model serving:** register the model via MLflow Model Registry and serve through Databricks Model Serving for low-latency inference, instead of loading a `.joblib` file in a notebook or Streamlit app.
- **CI/CD:** Databricks Asset Bundles or repo integration with GitHub Actions, so pipeline and model changes deploy automatically rather than being manually copied between platforms.
