"""
Phase 6 — Chronological Train/Test Split
NIFTY 50 Binary Market Trend Classification

Input : NIFTY50_FEATURED.csv (already has target + engineered features)
Output: nifty50_train.csv, nifty50_test.csv, split_summary.md

Rules followed (per project workflow doc):
- NO random shuffling — a time-based split only.
- Earlier observations -> train, later observations -> test.
- Split boundary is chosen and documented after inspecting the dataset.
- Test period is left completely untouched (no scaling/fitting on it here).
- A TimeSeriesSplit object is also built for time-aware CV, in case
  Member 3 needs to tune hyperparameters using only the training period.
"""

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

# ------------------------------------------------------------------
# 1. Load data and sort chronologically (safety check — should already
#    be sorted, but we never assume that for a time split).
# ------------------------------------------------------------------
INPUT_PATH = "NIFTY50_FEATURED.csv"

df = pd.read_csv(INPUT_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

n_total = len(df)
print(f"Total rows: {n_total}")
print(f"Date range: {df['Date'].min().date()} -> {df['Date'].max().date()}")

# ------------------------------------------------------------------
# 2. Choose the split boundary.
#    Using an 80/20 chronological split, which is standard for this
#    kind of walk-forward setup and leaves a reasonably sized,
#    contiguous test window (~1 year of trading days here).
#    Change SPLIT_RATIO if the team agrees on a different ratio.
# ------------------------------------------------------------------
SPLIT_RATIO = 0.80
split_idx = int(n_total * SPLIT_RATIO)

train_df = df.iloc[:split_idx].reset_index(drop=True)
test_df = df.iloc[split_idx:].reset_index(drop=True)

split_date = test_df["Date"].iloc[0]

print(f"\nSplit index: {split_idx} (row {split_idx} of {n_total})")
print(f"Split boundary date: {split_date.date()}")
print(f"Train period: {train_df['Date'].min().date()} -> {train_df['Date'].max().date()} "
      f"({len(train_df)} rows)")
print(f"Test period : {test_df['Date'].min().date()} -> {test_df['Date'].max().date()} "
      f"({len(test_df)} rows)")

# ------------------------------------------------------------------
# 3. Sanity checks
# ------------------------------------------------------------------
# a) No overlap / no leakage of dates between the two sets
assert train_df["Date"].max() < test_df["Date"].min(), \
    "Leakage: train period overlaps with test period!"

# b) Train set must precede test set for every row (chronological, not shuffled)
assert train_df["Date"].is_monotonic_increasing
assert test_df["Date"].is_monotonic_increasing

# c) Class balance in each split (important to report — accuracy alone
#    can be misleading if a split is imbalanced)
train_balance = train_df["Target"].value_counts(normalize=True).sort_index()
test_balance = test_df["Target"].value_counts(normalize=True).sort_index()

print("\nTrain class balance (0=DOWN, 1=UP):")
print(train_balance)
print("\nTest class balance (0=DOWN, 1=UP):")
print(test_balance)

# ------------------------------------------------------------------
# 4. Save the split datasets (raw data is left untouched elsewhere —
#    these are just chronological subsets of the feature-engineered file)
# ------------------------------------------------------------------
train_df.to_csv("nifty50_train.csv", index=False)
test_df.to_csv("nifty50_test.csv", index=False)
print("\nSaved: nifty50_train.csv, nifty50_test.csv")

# ------------------------------------------------------------------
# 5. Time-aware CV splitter for hyperparameter tuning
#    (use ONLY on train_df — never touch test_df until final evaluation)
# ------------------------------------------------------------------
N_SPLITS = 5
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

print(f"\nTimeSeriesSplit ready with n_splits={N_SPLITS} for tuning on train_df only.")
print("Example fold boundaries (row indices within train_df):")
for fold, (tr_idx, val_idx) in enumerate(tscv.split(train_df), start=1):
    tr_start, tr_end = train_df["Date"].iloc[tr_idx[0]].date(), train_df["Date"].iloc[tr_idx[-1]].date()
    val_start, val_end = train_df["Date"].iloc[val_idx[0]].date(), train_df["Date"].iloc[val_idx[-1]].date()
    print(f"  Fold {fold}: train {tr_start}->{tr_end} | validate {val_start}->{val_end}")

# ------------------------------------------------------------------
# 6. Write a short markdown summary for the report / PR description
# ------------------------------------------------------------------
summary = f"""# Phase 6 — Chronological Train/Test Split Summary

**Method:** Time-based split (no shuffling). Earlier rows -> train, later rows -> test.

- Total rows: {n_total}
- Split ratio: {int(SPLIT_RATIO*100)}/{int((1-SPLIT_RATIO)*100)}
- Split boundary date: **{split_date.date()}**
- Train period: {train_df['Date'].min().date()} to {train_df['Date'].max().date()} ({len(train_df)} rows)
- Test period: {test_df['Date'].min().date()} to {test_df['Date'].max().date()} ({len(test_df)} rows)

## Class balance (Target: 0 = DOWN, 1 = UP)

| Split | DOWN (0) | UP (1) |
|---|---|---|
| Train | {train_balance.get(0, 0):.3f} | {train_balance.get(1, 0):.3f} |
| Test  | {test_balance.get(0, 0):.3f} | {test_balance.get(1, 0):.3f} |

## Leakage checks passed
- Train dates strictly precede test dates (max train date < min test date).
- Both splits internally sorted chronologically (never shuffled).
- Test set is not touched again until final model evaluation (Phase 8).

## For Member 3 (tuning)
Use `TimeSeriesSplit(n_splits={N_SPLITS})` on `nifty50_train.csv` only for any
hyperparameter search / validation. Do not use `nifty50_test.csv` until the
final models are locked in.

## Files produced
- `nifty50_train.csv`
- `nifty50_test.csv`
"""

with open("split_summary.md", "w") as f:
    f.write(summary)

print("\nSaved: split_summary.md")
