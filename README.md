# 📈 NIFTY 50 Binary Market Trend Classification & Risk Analysis

A reproducible machine learning project designed to classify the next-day price direction (**UP vs DOWN**) of the NIFTY 50 index using approximately five years of historical daily OHLC data. The system evaluates and compares three supervised learning models (Logistic Regression, Random Forest, and Gradient Boosting) while tracking key risk metrics.

---

## 👥 Team Roles & Task Ownership

### 📊 Member 1 — Data & EDA
* **Ownership:** Data Collection, Cleaning & Exploratory Data Analysis (EDA).
* **Key Tasks:** Collect ~5 years of daily NIFTY 50 data, validate prices, handle missing values, and create 5–7 baseline charts (price, returns, volatility).
* **Deliverables:** `data/raw/nifty50_raw.csv`, `data/processed/nifty50_cleaned.csv`, and `notebooks/1_data_cleaning.ipynb`.

### 💡 Member 2 (Repo Owner) — Features & Target
* **Ownership:** Target Definition & Feature Engineering.
* **Key Tasks:** Define the binary target, implement technical indicators (SMA 5/10/20/50, RSI, MACD, Momentum, Volatility), and perform strict data-leakage checks.
* **Deliverables:** Feature formulas documentation, data-leakage check logs, and `data/processed/nifty50_features.csv`.

### 🤖 Member 3 — ML & Evaluation
* **Ownership:** Model Preparation, Training & Evaluation.
* **Key Tasks:** Design a chronological (time-based) train/test split, train the classifiers, generate confusion matrices, and run the risk analysis (VaR, Max Drawdown).
* **Deliverables:** Trained model artifacts, model comparison charts, evaluation metrics tables, and `notebooks/3_model_training.ipynb`.

---

## 📂 Repository Layout

```text
nifty50-trend-classification/
│
├── data/
│   ├── raw/          <-- Raw, untouched historical source data [1]
│   └── processed/    <-- Cleaned data & engineered feature datasets [1]
│
├── notebooks/
│   ├── 1_data_cleaning.ipynb  <-- Phase 1 & 2 Data work [1]
│   ├── 2_feature_eng.ipynb   <-- Phase 3 & 4 Math & Indicators [1]
│   └── 3_model_training.ipynb <-- Phase 6 to 10 ML & Evaluation [1]
│
├── .gitignore        <-- Excludes environment and cache files
└── README.md         <-- Project overview and guidelines [1]
```

---

## ⚠️ Pipeline Rules & Guardrails
To protect the integrity of the machine learning experiments, all team members must respect these core engineering parameters:
1. **No Shuffling:** The dataset **must** use a chronological split (past data for training, future data for testing). Random train/test splits are strictly forbidden to maintain time-series integrity.
2. **Leakage Prevention:** Features computed at time $t$ must only use information available at or before time $t$. No lookahead data can be used to predict time $t+1$.
3. **Target Rigidity:** The pipeline is strictly binary (1 for UP, 0 for DOWN). Do not add a third neutral/stable class.
