# nordic_assure

**nordic_assure** is a machine-learning project that builds a **binary classifier** to predict whether an automobile insurance claim is **fraudulent** or **legitimate**.

## Quickstart

### 1) Clone
```bash
git clone https://github.com/albino01/nordic_assure.git
cd nordic_assure
```

# nordic_assure

## What the training script does

The training script builds and evaluates an **XGBoost-based vehicle fraud classifier** using `fraud_oracle.csv`, then exports the trained model and lightweight metadata for use in an API.

### 1) Load and split data
- Loads `fraud_oracle.csv` with pandas
- Uses `FraudFound_P` as the binary target (converted to `0/1`)
- Splits into **train (80%)** and **test (20%)** using a **stratified split** so the fraud rate stays consistent

### 2) Preprocess numeric and categorical features
Creates a scikit-learn preprocessing pipeline:
- **Numeric columns:** missing values → **median** imputation  
- **Categorical columns:** missing values → **most frequent** imputation, then **One-Hot Encoding**  
  - `handle_unknown="ignore"` prevents inference from breaking on unseen categories

Preprocessing is wrapped in a `ColumnTransformer` and combined with the model in a single `Pipeline`, ensuring the same transforms are applied during training and inference.

### 3) Train an imbalanced fraud model (XGBoost)
Fraud is typically class-imbalanced. The script:
- Computes `scale_pos_weight = (# non-fraud) / (# fraud)` on the training set
- Passes it to `XGBClassifier` to increase sensitivity to the fraud class
- Optimizes using **PR-AUC** (`eval_metric="aucpr"`), which is often more informative than ROC-AUC for rare-event detection

### 4) Cross-validation and test evaluation (PR-AUC)
- Runs **5-fold Stratified Cross-Validation** on the training split and prints mean ± std **PR-AUC**
- Fits the final model on the training set and reports **Test PR-AUC** on the holdout set

### 5) Threshold selection for a precision-first fraud screen
Instead of the default 0.5 cutoff, the script:
- Computes the **precision–recall curve** on the test set
- Selects a threshold targeting ~**70% precision** (if achievable)
- Prints a `classification_report` at that threshold

This supports workflows where **false positives are costly** and cases are sent to a manual review queue.

### 6) Risk bucketing for triage
Converts predicted fraud probability `p` into 3 risk tiers:
- **Low:** `p < 0.30`
- **Medium:** `0.30 ≤ p < 0.70`
- **High:** `p ≥ 0.70`

Prints a summary table with:
- number of claims per tier
- number of frauds per tier
- fraud rate per tier

Also reports **fraud recall in Medium + High**, i.e., how many fraud cases would be flagged for investigation.

### 7) Export artifacts for deployment
Saves:
- `artifacts/fraud_model.pkl` — full preprocessing + model pipeline (ready for inference)
- `artifacts/model_meta.json` — model version and thresholds for API routing/versioning