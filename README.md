# nordic_assure
Auto-Insurance-Fraud-Detection is a machine-learning project that builds a binary classifier to predict whether an automobile insurance claim is fraudulent or legitimate.

## What this script does (README-ready)

This script trains and evaluates an XGBoost-based vehicle insurance fraud classifier using the dataset in fraud_oracle.csv, then exports the trained model and simple routing metadata for use in an API.

1) Load and split data

Loads fraud_oracle.csv with pandas.

Uses FraudFound_P as the binary target (converted to 0/1).

Splits the dataset into train (80%) and test (20%) using a stratified split so the fraud rate stays consistent across splits.

2) Preprocess numeric + categorical features

Builds a scikit-learn preprocessing pipeline:

Numeric columns: missing values are filled using the median.

Categorical columns: missing values are filled using the most frequent value, then categories are converted to machine-readable features using One-Hot Encoding (handle_unknown="ignore" ensures unseen categories won’t break inference).

This preprocessing is wrapped in a ColumnTransformer and combined with the model using a single Pipeline, so training and inference always apply the same transformations.

3) Train an imbalanced fraud model (XGBoost)

Fraud datasets are typically imbalanced (many more non-fraud than fraud). The script:

Computes scale_pos_weight = (# non-fraud) / (# fraud) on the training set.

Passes this value into XGBClassifier to make the model pay more attention to the minority (fraud) class.

XGBoost is configured for binary classification and optimized using PR-AUC (eval_metric="aucpr"), which is more informative than ROC-AUC when positives are rare.

4) Cross-validation + test evaluation (PR-AUC)

Runs 5-fold Stratified CV on the training split and prints mean ± std of PR-AUC (Average Precision).

Fits on the full training set and evaluates on the holdout test set, printing Test PR-AUC.

5) Threshold selection for a “precision-first” fraud screen

Instead of using the default 0.5 cutoff, the script:

Computes the precision–recall curve on the test set.

Picks a decision threshold that achieves approximately 70% precision (if possible).

Prints a classification_report at that threshold.

This is useful in fraud detection where false positives are costly and you often want high precision for review queues.

6) Risk bucketing for triage

It converts predicted fraud probability p into 3 human-friendly risk tiers:

Low: p < 0.30

Medium: 0.30 ≤ p < 0.70

High: p ≥ 0.70

Then it prints a summary table showing:

number of claims in each tier,

number of frauds captured,

fraud rate per tier,

and also reports fraud recall captured by Medium + High, i.e., how many fraud cases would be flagged for investigation.

7) Export artifacts for deployment

Finally, it saves:

artifacts/fraud_model.pkl — the full preprocessing + model pipeline (ready for inference)

artifacts/model_meta.json — simple metadata (model version and thresholds) useful for API routing/versioning