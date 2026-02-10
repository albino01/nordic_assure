import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, precision_recall_curve, classification_report

from xgboost import XGBClassifier

# --- Load data ---
df = pd.read_csv("fraud_oracle.csv")

#target_col = "FraudFound_P"
#y = df[target_col]

target_col = "FraudFound_P"
y = df[target_col].astype(int)   # keeps 0/1
X = df.drop(columns=[target_col])

"""Assuming positive = fraud:

TP (True Positive): predicted fraud and actually fraud

FP (False Positive): predicted fraud but actually not fraud (false alarm)

FN (False Negative): predicted not-fraud but actually fraud (missed fraud)

TN (True Negative): predicted not-fraud and actually not-fraud

Quick mapping:

	Actual Fraud (1)	Actual Not Fraud (0)
Predicted Fraud (1)	TP	FP
Predicted Not Fraud (0)	FN	TN

And the key formulas:

Precision = TP / (TP + FP)

Recall (TPR) = TP / (TP + FN)

FPR = FP / (FP + TN)
"""

### To be removed
## Convert target to 0/1 if it's Yes/No strings
#if y.dtype == "object":
#    y = (
#        y.astype(str)
#         .str.strip()
#         .str.lower()
#         .map({"yes": 1, "no": 0, "y": 1, "n": 0, "true": 1, "false": 0, "1": 1, "0": 0})
#    )

# Ensure numeric 0/1
y = y.astype(int)

X = df.drop(columns=[target_col])

feature_columns = list(X.columns)



def build_xgb_baseline(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    num_cols = X.columns.difference(cat_cols)

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])

    pre = ColumnTransformer([
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols),
    ])

    # Handle class imbalance for XGBoost
    # scale_pos_weight = (#negative / #positive)
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    clf = XGBClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        min_child_weight=1,
        gamma=0,
        objective="binary:logistic",
        eval_metric="aucpr",          # PR-AUC inside XGBoost training
        tree_method="hist",           # fast CPU training; use "gpu_hist" if you have GPU
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight
    )

    model = Pipeline([
        ("pre", pre),
        ("clf", clf)
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pr_auc_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="average_precision")
    print("CV PR-AUC:", pr_auc_scores.mean(), "+/-", pr_auc_scores.std())

    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]

    ap = average_precision_score(y_test, proba)
    print("Test PR-AUC:", ap)

    # Thresholding: pick threshold that yields ~80% precision (if possible)
    precision, recall, thr = precision_recall_curve(y_test, proba)
    target_precision = 0.70
    idx = np.where(precision[:-1] >= target_precision)[0]
    if len(idx) > 0:
        t = thr[idx[-1]]
        preds = (proba >= t).astype(int)
        print(f"\nThreshold for ~{target_precision:.0%} precision: {t:.3f}")
        print(classification_report(y_test, preds))
    else:
        print("\nCould not reach target precision on test set.")



    # extra evaluation
    proba = model.predict_proba(X_test)[:,1]
    out = pd.DataFrame({"y": y_test.values, "p": proba})

    bins = pd.cut(out["p"], bins=[-np.inf, 0.30, 0.70, np.inf], labels=["Low", "Medium", "High"])
    out["risk"] = bins

    summary = out.groupby("risk").agg(
        n=("y","size"),
        frauds=("y","sum"),
    )
    summary["fraud_rate"] = summary["frauds"] / summary["n"]
    print(summary)

    # Fraud recall captured by Medium+High
    captured = out[out["risk"].isin(["Medium","High"])]["y"].sum()
    total_fraud = out["y"].sum()
    print("Fraud recall in Medium+High:", captured / total_fraud)

    return model


model = build_xgb_baseline(X, y)

import os
import pickle
import json

# Make sure artifacts/ exists
os.makedirs("artifacts", exist_ok=True)

# Save model as pickle
with open("artifacts/fraud_model.pkl", "wb") as f:
    pickle.dump(model, f)

## >>> UPDATED: Save feature_columns in metadata
meta = {
    "model_version": "1.0",
    "low_threshold": 0.30,
    "high_threshold": 0.70,
    "feature_columns": feature_columns
}
with open("artifacts/model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("Saved artifacts/fraud_model.pkl and artifacts/model_meta.json")