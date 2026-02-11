import os
import json
import pickle
from datetime import datetime

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, precision_recall_curve, classification_report

from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# Config
# ============================================================
DATA_PATH = "fraud_oracle.csv"
TARGET_COL = "FraudFound_P"

MODEL_NAME = "NORDIC-ASSURE"
MODELS_DIR = "artifacts/models"          # versioned bundles live here
CURRENT_PTR = "artifacts/current.json"   # points to the active bundle
REGISTRY_CSV = "artifacts/registry.csv"  # append-only registry index

TARGET_PRECISION = 0.70                 # pick HIGH threshold to satisfy this precision (if possible)
DEFAULT_LOW = 0.30                      # LOW/MEDIUM cutoff
DEFAULT_HIGH = 0.70                     # fallback MEDIUM/HIGH cutoff if target precision not achievable


# ============================================================
# Versioning helpers (NORDIC-ASSURE_0001, NORDIC-ASSURE_0002, ...)
# ============================================================
def utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def next_model_version(models_dir: str, model_name: str) -> str:
    ensure_dir(models_dir)
    prefix = f"{model_name}_"
    max_n = 0
    for d in os.listdir(models_dir):
        full = os.path.join(models_dir, d)
        if not os.path.isdir(full):
            continue
        if not d.startswith(prefix):
            continue
        suffix = d[len(prefix):]
        if suffix.isdigit():
            max_n = max(max_n, int(suffix))
    return f"{model_name}_{max_n+1:04d}"

def append_registry_row(row: dict, path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    df = pd.DataFrame([row])
    if os.path.exists(path):
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


# ============================================================
# Training
# ============================================================
def build_xgb_baseline(X: pd.DataFrame, y: pd.Series, target_precision: float = 0.70):
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Column types
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

    # class imbalance
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
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
        eval_metric="aucpr",
        tree_method="hist",     # change to "gpu_hist" for GPU
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight
    )

    model = Pipeline([
        ("pre", pre),
        ("clf", clf)
    ])

    # CV on train only
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pr_auc_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="average_precision")

    cv_pr_auc_mean = float(pr_auc_scores.mean())
    cv_pr_auc_std = float(pr_auc_scores.std())
    print("CV PR-AUC:", cv_pr_auc_mean, "+/-", cv_pr_auc_std)

    # Fit + evaluate on holdout
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    test_pr_auc = float(average_precision_score(y_test, proba))
    print("Test PR-AUC:", test_pr_auc)

    # Threshold selection: choose threshold giving >= target_precision if possible
    precision, recall, thr = precision_recall_curve(y_test, proba)
    idx = np.where(precision[:-1] >= target_precision)[0]
    chosen_thr = None
    if len(idx) > 0:
        chosen_thr = float(thr[idx[-1]])
        preds = (proba >= chosen_thr).astype(int)
        print(f"\nThreshold for ~{target_precision:.0%} precision: {chosen_thr:.3f}")
        print(classification_report(y_test, preds))
    else:
        print(f"\nCould not reach target precision ({target_precision:.0%}) on test set.")

    # Extra evaluation using default routing thresholds (for insight)
    out = pd.DataFrame({"y": y_test.values, "p": proba})
    bins = pd.cut(out["p"], bins=[-np.inf, DEFAULT_LOW, DEFAULT_HIGH, np.inf], labels=["Low", "Medium", "High"])
    out["risk"] = bins

    summary = out.groupby("risk").agg(n=("y", "size"), frauds=("y", "sum"))
    summary["fraud_rate"] = summary["frauds"] / summary["n"]
    print(summary)

    captured = int(out[out["risk"].isin(["Medium", "High"])]["y"].sum())
    total_fraud = int(out["y"].sum())
    fraud_recall_medium_high = float(captured / total_fraud) if total_fraud > 0 else float("nan")
    print("Fraud recall in Medium+High (default thresholds):", fraud_recall_medium_high)

    metrics = {
        "cv_pr_auc_mean": cv_pr_auc_mean,
        "cv_pr_auc_std": cv_pr_auc_std,
        "test_pr_auc": test_pr_auc,
        "target_precision": float(target_precision),
        "threshold_at_target_precision": chosen_thr,
        "fraud_recall_medium_high_default_thresholds": fraud_recall_medium_high,
        "class_balance_train": {"neg": neg, "pos": pos, "scale_pos_weight": float(scale_pos_weight)},
    }

    return model, metrics


# ============================================================
# Run
# ============================================================
df = pd.read_csv(DATA_PATH)

# Ensure numeric 0/1 target
y = df[TARGET_COL].astype(int)
X = df.drop(columns=[TARGET_COL])

model, metrics = build_xgb_baseline(X, y, target_precision=TARGET_PRECISION)

# ============================================================
# Save versioned model bundle
# artifacts/models/NORDIC-ASSURE_0001/{model.pkl, meta.json}
# and update artifacts/current.json
# ============================================================
ensure_dir(MODELS_DIR)

model_version = next_model_version(MODELS_DIR, MODEL_NAME)
bundle_dir = os.path.join(MODELS_DIR, model_version)
os.makedirs(bundle_dir, exist_ok=False)

MODEL_FILE = "model.pkl"
META_FILE = "meta.json"

model_path = os.path.join(bundle_dir, MODEL_FILE)
meta_path = os.path.join(bundle_dir, META_FILE)

with open(model_path, "wb") as f:
    pickle.dump(model, f)

# Thresholds stored for the API routing:
# - LOW/MEDIUM cutoff stays DEFAULT_LOW
# - MEDIUM/HIGH cutoff: use learned threshold if available, else DEFAULT_HIGH
opt_high = metrics.get("threshold_at_target_precision")
low_threshold = float(DEFAULT_LOW)
high_threshold = float(opt_high) if opt_high is not None else float(DEFAULT_HIGH)

meta = {
    "model_name": MODEL_NAME,
    "model_version": model_version,
    "created_utc": utc_now_iso(),

    "model_file": MODEL_FILE,
    "meta_file": META_FILE,

    "low_threshold": low_threshold,
    "high_threshold": high_threshold,

    # IMPORTANT: stable production schema
    "feature_columns": list(X.columns),

    # Useful tracking info
    "metrics": metrics,
    "training_data": {
        "source": DATA_PATH,
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "target_col": TARGET_COL,
        "positive_class": "fraud (1)",
    },
}

with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)

# Pointer for the API to load "current" model
ensure_dir(os.path.dirname(CURRENT_PTR))
with open(CURRENT_PTR, "w") as f:
    json.dump(
        {
            "current_model_version": model_version,
            "model_dir": bundle_dir,
            "updated_utc": utc_now_iso(),
        },
        f,
        indent=2,
    )

# Append registry row
append_registry_row(
    {
        "model_name": MODEL_NAME,
        "model_version": model_version,
        "created_utc": meta["created_utc"],
        "model_dir": bundle_dir,
        "model_path": model_path,
        "meta_path": meta_path,
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "cv_pr_auc_mean": metrics.get("cv_pr_auc_mean"),
        "test_pr_auc": metrics.get("test_pr_auc"),
    },
    path=REGISTRY_CSV,
)

print("\n✅ Saved versioned model bundle:")
print(f"   - {model_path}")
print(f"   - {meta_path}")
print(f"✅ Updated pointer: {CURRENT_PTR} -> {model_version}")
print(f"✅ Registry: {REGISTRY_CSV}")


# ------------------------------------------------------------
# Backwards-compatible "latest" outputs for existing API paths
# ------------------------------------------------------------
ensure_dir("artifacts")

LATEST_MODEL_PATH = os.path.join("artifacts", "fraud_model.pkl")
LATEST_META_PATH  = os.path.join("artifacts", "model_meta.json")

# Save latest model (same model object)
with open(LATEST_MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

# Save latest meta (same meta dict)
with open(LATEST_META_PATH, "w") as f:
    json.dump(meta, f, indent=2)

print(f"✅ Latest files for API compatibility:")
print(f"   - {LATEST_MODEL_PATH}")
print(f"   - {LATEST_META_PATH}")