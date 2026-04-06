# ============================================================
# Kaggle Playground Series S6E4 — Predicting Irrigation Need
# Metric: Balanced Accuracy | Classes: Low / Medium / High
# ============================================================

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import minimize

import lightgbm as lgb
from catboost import CatBoostClassifier

# ── Optional: XGBoost
# import xgboost as xgb

# ============================================================
# 1. LOAD DATA
# ============================================================
train = pd.read_csv("/kaggle/input/playground-series-s6e4/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s6e4/test.csv")
sub   = pd.read_csv("/kaggle/input/playground-series-s6e4/sample_submission.csv")

print("Train shape:", train.shape)
print("Test  shape:", test.shape)
print("\nTarget distribution:")
print(train["Irrigation_Need"].value_counts(normalize=True).round(3))
print("\nColumns:", train.columns.tolist())
print("\nDtypes:\n", train.dtypes)
print("\nMissing values:\n", train.isnull().sum())

# ============================================================
# 2. BASIC EDA
# ============================================================
print("\nDescribe (numeric):")
print(train.describe())

# Check categorical columns
cat_cols = train.select_dtypes(include="object").columns.tolist()
cat_cols = [c for c in cat_cols if c != "Irrigation_Need"]
print(f"\nCategorical columns: {cat_cols}")
for col in cat_cols:
    print(f"  {col}: {train[col].nunique()} unique → {train[col].value_counts().head(5).to_dict()}")

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
TARGET = "Irrigation_Need"

def engineer_features(df):
    df = df.copy()

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    num_cols = [c for c in num_cols if c != "id"]

    # Common irrigation-relevant interactions
    # These will be created if the columns exist — adjust based on actual features!
    feature_pairs = [
        ("Temperature", "Humidity"),
        ("Temperature", "Soil_Moisture"),
        ("Humidity", "Soil_Moisture"),
        ("Wind_Speed", "Temperature"),
        ("Solar_Radiation", "Temperature"),
    ]
    for a, b in feature_pairs:
        if a in df.columns and b in df.columns:
            df[f"{a}_x_{b}"] = df[a] * df[b]
            df[f"{a}_div_{b}"] = df[a] / (df[b] + 1e-9)

    # Ratio features for common irrigation domains
    for col in num_cols:
        if col in df.columns:
            df[f"{col}_sq"] = df[col] ** 2

    return df

train = engineer_features(train)
test  = engineer_features(test)

# ============================================================
# 4. LABEL ENCODING
# ============================================================
le = LabelEncoder()
train[TARGET] = le.fit_transform(train[TARGET])
# Classes: Low=0, Medium=1, High=2 (alphabetical by LabelEncoder)
print(f"\nClass mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Encode categoricals
cat_cols = train.select_dtypes(include="object").columns.tolist()
for col in cat_cols:
    le_col = LabelEncoder()
    combined = pd.concat([train[col], test[col]], axis=0).astype(str)
    le_col.fit(combined)
    train[col] = le_col.transform(train[col].astype(str))
    test[col]  = le_col.transform(test[col].astype(str))

FEATURES = [c for c in train.columns if c not in ["id", TARGET]]
X = train[FEATURES]
y = train[TARGET]
X_test = test[FEATURES]

print(f"\nFeatures used: {len(FEATURES)}")

# ============================================================
# 5. CROSS-VALIDATION + LIGHTGBM (multi-seed)
# ============================================================
N_SPLITS = 5
SEEDS    = [42, 2024, 7]

lgb_oof   = np.zeros((len(train), 3))
lgb_preds = np.zeros((len(test),  3))

lgb_params_base = {
    "objective":         "multiclass",
    "num_class":         3,
    "metric":            "multi_logloss",
    "learning_rate":     0.05,
    "num_leaves":        127,
    "max_depth":         -1,
    "min_child_samples": 20,
    "feature_fraction":  0.8,
    "bagging_fraction":  0.8,
    "bagging_freq":      1,
    "reg_alpha":         0.1,
    "reg_lambda":        0.1,
    "n_estimators":      2000,
    "verbose":           -1,
    # Multiclass-correct imbalance handling (is_unbalance is binary-only)
    "class_weight":      "balanced",
}

print("\n── LightGBM Cross-Validation (multi-seed) ──")
for seed in SEEDS:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(**lgb_params_base, random_state=seed)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )

        # Average across seeds: each seed contributes 1/len(SEEDS) to OOF
        lgb_oof[val_idx] += model.predict_proba(X_val) / len(SEEDS)
        lgb_preds        += model.predict_proba(X_test) / (N_SPLITS * len(SEEDS))

    print(f"  Seed {seed} done")

lgb_cv_score = balanced_accuracy_score(y, np.argmax(lgb_oof, axis=1))
print(f"\n✅ LightGBM OOF Balanced Accuracy: {lgb_cv_score:.5f}")

# ============================================================
# 6. CROSS-VALIDATION + CATBOOST (multi-seed)
# ============================================================
cb_oof   = np.zeros((len(train), 3))
cb_preds = np.zeros((len(test),  3))

cb_params_base = dict(
    iterations=2000,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=3,
    loss_function="MultiClass",
    # TotalF1 (macro-averaged) is the closest multiclass eval metric to
    # Balanced Accuracy that CatBoost ships natively → better early stopping.
    eval_metric="TotalF1:average=Macro",
    early_stopping_rounds=100,
    verbose=False,
    auto_class_weights="Balanced",
)

print("\n── CatBoost Cross-Validation (multi-seed) ──")
for seed in SEEDS:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = CatBoostClassifier(**cb_params_base, random_seed=seed)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val))

        cb_oof[val_idx] += model.predict_proba(X_val) / len(SEEDS)
        cb_preds        += model.predict_proba(X_test) / (N_SPLITS * len(SEEDS))

    print(f"  Seed {seed} done")

cb_cv_score = balanced_accuracy_score(y, np.argmax(cb_oof, axis=1))
print(f"\n✅ CatBoost OOF Balanced Accuracy: {cb_cv_score:.5f}")

# ============================================================
# 7. ENSEMBLE — search optimal weight for Balanced Accuracy
# ============================================================
def neg_balacc(w_arr, oof_a, oof_b, y_true):
    w = np.clip(w_arr[0], 0.0, 1.0)
    blended = w * oof_a + (1 - w) * oof_b
    return -balanced_accuracy_score(y_true, np.argmax(blended, axis=1))

best = minimize(neg_balacc, x0=[0.5], args=(lgb_oof, cb_oof, y),
                method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-5})
w_opt = float(np.clip(best.x[0], 0.0, 1.0))
print(f"\nOptimal LGB weight: {w_opt:.3f}  (CB weight: {1-w_opt:.3f})")

ensemble_oof   = w_opt * lgb_oof   + (1 - w_opt) * cb_oof
ensemble_preds = w_opt * lgb_preds + (1 - w_opt) * cb_preds

ensemble_score = balanced_accuracy_score(y, np.argmax(ensemble_oof, axis=1))
print(f"🏆 Ensemble OOF Balanced Accuracy (raw argmax): {ensemble_score:.5f}")

# ============================================================
# 7b. THRESHOLD / PRIOR TUNING for Balanced Accuracy
# ------------------------------------------------------------
# Multiply class probabilities by per-class scaling factors and pick
# the argmax. This compensates residual class bias and typically adds
# +0.005–0.02 on Balanced Accuracy without retraining.
# ============================================================
def neg_balacc_scale(scale, oof, y_true):
    s = np.clip(scale, 1e-3, None)
    return -balanced_accuracy_score(y_true, np.argmax(oof * s, axis=1))

res = minimize(neg_balacc_scale, x0=np.ones(3), args=(ensemble_oof, y),
               method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-5, "maxiter": 500})
class_scale = np.clip(res.x, 1e-3, None)
class_scale = class_scale / class_scale.sum() * 3  # normalise for readability
print(f"Optimal class scaling: {dict(zip(le.classes_, class_scale.round(3)))}")

tuned_oof_score = balanced_accuracy_score(y, np.argmax(ensemble_oof * class_scale, axis=1))
print(f"🏆 Ensemble OOF Balanced Accuracy (tuned):     {tuned_oof_score:.5f}")

# ============================================================
# 8. SUBMISSION
# ============================================================
final_preds  = np.argmax(ensemble_preds * class_scale, axis=1)
final_labels = le.inverse_transform(final_preds)

sub["Irrigation_Need"] = final_labels
sub.to_csv("submission.csv", index=False)
print("\n✅ submission.csv gespeichert!")
print(sub["Irrigation_Need"].value_counts())
