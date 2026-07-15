"""
Predictive Modeling Using Machine Learning
===========================================
Business problem: predict whether a telecom customer will churn (Yes/No)
based on their account and usage attributes.

Pipeline:
  1. Load data
  2. Preprocess (encode categoricals, scale numerics)
  3. Train/test split
  4. Train three models: Logistic Regression, Decision Tree, Random Forest
  5. Evaluate with accuracy / precision / recall / F1 / ROC-AUC
  6. Visualize: confusion matrices + ROC curves + feature importance
  7. Save a text summary of results

Run:
    python3 churn_prediction.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score, classification_report
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "customer_churn.csv")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42

# -----------------------------------------------------------------------------
# 1. Load data
# -----------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

target_col = "churn"
id_col = "customer_id"

X = df.drop(columns=[target_col, id_col])
y = (df[target_col] == "Yes").astype(int)

numeric_features = [
    "age", "senior_citizen", "tenure_months", "monthly_charges",
    "total_charges", "num_support_calls",
]
categorical_features = [c for c in X.columns if c not in numeric_features]

# -----------------------------------------------------------------------------
# 2. Train/test split
# -----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# -----------------------------------------------------------------------------
# 3. Preprocessing pipeline
# -----------------------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# -----------------------------------------------------------------------------
# 4. Define models
# -----------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1
    ),
}

results = {}
fitted_pipelines = {}

for name, model in models.items():
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    fitted_pipelines[name] = pipe

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    results[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "y_pred": y_pred,
        "y_prob": y_prob,
        "report": classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]),
    }
    print(f"\n=== {name} ===")
    print(f"Accuracy:  {results[name]['accuracy']:.3f}")
    print(f"Precision: {results[name]['precision']:.3f}")
    print(f"Recall:    {results[name]['recall']:.3f}")
    print(f"F1 score:  {results[name]['f1']:.3f}")
    print(f"ROC AUC:   {results[name]['roc_auc']:.3f}")

# -----------------------------------------------------------------------------
# 5. Visualizations
# -----------------------------------------------------------------------------

# --- Confusion matrices (one figure, 3 subplots) ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res["y_pred"])
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"{name}\nConfusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["No Churn", "Churn"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["No Churn", "Churn"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrices.png"), dpi=150)
plt.close()

# --- ROC curves (overlay) ---
plt.figure(figsize=(6.5, 6))
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    plt.plot(fpr, tpr, label=f"{name} (AUC = {res['roc_auc']:.3f})", linewidth=2)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — Model Comparison")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "roc_curves.png"), dpi=150)
plt.close()

# --- Model comparison bar chart ---
metrics_df = pd.DataFrame({
    name: {k: v for k, v in res.items() if k in ["accuracy", "precision", "recall", "f1", "roc_auc"]}
    for name, res in results.items()
}).T

ax = metrics_df.plot(kind="bar", figsize=(9, 5), rot=0)
ax.set_title("Model Performance Comparison")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "model_comparison.png"), dpi=150)
plt.close()

# --- Feature importance (Random Forest) ---
rf_pipe = fitted_pipelines["Random Forest"]
feature_names = rf_pipe.named_steps["preprocess"].get_feature_names_out()
importances = rf_pipe.named_steps["model"].feature_importances_
imp_series = pd.Series(importances, index=feature_names).sort_values(ascending=True).tail(15)

plt.figure(figsize=(8, 6))
imp_series.plot(kind="barh", color="#4C72B0")
plt.title("Top 15 Feature Importances — Random Forest")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "feature_importance.png"), dpi=150)
plt.close()

# -----------------------------------------------------------------------------
# 6. Save results summary
# -----------------------------------------------------------------------------
summary_path = os.path.join(OUT_DIR, "results_summary.txt")
with open(summary_path, "w") as f:
    f.write("PREDICTIVE MODELING RESULTS — CUSTOMER CHURN\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Dataset: {DATA_PATH}\n")
    f.write(f"Total rows: {len(df)} | Train: {len(X_train)} | Test: {len(X_test)}\n")
    f.write(f"Churn rate: {y.mean():.1%}\n\n")

    f.write(metrics_df.round(3).to_string())
    f.write("\n\n")

    best_model = metrics_df["roc_auc"].idxmax()
    f.write(f"Best model by ROC-AUC: {best_model} ({metrics_df.loc[best_model, 'roc_auc']:.3f})\n\n")

    for name, res in results.items():
        f.write(f"\n--- {name}: classification report ---\n")
        f.write(res["report"])
        f.write("\n")

print(f"\nAll outputs saved to: {OUT_DIR}")
print(f"Summary written to: {summary_path}")
