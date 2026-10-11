from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             f1_score, mean_squared_error, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.features import FEATURES, clean_shots

THRESHOLD = 0.5      # grensen for å si "mål". Prøv også f.eks. 0.2

# ---------- Data ----------
df = clean_shots(pd.read_csv("data/raw/shots.csv"))
X = df[FEATURES]
y = df["goal"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
sb_xg = df.loc[X_test.index, "shot_statsbomb_xg"]      # kun til sammenligning


def evaluate(name, p, threshold=THRESHOLD):
    """p = predikert sannsynlighet for mål"""
    pred = (p >= threshold).astype(int)
    return {"modell": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, p),
            "mse": mean_squared_error(y_test, p)}


results = []

# ---------- Baseline: samme sannsynlighet for alle skudd ----------
results.append(evaluate("Baseline (gjennomsnitt)", np.full(len(y_test), y_train.mean())))

# ---------- Modeller ----------
models = {
    "Logistisk regresjon": make_pipeline(StandardScaler(), LogisticRegression()),
    "Beslutningstre (dybde 5)": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
}

fitted = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    fitted[name] = model
    results.append(evaluate(name, model.predict_proba(X_test)[:, 1]))

# ---------- Hyperparameter-optimalisering av random forest ----------
grid = GridSearchCV(
    RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    param_grid={"max_depth": [4, 6, 8, 12], "min_samples_leaf": [5, 20, 50]},
    cv=5, scoring="roc_auc")
grid.fit(X_train, y_train)
print("Beste parametre:", grid.best_params_)
fitted["Random forest (tunet)"] = grid.best_estimator_
results.append(evaluate("Random forest (tunet)", grid.predict_proba(X_test)[:, 1]))

# ---------- StatsBombs egen xG som referanse (IKKE brukt som feature) ----------
results.append(evaluate("StatsBomb xG (referanse)", sb_xg))

table = pd.DataFrame(results).round(4)
print(table.to_string(index=False))

# ---------- Feature importance (tunet random forest) ----------
imp = pd.Series(grid.best_estimator_.feature_importances_, index=FEATURES).sort_values()
print(imp)

# ---------- Velg og lagre beste modell (høyest ROC AUC) ----------
best = max(fitted, key=lambda n: roc_auc_score(y_test, fitted[n].predict_proba(X_test)[:, 1]))
print("Lagrer:", best)
Path("models").mkdir(exist_ok=True)
joblib.dump(fitted[best], "models/model.joblib")

# ---------- Confusion matrix og ulike grenser for beste modell ----------
p_best = fitted[best].predict_proba(X_test)[:, 1]
pred_best = (p_best >= THRESHOLD).astype(int)
print(confusion_matrix(y_test, pred_best))

Path("report").mkdir(exist_ok=True)
ConfusionMatrixDisplay.from_predictions(
    y_test, pred_best, display_labels=["Ikke mål", "Mål"])
plt.title(f"{best} (grense {THRESHOLD})")
plt.savefig("report/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()

thresholds = pd.DataFrame([evaluate(best, p_best, threshold=t) | {"grense": t}
                           for t in [0.1, 0.15, 0.2, 0.3, 0.5]])
print(thresholds[["grense", "accuracy", "precision", "recall", "f1"]].round(3).to_string(index=False))