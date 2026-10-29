from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             f1_score, log_loss, mean_squared_error, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.features import FEATURES, clean_shots

ROOT = Path(__file__).resolve().parent.parent
RANDOM_STATE = 42
THRESHOLD = 0.5      # gjelder bare accuracy/precision/recall/F1 og confusion matrix, ikke xG-verdien

# ---------- Data ----------
df = clean_shots(pd.read_csv(ROOT / "data" / "raw" / "shots.csv"))
print(f"Antall skudd: {len(df)}, kamper: {df['match_id'].nunique()}, "
      f"målandel: {df['goal'].mean():.4f}")

# Tilfeldig splitt av skudd (80/20)
X, y = df[FEATURES], df["goal"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
train, test = df.loc[X_train.index], df.loc[X_test.index]   # brukes i utforsking og feilanalyse
sb_xg = test["shot_statsbomb_xg"]          # kun til sammenligning, ikke feature
print(f"Trening: {len(train)} skudd, test: {len(test)} skudd")

# ---------- Utforsking (kun treningsdata) ----------
print("\nManglende verdier i treningsdata:")
print(train[FEATURES + ["goal"]].isna().sum())
print("\nGjennomsnitt av features for mål (1) og ikke mål (0):")
print(train.groupby("goal")[FEATURES].mean().round(3))


def evaluate(name, p, threshold=THRESHOLD):
    """p = predikert sannsynlighet for mål"""
    pred = (p >= threshold).astype(int)
    return {"modell": name,
            "log_loss": log_loss(y_test, p),
            "roc_auc": roc_auc_score(y_test, p),
            "brier": mean_squared_error(y_test, p),
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0)}


# flere modelltyper, kryssvalidering på treningsdata
models = {
    "Logistisk regresjon": make_pipeline(StandardScaler(), LogisticRegression()),
    "Beslutningstre (dybde 5)": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
    "Random forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    "Gradient boosting": HistGradientBoostingClassifier(
        max_depth=4, learning_rate=0.05, max_iter=200, random_state=RANDOM_STATE),
}

cv_scores = {}
for name, model in models.items():
    s = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
    cv_scores[name] = s.mean()

# ---------- 2. Finjuster random forest med GridSearchCV ----------
grid = GridSearchCV(
    RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    param_grid={"max_depth": [4, 6, 8, 12], "min_samples_leaf": [20, 50, 100, 200]},
    cv=5, scoring="roc_auc")
grid.fit(X_train, y_train)
print("\nBeste parametre:", grid.best_params_)
cv_scores["Random forest (tunet)"] = grid.best_score_

# ---------- 3. Velg modell ut fra kryssvalidering  ----------
print("\nCV ROC AUC (treningsdata):")
for name, score in cv_scores.items():
    print(f"  {name}: {score:.4f}")
best = max(cv_scores, key=cv_scores.get)
print("Valgt modell:", best)

# ---------- 4. Tren alle på hele treningssettet og evaluer på testsettet ----------
fitted = {name: model.fit(X_train, y_train) for name, model in models.items()}
fitted["Random forest (tunet)"] = grid.best_estimator_

results = [evaluate("Baseline (gjennomsnitt)", np.full(len(y_test), y_train.mean()))]
for name, model in fitted.items():
    results.append(evaluate(name, model.predict_proba(X_test)[:, 1]))
results.append(evaluate("StatsBomb xG (referanse)", sb_xg))

table = pd.DataFrame(results).round(4)
print("\nTestresultater:")
print(table.to_string(index=False))
(ROOT / "report").mkdir(exist_ok=True)
table.to_csv(ROOT / "report" / "resultater.csv", index=False)

# ---------- Feature importance (tunet random forest) ----------
imp = pd.Series(grid.best_estimator_.feature_importances_, index=FEATURES).sort_values()
print("\nFeature importance:")
print(imp)

# ---------- Lagre valgt modell ----------
(ROOT / "models").mkdir(exist_ok=True)
joblib.dump(fitted[best], ROOT / "models" / "model.joblib")

# ---------- Confusion matrix og ulike grenser ----------
p_best = fitted[best].predict_proba(X_test)[:, 1]
pred_best = (p_best >= THRESHOLD).astype(int)
print("\n", confusion_matrix(y_test, pred_best))

ConfusionMatrixDisplay.from_predictions(
    y_test, pred_best, display_labels=["Ikke mål", "Mål"])
plt.title(f"{best} (grense {THRESHOLD})")
plt.savefig(ROOT / "report" / "confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

thresholds = pd.DataFrame([evaluate(best, p_best, threshold=t) | {"grense": t}
                           for t in [0.1, 0.15, 0.2, 0.3, 0.5]])
print(thresholds[["grense", "accuracy", "precision", "recall", "f1"]].round(3).to_string(index=False))

# ---------- Kalibreringskurve ----------
frac_pos, mean_pred = calibration_curve(y_test, p_best, n_bins=10, strategy="quantile")
plt.figure(figsize=(5, 5))
plt.plot(mean_pred, frac_pos, "o-", label=best)
plt.plot([0, 1], [0, 1], "k--", label="Perfekt kalibrert")
plt.xlabel("Predikert sannsynlighet")
plt.ylabel("Faktisk andel mål")
plt.legend()
plt.savefig(ROOT / "report" / "kalibrering.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------- Feilanalyse: predikert vs. faktisk målandel per skuddtype ----------
test = test.assign(p=p_best)


def error_table(by):
    return test.groupby(by, observed=True).agg(
        n=("goal", "size"), faktisk=("goal", "mean"), predikert=("p", "mean")).round(3)


print("\nHode (1) vs. fot (0):")
print(error_table("is_header"))
print("\nDødball (1) vs. åpent spill (0):")
print(error_table("set_piece"))
print("\nAvstand (StatsBomb-koordinater):")
print(error_table(pd.cut(test["dist"], [0, 6, 12, 18, 25, 200])))