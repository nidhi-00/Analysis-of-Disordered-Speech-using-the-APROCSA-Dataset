import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent

stats = pd.read_csv(PROJECT_ROOT / "Outputs/Statistics/a3_table.csv")
ratings = pd.read_csv(PROJECT_ROOT / "Outputs/perceptual_ratings.csv")

par = stats[stats["speaker"] == "PAR"]
df = par.merge(ratings, on="file")

features = [
    "utterances",
    "tokens",
    "mean_utterance_length",
    "ttr",
    "pause_markers_per_100_tokens",
    "filled_pauses_per_100_tokens",
]

X = df[features].to_numpy()
y = df["anomia"].to_numpy()

# ---- nested LOOCV to pick Ridge alpha on each train fold (small but correct) ----
outer = LeaveOneOut()
alphas = np.logspace(-3, 3, 25)

preds = []
preds_clipped = []
best_alphas = []

for train_idx, test_idx in outer.split(X):
    X_train, y_train = X[train_idx], y[train_idx]
    X_test = X[test_idx]

    # inner LOOCV to choose alpha
    inner = LeaveOneOut()
    best_alpha = None
    best_mae = float("inf")

    for a in alphas:
        inner_preds = []
        for tr2, te2 in inner.split(X_train):
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=a))
            ])
            pipe.fit(X_train[tr2], y_train[tr2])
            inner_preds.append(pipe.predict(X_train[te2])[0])

        mae_inner = mean_absolute_error(y_train, inner_preds)
        if mae_inner < best_mae:
            best_mae = mae_inner
            best_alpha = a

    best_alphas.append(best_alpha)

    # fit with chosen alpha on full train fold
    final_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=best_alpha))
    ])
    final_pipe.fit(X_train, y_train)

    p = float(final_pipe.predict(X_test)[0])
    preds.append(p)

    # optional: clip to rating scale 0–4
    preds_clipped.append(float(np.clip(p, 0, 4)))

preds = np.array(preds)
preds_clipped = np.array(preds_clipped)

mae = mean_absolute_error(y, preds)
mae_clip = mean_absolute_error(y, preds_clipped)
rho, pval = spearmanr(y, preds_clipped)

print("True Anomia:", y)
print("Predicted Anomia (raw):", np.round(preds, 2))
print("Predicted Anomia (clipped 0–4):", np.round(preds_clipped, 2))
print("Chosen alphas per fold:", np.round(best_alphas, 5))
print("MAE (raw):", round(mae, 3))
print("MAE (clipped):", round(mae_clip, 3))
print("Spearman corr (true vs clipped preds): rho =", round(rho, 3), "p =", round(pval, 3))

# ---- B4 coefficients (fit once on full data, using a reasonable alpha) ----
# pick median alpha chosen across folds (stable choice for reporting)
alpha_report = float(np.median(best_alphas))

pipe_full = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=alpha_report))
])
pipe_full.fit(X, y)

coefs = pipe_full.named_steps["ridge"].coef_
print("\nB4 standardized coefficients (Ridge, alpha={}):".format(round(alpha_report, 5)))
for f, c in sorted(zip(features, coefs), key=lambda t: abs(t[1]), reverse=True):
    print(f"{f:30s} {c:+.3f}")
