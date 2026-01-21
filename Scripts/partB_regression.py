import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from pathlib import Path

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
    "filled_pauses_per_100_tokens"
]

X = df[features].values
y = df["anomia"].values

loo = LeaveOneOut()
model = LinearRegression()

preds = []

for train, test in loo.split(X):
    model.fit(X[train], y[train])
    preds.append(model.predict(X[test])[0])

mae = mean_absolute_error(y, preds)

print("True Anomia:", y)
print("Predicted Anomia:", np.round(preds, 2))
print("MAE:", round(mae, 3))

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

model.fit(X_std, y)
coefs = model.coef_

for f, c in zip(features, coefs):
    print(f, round(c, 3))
