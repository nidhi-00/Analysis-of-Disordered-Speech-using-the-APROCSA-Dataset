import pandas as pd
from scipy.stats import spearmanr

# Load A3 stats
stats = pd.read_csv("../Outputs/Statistics/a3_table.csv")

# Keep only participant rows
par = stats[stats["speaker"] == "PAR"]

# Load perceptual ratings (example filename)
ratings = pd.read_csv("../Outputs/perceptual_ratings.csv")

# Merge on participant / file id
df = par.merge(ratings, on="file")

# 1. Pauses within utterances
rho1, p1 = spearmanr(
    df["pause_markers_per_100_tokens"],
    df["pauses_within_utterances"]
)

# 2. Reduced speech rate
rho2, p2 = spearmanr(
    df["filled_pauses_per_100_tokens"],
    df["reduced_speech_rate"]
)

# 3. Anomia
rho3, p3 = spearmanr(
    df["ttr"],
    df["anomia"]
)

print("Spearman correlations (PAR only):\n")

print(
    f"Pauses within utterances vs pause markers per 100 tokens: "
    f"rho = {rho1:.3f}, p = {p1:.3f}"
)

print(
    f"Reduced speech rate vs filled pauses per 100 tokens: "
    f"rho = {rho2:.3f}, p = {p2:.3f}"
)

print(
    f"Anomia vs lexical diversity (TTR): "
    f"rho = {rho3:.3f}, p = {p3:.3f}"
)
