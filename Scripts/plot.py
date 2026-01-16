import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load data
stats = pd.read_csv(PROJECT_ROOT / "Outputs" / "Statistics" / "a3_table.csv")
ratings = pd.read_csv(PROJECT_ROOT / "Outputs" / "perceptual_ratings.csv")

# Keep only participant rows
par = stats[stats["speaker"] == "PAR"]
df = par.merge(ratings, on="file")

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: Pauses within utterances
axes[0].scatter(df["pause_markers_per_100_tokens"], df["pauses_within_utterances"])
axes[0].set_xlabel("Pause markers per 100 tokens")
axes[0].set_ylabel("Pauses within utterances (rating)")
axes[0].set_title("Pausing")

# Plot 2: Reduced speech rate
axes[1].scatter(df["filled_pauses_per_100_tokens"], df["reduced_speech_rate"])
axes[1].set_xlabel("Filled pauses per 100 tokens")
axes[1].set_ylabel("Reduced speech rate (rating)")
axes[1].set_title("Speech rate")

# Plot 3: Anomia
axes[2].scatter(df["ttr"], df["anomia"])
axes[2].set_xlabel("Type-Token Ratio (TTR)")
axes[2].set_ylabel("Anomia (rating)")
axes[2].set_title("Lexical diversity")

plt.tight_layout()

# Save figure
output_dir = PROJECT_ROOT / "Outputs" / "Figures"
output_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(output_dir / "A5_scatter_plots.png", dpi=300, bbox_inches="tight")
plt.close()

print("Figure saved to Outputs/Figures/A5_scatter_plots.png")
