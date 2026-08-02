"""
Stratified Sampling for Chicago Crime Dataset
==============================================
Creates a representative subset (~100K rows) from the full 8.5M-row dataset
using stratified sampling on Primary Type × Year to preserve both crime
category and temporal distributions.

Usage:
    python src/sampling.py

Output:
    - data/sampled_dataset.csv          (the sampled subset)
    - plots/sampling_validation.png     (validation plots)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
SAMPLE_FRACTION = 0.012  # ~1.2% → ~100 K rows from 8.5 M
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PLOTS_DIR = BASE_DIR / "plots"

PLOTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("Loading raw dataset ...")
df = pd.read_csv(DATA_DIR / "raw_dataset.csv")
print(f"  Loaded {len(df):,} rows × {df.shape[1]} columns\n")

# ---------------------------------------------------------------------------
# 2. Parse dates & create strata
# ---------------------------------------------------------------------------
print("Parsing dates and creating strata ...")
df["Parsed_Date"] = pd.to_datetime(
    df["Date"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce"
)
df["Month"] = df["Parsed_Date"].dt.month
df["Strata"] = df["Primary Type"] + "_|_" + df["Year"].astype(str)

n_strata = df["Strata"].nunique()
print(f"  Unique strata (Primary Type × Year): {n_strata}\n")

# ---------------------------------------------------------------------------
# 3. Stratified sampling
# ---------------------------------------------------------------------------
print(f"Performing stratified sampling (fraction={SAMPLE_FRACTION}) ...")
df_sampled = df.groupby(["Primary Type", "Year"], group_keys=False).sample(
    frac=SAMPLE_FRACTION, random_state=RANDOM_STATE
)

print(f"  Original size : {len(df):>12,}")
print(f"  Sampled size  : {len(df_sampled):>12,}")
print(f"  Actual ratio  : {len(df_sampled) / len(df) * 100:.2f}%\n")

# ---------------------------------------------------------------------------
# 4. Statistical validation
# ---------------------------------------------------------------------------
print("=" * 65)
print("DISTRIBUTION COMPARISON SUMMARY")
print("=" * 65)


def compare_distributions(orig: pd.Series, sampled: pd.Series, name: str):
    """Compare two normalised value_counts series and print metrics."""
    o = orig.value_counts(normalize=True).sort_index()
    s = sampled.value_counts(normalize=True).sort_index()
    # align on common index
    idx = o.index.union(s.index)
    o = o.reindex(idx, fill_value=0)
    s = s.reindex(idx, fill_value=0)
    max_diff = (s - o).abs().max()
    mean_diff = (s - o).abs().mean()
    print(f"\n  {name}")
    print(f"    Max  absolute diff : {max_diff:.4f}")
    print(f"    Mean absolute diff : {mean_diff:.4f}")
    return o, s


compare_distributions(df["Primary Type"], df_sampled["Primary Type"], "Primary Type")
compare_distributions(df["Year"], df_sampled["Year"], "Year")
compare_distributions(df["District"], df_sampled["District"], "District")
compare_distributions(
    df["Location Description"], df_sampled["Location Description"], "Location Description"
)

print(f"\n  Arrest rate")
print(f"    Original : {df['Arrest'].mean():.4f}")
print(f"    Sampled  : {df_sampled['Arrest'].mean():.4f}")
print(f"    Diff     : {abs(df['Arrest'].mean() - df_sampled['Arrest'].mean()):.4f}")

print(f"\n  Domestic rate")
print(f"    Original : {df['Domestic'].mean():.4f}")
print(f"    Sampled  : {df_sampled['Domestic'].mean():.4f}")
print(f"    Diff     : {abs(df['Domestic'].mean() - df_sampled['Domestic'].mean()):.4f}")
print()

# ---------------------------------------------------------------------------
# 5. Validation plots
# ---------------------------------------------------------------------------
print("Generating validation plots ...")

sns.set_style("whitegrid")
fig, axes = plt.subplots(3, 2, figsize=(18, 22))
fig.suptitle(
    "Distribution Comparison: Original vs Sampled Dataset",
    fontsize=18,
    fontweight="bold",
    y=1.005,
)

width = 0.35

# ---- Panel 1: Top 15 Primary Crime Types ----
ax = axes[0, 0]
orig_pt = df["Primary Type"].value_counts(normalize=True).sort_values(ascending=False).head(15)
samp_pt = df_sampled["Primary Type"].value_counts(normalize=True).reindex(orig_pt.index)
y_pos = np.arange(len(orig_pt))
ax.barh(y_pos - width / 2, orig_pt.values, width, label="Original", color="steelblue", alpha=0.85)
ax.barh(y_pos + width / 2, samp_pt.values, width, label="Sampled", color="coral", alpha=0.85)
ax.set_yticks(y_pos)
ax.set_yticklabels(orig_pt.index, fontsize=9)
ax.set_xlabel("Proportion")
ax.set_title("Top 15 Primary Crime Types", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.invert_yaxis()

# ---- Panel 2: Year Distribution ----
ax = axes[0, 1]
orig_yr = df["Year"].value_counts(normalize=True).sort_index()
samp_yr = df_sampled["Year"].value_counts(normalize=True).sort_index()
ax.plot(orig_yr.index, orig_yr.values, "o-", label="Original", color="steelblue", linewidth=2)
ax.plot(samp_yr.index, samp_yr.values, "s-", label="Sampled", color="coral", linewidth=2)
ax.set_xlabel("Year")
ax.set_ylabel("Proportion")
ax.set_title("Year Distribution", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# ---- Panel 3: Arrest Rate by Crime Type (Top 10) ----
ax = axes[1, 0]
top10 = df["Primary Type"].value_counts().head(10).index
orig_arrest = (
    df[df["Primary Type"].isin(top10)].groupby("Primary Type")["Arrest"].mean().reindex(top10)
)
samp_arrest = (
    df_sampled[df_sampled["Primary Type"].isin(top10)]
    .groupby("Primary Type")["Arrest"]
    .mean()
    .reindex(top10)
)
x_pos = np.arange(len(top10))
ax.bar(x_pos - width / 2, orig_arrest.values, width, label="Original", color="steelblue", alpha=0.85)
ax.bar(x_pos + width / 2, samp_arrest.values, width, label="Sampled", color="coral", alpha=0.85)
ax.set_xticks(x_pos)
ax.set_xticklabels(top10, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Arrest Rate")
ax.set_title("Arrest Rate by Crime Type (Top 10)", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)

# ---- Panel 4: District Distribution ----
ax = axes[1, 1]
orig_dist = df["District"].value_counts(normalize=True).sort_index()
samp_dist = df_sampled["District"].value_counts(normalize=True).sort_index()
all_dist = sorted(set(orig_dist.index) | set(samp_dist.index))
orig_vals = [orig_dist.get(d, 0) for d in all_dist]
samp_vals = [samp_dist.get(d, 0) for d in all_dist]
x_pos = np.arange(len(all_dist))
ax.bar(x_pos - width / 2, orig_vals, width, label="Original", color="steelblue", alpha=0.85)
ax.bar(x_pos + width / 2, samp_vals, width, label="Sampled", color="coral", alpha=0.85)
ax.set_xticks(x_pos)
ax.set_xticklabels([str(int(d)) for d in all_dist], rotation=45, fontsize=8)
ax.set_xlabel("District")
ax.set_ylabel("Proportion")
ax.set_title("District Distribution", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)

# ---- Panel 5: Geographic Scatter ----
ax = axes[2, 0]
plot_orig = df.dropna(subset=["Latitude", "Longitude"]).sample(
    n=min(50_000, len(df.dropna(subset=["Latitude", "Longitude"]))),
    random_state=RANDOM_STATE,
)
plot_samp = df_sampled.dropna(subset=["Latitude", "Longitude"])
ax.scatter(
    plot_orig["Longitude"], plot_orig["Latitude"],
    s=1, alpha=0.08, label=f"Original (n={len(plot_orig):,})", color="steelblue",
)
ax.scatter(
    plot_samp["Longitude"], plot_samp["Latitude"],
    s=3, alpha=0.25, label=f"Sampled (n={len(plot_samp):,})", color="coral",
)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Geographic Distribution", fontsize=13, fontweight="bold")
ax.legend(fontsize=10, markerscale=5)

# ---- Panel 6: Top 10 Location Descriptions ----
ax = axes[2, 1]
orig_ld = df["Location Description"].value_counts(normalize=True).head(10)
samp_ld = df_sampled["Location Description"].value_counts(normalize=True).reindex(orig_ld.index)
y_pos = np.arange(len(orig_ld))
ax.barh(y_pos - width / 2, orig_ld.values, width, label="Original", color="steelblue", alpha=0.85)
ax.barh(y_pos + width / 2, samp_ld.values, width, label="Sampled", color="coral", alpha=0.85)
ax.set_yticks(y_pos)
ax.set_yticklabels(orig_ld.index, fontsize=9)
ax.set_xlabel("Proportion")
ax.set_title("Top 10 Location Descriptions", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.invert_yaxis()

plt.tight_layout()
fig.savefig(PLOTS_DIR / "sampling_validation.png", dpi=150, bbox_inches="tight")
print(f"  Saved → {PLOTS_DIR / 'sampling_validation.png'}")
plt.show()

# ---------------------------------------------------------------------------
# 6. Save sampled dataset
# ---------------------------------------------------------------------------
output_path = DATA_DIR / "sampled_dataset.csv"
df_sampled.drop(columns=["Parsed_Date", "Month", "Strata"], inplace=True)
df_sampled.to_csv(output_path, index=False)
print(f"\nSampled dataset saved → {output_path}  ({len(df_sampled):,} rows)")
