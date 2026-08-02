import pandas as pd
import numpy as np
from pathlib import Path

RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

print("=" * 65)
print("FEATURE ENGINEERING — Chicago Crime Dataset")
print("=" * 65)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("\n[1] Loading sampled dataset ...")
df = pd.read_csv(DATA_DIR / "sampled_dataset.csv")
print(f"    Loaded {len(df):,} rows x {df.shape[1]} columns")

# ---------------------------------------------------------------------------
# 2. Parse Date
# ---------------------------------------------------------------------------
print("\n[2] Parsing Date ...")
df["Parsed_Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")

# ===========================================================================
# FEATURE 1: Time-based features
# ===========================================================================
print("\n[Feature 1] Time-based features ...")
df["Hour"] = df["Parsed_Date"].dt.hour
df["DayOfWeek"] = df["Parsed_Date"].dt.dayofweek  # 0=Monday
df["Month"] = df["Parsed_Date"].dt.month
df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)

def part_of_day(hour):
    if 5 <= hour < 9:    return "dawn"
    if 9 <= hour < 12:   return "morning"
    if 12 <= hour < 17:  return "afternoon"
    if 17 <= hour < 21:  return "evening"
    return "night"

df["PartOfDay"] = df["Hour"].map(part_of_day)
print(f"    Hour range: {df['Hour'].min()}-{df['Hour'].max()}")
print(f"    Weekend proportion: {df['IsWeekend'].mean():.3f}")
print(f"    PartOfDay categories: {df['PartOfDay'].unique()}")

# ===========================================================================
# FEATURE 2: Spatial crime density per Community Area
# ===========================================================================
print("\n[Feature 2] Spatial crime density per Community Area ...")
# Count crimes per community area, then map back
comm_area_counts = df.groupby("Community Area")["ID"].count().to_dict()
df["CrimeDensity_Community"] = df["Community Area"].map(comm_area_counts)
# Fill NaN for missing community areas with the overall median
median_density = df["CrimeDensity_Community"].median()
df["CrimeDensity_Community"] = df["CrimeDensity_Community"].fillna(median_density)
print(f"    Unique community areas: {df['Community Area'].nunique()}")
print(f"    Density range: {df['CrimeDensity_Community'].min():.0f} - {df['CrimeDensity_Community'].max():.0f}")

# Also compute beat-level density (finer granularity)
beat_counts = df.groupby("Beat")["ID"].count().to_dict()
df["CrimeDensity_Beat"] = df["Beat"].map(beat_counts)
median_beat_density = df["CrimeDensity_Beat"].median()
df["CrimeDensity_Beat"] = df["CrimeDensity_Beat"].fillna(median_beat_density)

# ===========================================================================
# FEATURE 3: Crime-type x Location interaction (frequency encoding)
# ===========================================================================
print("\n[Feature 3] Crime-type x Location interaction (frequency encoding) ...")
type_location_counts = df.groupby(["Primary Type", "Location Description"])["ID"].count().reset_index()
type_location_counts.columns = ["Primary Type", "Location Description", "TypeLocationFreq"]
# Normalize: proportion of each crime type that occurs at each location
type_totals = df.groupby("Primary Type")["ID"].count().reset_index()
type_totals.columns = ["Primary Type", "TypeTotal"]
type_location_counts = type_location_counts.merge(type_totals, on="Primary Type")
type_location_counts["TypeLocationProb"] = (
    type_location_counts["TypeLocationFreq"] / type_location_counts["TypeTotal"]
)
df = df.merge(
    type_location_counts[["Primary Type", "Location Description", "TypeLocationProb", "TypeLocationFreq"]],
    on=["Primary Type", "Location Description"],
    how="left"
)
print(f"    Unique (Primary Type, Location Description) pairs: {(~df['TypeLocationProb'].isna()).sum():,}")

# ===========================================================================
# FEATURE 4: Recency — DaysSinceLastCrime on same block
# ===========================================================================
print("\n[Feature 4] Days since last crime on same block ...")
df_sorted = df.sort_values(["Block", "Parsed_Date"]).copy()
df_sorted["DaysSinceLastCrime_Block"] = (
    df_sorted.groupby("Block")["Parsed_Date"].diff().dt.total_seconds() / (60 * 60 * 24)
)
df = df.merge(
    df_sorted[["ID", "DaysSinceLastCrime_Block"]],
    on="ID",
    how="left"
)
# First occurrence on a block gets NaN — fill with a large sentinel
max_block_days = df["DaysSinceLastCrime_Block"].max()
df["DaysSinceLastCrime_Block"] = df["DaysSinceLastCrime_Block"].fillna(max_block_days * 2)
print(f"    Days range: {df['DaysSinceLastCrime_Block'].min():.1f} - {df['DaysSinceLastCrime_Block'].max():.1f}")

# ===========================================================================
# FEATURE 5: Arrest rate by District (deterrence)
# ===========================================================================
print("\n[Feature 5] Arrest rate by District (deterrence feature) ...")
district_arrest_rate = df.groupby("District")["Arrest"].mean().to_dict()
df["DistrictArrestRate"] = df["District"].map(district_arrest_rate)
# For the tiny number of missing District values, fill with overall rate
overall_arrest_rate = df["Arrest"].mean()
df["DistrictArrestRate"] = df["DistrictArrestRate"].fillna(overall_arrest_rate)
print(f"    District arrest rate range: {df['DistrictArrestRate'].min():.3f} - {df['DistrictArrestRate'].max():.3f}")

# ===========================================================================
# FEATURE 6: Season (broader temporal aggregation)
# ===========================================================================
print("\n[Feature 6] Season encoding ...")
def month_to_season(m):
    if m in [12, 1, 2]:   return "winter"
    if m in [3, 4, 5]:    return "spring"
    if m in [6, 7, 8]:    return "summer"
    return "fall"

df["Season"] = df["Month"].map(month_to_season)
print(f"    Season categories: {df['Season'].unique()}")

# ===========================================================================
# FEATURE 7: Crime severity proxy — FBI Code categorization
# ===========================================================================
print("\n[Feature 7] Crime severity from FBI Code ...")
# FBI code groupings: first character indicates broad category
# 0x = violent (homicide, criminal sexual assault, robbery, assault)
# 1x = property (burglary, theft, motor vehicle theft, arson)
# 2x = drug/alcohol related
# 3x = sex offenses
# 4x-5x = deception/fraud
# 6x-7x = other
# 8x = weapons
# 9x-99 = other
def fbi_severity(code):
    code_str = str(code).strip()
    if not code_str:
        return "unknown"
    first = code_str[0]
    if first in "0123":
        cat_map = {
            "01": "violent",
            "02": "violent",
            "03": "violent",
            "04": "property",
            "05": "property",
            "06": "property",
            "07": "property",
            "08": "property",
            "09": "property",
            "10": "drug",
            "11": "drug",
            "12": "deception",
            "13": "deception",
            "14": "deception",
            "15": "deception",
            "16": "other",
            "17": "other",
            "18": "weapons",
            "20": "other",
            "26": "other",
        }
        return cat_map.get(code_str[:2], "other")
    return "other"

df["FBI_Severity"] = df["FBI Code"].astype(str).map(fbi_severity)
print(f"    Severity categories: {df['FBI_Severity'].value_counts().to_dict()}")

# ---------------------------------------------------------------------------
# Drop temporary columns
# ---------------------------------------------------------------------------
df.drop(columns=["Parsed_Date"], inplace=True, errors="ignore")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
output_path = DATA_DIR / "engineered_dataset.csv"
df.to_csv(output_path, index=False)
print(f"\n{'=' * 65}")
print(f"Engineered dataset saved → {output_path}")
print(f"    Rows: {len(df):,}")
print(f"    Columns ({df.shape[1]}): {list(df.columns)}")
print(f"{'=' * 65}")

# Quick validation of new features
print("\nNew features summary:")
new_cols = [
    "Hour", "DayOfWeek", "Month", "IsWeekend", "PartOfDay",
    "CrimeDensity_Community", "CrimeDensity_Beat",
    "TypeLocationProb", "TypeLocationFreq",
    "DaysSinceLastCrime_Block",
    "DistrictArrestRate",
    "Season",
    "FBI_Severity",
]
for col in new_cols:
    if col in df.columns:
        dtype = df[col].dtype
        nunique = df[col].nunique()
        missing = df[col].isna().sum()
        print(f"  {col:30s}  dtype={str(dtype):10s}  unique={nunique:>5}  missing={missing:>5}")
