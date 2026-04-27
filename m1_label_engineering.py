"""
M1 — Label Engineering Pipeline
Source dataset   : cephasax DS3 (OBD-II, 55 columns, no labels)
Label reference  : outofskills/driving-behavior (AccX/Y/Z + Gyro -> SLOW/NORMAL/AGGRESSIVE)

How thresholds were derived
---------------------------
The outofskills test file (3,084 rows) was analyzed and the following
class boundaries were extracted:

  Metric              SLOW       NORMAL     AGGRESSIVE
  AccMag (m/s²) p95   2.46       3.01       3.82
  AccY min (m/s²)    -3.03      -3.07      -7.62   ← hard brake axis
  AccX max (m/s²)     2.83       3.30       5.87   ← lateral axis

OBD-II translation (1 m/s² = 3.6 km/h per second):
  AccY < -3.5 m/s²  ->  speed_delta < -12.6 km/h/s  (hard braking event)
  AccY >  2.5 m/s²  ->  speed_delta >   9.0 km/h/s  (rapid acceleration)
  AccX > 3.0 m/s²   ->  no direct OBD PID -> proxied by throttle_delta

These are applied per 30-second window to label each window as
SLOW / NORMAL / AGGRESSIVE.

Directory layout expected:
    data/ds3/
        *.csv   ← cephasax DS3 files (and DS1/DS2 if available)
"""

import os, glob, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

OUT = Path("outputs/m1_labels")
OUT.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# SECTION 1 — THRESHOLDS (UPDATED FOR OBD-II EXP2 REALITIES)
# -------------------------------------------------------------

THRESHOLDS = {
    # Speed-based (km/h per second window)
    "hard_brake_kmh_per_s"     : -14.0,   # Was -12.0. Requires a true slam on the brakes.
    "rapid_accel_kmh_per_s"    :  12.0,   # Was 9.0. 

    # Event counts per 30-second window
    "hard_brake_events_aggressive" : 2,   
    "hard_brake_events_normal"     : 1,   
    "rapid_accel_events_aggressive": 2,
    "rapid_accel_events_normal"    : 1,

    # RPM thresholds (OBD-II direct reading)
    "rpm_aggressive"           : 3600,    # Was 3000. Now requires actual high-revving.
    "rpm_normal_upper"         : 2600,    # Was 2200.

    # Throttle delta (proxy for lateral/longitudinal aggression)
    "throttle_delta_aggressive": 35.0,    # Was 15.0%. Requires a deep, sudden pedal press.
    "throttle_delta_normal"    : 20.0,    # Was 8.0%. 

    # Engine load
    "load_aggressive"          : 85.0,    # Was 80.0.
    "load_normal_upper"        : 70.0,    # Was 60.0.

    # SLOW class: low speed + gentle inputs
    "slow_max_mean_speed"      : 35.0,    # Was 25.0. Catch more urban/traffic driving.
    "slow_max_speed_std"       : 12.0,    # Was 8.0.

    # Aggression score weights
    "w_hard_brake"             : 3.0,
    "w_rapid_accel"            : 2.5,
    "w_high_rpm"               : 1.5,
    "w_high_throttle_delta"    : 2.0,
    "w_high_load"              : 1.0,

    # Aggression score thresholds -> final label
    "score_aggressive"         : 4.5,    # BUMPED UP: Was 4.0. Harder to get flagged aggressive.
    "score_slow"               : 1.0,    # BUMPED UP: Was 0.5. Allows minor inputs in traffic to still be 'SLOW'.

    # Window settings
    "window_seconds"           : 30,
    "min_window_rows"          : 10,     
}

# -------------------------------------------------------------
# SECTION 2 — DATA LOADING
# -------------------------------------------------------------

# DS3 column name variants (the dataset uses verbose names)
COL_MAP = {
    # Canonical name -> possible variants in DS3
    "speed"   : ["VEHICLE_SPEED_SENSOR", "SPEED", "VEHICLE SPEED", "GPS_SPEED",
                 "Vehicle Speed Sensor [km/h]", "vehicle_speed"],
    "rpm"     : ["ENGINE_RPM", "RPM", "Engine Speed", "ENGINE SPEED",
                 "Engine RPM [RPM]", "engine_rpm"],
    "throttle": ["ABSOLUTE_THROTTLE_POSITION", "THROTTLE", "THROTTLE_POS",
                 "Absolute Throttle Position [%]", "throttle_pos"],
    "load"    : ["CALCULATED_ENGINE_LOAD", "ENGINE_LOAD", "LOAD",
                 "Calculated LOAD value", "engine_load"],
    "coolant" : ["ENGINE_COOLANT_TEMP", "COOLANT_TEMP", "COOLANT",
                 "Engine Coolant Temperature [°C]", "coolant_temp"],
    "maf"     : ["AIR_FLOW_RATE", "MAF", "Air Flow Rate from Mass Flow Sensor [g/s]"],
}


def find_column(df: pd.DataFrame, canonical: str) -> str | None:
    """Find the actual column name in df for a given canonical OBD name."""
    variants = COL_MAP.get(canonical, [canonical])
    df_cols_upper = {c.upper().replace(" ", "_"): c for c in df.columns}
    for v in variants:
        v_norm = v.upper().replace(" ", "_")
        if v_norm in df_cols_upper:
            return df_cols_upper[v_norm]
    return None


def load_ds3(data_dir: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(data_dir, "**/*.csv"), recursive=True) + \
            glob.glob(os.path.join(data_dir, "*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files in {data_dir}")

    frames = []
    for f in files:
        try:
            # DS3 uses semicolons and European decimal commas in some versions
            try:
                df = pd.read_csv(f, sep=",", decimal=".", low_memory=False)
            except Exception:
                df = pd.read_csv(f, sep=";", decimal=",", low_memory=False)

            df["source_file"] = os.path.basename(f)
            frames.append(df)
            print(f"  [load] {os.path.basename(f):45s} -> {len(df):,} rows")
        except Exception as e:
            print(f"  [warn] skipping {f}: {e}")

    combined = pd.concat(frames, ignore_index=True)
    print(f"\n[load] total rows: {len(combined):,}")
    return combined


# -------------------------------------------------------------
# SECTION 3 — PREPROCESSING
# -------------------------------------------------------------

def extract_obd_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and rename the OBD columns we need.
    Strips units from string values (e.g. '2124RPM' -> 2124).
    """
    result = pd.DataFrame()

    for canonical, _ in COL_MAP.items():
        col = find_column(df, canonical)
        if col:
            series = df[col].copy().astype(str)
            # Strip common units: RPM, km/h, %, kPa, g/s, C, °C
            series = series.str.replace(
                r"[A-Za-z°%/\s]", "", regex=True
            ).str.replace(",", ".").str.strip()
            result[canonical] = pd.to_numeric(series, errors="coerce")
        else:
            result[canonical] = np.nan

    # Preserve source file for grouping
    if "source_file" in df.columns:
        result["source_file"] = df["source_file"].values

    # Preserve timestamp if available
    time_candidates = ["TIME", "TIMESTAMP", "Time", "time"]
    for tc in time_candidates:
        if tc in df.columns:
            result["timestamp_raw"] = df[tc].values
            break

    return result


def compute_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-row speed_delta and throttle_delta (change per second)."""
    df = df.copy()
    df["speed_delta"]    = df["speed"].diff().fillna(0)
    df["throttle_delta"] = df["throttle"].diff().fillna(0).abs()
    df["rpm_delta"]      = df["rpm"].diff().fillna(0).abs()
    return df


def clean_obd(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop rows where all key signals are NaN, clip extremes."""
    key_cols = ["speed", "rpm", "throttle", "load"]
    df = df.dropna(subset=[c for c in key_cols if c in df.columns], how="all")

    # Clip physically impossible values
    if "speed" in df.columns:
        df["speed"] = df["speed"].clip(0, 300)
    if "rpm" in df.columns:
        df["rpm"] = df["rpm"].clip(0, 8000)
    if "throttle" in df.columns:
        df["throttle"] = df["throttle"].clip(0, 100)
    if "load" in df.columns:
        df["load"] = df["load"].clip(0, 100)

    return df.reset_index(drop=True)


# -------------------------------------------------------------
# SECTION 4 — LABEL ENGINEERING
# Core logic: score-based per 30-second window
# -------------------------------------------------------------

def compute_window_features(window: pd.DataFrame) -> dict:
    """
    Compute aggregate features for a single 30-second window.
    These mirror the statistical features used by outofskills labeling.
    """
    T = THRESHOLDS
    feat = {}

    # Speed features
    if "speed" in window.columns and window["speed"].notna().sum() > 2:
        feat["mean_speed"]     = window["speed"].mean()
        feat["max_speed"]      = window["speed"].max()
        feat["speed_std"]      = window["speed"].std()
        feat["hard_brake_cnt"] = (window["speed_delta"] < T["hard_brake_kmh_per_s"]).sum()
        feat["rapid_accel_cnt"]= (window["speed_delta"] > T["rapid_accel_kmh_per_s"]).sum()
    else:
        feat.update({"mean_speed": np.nan, "max_speed": np.nan,
                     "speed_std": np.nan, "hard_brake_cnt": 0, "rapid_accel_cnt": 0})

    # RPM features
    if "rpm" in window.columns and window["rpm"].notna().sum() > 2:
        feat["mean_rpm"]      = window["rpm"].mean()
        feat["max_rpm"]       = window["rpm"].max()
        feat["high_rpm_ratio"]= (window["rpm"] > T["rpm_aggressive"]).mean()
    else:
        feat.update({"mean_rpm": np.nan, "max_rpm": np.nan, "high_rpm_ratio": 0})

    # Throttle features
    if "throttle" in window.columns and window["throttle"].notna().sum() > 2:
        feat["mean_throttle"]  = window["throttle"].mean()
        feat["throttle_std"]   = window["throttle"].std()
        feat["max_throttle_delta"] = window["throttle_delta"].max()
    else:
        feat.update({"mean_throttle": np.nan, "throttle_std": np.nan,
                     "max_throttle_delta": 0})

    # Engine load features
    if "load" in window.columns and window["load"].notna().sum() > 2:
        feat["mean_load"] = window["load"].mean()
    else:
        feat["mean_load"] = np.nan

    return feat


def compute_aggression_score(feat: dict) -> float:
    """
    Weighted aggression score that mirrors the intensity separation
    observed in outofskills data (AccMag SLOW<NORMAL<AGGRESSIVE).

    Score -> label:
        >= score_aggressive  -> AGGRESSIVE
        <= score_slow        -> SLOW (if also low speed)
        else                 -> NORMAL
    """
    T = THRESHOLDS
    score = 0.0

    # Hard braking (strongest signal — AccY < -3.5 m/s² in outofskills)
    hb = feat.get("hard_brake_cnt", 0)
    if hb >= T["hard_brake_events_aggressive"]:
        score += T["w_hard_brake"]
    elif hb >= T["hard_brake_events_normal"]:
        score += T["w_hard_brake"] * 0.4

    # Rapid acceleration
    ra = feat.get("rapid_accel_cnt", 0)
    if ra >= T["rapid_accel_events_aggressive"]:
        score += T["w_rapid_accel"]
    elif ra >= T["rapid_accel_events_normal"]:
        score += T["w_rapid_accel"] * 0.4

    # High RPM
    if not np.isnan(feat.get("mean_rpm", np.nan)):
        if feat["mean_rpm"] > T["rpm_aggressive"]:
            score += T["w_high_rpm"]
        elif feat["mean_rpm"] > T["rpm_normal_upper"]:
            score += T["w_high_rpm"] * 0.3

    # Aggressive throttle changes (proxy for lateral AccX in outofskills)
    if not np.isnan(feat.get("max_throttle_delta", np.nan)):
        if feat["max_throttle_delta"] > T["throttle_delta_aggressive"]:
            score += T["w_high_throttle_delta"]
        elif feat["max_throttle_delta"] > T["throttle_delta_normal"]:
            score += T["w_high_throttle_delta"] * 0.3

    # High engine load
    if not np.isnan(feat.get("mean_load", np.nan)):
        if feat["mean_load"] > T["load_aggressive"]:
            score += T["w_high_load"]
        elif feat["mean_load"] > T["load_normal_upper"]:
            score += T["w_high_load"] * 0.4

    return score


def assign_label(feat: dict) -> str:
    T = THRESHOLDS
    score = compute_aggression_score(feat)

    # AGGRESSIVE: high score regardless of speed
    if score >= T["score_aggressive"]:
        return "AGGRESSIVE"

    # SLOW: low score AND genuinely slow driving
    mean_speed = feat.get("mean_speed", np.nan)
    speed_std  = feat.get("speed_std", np.nan)
    if (score <= T["score_slow"]
            and not np.isnan(mean_speed)
            and mean_speed <= T["slow_max_mean_speed"]
            and (np.isnan(speed_std) or speed_std <= T["slow_max_speed_std"])):
        return "SLOW"

    return "NORMAL"


def label_dataset(df: pd.DataFrame, window_seconds: int = 30) -> pd.DataFrame:
    """
    Slide a fixed-size window over the dataset and assign a label per window.
    Groups by source_file so windows don't cross file boundaries.
    """
    T = THRESHOLDS
    records = []
    groups = df.groupby("source_file") if "source_file" in df.columns \
             else [("all", df)]

    for file_name, group in groups:
        group = group.reset_index(drop=True)
        n_rows = len(group)

        for start in range(0, n_rows, window_seconds):
            window = group.iloc[start : start + window_seconds]
            if len(window) < T["min_window_rows"]:
                continue

            feat = compute_window_features(window)
            label = assign_label(feat)

            record = {
                "source_file"        : file_name,
                "window_start_idx"   : start,
                "window_size"        : len(window),
                "label"              : label,
                "aggression_score"   : round(compute_aggression_score(feat), 3),
                **{k: round(v, 4) if isinstance(v, float) else v
                   for k, v in feat.items()}
            }
            records.append(record)

    labeled = pd.DataFrame(records)
    print(f"\n[label] total windows: {len(labeled):,}")
    print(f"[label] distribution:\n{labeled['label'].value_counts()}")
    print(f"[label] %:\n{labeled['label'].value_counts(normalize=True).round(3)*100}")
    return labeled


# -------------------------------------------------------------
# SECTION 5 — VALIDATION & VISUALISATION
# -------------------------------------------------------------

def plot_label_distribution(labeled: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Class count bar
    counts = labeled["label"].value_counts().reindex(["SLOW","NORMAL","AGGRESSIVE"])
    colors = ["#4CAF50", "#FF9800", "#F44336"]
    axes[0].bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=0.5)
    axes[0].set_title("Window Label Distribution", fontsize=13)
    axes[0].set_ylabel("Count")
    for i, (label, val) in enumerate(counts.items()):
        axes[0].text(i, val + 5, str(val), ha="center", fontsize=11, fontweight="bold")

    # Aggression score distribution per label
    for label, color in zip(["SLOW","NORMAL","AGGRESSIVE"], colors):
        sub = labeled[labeled["label"] == label]["aggression_score"]
        if len(sub) > 0:
            axes[1].hist(sub, bins=40, alpha=0.6, label=label, color=color, density=True)
    axes[1].set_title("Aggression Score Distribution per Label", fontsize=13)
    axes[1].set_xlabel("Aggression Score")
    axes[1].set_ylabel("Density")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle("M1 — Label Engineering Results", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "label_distribution.png", dpi=150)
    plt.close(fig)
    print(f"[plot] label distribution -> {OUT}/label_distribution.png")


def plot_feature_boxplots(labeled: pd.DataFrame):
    features = ["mean_speed", "mean_rpm", "mean_throttle", "mean_load",
                "hard_brake_cnt", "rapid_accel_cnt", "max_throttle_delta"]
    features = [f for f in features if f in labeled.columns]
    colors = {"SLOW": "#4CAF50", "NORMAL": "#FF9800", "AGGRESSIVE": "#F44336"}

    n = len(features)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(16, 10))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        data = [labeled[labeled["label"] == cls][feat].dropna().values
                for cls in ["SLOW", "NORMAL", "AGGRESSIVE"]]
        bp = axes[i].boxplot(data, patch_artist=True, labels=["SLOW","NORMAL","AGGRESSIVE"])
        for patch, color in zip(bp["boxes"], colors.values()):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[i].set_title(feat, fontsize=11)
        axes[i].grid(axis="y", alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions per Label Class — M1", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "feature_boxplots.png", dpi=150)
    plt.close(fig)
    print(f"[plot] feature boxplots -> {OUT}/feature_boxplots.png")


def validate_against_outofskills(labeled: pd.DataFrame):
    """
    Sanity check: compare class % to outofskills reference.
    Reference: SLOW=41.3%, NORMAL=32.3%, AGGRESSIVE=26.4%
    """
    ref = {"SLOW": 41.3, "NORMAL": 32.3, "AGGRESSIVE": 26.4}
    our = labeled["label"].value_counts(normalize=True).round(3) * 100

    print("\n[validate] Class distribution comparison:")
    print(f"{'Class':<12} {'outofskills':>12} {'DS3 (ours)':>12} {'Delta':>8}")
    print("-" * 46)
    for cls in ["SLOW", "NORMAL", "AGGRESSIVE"]:
        ref_pct = ref.get(cls, 0)
        our_pct = our.get(cls, 0)
        diff = our_pct - ref_pct
        flag = " ✓" if abs(diff) < 15 else " ! (adjust thresholds)"
        print(f"{cls:<12} {ref_pct:>11.1f}% {our_pct:>11.1f}% {diff:>+7.1f}%{flag}")

    print("\n[validate] If Delta > 15% for any class, adjust THRESHOLDS in Section 1.")
    print("  Too many AGGRESSIVE -> raise score_aggressive or hard_brake_kmh_per_s")
    print("  Too few  AGGRESSIVE -> lower score_aggressive or hard_brake_kmh_per_s")
    print("  Too many SLOW       -> raise slow_max_mean_speed or slow_max_speed_std")


# -------------------------------------------------------------
# SECTION 6 — EXPORT
# -------------------------------------------------------------

def export(labeled: pd.DataFrame):
    # Full labeled window dataset (features + label)
    labeled.to_csv(OUT / "ds3_labeled_windows.csv", index=False)
    print(f"\n[export] labeled windows -> {OUT}/ds3_labeled_windows.csv")

    # Label-only mapping (for joining back to raw rows if needed)
    label_map = labeled[["source_file", "window_start_idx", "window_size", "label"]]
    label_map.to_csv(OUT / "ds3_label_map.csv", index=False)
    print(f"[export] label map       -> {OUT}/ds3_label_map.csv")

    # Save thresholds used (for reproducibility)
    with open(OUT / "thresholds_used.json", "w") as f:
        json.dump(THRESHOLDS, f, indent=2)
    print(f"[export] thresholds      -> {OUT}/thresholds_used.json")


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------

def main():
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/ds3"

    print("=" * 65)
    print("M1 — Label Engineering Pipeline")
    print(f"Data directory: {data_dir}")
    print("=" * 65)

    # 1. Load
    raw = load_ds3(data_dir)

    # 2. Extract OBD columns
    print("\n-- Extracting OBD columns ------------------------------")
    obd = extract_obd_columns(raw)
    found = {k: v for k, v in {
        c: find_column(raw, c) for c in COL_MAP
    }.items() if v}
    print(f"Found columns: {found}")

    # 3. Clean + compute deltas
    obd = clean_obd(obd)
    obd = compute_deltas(obd)
    print(f"[preprocess] clean rows: {len(obd):,}")

    # 4. Label
    print("\n-- Assigning labels (30-second windows) ----------------")
    labeled = label_dataset(obd, THRESHOLDS["window_seconds"])

    # 5. Validate
    validate_against_outofskills(labeled)

    # 6. Visualise
    print("\n-- Generating plots ------------------------------------")
    plot_label_distribution(labeled)
    plot_feature_boxplots(labeled)

    # 7. Export
    print("\n-- Exporting --------------------------------------------")
    export(labeled)

    print(f"\n{'='*65}")
    print(f"Done. Labeled dataset ready for M1 training.")
    print(f"Next step: run m1_training.py with {OUT}/ds3_labeled_windows.csv")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
