"""
M1 — Label Engineering Pipeline (v2.5 - Utilization-Based, Quantization-Robust)
================================================================================
Source dataset   : cephasax DS3 (OBD-II, 55 columns, no labels)
Label reference  : outofskills/driving-behavior (AccX/Y/Z + Gyro -> SLOW/NORMAL/AGGRESSIVE)

CRITICAL FIXES in v2.5:
-----------------------
1. ABANDONED BROKEN ACCEL FEATURES: Speed is quantized (~1 km/h steps at 10 Hz).
   Point-to-point accel computation produces ±23 m/s² noise. Removed from scoring.
   (was: every window scored 0.74-0.88 due to fake "extreme events")

2. UTILIZATION-ONLY SCORING: Score based on RPM/speed/load/throttle utilization
   and speed variation. These are naturally robust to quantization.

3. SPEED VARIATION FEATURES: Coefficient of variation + acceleration/deceleration
   ratio computed from speed percentiles (not derivatives).

4. THROTTLE PATTERN DETECTION: Rapid throttle changes = aggressive, steady = calm.

5. IDLE + STOPPED TIME: Primary SLOW indicator.

6. MEDIAN SMOOTHING: Replaced mean MA with median filter (robust to quantization).

Directory layout expected:
    data/ds3/
        *.csv   ← cephasax DS3 files
"""

import os, glob, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("outputs/m1_labels")
OUT.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# SECTION 1 — THRESHOLDS
# -------------------------------------------------------------
THRESHOLDS = {
    # --- Utilization thresholds ---
    "rpm_util_aggressive"      : 0.75,     # >75% of car's p95 RPM
    "rpm_util_normal"          : 0.45,     # 45-75% = normal
    "speed_util_aggressive"    : 0.80,     # >80% of car's p95 speed
    "speed_util_normal"        : 0.55,     # 55-80% = normal cruising
    "speed_util_slow"          : 0.30,     # <30% = slow/city
    "load_util_aggressive"     : 0.80,     # >80% of car's p95 load
    "load_util_normal"         : 0.50,
    "throttle_util_aggressive" : 0.75,     # >75% of car's p95 throttle delta
    "throttle_util_normal"     : 0.35,

    # --- Speed variation (CV = coefficient of variation) ---
    "speed_cv_aggressive"      : 0.35,     # >35% CV = erratic speed = aggressive
    "speed_cv_normal_high"     : 0.20,     # 20-35% CV = normal variation
    "speed_cv_slow_max"        : 0.15,     # <15% CV = steady speed = slow/normal

    # --- Speed change patterns (percentile-based, not derivative) ---
    "speed_drop_aggressive"    : 0.40,     # p10 speed drops >40% from max = hard braking
    "speed_rise_aggressive"    : 0.40,     # p90 speed rises >40% from min = rapid accel

    # --- Throttle patterns ---
    "throttle_changes_aggr"    : 15,      # >15 significant throttle changes per 30s
    "throttle_changes_normal"  : 8,       # 8-15 = normal
    "throttle_cv_aggressive"   : 0.30,     # high throttle variation = aggressive

    # --- Idle / stopped ---
    "idle_ratio_slow"          : 0.30,     # >30% stopped = likely SLOW
    "idle_ratio_normal_max"    : 0.15,     # <15% stopped = moving

    # --- SLOW class ---
    "slow_speed_util_max"      : 0.30,
    "slow_rpm_util_max"        : 0.40,
    "slow_load_util_max"       : 0.45,
    "slow_max_cv_speed"        : 0.20,
    "score_slow"               : 0.25,     # max score for SLOW

    # --- Aggression score weights (utilization-based only) ---
    "w_speed_util"             : 0.20,
    "w_rpm_util"               : 0.20,
    "w_load_util"              : 0.15,
    "w_throttle_util"          : 0.15,
    "w_speed_cv"               : 0.15,     # high CV = aggressive
    "w_throttle_changes"       : 0.10,
    "w_speed_drop"             : 0.05,     # small bonus for hard braking pattern

    # --- Final score thresholds ---
    "score_aggressive"         : 0.50,
    "score_normal_max"         : 0.35,     # 0.35-0.50 = normal
    "score_slow"               : 0.25,

    # --- Window settings ---
    "window_seconds"           : 30,
    "window_overlap_ratio"     : 0.50,
    "min_window_rows"          : 10,
    "time_gap_max"             : 10.0,
    "time_gap_min"             : 0.1,

    # --- Smoothing ---
    "speed_median_window"      : 11,        # median filter (robust to quantization)
}

# -------------------------------------------------------------
# SECTION 2 — DATA LOADING
# -------------------------------------------------------------

COL_MAP = {
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
    variants = COL_MAP.get(canonical, [canonical])
    df_cols_upper = {c.upper().replace(" ", "_"): c for c in df.columns}
    for v in variants:
        v_norm = v.upper().replace(" ", "_")
        if v_norm in df_cols_upper:
            return df_cols_upper[v_norm]
    return None


def load_ds3(data_dir: str) -> pd.DataFrame:
    files = set(
        glob.glob(os.path.join(data_dir, "**/*.csv"), recursive=True) +
        glob.glob(os.path.join(data_dir, "*.csv"))
    )
    if not files:
        raise FileNotFoundError(f"No CSV files in {data_dir}")

    frames = []
    for f in sorted(files):
        try:
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
    print(f"\n[load] total rows: {len(combined):,}  |  unique files: {len(files)}")
    return combined


# -------------------------------------------------------------
# SECTION 3 — PREPROCESSING
# -------------------------------------------------------------

def extract_obd_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame()
    for canonical, _ in COL_MAP.items():
        col = find_column(df, canonical)
        if col:
            series = df[col].copy().astype(str)
            series = series.str.replace(
                r"[A-Za-z°%/\s]", "", regex=True
            ).str.replace(",", ".").str.strip()
            result[canonical] = pd.to_numeric(series, errors="coerce")
        else:
            result[canonical] = np.nan

    if "source_file" in df.columns:
        result["source_file"] = df["source_file"].values

    time_candidates = ["TIME", "TIMESTAMP", "Time", "time"]
    for tc in time_candidates:
        if tc in df.columns:
            ts_raw = df[tc].astype(str).str.replace(",", ".", regex=False)
            result["timestamp_sec"] = pd.to_numeric(ts_raw, errors="coerce")
            break

    return result


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    v2.5: Minimal preprocessing. No acceleration computation.
    Only compute time deltas and median-smoothed speed for visualization.
    """
    df = df.copy()
    T = THRESHOLDS

    if "timestamp_sec" in df.columns and df["timestamp_sec"].notna().sum() > 2:
        df["time_delta"] = df["timestamp_sec"].diff()
    else:
        df["time_delta"] = 1.0

    df["time_delta"] = df["time_delta"].clip(
        lower=T["time_gap_min"], 
        upper=T["time_gap_max"]
    ).fillna(1.0)

    # Median-smoothed speed (robust to quantization)
    if "speed" in df.columns:
        df["speed_median"] = df["speed"].rolling(
            window=T["speed_median_window"], 
            min_periods=1, 
            center=True
        ).median()
    else:
        df["speed_median"] = np.nan

    return df


def clean_obd(df: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["speed", "rpm", "throttle", "load"]
    df = df.dropna(subset=[c for c in key_cols if c in df.columns], how="all")
    df = df.copy()
    if "speed" in df.columns:
        df.loc[:, "speed"] = df["speed"].clip(0, 300)
    if "rpm" in df.columns:
        df.loc[:, "rpm"] = df["rpm"].clip(0, 8000)
    if "throttle" in df.columns:
        df.loc[:, "throttle"] = df["throttle"].clip(0, 100)
    if "load" in df.columns:
        df.loc[:, "load"] = df["load"].clip(0, 100)
    return df.reset_index(drop=True)


# -------------------------------------------------------------
# SECTION 4 — CAR PROFILES
# -------------------------------------------------------------

def compute_car_profiles(df: pd.DataFrame) -> dict:
    profiles = {}
    for file_name, group in df.groupby("source_file"):
        moving = group[group["speed"] > 0]
        if len(moving) < 10:
            profiles[file_name] = {
                "p95_speed": 120.0, "p99_speed": 150.0,
                "p95_rpm": 4000.0, "p99_rpm": 5000.0,
                "p95_load": 70.0,
                "p95_throttle_delta": 10.0,
                "max_throttle_observed": 50.0,
                "median_speed": 40.0, "median_rpm": 1500.0,
            }
            continue

        # v2.5: Compute throttle delta for profile
        throttle_delta = group["throttle"].diff().abs()
        throttle_delta_p95 = throttle_delta.quantile(0.95)
        if pd.isna(throttle_delta_p95) or throttle_delta_p95 <= 0:
            throttle_delta_p95 = 10.0

        profiles[file_name] = {
            "p95_speed": moving["speed"].quantile(0.95),
            "p99_speed": moving["speed"].quantile(0.99),
            "p95_rpm": moving["rpm"].quantile(0.95),
            "p99_rpm": moving["rpm"].quantile(0.99),
            "p95_load": moving["load"].quantile(0.95),
            "p95_throttle_delta": throttle_delta_p95,
            "max_throttle_observed": min(throttle_delta.max() or 50.0, 100.0),
            "median_speed": moving["speed"].median(),
            "median_rpm": moving["rpm"].median(),
        }
    print(f"[profile] Computed profiles for {len(profiles)} files")
    return profiles


# -------------------------------------------------------------
# SECTION 5 — WINDOW FEATURES (QUANTIZATION-ROBUST)
# -------------------------------------------------------------

def compute_window_features(window: pd.DataFrame, car_profile: dict) -> dict:
    """
    v2.5: Features that DON'T depend on noisy acceleration derivatives.
    Uses: utilization ratios, percentiles, variation, patterns.
    """
    T = THRESHOLDS
    cp = car_profile
    feat = {}

    total_rows = len(window)
    stopped_rows = (window["speed"] == 0).sum() if "speed" in window.columns else 0
    feat["idle_time_ratio"] = stopped_rows / total_rows if total_rows > 0 else 0

    # --- Speed features ---
    if "speed" in window.columns and window["speed"].notna().sum() > 2:
        speed = window["speed"].dropna()
        feat["mean_speed"] = speed.mean()
        feat["max_speed"] = speed.max()
        feat["min_speed"] = speed.min()
        feat["speed_std"] = speed.std()
        feat["speed_p10"] = speed.quantile(0.10)
        feat["speed_p90"] = speed.quantile(0.90)
        feat["speed_range"] = feat["max_speed"] - feat["min_speed"]

        if cp["p95_speed"] > 0:
            feat["speed_utilization"] = feat["mean_speed"] / cp["p95_speed"]
            feat["max_speed_utilization"] = feat["max_speed"] / cp["p95_speed"]
            feat["speed_cv"] = feat["speed_std"] / feat["mean_speed"] if feat["mean_speed"] > 0 else 0
        else:
            feat["speed_utilization"] = 0
            feat["max_speed_utilization"] = 0
            feat["speed_cv"] = 0

        # Speed drop/rise patterns (percentile-based, not derivative)
        if feat["max_speed"] > 0:
            feat["speed_drop_ratio"] = (feat["max_speed"] - feat["speed_p10"]) / feat["max_speed"]
            feat["speed_rise_ratio"] = (feat["speed_p90"] - feat["min_speed"]) / feat["max_speed"] if feat["max_speed"] > 0 else 0
        else:
            feat["speed_drop_ratio"] = 0
            feat["speed_rise_ratio"] = 0
    else:
        feat.update({
            "mean_speed": np.nan, "max_speed": np.nan, "min_speed": np.nan,
            "speed_std": np.nan, "speed_p10": np.nan, "speed_p90": np.nan,
            "speed_range": 0, "speed_utilization": 0,
            "max_speed_utilization": 0, "speed_cv": 0,
            "speed_drop_ratio": 0, "speed_rise_ratio": 0,
        })

    # --- RPM features ---
    if "rpm" in window.columns and window["rpm"].notna().sum() > 2:
        rpm = window["rpm"].dropna()
        feat["mean_rpm"] = rpm.mean()
        feat["max_rpm"] = rpm.max()
        feat["rpm_std"] = rpm.std()

        if cp["p95_rpm"] > 0:
            feat["rpm_utilization"] = feat["mean_rpm"] / cp["p95_rpm"]
            feat["high_rpm_ratio"] = (rpm / cp["p95_rpm"] > T["rpm_util_aggressive"]).mean()
            feat["rpm_cv"] = feat["rpm_std"] / feat["mean_rpm"] if feat["mean_rpm"] > 0 else 0
        else:
            feat["rpm_utilization"] = 0
            feat["high_rpm_ratio"] = 0
            feat["rpm_cv"] = 0
    else:
        feat.update({
            "mean_rpm": np.nan, "max_rpm": np.nan, "rpm_std": np.nan,
            "rpm_utilization": 0, "high_rpm_ratio": 0, "rpm_cv": 0,
        })

    # --- Throttle features ---
    if "throttle" in window.columns and window["throttle"].notna().sum() > 2:
        throttle = window["throttle"].dropna()
        feat["mean_throttle"] = throttle.mean()
        feat["throttle_std"] = throttle.std()
        feat["throttle_cv"] = feat["throttle_std"] / feat["mean_throttle"] if feat["mean_throttle"] > 0 else 0

        # Count significant throttle changes (>5% change)
        throttle_changes = (throttle.diff().abs() > 5.0).sum()
        feat["throttle_changes"] = int(throttle_changes)

        # Max throttle delta
        feat["max_throttle_delta"] = throttle.diff().abs().max()

        throttle_ref = cp.get("p95_throttle_delta", cp.get("max_throttle_observed", 50.0))
        if throttle_ref > 0:
            feat["throttle_utilization"] = feat["max_throttle_delta"] / throttle_ref
            feat["throttle_utilization"] = min(feat["throttle_utilization"], 1.0)
        else:
            feat["throttle_utilization"] = 0
    else:
        feat.update({
            "mean_throttle": np.nan, "throttle_std": np.nan, "throttle_cv": 0,
            "throttle_changes": 0, "max_throttle_delta": 0, "throttle_utilization": 0,
        })

    # --- Load features ---
    if "load" in window.columns and window["load"].notna().sum() > 2:
        load = window["load"].dropna()
        feat["mean_load"] = load.mean()
        feat["load_std"] = load.std()

        if cp["p95_load"] > 0:
            feat["load_utilization"] = feat["mean_load"] / cp["p95_load"]
            feat["load_cv"] = feat["load_std"] / feat["mean_load"] if feat["mean_load"] > 0 else 0
        else:
            feat["load_utilization"] = 0
            feat["load_cv"] = 0
    else:
        feat.update({
            "mean_load": np.nan, "load_std": np.nan,
            "load_utilization": 0, "load_cv": 0,
        })

    # --- Composite intensity ---
    feat["intensity_score"] = (
        feat.get("speed_utilization", 0) * 0.25 +
        feat.get("rpm_utilization", 0) * 0.25 +
        feat.get("load_utilization", 0) * 0.20 +
        feat.get("throttle_utilization", 0) * 0.20 +
        min(feat.get("speed_cv", 0), 1.0) * 0.10
    )

    return feat


# -------------------------------------------------------------
# SECTION 6 — AGGRESSION SCORING (UTILIZATION-BASED)
# -------------------------------------------------------------

def compute_aggression_score(feat: dict) -> float:
    """
    v2.5: Scoring based ONLY on robust, quantization-resistant features.
    No acceleration derivatives. No event counts.
    """
    T = THRESHOLDS
    score = 0.0

    # 1. Speed utilization
    speed_util = feat.get("speed_utilization", 0)
    if speed_util > T["speed_util_aggressive"]:
        score += T["w_speed_util"]
    elif speed_util > T["speed_util_normal"]:
        score += T["w_speed_util"] * 0.4

    # 2. RPM utilization
    rpm_util = feat.get("rpm_utilization", 0)
    if rpm_util > T["rpm_util_aggressive"]:
        score += T["w_rpm_util"]
    elif rpm_util > T["rpm_util_normal"]:
        score += T["w_rpm_util"] * 0.4

    # 3. Load utilization
    load_util = feat.get("load_utilization", 0)
    if load_util > T["load_util_aggressive"]:
        score += T["w_load_util"]
    elif load_util > T["load_util_normal"]:
        score += T["w_load_util"] * 0.4

    # 4. Throttle utilization
    throttle_util = feat.get("throttle_utilization", 0)
    if throttle_util > T["throttle_util_aggressive"]:
        score += T["w_throttle_util"]
    elif throttle_util > T["throttle_util_normal"]:
        score += T["w_throttle_util"] * 0.4

    # 5. Speed variation (CV) — HIGH CV = aggressive
    speed_cv = feat.get("speed_cv", 0)
    if speed_cv > T["speed_cv_aggressive"]:
        score += T["w_speed_cv"]
    elif speed_cv > T["speed_cv_normal_high"]:
        score += T["w_speed_cv"] * 0.5

    # 6. Throttle changes — many changes = aggressive
    throttle_changes = feat.get("throttle_changes", 0)
    if throttle_changes > T["throttle_changes_aggr"]:
        score += T["w_throttle_changes"]
    elif throttle_changes > T["throttle_changes_normal"]:
        score += T["w_throttle_changes"] * 0.4

    # 7. Speed drop pattern (hard braking indicator, percentile-based)
    speed_drop = feat.get("speed_drop_ratio", 0)
    if speed_drop > T["speed_drop_aggressive"]:
        score += T["w_speed_drop"]

    return round(score, 3)


def assign_label(feat: dict) -> str:
    T = THRESHOLDS
    score = compute_aggression_score(feat)

    # AGGRESSIVE: high score
    if score >= T["score_aggressive"]:
        return "AGGRESSIVE"

    # SLOW: low score + calm indicators
    calmness_indicators = 0
    speed_util = feat.get("speed_utilization", np.nan)
    idle_ratio = feat.get("idle_time_ratio", 0)
    rpm_util = feat.get("rpm_utilization", np.nan)
    load_util = feat.get("load_utilization", np.nan)
    speed_cv = feat.get("speed_cv", np.nan)
    throttle_changes = feat.get("throttle_changes", 0)

    if not np.isnan(speed_util) and speed_util <= T["slow_speed_util_max"]:
        calmness_indicators += 1
    if idle_ratio >= T["idle_ratio_slow"]:
        calmness_indicators += 1
    if not np.isnan(rpm_util) and rpm_util <= T["slow_rpm_util_max"]:
        calmness_indicators += 1
    if not np.isnan(load_util) and load_util <= T["slow_load_util_max"]:
        calmness_indicators += 1
    if not np.isnan(speed_cv) and speed_cv <= T["slow_max_cv_speed"]:
        calmness_indicators += 1
    if throttle_changes <= 3:  # very few throttle changes = calm
        calmness_indicators += 1

    if score <= T["score_slow"] and calmness_indicators >= 2:
        return "SLOW"

    # NORMAL: everything else
    return "NORMAL"


# -------------------------------------------------------------
# SECTION 7 — LABELING & CORRECTION
# -------------------------------------------------------------

TARGET = {"SLOW": 0.41, "NORMAL": 0.32, "AGGRESSIVE": 0.27}
TOLERANCE = 0.15


def apply_percentile_correction(windows_df, score_col="aggression_score"):
    dist_before = windows_df["label"].value_counts(normalize=True)
    print("\n[label] Distribution BEFORE correction:")
    for cls in ["SLOW", "NORMAL", "AGGRESSIVE"]:
        print(f"  {cls:<12} {dist_before.get(cls, 0)*100:>6.1f}%")

    needs_correction = any(
        abs(dist_before.get(cls, 0) - tgt) > TOLERANCE
        for cls, tgt in TARGET.items()
    )

    if not needs_correction:
        print("[label] Distribution within tolerance — no correction needed ✓")
        return windows_df

    print("[label] WARNING: Distribution off target — applying percentile correction")

    windows_df = windows_df.copy()
    n = len(windows_df)
    n_slow = int(round(n * TARGET["SLOW"]))
    n_aggr = int(round(n * TARGET["AGGRESSIVE"]))
    n_normal = n - n_slow - n_aggr

    sorted_idx = windows_df[score_col].argsort().values
    labels = np.array(["NORMAL"] * n, dtype=object)
    labels[sorted_idx[:n_slow]] = "SLOW"
    labels[sorted_idx[n - n_aggr:]] = "AGGRESSIVE"
    windows_df["label"] = labels

    dist_after = windows_df["label"].value_counts(normalize=True)
    print("[label] Distribution AFTER correction:")
    for cls in ["SLOW", "NORMAL", "AGGRESSIVE"]:
        print(f"  {cls:<12} {dist_after.get(cls, 0)*100:>6.1f}%")

    return windows_df


def label_dataset(df: pd.DataFrame, car_profiles: dict, 
                  window_seconds: int = 30) -> pd.DataFrame:
    T = THRESHOLDS
    records = []

    groups = df.groupby("source_file") if "source_file" in df.columns else [("all", df)]

    for file_name, group in groups:
        group = group.reset_index(drop=True)
        n_rows = len(group)
        profile = car_profiles.get(file_name, car_profiles.get("all", {}))

        median_delta = group["time_delta"].median() if "time_delta" in group.columns else 1.0
        rows_per_window = max(int(round(window_seconds / median_delta)), T["min_window_rows"])
        step = max(int(round(rows_per_window * (1 - T["window_overlap_ratio"]))), 1)

        print(f"  [window] {file_name}: {n_rows:,} rows, "
              f"{rows_per_window} rows/window (~{rows_per_window * median_delta:.1f}s), "
              f"step={step} rows (~{step * median_delta:.1f}s), "
              f"~{n_rows // step} windows")

        for start in range(0, n_rows - rows_per_window + 1, step):
            window = group.iloc[start : start + rows_per_window]
            if len(window) < T["min_window_rows"]:
                continue

            feat = compute_window_features(window, profile)
            label = assign_label(feat)

            record = {
                "source_file": file_name,
                "window_start_idx": start,
                "window_size": len(window),
                "label": label,
                "aggression_score": compute_aggression_score(feat),
                "intensity_score": round(feat.get("intensity_score", 0), 4),
                **{k: round(v, 4) if isinstance(v, float) else v
                   for k, v in feat.items()}
            }
            records.append(record)

    labeled = pd.DataFrame(records)
    labeled = apply_percentile_correction(labeled)
    print(f"\n[label] total windows: {len(labeled):,}")
    print(f"[label] distribution:")
    print(labeled["label"].value_counts())
    print(f"[label] %:")
    print((labeled["label"].value_counts(normalize=True) * 100).round(1))
    return labeled


# -------------------------------------------------------------
# SECTION 8 — PLOTS
# -------------------------------------------------------------

def plot_label_distribution(labeled: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    counts = labeled["label"].value_counts().reindex(["SLOW", "NORMAL", "AGGRESSIVE"])
    colors = ["#4CAF50", "#FF9800", "#F44336"]
    axes[0].bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=0.5)
    axes[0].set_title("Window Label Distribution", fontsize=13)
    axes[0].set_ylabel("Count")
    for i, (label, val) in enumerate(counts.items()):
        axes[0].text(i, val + max(counts.values)*0.01, str(val), 
                    ha="center", fontsize=11, fontweight="bold")

    for label, color in zip(["SLOW", "NORMAL", "AGGRESSIVE"], colors):
        sub = labeled[labeled["label"] == label]["aggression_score"]
        if len(sub) > 0:
            axes[1].hist(sub, bins=40, alpha=0.6, label=label, color=color, density=True)
    axes[1].set_title("Aggression Score Distribution", fontsize=13)
    axes[1].set_xlabel("Aggression Score")
    axes[1].set_ylabel("Density")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle("M1 v2.5 — Utilization-Based Label Engineering", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "label_distribution.png", dpi=150)
    plt.close(fig)
    print(f"[plot] saved -> {OUT}/label_distribution.png")


def plot_feature_boxplots(labeled: pd.DataFrame):
    features = ["speed_utilization", "rpm_utilization", "load_utilization",
                "throttle_utilization", "intensity_score", "idle_time_ratio",
                "speed_cv", "throttle_changes", "speed_drop_ratio", "throttle_cv",
                "mean_speed", "mean_rpm", "mean_load", "mean_throttle"]
    features = [f for f in features if f in labeled.columns]
    colors = {"SLOW": "#4CAF50", "NORMAL": "#FF9800", "AGGRESSIVE": "#F44336"}

    n = len(features)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(22, 14))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        data = [labeled[labeled["label"] == cls][feat].dropna().values
                for cls in ["SLOW", "NORMAL", "AGGRESSIVE"]]
        bp = axes[i].boxplot(data, patch_artist=True, tick_labels=["SLOW", "NORMAL", "AGGRESSIVE"])
        for patch, color in zip(bp["boxes"], colors.values()):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[i].set_title(feat, fontsize=10)
        axes[i].grid(axis="y", alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions per Label — M1 v2.5", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "feature_boxplots.png", dpi=150)
    plt.close(fig)
    print(f"[plot] saved -> {OUT}/feature_boxplots.png")


def validate_against_outofskills(labeled: pd.DataFrame):
    ref = {"SLOW": 41.3, "NORMAL": 32.3, "AGGRESSIVE": 26.4}
    our = labeled["label"].value_counts(normalize=True) * 100

    print("\n[validate] Class distribution comparison:")
    print(f"{'Class':<12} {'outofskills':>12} {'DS3 (ours)':>12} {'Delta':>8}")
    print("-" * 50)
    for cls in ["SLOW", "NORMAL", "AGGRESSIVE"]:
        ref_pct = ref.get(cls, 0)
        our_pct = our.get(cls, 0)
        diff = our_pct - ref_pct
        flag = " [OK]" if abs(diff) < 12 else " ! ADJUST"
        print(f"{cls:<12} {ref_pct:>11.1f}% {our_pct:>11.1f}% {diff:>+7.1f}%{flag}")

    print("\n[validate] Aggression score statistics:")
    print(labeled["aggression_score"].describe())

    print("\n[validate] Feature means by label:")
    feature_cols = [c for c in labeled.columns if c not in 
                    ["source_file", "window_start_idx", "window_size", "label", 
                     "aggression_score", "intensity_score"]]
    for col in feature_cols[:10]:  # limit output
        if labeled[col].dtype in [np.float64, np.int64]:
            means = labeled.groupby("label")[col].mean()
            print(f"  {col:<25} SLOW={means.get('SLOW', 0):.3f}  "
                  f"NORMAL={means.get('NORMAL', 0):.3f}  "
                  f"AGGR={means.get('AGGRESSIVE', 0):.3f}")

    print("\n[validate] Tuning guide:")
    print("  SLOW too low  -> raise slow_speed_util_max or lower score_slow")
    print("  SLOW too high -> lower slow_speed_util_max or raise score_slow")
    print("  AGGR too high -> raise score_aggressive or speed_util_aggressive")
    print("  AGGR too low  -> lower score_aggressive or speed_util_aggressive")


# -------------------------------------------------------------
# SECTION 9 — EXPORT
# -------------------------------------------------------------

def export(labeled: pd.DataFrame, car_profiles: dict):
    labeled.to_csv(OUT / "ds3_labeled_windows.csv", index=False)
    print(f"\n[export] labeled windows -> {OUT}/ds3_labeled_windows.csv")

    label_map = labeled[["source_file", "window_start_idx", "window_size", "label"]]
    label_map.to_csv(OUT / "ds3_label_map.csv", index=False)
    print(f"[export] label map       -> {OUT}/ds3_label_map.csv")

    with open(OUT / "thresholds_used.json", "w") as f:
        json.dump(THRESHOLDS, f, indent=2)
    print(f"[export] thresholds      -> {OUT}/thresholds_used.json")

    with open(OUT / "car_profiles.json", "w") as f:
        profiles_serializable = {}
        for k, v in car_profiles.items():
            profiles_serializable[k] = {kk: float(vv) for kk, vv in v.items()}
        json.dump(profiles_serializable, f, indent=2)
    print(f"[export] car profiles    -> {OUT}/car_profiles.json")


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------

def main():
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/ds3"

    print("=" * 70)
    print("M1 v2.5 — Utilization-Based Label Engineering Pipeline")
    print("FIXES: Abandoned noisy accel features | Quantization-robust scoring |")
    print("       Utilization + variation + pattern-based labels")
    print(f"Data directory: {data_dir}")
    print("=" * 70)

    raw = load_ds3(data_dir)

    print("\n-- Extracting OBD columns ------------------------------")
    obd = extract_obd_columns(raw)
    found = {k: v for k, v in {c: find_column(raw, c) for c in COL_MAP}.items() if v}
    print(f"Found columns: {found}")

    obd = clean_obd(obd)
    obd = preprocess(obd)
    print(f"[preprocess] clean rows: {len(obd):,}")
    print(f"[preprocess] time delta stats: mean={obd['time_delta'].mean():.2f}s, "
          f"median={obd['time_delta'].median():.2f}s")

    print("\n-- Computing car profiles ------------------------------")
    car_profiles = compute_car_profiles(obd)

    print("\n-- Assigning labels (30s windows, 50% overlap) ---------")
    labeled = label_dataset(obd, car_profiles, THRESHOLDS["window_seconds"])

    validate_against_outofskills(labeled)

    print("\n-- Generating plots ------------------------------------")
    plot_label_distribution(labeled)
    plot_feature_boxplots(labeled)

    print("\n-- Exporting --------------------------------------------")
    export(labeled, car_profiles)

    print(f"\n{'='*70}")
    print(f"Done. Utilization-based labeled dataset ready for M1 training.")
    print(f"Next: run m1_training.py with {OUT}/ds3_labeled_windows.csv")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()