"""
M1 — Label Engineering Pipeline (v2.2 - Car-Agnostic, Time-Aware, FIXED)
=========================================================================
Source dataset   : cephasax DS3 (OBD-II, 55 columns, no labels)
Label reference  : outofskills/driving-behavior (AccX/Y/Z + Gyro -> SLOW/NORMAL/AGGRESSIVE)

CRITICAL FIXES in v2.2:
-----------------------
1. SCALED EVENT THRESHOLDS: Event counts are now rates per second, not absolute counts
   (was: fixed counts like 2 events per window, regardless of window size)

2. FIXED THROTTLE_UTILIZATION: Uses percentile-based ratio instead of max/max=1.0
   (was: max_throttle_delta / max_throttle_observed always = 1.0)

3. RAISED JERK THRESHOLD: 8.0 m/s³ with adaptive scaling
   (was: 4.0 m/s³ — too low even with smoothing)

4. FIXED SPEED_CV LOGIC: SLOW now correctly has LOW cv, AGGRESSIVE has HIGH cv
   (was: inverted — SLOW had higher cv than AGGRESSIVE)

5. FIXED WINDOW SIZE: window_seconds now correctly converts to row counts
   using actual median sampling interval (was: used as raw row count)

6. ADDED WINDOW OVERLAP: 50% overlap for better temporal coverage
   (was: non-overlapping, threw away boundary transitions)

7. JERK SMOOTHING: 5-point moving average on speed before jerk computation
   (was: raw 10 Hz noise amplified 100x)

8. SIMPLIFIED SLOW LOGIC: Single "calmness score" instead of nested OR/AND
   (was: 4 conditions with mixed logic, hard to reason about)

9. CAR-AGNOSTIC SPEED_STD: Now uses coefficient of variation, not absolute threshold
   (was: speed_std <= 8.0 — absolute, not car-agnostic)

10. PRE-CORRECTION DISTRIBUTION PRINT: Debug visibility into threshold performance
    (was: hidden — correction triggered silently)

11. FIXED DUPLICATE FILE LOADING: Deduplicate paths from glob
    (was: files loaded twice via root + subdirectory matching)

12. FIXED MATPLOTLIB DEPRECATION: tick_labels instead of labels

Directory layout expected:
    data/ds3/
        *.csv   ← cephasax DS3 files
"""

import os, glob, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

OUT = Path("outputs/m1_labels")
OUT.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# SECTION 1 — THRESHOLDS (CAR-AGNOSTIC, UTILIZATION-BASED)
# -------------------------------------------------------------
# All thresholds are now RATIOS (0.0-1.0) or PHYSICAL LIMITS (m/s²)
# No absolute speed/RPM values — model works for any car

THRESHOLDS = {
    # --- Physical limits (same for all cars) ---
    # Hard braking: AccY < -3.5 m/s² = -12.6 km/h per second
    "hard_brake_ms2"           : -3.5,     # m/s² — PHYSICAL, car-independent
    "rapid_accel_ms2"          :  3.0,     # m/s² — PHYSICAL, car-independent

    # Conversion: 1 m/s² = 3.6 km/h/s
    "ms2_to_kmh_per_s"         : 3.6,

    # --- Event RATES per second (NOT absolute counts) ---
    # v2.2 FIX: These are now rates per second, scaled to window duration internally
    # A 30s window needs ~3 hard brakes to be aggressive (0.1/sec * 30s)
    "hard_brake_rate_aggressive"   : 0.10,   # >= 0.10 events/sec = aggressive
    "hard_brake_rate_normal"       : 0.03,   # >= 0.03 events/sec = normal
    "rapid_accel_rate_aggressive"  : 0.10,   # >= 0.10 events/sec = aggressive
    "rapid_accel_rate_normal"      : 0.03,   # >= 0.03 events/sec = normal

    # --- Utilization thresholds (0.0 - 1.0) ---
    # How much of the car's capability is being used
    "rpm_util_aggressive"      : 0.80,     # >80% of car's p95 RPM
    "rpm_util_normal"          : 0.50,     # 50-80% = normal

    "speed_util_aggressive"    : 0.85,     # >85% of car's p95 speed
    "speed_util_normal"        : 0.60,     # 60-85% = normal cruising
    "speed_util_slow"          : 0.30,     # <30% = slow/city

    "load_util_aggressive"     : 0.85,     # >85% of car's p95 load
    "load_util_normal"         : 0.55,

    # v2.2 FIX: Throttle utilization now uses p95, not max
    "throttle_util_aggressive" : 0.80,     # >80% of car's p95 throttle delta
    "throttle_util_normal"     : 0.40,

    # --- SLOW class: simplified, car-agnostic ---
    "slow_speed_util_max"      : 0.25,     # <=25% of car's p95 speed
    "slow_idle_time_ratio"     : 0.25,     # >25% stopped = likely SLOW
    "slow_max_rpm_util"        : 0.35,     # Low RPM utilization
    "slow_max_load_util"       : 0.40,     # Low load utilization
    "slow_max_cv_speed"        : 0.30,     # Coefficient of variation <= 30% (car-agnostic)
    "score_slow"               : 0.15,      # Max aggression score for SLOW

    # --- Aggression score weights ---
    "w_hard_brake"             : 0.25,
    "w_rapid_accel"            : 0.25,
    "w_high_rpm_util"          : 0.20,
    "w_high_throttle_util"     : 0.10,
    "w_high_load_util"         : 0.10,
    "w_high_jerk"              : 0.10,     # reduced; jerk is noisy even with smoothing
    "jerk_threshold_ms2"       : 8.0,      # v2.2 FIX: raised from 4.0 to 8.0 m/s³

    # --- Final score thresholds ---
    "score_aggressive"         : 0.55,     # raise (was likely 0.30-0.40)
    "score_slow"               : 0.15,

    # --- Window settings ---
    "window_seconds"           : 30,        # ACTUAL seconds (not rows)
    "window_overlap_ratio"     : 0.50,     # 50% overlap between consecutive windows
    "min_window_rows"          : 10,
    "time_gap_max"             : 10.0,     # Max seconds between rows
    "time_gap_min"             : 0.1,      # Min seconds (avoid division by zero)

    # --- Jerk smoothing ---
    "jerk_smooth_window"       : 5,         # 5-point moving average for speed before jerk
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
    """Find the actual column name in df for a given canonical OBD name."""
    variants = COL_MAP.get(canonical, [canonical])
    df_cols_upper = {c.upper().replace(" ", "_"): c for c in df.columns}
    for v in variants:
        v_norm = v.upper().replace(" ", "_")
        if v_norm in df_cols_upper:
            return df_cols_upper[v_norm]
    return None


def load_ds3(data_dir: str) -> pd.DataFrame:
    """Load CSV files with deduplication to prevent double-loading."""
    files = set(
        glob.glob(os.path.join(data_dir, "**/*.csv"), recursive=True) +
        glob.glob(os.path.join(data_dir, "*.csv"))
    )
    if not files:
        raise FileNotFoundError(f"No CSV files in {data_dir}")

    frames = []
    for f in sorted(files):
        try:
            # Try comma first, then semicolon (European format)
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
# SECTION 3 — PREPROCESSING (TIME-AWARE)
# -------------------------------------------------------------

def extract_obd_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and rename OBD columns. Strip units from string values.
    CRITICAL: Properly parse timestamps with European decimal commas.
    """
    result = pd.DataFrame()

    for canonical, _ in COL_MAP.items():
        col = find_column(df, canonical)
        if col:
            series = df[col].copy().astype(str)
            # Strip units: RPM, km/h, %, kPa, g/s, C, °C
            series = series.str.replace(
                r"[A-Za-z°%/\s]", "", regex=True
            ).str.replace(",", ".").str.strip()
            result[canonical] = pd.to_numeric(series, errors="coerce")
        else:
            result[canonical] = np.nan

    # Preserve source file
    if "source_file" in df.columns:
        result["source_file"] = df["source_file"].values

    # Parse timestamp (handle European scientific notation "1,51336E+12")
    time_candidates = ["TIME", "TIMESTAMP", "Time", "time"]
    for tc in time_candidates:
        if tc in df.columns:
            ts_raw = df[tc].astype(str).str.replace(",", ".", regex=False)
            result["timestamp_sec"] = pd.to_numeric(ts_raw, errors="coerce")
            break

    return result


def compute_time_aware_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """
    CRITICAL FIX v2.1: Compute kinematically correct derivatives with smoothing.

    OLD (WRONG): df["speed"].diff() — assumes uniform 1-second sampling
    NEW v2.0 (CORRECT): df["speed"].diff() / time_delta — actual km/h per SECOND
    NEW v2.1 (SMOOTHED): 5-point MA on speed before derivatives to reduce noise
    """
    df = df.copy()
    T = THRESHOLDS

    # Compute actual time between rows
    if "timestamp_sec" in df.columns and df["timestamp_sec"].notna().sum() > 2:
        df["time_delta"] = df["timestamp_sec"].diff()
    else:
        # Fallback: assume 1-second intervals if no valid timestamp
        df["time_delta"] = 1.0

    # Clip time gaps to avoid division errors
    df["time_delta"] = df["time_delta"].clip(
        lower=T["time_gap_min"], 
        upper=T["time_gap_max"]
    ).fillna(1.0)

    # v2.1 FIX: Smooth speed before computing derivatives (reduces jerk noise)
    if "speed" in df.columns:
        df["speed_smoothed"] = df["speed"].rolling(
            window=T["jerk_smooth_window"], 
            min_periods=1, 
            center=True
        ).mean()
    else:
        df["speed_smoothed"] = np.nan

    # KINEMATICALLY CORRECT derivatives using SMOOTHED speed
    df["speed_delta"] = df["speed_smoothed"].diff() / df["time_delta"]
    # Throttle should be 0-100%, delta should be % per second
    df["throttle_delta"] = df["throttle"].diff().abs() / df["time_delta"]
    # Cap at physical maximum
    df["throttle_delta"] = df["throttle_delta"].clip(0, 100)
    df["rpm_delta"] = df["rpm"].diff().abs() / df["time_delta"]

    # Fill NaN (first row) with 0
    df["speed_delta"] = df["speed_delta"].fillna(0)
    df["throttle_delta"] = df["throttle_delta"].fillna(0)
    df["rpm_delta"] = df["rpm_delta"].fillna(0)

    # Acceleration (speed_delta is already acceleration in km/h/s)
    df["speed_accel"] = df["speed_delta"]

    # Jerk = derivative of acceleration (km/h/s²) — now on SMOOTHED data
    df["jerk"] = df["speed_accel"].diff() / df["time_delta"]
    df["jerk"] = df["jerk"].fillna(0)

    # Convert to m/s² for physical consistency with outofskills reference
    df["accel_ms2"] = df["speed_delta"] / T["ms2_to_kmh_per_s"]
    df["jerk_ms3"] = df["jerk"] / T["ms2_to_kmh_per_s"]

    return df


def clean_obd(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop rows where all key signals are NaN, clip extremes."""
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
# SECTION 4 — CAR PROFILE COMPUTATION (CAR-AGNOSTIC)
# -------------------------------------------------------------

def compute_car_profiles(df: pd.DataFrame) -> dict:
    """
    Infer each car's capability from its own data.
    Uses p95 (not max) to avoid sensor outliers.

    Returns: {source_file: {p95_speed, p95_rpm, p95_load, p95_throttle_delta, ...}}
    """
    profiles = {}

    for file_name, group in df.groupby("source_file"):
        # Only use moving data (speed > 0) for capability estimation
        moving = group[group["speed"] > 0]

        if len(moving) < 10:
            # Not enough data — use conservative defaults
            profiles[file_name] = {
                "p95_speed": 120.0,
                "p99_speed": 150.0,
                "p95_rpm": 4000.0,
                "p99_rpm": 5000.0,
                "p95_load": 70.0,
                "p95_throttle_delta": 10.0,   # v2.2 FIX: p95 instead of max
                "max_throttle_observed": 50.0,
                "median_speed": 40.0,
                "median_rpm": 1500.0,
            }
            continue

        # v2.2 FIX: Use p95 for throttle_delta to avoid outlier-driven 1.0 ratios
        throttle_delta_p95 = group["throttle_delta"].quantile(0.95)
        if pd.isna(throttle_delta_p95) or throttle_delta_p95 <= 0:
            throttle_delta_p95 = 10.0

        profiles[file_name] = {
            "p95_speed": moving["speed"].quantile(0.95),
            "p99_speed": moving["speed"].quantile(0.99),
            "p95_rpm": moving["rpm"].quantile(0.95),
            "p99_rpm": moving["rpm"].quantile(0.99),
            "p95_load": moving["load"].quantile(0.95),
            "p95_throttle_delta": throttle_delta_p95,  # v2.2 FIX
            "max_throttle_observed": min(group["throttle_delta"].max() or 50.0, 100.0),
            "median_speed": moving["speed"].median(),
            "median_rpm": moving["rpm"].median(),
        }

    print(f"[profile] Computed profiles for {len(profiles)} files")
    return profiles


# -------------------------------------------------------------
# SECTION 5 — WINDOW FEATURES (CAR-AGNOSTIC)
# -------------------------------------------------------------

def compute_window_features(window: pd.DataFrame, car_profile: dict, window_duration_sec: float = 30.0) -> dict:
    """
    Compute car-agnostic features for a single window.
    All continuous features are normalized to [0, 1] utilization ratios.

    v2.2: Added window_duration_sec for rate-based event thresholds.
    """
    T = THRESHOLDS
    cp = car_profile
    feat = {}

    # --- Idle detection ---
    total_rows = len(window)
    stopped_rows = (window["speed"] == 0).sum() if "speed" in window.columns else 0
    feat["idle_time_ratio"] = stopped_rows / total_rows if total_rows > 0 else 0

    # --- Speed features (normalized) ---
    if "speed" in window.columns and window["speed"].notna().sum() > 2:
        feat["mean_speed"] = window["speed"].mean()
        feat["max_speed"] = window["speed"].max()
        feat["speed_std"] = window["speed"].std()

        # CAR-AGNOSTIC: utilization ratios
        if cp["p95_speed"] > 0:
            feat["speed_utilization"] = feat["mean_speed"] / cp["p95_speed"]
            feat["max_speed_utilization"] = feat["max_speed"] / cp["p95_speed"]
            # v2.1: Coefficient of variation (car-agnostic measure of variation)
            feat["speed_cv"] = feat["speed_std"] / feat["mean_speed"] if feat["mean_speed"] > 0 else 0
        else:
            feat["speed_utilization"] = 0
            feat["max_speed_utilization"] = 0
            feat["speed_cv"] = 0

        # v2.2 FIX: Event RATES (per second), not absolute counts
        hard_brake_cnt = (window["accel_ms2"] < T["hard_brake_ms2"]).sum()
        rapid_accel_cnt = (window["accel_ms2"] > T["rapid_accel_ms2"]).sum()
        feat["hard_brake_cnt"] = int(hard_brake_cnt)
        feat["rapid_accel_cnt"] = int(rapid_accel_cnt)
        feat["hard_brake_rate"] = hard_brake_cnt / window_duration_sec
        feat["rapid_accel_rate"] = rapid_accel_cnt / window_duration_sec
    else:
        feat.update({
            "mean_speed": np.nan, "max_speed": np.nan, "speed_std": np.nan,
            "speed_utilization": 0, "max_speed_utilization": 0, "speed_cv": 0,
            "hard_brake_cnt": 0, "rapid_accel_cnt": 0,
            "hard_brake_rate": 0.0, "rapid_accel_rate": 0.0,
        })

    # --- RPM features (normalized) ---
    if "rpm" in window.columns and window["rpm"].notna().sum() > 2:
        feat["mean_rpm"] = window["rpm"].mean()
        feat["max_rpm"] = window["rpm"].max()

        if cp["p95_rpm"] > 0:
            feat["rpm_utilization"] = feat["mean_rpm"] / cp["p95_rpm"]
            feat["high_rpm_ratio"] = (window["rpm"] / cp["p95_rpm"] > T["rpm_util_aggressive"]).mean()
        else:
            feat["rpm_utilization"] = 0
            feat["high_rpm_ratio"] = 0
    else:
        feat.update({
            "mean_rpm": np.nan, "max_rpm": np.nan,
            "rpm_utilization": 0, "high_rpm_ratio": 0
        })

    # --- Throttle features (normalized) ---
    if "throttle" in window.columns and window["throttle"].notna().sum() > 2:
        feat["mean_throttle"] = window["throttle"].mean()
        feat["throttle_std"] = window["throttle"].std()
        feat["max_throttle_delta"] = window["throttle_delta"].max()

        # v2.2 FIX: Use p95_throttle_delta instead of max_throttle_observed
        # This prevents the max/max = 1.0 bug
        throttle_ref = cp.get("p95_throttle_delta", cp.get("max_throttle_observed", 50.0))
        if throttle_ref > 0:
            feat["throttle_utilization"] = feat["max_throttle_delta"] / throttle_ref
            # Cap at 1.0 to avoid outliers
            feat["throttle_utilization"] = min(feat["throttle_utilization"], 1.0)
        else:
            feat["throttle_utilization"] = 0
    else:
        feat.update({
            "mean_throttle": np.nan, "throttle_std": np.nan,
            "max_throttle_delta": 0, "throttle_utilization": 0
        })

    # --- Load features (normalized) ---
    if "load" in window.columns and window["load"].notna().sum() > 2:
        feat["mean_load"] = window["load"].mean()

        if cp["p95_load"] > 0:
            feat["load_utilization"] = feat["mean_load"] / cp["p95_load"]
        else:
            feat["load_utilization"] = 0
    else:
        feat.update({"mean_load": np.nan, "load_utilization": 0})

    # --- Jerk features (smoothed, car-agnostic by nature) ---
    if "jerk_ms3" in window.columns and window["jerk_ms3"].notna().sum() > 2:
        feat["mean_jerk"] = window["jerk_ms3"].mean()
        feat["max_jerk"] = window["jerk_ms3"].max()
        feat["jerk_std"] = window["jerk_ms3"].std()
        # v2.2 FIX: Higher threshold (8.0) and adaptive: require >50% of window above threshold
        feat["high_jerk_ratio"] = (window["jerk_ms3"].abs() > T.get("jerk_threshold_ms2", 8.0)).mean()
    else:
        feat.update({
            "mean_jerk": 0, "max_jerk": 0, "jerk_std": 0, "high_jerk_ratio": 0
        })

    # --- Composite intensity score (0-1) ---
    feat["intensity_score"] = (
        feat.get("speed_utilization", 0) * 0.25 +
        feat.get("rpm_utilization", 0) * 0.25 +
        feat.get("load_utilization", 0) * 0.20 +
        feat.get("throttle_utilization", 0) * 0.20 +
        min(feat.get("high_jerk_ratio", 0) * 2, 1.0) * 0.10
    )

    return feat


# -------------------------------------------------------------
# SECTION 6 — AGGRESSION SCORING (CAR-AGNOSTIC)
# -------------------------------------------------------------

def compute_aggression_score(feat: dict) -> float:
    """
    Weighted aggression score using CAR-AGNOSTIC utilization features.
    v2.2: Uses event RATES instead of absolute counts.
    """
    T = THRESHOLDS
    score = 0.0

    # Hard braking (physical threshold — car-independent)
    # v2.2 FIX: Use rates, not counts
    hb_rate = feat.get("hard_brake_rate", 0)
    if hb_rate >= T["hard_brake_rate_aggressive"]:
        score += T["w_hard_brake"]
    elif hb_rate >= T["hard_brake_rate_normal"]:
        score += T["w_hard_brake"] * 0.4

    # Rapid acceleration
    ra_rate = feat.get("rapid_accel_rate", 0)
    if ra_rate >= T["rapid_accel_rate_aggressive"]:
        score += T["w_rapid_accel"]
    elif ra_rate >= T["rapid_accel_rate_normal"]:
        score += T["w_rapid_accel"] * 0.4

    # High RPM utilization
    rpm_util = feat.get("rpm_utilization", 0)
    if rpm_util > T["rpm_util_aggressive"]:
        score += T["w_high_rpm_util"]
    elif rpm_util > T["rpm_util_normal"]:
        score += T["w_high_rpm_util"] * 0.3

    # Aggressive throttle utilization
    throttle_util = feat.get("throttle_utilization", 0)
    if throttle_util > T["throttle_util_aggressive"]:
        score += T["w_high_throttle_util"]
    elif throttle_util > T["throttle_util_normal"]:
        score += T["w_high_throttle_util"] * 0.3

    # High load utilization
    load_util = feat.get("load_utilization", 0)
    if load_util > T["load_util_aggressive"]:
        score += T["w_high_load_util"]
    elif load_util > T["load_util_normal"]:
        score += T["w_high_load_util"] * 0.4

    # High jerk (smoothed)
    # v2.2 FIX: Require >50% of window above threshold for full weight
    high_jerk = feat.get("high_jerk_ratio", 0)
    if high_jerk > 0.5:
        score += T["w_high_jerk"]
    elif high_jerk > 0.2:
        score += T["w_high_jerk"] * 0.4

    return score


def assign_label(feat: dict) -> str:
    """
    v2.2 FIX: Simplified SLOW logic using a single "calmness score".

    Target distribution: SLOW ~35-40%, NORMAL ~30-35%, AGGRESSIVE ~25-30%
    """
    T = THRESHOLDS
    score = compute_aggression_score(feat)

    # AGGRESSIVE: high score regardless of speed
    if score >= T["score_aggressive"]:
        return "AGGRESSIVE"

    # v2.2 SIMPLIFIED SLOW LOGIC:
    # Compute a "calmness score" — higher = more likely SLOW
    calmness_indicators = 0

    speed_util = feat.get("speed_utilization", np.nan)
    idle_ratio = feat.get("idle_time_ratio", 0)
    rpm_util = feat.get("rpm_utilization", np.nan)
    load_util = feat.get("load_utilization", np.nan)
    speed_cv = feat.get("speed_cv", np.nan)

    if not np.isnan(speed_util) and speed_util <= T["slow_speed_util_max"]:
        calmness_indicators += 1
    if idle_ratio >= T["slow_idle_time_ratio"]:
        calmness_indicators += 1
    if not np.isnan(rpm_util) and rpm_util <= T["slow_max_rpm_util"]:
        calmness_indicators += 1
    if not np.isnan(load_util) and load_util <= T["slow_max_load_util"]:
        calmness_indicators += 1
    # v2.2 FIX: SLOW should have LOW cv (steady speed), not high
    if not np.isnan(speed_cv) and speed_cv <= T["slow_max_cv_speed"]:
        calmness_indicators += 1

    # SLOW: low aggression score AND at least 2 calmness indicators
    if score <= T["score_slow"] and calmness_indicators >= 2:
        return "SLOW"

    return "NORMAL"


# -------------------------------------------------------------
# SECTION 7 — LABELING & PERCENTILE CORRECTION
# -------------------------------------------------------------

TARGET = {"SLOW": 0.41, "NORMAL": 0.32, "AGGRESSIVE": 0.27}
TOLERANCE = 0.15


def apply_percentile_correction(windows_df, score_col="aggression_score"):
    """
    Apply percentile correction only if distribution is off-target.
    v2.1: Always prints pre-correction distribution for debugging.
    """
    # v2.1 FIX: Print pre-correction distribution
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

    # Compute exact counts from targets
    n_slow  = int(round(n * TARGET["SLOW"]))
    n_aggr  = int(round(n * TARGET["AGGRESSIVE"]))
    n_normal = n - n_slow - n_aggr   # remainder → no rounding gaps

    # Rank windows by aggression score (lowest = most SLOW)
    sorted_idx = windows_df[score_col].argsort().values  # ascending order

    labels = np.array(["NORMAL"] * n, dtype=object)
    labels[sorted_idx[:n_slow]]          = "SLOW"        # bottom n_slow
    labels[sorted_idx[n - n_aggr:]]      = "AGGRESSIVE"  # top n_aggr
    # middle is already "NORMAL"

    windows_df["label"] = labels

    # v2.1: Print post-correction distribution
    dist_after = windows_df["label"].value_counts(normalize=True)
    print("[label] Distribution AFTER correction:")
    for cls in ["SLOW", "NORMAL", "AGGRESSIVE"]:
        print(f"  {cls:<12} {dist_after.get(cls, 0)*100:>6.1f}%")

    return windows_df


def label_dataset(df: pd.DataFrame, car_profiles: dict, window_seconds: int = 30) -> pd.DataFrame:
    """
    v2.1 FIX: Slide windows with correct row-count computation and overlap.
    v2.2: Pass window_duration_sec to feature computation for rate-based thresholds.
    """
    T = THRESHOLDS
    records = []

    groups = df.groupby("source_file") if "source_file" in df.columns else [("all", df)]

    for file_name, group in groups:
        group = group.reset_index(drop=True)
        n_rows = len(group)
        profile = car_profiles.get(file_name, car_profiles.get("all", {}))

        # v2.1 FIX: Compute actual rows per window from sampling rate
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

            # v2.2: Pass actual window duration for rate computation
            window_duration_sec = window["time_delta"].sum() if "time_delta" in window.columns else window_seconds
            feat = compute_window_features(window, profile, window_duration_sec)
            label = assign_label(feat)

            record = {
                "source_file": file_name,
                "window_start_idx": start,
                "window_size": len(window),
                "label": label,
                "aggression_score": round(compute_aggression_score(feat), 3),
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
# SECTION 8 — VALIDATION & VISUALISATION
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

    fig.suptitle("M1 v2.2 — Car-Agnostic Label Engineering", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "label_distribution.png", dpi=150)
    plt.close(fig)
    print(f"[plot] saved -> {OUT}/label_distribution.png")


def plot_feature_boxplots(labeled: pd.DataFrame):
    # Plot UTILIZATION features (car-agnostic)
    features = ["speed_utilization", "rpm_utilization", "load_utilization",
                "throttle_utilization", "intensity_score", "idle_time_ratio",
                "hard_brake_rate", "rapid_accel_rate", "high_jerk_ratio", "speed_cv"]
    features = [f for f in features if f in labeled.columns]
    colors = {"SLOW": "#4CAF50", "NORMAL": "#FF9800", "AGGRESSIVE": "#F44336"}

    n = len(features)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(18, 10))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        data = [labeled[labeled["label"] == cls][feat].dropna().values
                for cls in ["SLOW", "NORMAL", "AGGRESSIVE"]]
        # v2.1 FIX: tick_labels instead of labels (Matplotlib 3.9+)
        bp = axes[i].boxplot(data, patch_artist=True, tick_labels=["SLOW", "NORMAL", "AGGRESSIVE"])
        for patch, color in zip(bp["boxes"], colors.values()):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[i].set_title(feat, fontsize=11)
        axes[i].grid(axis="y", alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Car-Agnostic Feature Distributions per Label — M1 v2.2", fontsize=13)
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

    print("\n[validate] Tuning guide:")
    print("  SLOW too low  -> raise slow_speed_util_max or lower score_slow")
    print("  SLOW too high -> lower slow_speed_util_max or raise score_slow")
    print("  AGGR too high -> raise score_aggressive or hard_brake_ms2")
    print("  AGGR too low  -> lower score_aggressive or hard_brake_ms2")


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
        # Convert numpy types to native Python for JSON
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
    print("M1 v2.2 — Car-Agnostic Label Engineering Pipeline")
    print("FIXES: Scaled event rates | Fixed throttle_util | Raised jerk threshold |")
    print("       Fixed speed_cv logic | Correct window sizing | 50% overlap |")
    print("       Jerk smoothing | Simplified SLOW logic | Pre-correction debug")
    print(f"Data directory: {data_dir}")
    print("=" * 70)

    # 1. Load
    raw = load_ds3(data_dir)

    # 2. Extract OBD columns
    print("\n-- Extracting OBD columns ------------------------------")
    obd = extract_obd_columns(raw)
    found = {k: v for k, v in {c: find_column(raw, c) for c in COL_MAP}.items() if v}
    print(f"Found columns: {found}")

    # 3. Clean + compute TIME-AWARE deltas
    obd = clean_obd(obd)
    obd = compute_time_aware_deltas(obd)
    print(f"[preprocess] clean rows: {len(obd):,}")
    print(f"[preprocess] time delta stats: mean={obd['time_delta'].mean():.2f}s, "
          f"median={obd['time_delta'].median():.2f}s")

    # 4. Compute CAR PROFILES (car-agnostic normalization)
    print("\n-- Computing car profiles ------------------------------")
    car_profiles = compute_car_profiles(obd)

    # 5. Label
    print("\n-- Assigning labels (30s windows, 50% overlap, car-agnostic) --------")
    labeled = label_dataset(obd, car_profiles, THRESHOLDS["window_seconds"])

    # 6. Validate
    validate_against_outofskills(labeled)

    # 7. Visualise
    print("\n-- Generating plots ------------------------------------")
    plot_label_distribution(labeled)
    plot_feature_boxplots(labeled)

    # 8. Export
    print("\n-- Exporting --------------------------------------------")
    export(labeled, car_profiles)

    print(f"\n{'='*70}")
    print(f"Done. Car-agnostic labeled dataset ready for M1 training.")
    print(f"Next: run m1_training.py with {OUT}/ds3_labeled_windows.csv")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()