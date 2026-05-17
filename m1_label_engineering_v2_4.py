"""
M1 — Label Engineering Pipeline (v2.4 - Car-Agnostic, ROBUST, Noise-Resistant)
=============================================================================
Source dataset   : cephasax DS3 (OBD-II, 55 columns, no labels)
Label reference  : outofskills/driving-behavior (AccX/Y/Z + Gyro -> SLOW/NORMAL/AGGRESSIVE)

CRITICAL FIXES in v2.4:
-----------------------
1. ROBUST ACCELERATION: Replaced point-to-point speed diff with 1-second 
   block-averaged acceleration. Eliminates 10 Hz noise amplification.
   (was: speed.diff() / 0.1s = ±16 m/s² noise)

2. NOISE-RESISTANT EVENT DETECTION: Events require sustained pattern 
   (3+ consecutive points) instead of single-point threshold crossing.

3. SIMPLIFIED SCORING: Removed multi-indicator overkill. Back to 
   interpretable weighted score with properly calibrated thresholds.

4. PERCENTILE-ADAPTIVE THRESHOLDS: Kept from v2.3 but applied to 
   properly smoothed data.

5. FIXED THROTTLE_UTILIZATION: Uses p95 (v2.2 carryover)
6. RAISED JERK THRESHOLD: Applied to smoothed data (v2.2 carryover)
7. FIXED SPEED_CV LOGIC: SLOW = low CV (v2.2 carryover)
8. CORRECT WINDOW SIZING (v2.1 carryover)
9. 50% OVERLAP (v2.1 carryover)

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
    # --- Physical limits (for properly smoothed 1s-block accel) ---
    # v2.4: These apply to 1-second averaged acceleration, not 0.1s point diff
    "hard_brake_ms2"           : -2.5,     # -2.5 m/s² = noticeable braking
    "hard_brake_extreme_ms2"   : -4.5,     # -4.5 m/s² = hard braking
    "rapid_accel_ms2"          :  2.0,     # 2.0 m/s² = noticeable acceleration
    "rapid_accel_extreme_ms2"  :  4.0,     # 4.0 m/s² = hard acceleration

    # Conversion
    "ms2_to_kmh_per_s"         : 3.6,

    # --- Event counts (for sustained events on smoothed data) ---
    "hard_brake_events_aggressive" : 2,    # 2+ sustained hard brakes per 30s
    "hard_brake_events_normal"     : 1,
    "rapid_accel_events_aggressive": 2,
    "rapid_accel_events_normal"    : 1,

    # --- Utilization thresholds ---
    "rpm_util_aggressive"      : 0.80,
    "rpm_util_normal"          : 0.50,
    "speed_util_aggressive"    : 0.85,
    "speed_util_normal"        : 0.60,
    "speed_util_slow"          : 0.30,
    "load_util_aggressive"     : 0.85,
    "load_util_normal"         : 0.55,
    "throttle_util_aggressive" : 0.80,
    "throttle_util_normal"     : 0.40,

    # --- SLOW class ---
    "slow_speed_util_max"      : 0.25,
    "slow_idle_time_ratio"     : 0.25,
    "slow_max_rpm_util"        : 0.35,
    "slow_max_load_util"       : 0.40,
    "slow_max_cv_speed"        : 0.30,
    "score_slow"               : 0.20,      # v2.4: raised

    # --- Aggression score weights ---
    "w_hard_brake"             : 0.25,
    "w_rapid_accel"            : 0.25,
    "w_high_rpm_util"          : 0.15,
    "w_high_throttle_util"     : 0.10,
    "w_high_load_util"         : 0.10,
    "w_high_jerk"              : 0.10,
    "w_extreme_events"         : 0.05,     # small bonus for extreme events

    # --- Jerk ---
    "jerk_threshold_ms2"       : 6.0,      # v2.4: on smoothed data this is reasonable
    "jerk_sustained_ratio"     : 0.20,

    # --- Final score thresholds ---
    "score_aggressive"         : 0.50,     # v2.4: lowered — with proper smoothing, this should work
    "score_slow"               : 0.20,

    # --- Window settings ---
    "window_seconds"           : 30,
    "window_overlap_ratio"     : 0.50,
    "min_window_rows"          : 10,
    "time_gap_max"             : 10.0,
    "time_gap_min"             : 0.1,

    # --- Smoothing settings ---
    "speed_smooth_window"      : 11,        # v2.4: 11-point (~1s) MA for speed
    "accel_smooth_window"      : 10,        # v2.4: 10-point block average for accel
    "jerk_smooth_window"       : 5,
    "sustained_event_min_pts"  : 3,         # v2.4: event must last 3+ points
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
# SECTION 3 — PREPROCESSING (ROBUST, NOISE-RESISTANT)
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


def compute_robust_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """
    v2.4 CRITICAL FIX: Robust acceleration computation.

    OLD (BROKEN): speed.diff() / 0.1s → ±16 m/s² noise
    NEW (ROBUST): 
      1. Smooth speed with 11-point (~1s) MA
      2. Compute accel over 1-second blocks (10 points)
      3. Jerk from block-averaged accel

    This eliminates 10 Hz quantization noise.
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

    # Step 1: Smooth speed with wider window (~1 second)
    if "speed" in df.columns:
        df["speed_smooth"] = df["speed"].rolling(
            window=T["speed_smooth_window"], 
            min_periods=1, 
            center=True
        ).mean()
    else:
        df["speed_smooth"] = np.nan

    # Step 2: Compute 1-second block averages for acceleration
    # Instead of point-to-point diff, use block averages
    block_size = T["accel_smooth_window"]

    # Speed delta per second (not per 0.1s)
    # Use rolling difference over ~1 second
    df["speed_delta_1s"] = df["speed_smooth"].diff(block_size) / df["time_delta"].rolling(block_size).sum()
    df["speed_delta_1s"] = df["speed_delta_1s"].fillna(0)

    # Also compute point accel for sustained event detection
    df["speed_delta"] = df["speed_smooth"].diff() / df["time_delta"]
    df["speed_delta"] = df["speed_delta"].fillna(0)

    # Throttle and RPM deltas
    df["throttle_delta"] = df["throttle"].diff().abs() / df["time_delta"]
    df["throttle_delta"] = df["throttle_delta"].clip(0, 100)
    df["rpm_delta"] = df["rpm"].diff().abs() / df["time_delta"]

    # Acceleration in m/s² (from 1-second block)
    df["accel_ms2"] = df["speed_delta_1s"] / T["ms2_to_kmh_per_s"]

    # Also raw accel for reference
    df["accel_raw_ms2"] = df["speed_delta"] / T["ms2_to_kmh_per_s"]

    # Jerk from block-averaged accel
    df["jerk_ms3"] = df["accel_ms2"].diff() / df["time_delta"].rolling(3).mean()
    df["jerk_ms3"] = df["jerk_ms3"].fillna(0)

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

        throttle_delta_p95 = group["throttle_delta"].quantile(0.95)
        if pd.isna(throttle_delta_p95) or throttle_delta_p95 <= 0:
            throttle_delta_p95 = 10.0

        profiles[file_name] = {
            "p95_speed": moving["speed"].quantile(0.95),
            "p99_speed": moving["speed"].quantile(0.99),
            "p95_rpm": moving["rpm"].quantile(0.95),
            "p99_rpm": moving["rpm"].quantile(0.99),
            "p95_load": moving["load"].quantile(0.95),
            "p95_throttle_delta": throttle_delta_p95,
            "max_throttle_observed": min(group["throttle_delta"].max() or 50.0, 100.0),
            "median_speed": moving["speed"].median(),
            "median_rpm": moving["rpm"].median(),
        }
    print(f"[profile] Computed profiles for {len(profiles)} files")
    return profiles


# -------------------------------------------------------------
# SECTION 5 — SUSTAINED EVENT DETECTION (v2.4)
# -------------------------------------------------------------

def count_sustained_events(series: pd.Series, threshold: float, 
                           min_consecutive: int = 3, above: bool = True) -> int:
    """
    v2.4: Count sustained events (3+ consecutive points above/below threshold).
    This filters out single-point noise spikes.
    """
    if above:
        mask = series > threshold
    else:
        mask = series < threshold

    if not mask.any():
        return 0

    # Find consecutive runs
    runs = []
    current_run = 0
    for val in mask:
        if val:
            current_run += 1
        else:
            if current_run >= min_consecutive:
                runs.append(current_run)
            current_run = 0
    if current_run >= min_consecutive:
        runs.append(current_run)

    return len(runs)


# -------------------------------------------------------------
# SECTION 6 — WINDOW FEATURES
# -------------------------------------------------------------

def compute_window_features(window: pd.DataFrame, car_profile: dict, 
                            window_duration_sec: float = 30.0) -> dict:
    T = THRESHOLDS
    cp = car_profile
    feat = {}

    total_rows = len(window)
    stopped_rows = (window["speed"] == 0).sum() if "speed" in window.columns else 0
    feat["idle_time_ratio"] = stopped_rows / total_rows if total_rows > 0 else 0

    # Speed features
    if "speed" in window.columns and window["speed"].notna().sum() > 2:
        feat["mean_speed"] = window["speed"].mean()
        feat["max_speed"] = window["speed"].max()
        feat["speed_std"] = window["speed"].std()

        if cp["p95_speed"] > 0:
            feat["speed_utilization"] = feat["mean_speed"] / cp["p95_speed"]
            feat["max_speed_utilization"] = feat["max_speed"] / cp["p95_speed"]
            feat["speed_cv"] = feat["speed_std"] / feat["mean_speed"] if feat["mean_speed"] > 0 else 0
        else:
            feat["speed_utilization"] = 0
            feat["max_speed_utilization"] = 0
            feat["speed_cv"] = 0

        # v2.4: Sustained event detection on properly smoothed accel
        feat["hard_brake_cnt"] = count_sustained_events(
            window["accel_ms2"], T["hard_brake_ms2"], 
            T["sustained_event_min_pts"], above=False
        )
        feat["rapid_accel_cnt"] = count_sustained_events(
            window["accel_ms2"], T["rapid_accel_ms2"], 
            T["sustained_event_min_pts"], above=True
        )
        feat["extreme_brake_cnt"] = count_sustained_events(
            window["accel_ms2"], T["hard_brake_extreme_ms2"], 
            T["sustained_event_min_pts"], above=False
        )
        feat["extreme_accel_cnt"] = count_sustained_events(
            window["accel_ms2"], T["rapid_accel_extreme_ms2"], 
            T["sustained_event_min_pts"], above=True
        )

        # Also track max/min for reference
        feat["min_accel_ms2"] = window["accel_ms2"].min()
        feat["max_accel_ms2"] = window["accel_ms2"].max()
    else:
        feat.update({
            "mean_speed": np.nan, "max_speed": np.nan, "speed_std": np.nan,
            "speed_utilization": 0, "max_speed_utilization": 0, "speed_cv": 0,
            "hard_brake_cnt": 0, "rapid_accel_cnt": 0,
            "extreme_brake_cnt": 0, "extreme_accel_cnt": 0,
            "min_accel_ms2": 0, "max_accel_ms2": 0,
        })

    # RPM features
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

    # Throttle features
    if "throttle" in window.columns and window["throttle"].notna().sum() > 2:
        feat["mean_throttle"] = window["throttle"].mean()
        feat["throttle_std"] = window["throttle"].std()
        feat["max_throttle_delta"] = window["throttle_delta"].max()
        throttle_ref = cp.get("p95_throttle_delta", cp.get("max_throttle_observed", 50.0))
        if throttle_ref > 0:
            feat["throttle_utilization"] = feat["max_throttle_delta"] / throttle_ref
            feat["throttle_utilization"] = min(feat["throttle_utilization"], 1.0)
        else:
            feat["throttle_utilization"] = 0
    else:
        feat.update({
            "mean_throttle": np.nan, "throttle_std": np.nan,
            "max_throttle_delta": 0, "throttle_utilization": 0
        })

    # Load features
    if "load" in window.columns and window["load"].notna().sum() > 2:
        feat["mean_load"] = window["load"].mean()
        if cp["p95_load"] > 0:
            feat["load_utilization"] = feat["mean_load"] / cp["p95_load"]
        else:
            feat["load_utilization"] = 0
    else:
        feat.update({"mean_load": np.nan, "load_utilization": 0})

    # Jerk features (on smoothed data)
    if "jerk_ms3" in window.columns and window["jerk_ms3"].notna().sum() > 2:
        feat["mean_jerk"] = window["jerk_ms3"].mean()
        feat["max_jerk"] = window["jerk_ms3"].max()
        feat["jerk_std"] = window["jerk_ms3"].std()
        feat["high_jerk_ratio"] = (window["jerk_ms3"].abs() > T["jerk_threshold_ms2"]).mean()
    else:
        feat.update({
            "mean_jerk": 0, "max_jerk": 0, "jerk_std": 0, "high_jerk_ratio": 0
        })

    # Composite intensity
    feat["intensity_score"] = (
        feat.get("speed_utilization", 0) * 0.25 +
        feat.get("rpm_utilization", 0) * 0.25 +
        feat.get("load_utilization", 0) * 0.20 +
        feat.get("throttle_utilization", 0) * 0.20 +
        min(feat.get("high_jerk_ratio", 0) * 2, 1.0) * 0.10
    )

    return feat


# -------------------------------------------------------------
# SECTION 7 — AGGRESSION SCORING (v2.4: SIMPLIFIED)
# -------------------------------------------------------------

def compute_aggression_score(feat: dict) -> float:
    """
    v2.4: Simplified scoring. Back to weighted sum but with:
    - Sustained event counts (not single-point)
    - Properly smoothed features
    - Calibrated thresholds
    """
    T = THRESHOLDS
    score = 0.0

    # Hard braking (sustained events)
    hb = feat.get("hard_brake_cnt", 0)
    eb = feat.get("extreme_brake_cnt", 0)
    if eb >= 1:
        score += T["w_hard_brake"] + T["w_extreme_events"]
    elif hb >= T["hard_brake_events_aggressive"]:
        score += T["w_hard_brake"]
    elif hb >= T["hard_brake_events_normal"]:
        score += T["w_hard_brake"] * 0.4

    # Rapid acceleration (sustained events)
    ra = feat.get("rapid_accel_cnt", 0)
    ea = feat.get("extreme_accel_cnt", 0)
    if ea >= 1:
        score += T["w_rapid_accel"] + T["w_extreme_events"]
    elif ra >= T["rapid_accel_events_aggressive"]:
        score += T["w_rapid_accel"]
    elif ra >= T["rapid_accel_events_normal"]:
        score += T["w_rapid_accel"] * 0.4

    # High RPM
    rpm_util = feat.get("rpm_utilization", 0)
    if rpm_util > T["rpm_util_aggressive"]:
        score += T["w_high_rpm_util"]
    elif rpm_util > T["rpm_util_normal"]:
        score += T["w_high_rpm_util"] * 0.3

    # Throttle
    throttle_util = feat.get("throttle_utilization", 0)
    if throttle_util > T["throttle_util_aggressive"]:
        score += T["w_high_throttle_util"]
    elif throttle_util > T["throttle_util_normal"]:
        score += T["w_high_throttle_util"] * 0.3

    # Load
    load_util = feat.get("load_utilization", 0)
    if load_util > T["load_util_aggressive"]:
        score += T["w_high_load_util"]
    elif load_util > T["load_util_normal"]:
        score += T["w_high_load_util"] * 0.4

    # Jerk
    high_jerk = feat.get("high_jerk_ratio", 0)
    if high_jerk > T["jerk_sustained_ratio"]:
        score += T["w_high_jerk"]
    elif high_jerk > 0.05:
        score += T["w_high_jerk"] * 0.4

    return round(score, 3)


def assign_label(feat: dict) -> str:
    T = THRESHOLDS
    score = compute_aggression_score(feat)

    if score >= T["score_aggressive"]:
        return "AGGRESSIVE"

    # SLOW logic
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
    if not np.isnan(speed_cv) and speed_cv <= T["slow_max_cv_speed"]:
        calmness_indicators += 1

    if score <= T["score_slow"] and calmness_indicators >= 2:
        return "SLOW"

    return "NORMAL"


# -------------------------------------------------------------
# SECTION 8 — LABELING & CORRECTION
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

            window_duration_sec = window["time_delta"].sum() if "time_delta" in window.columns else window_seconds
            feat = compute_window_features(window, profile, window_duration_sec)
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
# SECTION 9 — PLOTS
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

    fig.suptitle("M1 v2.4 — Noise-Resistant Label Engineering", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "label_distribution.png", dpi=150)
    plt.close(fig)
    print(f"[plot] saved -> {OUT}/label_distribution.png")


def plot_feature_boxplots(labeled: pd.DataFrame):
    features = ["speed_utilization", "rpm_utilization", "load_utilization",
                "throttle_utilization", "intensity_score", "idle_time_ratio",
                "hard_brake_cnt", "rapid_accel_cnt", "high_jerk_ratio", "speed_cv",
                "min_accel_ms2", "max_accel_ms2"]
    features = [f for f in features if f in labeled.columns]
    colors = {"SLOW": "#4CAF50", "NORMAL": "#FF9800", "AGGRESSIVE": "#F44336"}

    n = len(features)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(20, 12))
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

    fig.suptitle("Feature Distributions per Label — M1 v2.4", fontsize=13)
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
    print(f"\n[validate] Event count statistics:")
    for col in ["hard_brake_cnt", "rapid_accel_cnt", "extreme_brake_cnt", "extreme_accel_cnt"]:
        if col in labeled.columns:
            print(f"  {col}: {labeled[col].describe().to_dict()}")

    print("\n[validate] Tuning guide:")
    print("  SLOW too low  -> raise slow_speed_util_max or lower score_slow")
    print("  SLOW too high -> lower slow_speed_util_max or raise score_slow")
    print("  AGGR too high -> raise score_aggressive or hard_brake_ms2")
    print("  AGGR too low  -> lower score_aggressive or hard_brake_ms2")


# -------------------------------------------------------------
# SECTION 10 — EXPORT
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
    print("M1 v2.4 — Noise-Resistant Label Engineering Pipeline")
    print("FIXES: Robust 1s-block acceleration | Sustained event detection |")
    print("       Simplified scoring | Properly calibrated thresholds")
    print(f"Data directory: {data_dir}")
    print("=" * 70)

    raw = load_ds3(data_dir)

    print("\n-- Extracting OBD columns ------------------------------")
    obd = extract_obd_columns(raw)
    found = {k: v for k, v in {c: find_column(raw, c) for c in COL_MAP}.items() if v}
    print(f"Found columns: {found}")

    obd = clean_obd(obd)
    obd = compute_robust_deltas(obd)
    print(f"[preprocess] clean rows: {len(obd):,}")
    print(f"[preprocess] time delta stats: mean={obd['time_delta'].mean():.2f}s, "
          f"median={obd['time_delta'].median():.2f}s")
    print(f"[preprocess] accel stats: mean={obd['accel_ms2'].mean():.2f}, "
          f"std={obd['accel_ms2'].std():.2f}, "
          f"min={obd['accel_ms2'].min():.2f}, max={obd['accel_ms2'].max():.2f}")

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
    print(f"Done. Noise-resistant labeled dataset ready for M1 training.")
    print(f"Next: run m1_training.py with {OUT}/ds3_labeled_windows.csv")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
