"""
Per-session direction + posture summary for the TRex-tracked Maries videos.

Direction fields follow the same convention as the existing
results/analysis_summary_dataset.csv (bearings in degrees, 0-360,
atan2(y_mm, x_mm) measured counter-clockwise from +x, at the frame of the
first/last zone-crossing event). Posture fields are aggregated from the
TRex kinematics columns that were carried through into each bee_track_*.csv
(speed_cm_s, angle_rad, angular_velocity_centroid, ...).

Output: results/maries_direction_posture_summary.csv (one row per session)
"""
import os
import numpy as np
import pandas as pd

RESULTS_DIR = "results"
SESSION_LIST_FILE = "/tmp/maries_sessions.txt"
OUTPUT_CSV = os.path.join(RESULTS_DIR, "maries_direction_posture_summary.csv")


def bearing_deg(x_mm, y_mm):
    return float(np.degrees(np.arctan2(y_mm, x_mm)) % 360.0)


def circular_mean_deg(angles_rad):
    angles_rad = angles_rad[np.isfinite(angles_rad)]
    if len(angles_rad) == 0:
        return np.nan
    return float(np.degrees(np.arctan2(np.mean(np.sin(angles_rad)), np.mean(np.cos(angles_rad)))) % 360.0)


def first_event_row(df, event):
    hits = df.index[df["transition_event"] == event]
    return df.loc[hits[0]] if len(hits) else None


def last_event_row(df, event):
    hits = df.index[df["transition_event"] == event]
    return df.loc[hits[-1]] if len(hits) else None


def read_trial_outcome(session_dir):
    path = os.path.join(session_dir, "trial_outcome.txt")
    bee_id, outcome = "unknown", "Unknown"
    if os.path.exists(path):
        for line in open(path):
            if line.startswith("Bee ID:"):
                bee_id = line.split(":", 1)[1].strip()
            elif line.startswith("Outcome:"):
                outcome = line.split(":", 1)[1].strip()
    return bee_id, outcome


def find_session_path(session_name, search_dirs=None):
    if search_dirs is None:
        search_dirs = [
            "results/maries data",
            "results/F",
            "results/maries/output",
            "results/m",
            "results"
        ]
    for s_dir in search_dirs:
        candidate = os.path.join(s_dir, session_name)
        if os.path.isdir(candidate):
            return candidate
    return None


def summarize_session(session_name):
    session_dir = find_session_path(session_name)
    if not session_dir:
        return None
    track_csv = None
    for f in os.listdir(session_dir):
        if f.startswith("bee_track_") and f.endswith(".csv"):
            track_csv = os.path.join(session_dir, f)
            break
    if not track_csv:
        return None

    df = pd.read_csv(track_csv, low_memory=False)
    bee_id, outcome = read_trial_outcome(session_dir)

    row = {
        "session_name": session_name,
        "bee_id": bee_id,
        "orientation": str(df["orientation"].iloc[0]) if "orientation" in df.columns else "unknown",
        "trial_outcome": outcome,
        "n_frames": len(df),
        "duration_sec": float(df["time_sec"].iloc[-1] - df["time_sec"].iloc[0]),
    }

    # --- Direction: bearings (deg) at first/last zone-crossing events ---
    r = first_event_row(df, "Exited -> Outer")
    row["outer_entry_angle_deg"] = bearing_deg(r["x_mm"], r["y_mm"]) if r is not None else np.nan

    r = first_event_row(df, "Outer -> Inner")
    row["inner_entry_angle_deg"] = bearing_deg(r["x_mm"], r["y_mm"]) if r is not None else np.nan
    row["time_to_inner_entry_sec"] = float(r["time_sec"]) if r is not None else np.nan

    r = first_event_row(df, "Inner -> Outer")
    row["inner_exit_angle_deg"] = bearing_deg(r["x_mm"], r["y_mm"]) if r is not None else np.nan
    row["time_to_inner_exit_sec"] = float(r["time_sec"]) if r is not None else np.nan

    r = last_event_row(df, "Outer -> Exited")
    row["outer_exit_angle_deg"] = bearing_deg(r["x_mm"], r["y_mm"]) if r is not None else np.nan

    # Overall travel-direction bearing: displacement vector from first to last point
    row["net_travel_bearing_deg"] = bearing_deg(
        df["x_mm"].iloc[-1] - df["x_mm"].iloc[0],
        df["y_mm"].iloc[-1] - df["y_mm"].iloc[0],
    )

    # --- Posture: aggregate TRex kinematics passthrough columns ---
    def finite(col):
        if col not in df.columns:
            return np.array([])
        v = pd.to_numeric(df[col], errors="coerce").values
        return v[np.isfinite(v)]

    speed = finite("speed_cm_s")
    row["mean_speed_cm_s"] = float(np.mean(speed)) if len(speed) else np.nan
    row["median_speed_cm_s"] = float(np.median(speed)) if len(speed) else np.nan
    row["max_speed_cm_s"] = float(np.max(speed)) if len(speed) else np.nan

    ang_vel = finite("angular_velocity_centroid")
    row["mean_angular_velocity"] = float(np.mean(ang_vel)) if len(ang_vel) else np.nan

    body_angle = finite("angle_rad")
    row["mean_body_angle_deg"] = circular_mean_deg(body_angle)

    # --- Path geometry ---
    dx = np.diff(df["x_mm"].values)
    dy = np.diff(df["y_mm"].values)
    step_dist = np.sqrt(dx**2 + dy**2)
    total_dist_cm = float(np.sum(step_dist)) / 10.0
    row["total_distance_cm"] = total_dist_cm

    net_disp_mm = np.hypot(df["x_mm"].iloc[-1] - df["x_mm"].iloc[0],
                            df["y_mm"].iloc[-1] - df["y_mm"].iloc[0])
    row["straightness"] = float((net_disp_mm / 10.0) / total_dist_cm) if total_dist_cm > 0 else np.nan

    # --- Feeder time & tracking quality ---
    if "time_sec" in df.columns and len(df) > 1:
        median_dt = float(np.median(np.diff(df["time_sec"].values)))
    else:
        median_dt = np.nan
    row["time_on_feeder_sec"] = float(df["on_feeder"].sum() * median_dt) if np.isfinite(median_dt) else np.nan

    if "trex_missing" in df.columns:
        row["pct_frames_detected"] = float(100.0 * (1.0 - df["trex_missing"].mean()))
    else:
        row["pct_frames_detected"] = np.nan

    return row


def main():
    names = []
    if os.path.exists(SESSION_LIST_FILE):
        names = [l.rstrip("\n") for l in open(SESSION_LIST_FILE) if l.strip()]
    
    if not names:
        search_dirs = ["results/maries data", "results/F", "results/maries/output", "results/m", "results"]
        for s_dir in search_dirs:
            if os.path.exists(s_dir):
                for item in sorted(os.listdir(s_dir)):
                    if item not in names and item not in ("paper_plots", "F", "maries data", "maries", "m") and os.path.isdir(os.path.join(s_dir, item)):
                        names.append(item)

    rows = []
    for name in names:
        try:
            row = summarize_session(name)
            if row is not None:
                rows.append(row)
                print(f"[{name}] ok")
            else:
                print(f"[{name}] SKIP - no bee_track csv")
        except Exception as e:
            print(f"[{name}] ERROR: {e}")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote {len(out_df)} rows -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
