#!/usr/bin/env python3
"""
Convert TRex tracking output (Marie's videos) into the same bee_track CSV
schema used for the manually-tracked (app.py) sessions under results/.

Pipeline per TRex session folder:
  data/<name>_id0.csv  -->  bee_track_<name>.csv
                            zone_transitions_<name>.csv
                            help_points_<name>.csv   (empty - no manual points)
                            trial_outcome.txt

Arena calibration (confirmed with user):
  - Fixed pixel arena center across all of Marie's videos (single camera rig):
    measured via ring-fit on 4 sample sessions -> (599.5, 368.3) px.
  - Same physical arena size as the R13 dataset: Inner = 210 mm, Outer = 420 mm.
  - Feeder sits at the arena center (0, 0) in mm, feeder radius = 40 mm.
  - cm_per_pixel = 0.0773 (constant across all Marie's TRex .settings files).

Unresolved for now (per user instructions):
  - bee_id / orientation / P / U / calibration-lookup fields -> "unknown"
    (to be filled in later once Marie's metadata is matched up).
  - trial_outcome -> "Unknown" (user will fill in manually per video).
  - Posture/kinematics columns from TRex (speed, angle, acceleration, etc.)
    are carried through as extra columns for later discussion, not dropped.
  - Frames TRex marked "missing" (no detection) are forward-filled so the
    zone/transition logic stays continuous.
"""
import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

INNER_RADIUS_MM = 210.0
OUTER_RADIUS_MM = 420.0
FEEDER_RADIUS_MM = 40.0
CM_PER_PIXEL_DEFAULT = 0.0773
ARENA_CENTER_PX = (599.5, 368.3)  # fixed across Marie's sessions (measured)

# TRex posture/kinematics columns to carry through untouched for later use.
PASSTHROUGH_COLUMNS = {
    "ACCELERATION#pcentroid (cm/s2)": "acceleration_pcentroid_cm_s2",
    "ACCELERATION#wcentroid (cm/s2)": "acceleration_wcentroid_cm_s2",
    "ANGLE": "angle_rad",
    "ANGULAR_A#centroid": "angular_acceleration_centroid",
    "ANGULAR_V#centroid": "angular_velocity_centroid",
    "SPEED#wcentroid (cm/s)": "speed_wcentroid_cm_s",
    "SPEED#pcentroid (cm/s)": "speed_pcentroid_cm_s",
    "SPEED (cm/s)": "speed_cm_s",
    "VX (cm/s)": "vx_cm_s",
    "VY (cm/s)": "vy_cm_s",
    "midline_length": "midline_length",
    "midline_segment_length": "midline_segment_length",
    "num_pixels": "num_pixels",
    "missing": "trex_missing",
    "visual_identification_p": "visual_identification_p",
}


def parse_cm_per_pixel(settings_path: Path) -> float:
    if not settings_path.exists():
        return CM_PER_PIXEL_DEFAULT
    for line in settings_path.read_text().splitlines():
        if line.strip().startswith("cm_per_pixel"):
            try:
                return float(line.split("=", 1)[1].strip())
            except ValueError:
                pass
    return CM_PER_PIXEL_DEFAULT


def build_bee_track_df(id0_csv: Path, cm_per_pixel: float) -> pd.DataFrame:
    raw = pd.read_csv(id0_csv)
    cx_px, cy_px = ARENA_CENTER_PX

    x_px = raw["X (cm)"] / cm_per_pixel
    y_px = raw["Y (cm)"] / cm_per_pixel

    x_px = x_px.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    y_px = y_px.replace([np.inf, -np.inf], np.nan).ffill().bfill()

    x_mm = (x_px - cx_px) * cm_per_pixel * 10.0
    y_mm = -(y_px - cy_px) * cm_per_pixel * 10.0
    dist_mm = np.hypot(x_mm, y_mm)

    zone = np.where(dist_mm <= INNER_RADIUS_MM, "Inner",
            np.where(dist_mm <= OUTER_RADIUS_MM, "Outer", "Exited"))

    df = pd.DataFrame({
        "frame": raw["frame"].astype(int),
        "time_sec": raw["time"],
        "x_pixel": x_px,
        "y_pixel": y_px,
        "x_mm": x_mm,
        "y_mm": y_mm,
        "distance_from_center_mm": dist_mm,
        "current_zone": zone,
        "in_arena": zone != "Exited",
        "on_feeder": dist_mm <= FEEDER_RADIUS_MM,
    })

    prev_zone = df["current_zone"].shift(1)
    changed = df["current_zone"] != prev_zone
    changed.iloc[0] = False
    df["transition_event"] = np.where(changed, prev_zone + " -> " + df["current_zone"], "")

    bee_id = parse_bee_id(id0_csv.parent.parent.name) if id0_csv else "Unknown"
    x_last = df["x_mm"].iloc[-1] if len(df) > 0 else 0.0
    y_last = df["y_mm"].iloc[-1] if len(df) > 0 else 0.0
    outcome = determine_outcome(x_last, y_last)

    df["tag_type"] = "auto"
    df["bee_id"] = bee_id
    df["trial_outcome"] = outcome
    df["orientation"] = "unknown"
    df["x"] = "unknown"
    df["xdop"] = "unknown"
    df["xhdr"] = "unknown"
    df["hdr"] = "unknown"
    df["intensity"] = "unknown"
    df["edge_intensity"] = "unknown"

    for src, dst in PASSTHROUGH_COLUMNS.items():
        if src in raw.columns:
            df[dst] = raw[src]

    return df


def parse_bee_id(folder_name: str) -> str:
    m = re.search(r"\b(\d+[wWgG])\b", folder_name)
    if m:
        return m.group(1).upper()
    m2 = re.search(r"(\d+[wWgG])", folder_name)
    if m2:
        return m2.group(1).upper()
    return "Unknown"


def determine_outcome(x_last: float, y_last: float, hive_pos=(0.0, -450.0)) -> str:
    hx, hy = hive_pos
    dist_to_hive = np.hypot(x_last - hx, y_last - hy)
    if dist_to_hive < 280.0 or (y_last < -150.0 and abs(x_last) < 200.0):
        return "Returned to Hive"
    else:
        return "Still in Arena"


def write_outputs(df: pd.DataFrame, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    bee_id = parse_bee_id(name)
    df["bee_id"] = bee_id

    x_last = df["x_mm"].iloc[-1] if len(df) > 0 else 0.0
    y_last = df["y_mm"].iloc[-1] if len(df) > 0 else 0.0
    outcome = determine_outcome(x_last, y_last)
    df["trial_outcome"] = outcome

    bee_track_path = out_dir / f"bee_track_{name}.csv"
    df.to_csv(bee_track_path, index=False)

    trans = df.loc[df["transition_event"] != "", ["frame", "time_sec", "transition_event"]]
    trans.to_csv(out_dir / f"zone_transitions_{name}.csv", index=False)

    # No manual "help" corrections exist for TRex-tracked sessions.
    pd.DataFrame(columns=["frame", "time_sec", "x_mm", "y_mm"]).to_csv(
        out_dir / f"help_points_{name}.csv", index=False
    )

    (out_dir / "trial_outcome.txt").write_text(
        f"Bee ID: {bee_id}\nOutcome: {outcome}\n"
    )

    print(f"[{name}] wrote {bee_track_path.name} ({len(df)} rows), "
          f"{len(trans)} zone transitions, Bee ID: {bee_id}, Outcome: {outcome} -> {out_dir}")


def convert_session(session_dir: Path, results_root: Path):
    name = session_dir.name
    id0_candidates = sorted((session_dir / "data").glob("*_id0.csv"))
    if not id0_candidates:
        print(f"[{name}] SKIP - no data/*_id0.csv found")
        return
    id0_csv = id0_candidates[0]

    settings_candidates = sorted(session_dir.glob("*.settings"))
    cm_per_pixel = parse_cm_per_pixel(settings_candidates[0]) if settings_candidates else CM_PER_PIXEL_DEFAULT

    df = build_bee_track_df(id0_csv, cm_per_pixel)
    out_dir = results_root / name
    write_outputs(df, out_dir, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="+", help="Session folder name(s) under results/maries/output/")
    ap.add_argument("--maries-root",
                    default="/Users/yakkshit/Downloads/project/hiwi2/p1/Videos/BBP2025/working/results/maries/output")
    ap.add_argument("--results-root",
                    default="/Users/yakkshit/Downloads/project/hiwi2/p1/Videos/BBP2025/working/results/m")
    args = ap.parse_args()

    maries_root = Path(args.maries_root)
    results_root = Path(args.results_root)

    for session_name in args.sessions:
        session_dir = maries_root / session_name
        if not session_dir.exists():
            print(f"[{session_name}] SKIP - folder not found")
            continue
        convert_session(session_dir, results_root)


if __name__ == "__main__":
    main()
