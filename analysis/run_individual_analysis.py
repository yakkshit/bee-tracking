#!/usr/bin/env python3
"""
Run all individual trajectory analysis plots for all sessions found in a given folder.

Usage:
  python analysis/run_individual_analysis.py [path_to_tracking_folder]
"""

import os
import sys
import pandas as pd

# Add repo root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis import generate_analysis_plots as p1
from analysis import generate_feeder_to_inner_circle_plot as p2
from analysis import generate_feeder_to_final_exit_plot as p3
from analysis import generate_single_colour_plot as p4
from analysis import generate_gradient_coloured_plot as p5
from analysis import generate_60fps_gradient_plots as p6
from analysis import generate_smooth_path_gradient_plots as p7
from analysis import generate_multiple_entries_plot as p8
from analysis import generate_fixed_clean_trajectory_plot as p9
from analysis import fixed_version as p10
from analysis import gradient_fixed as p11

PLOTTING_STEPS = [
    ("Complete trajectory (Prompt 3)", p1.generate_figure),
    ("Feeder to inner circle (Prompt 4)", p2.generate_feeder_to_inner_plot),
    ("Feeder to final exit (Prompt 5)", p3.generate_feeder_to_final_exit_plot),
    ("Single colour inbound/outbound (Prompt 6)", p4.generate_single_colour),
    ("Gradient coloured trial (Prompt 7)", p5.generate_gradient_coloured_plot),
    ("60fps gradient plots", p6.generate_60fps_gradient_plots),
    ("Smooth path gradient plots", p7.generate_smooth_path_gradient_plots),
    ("Multiple entries/exits with mean path", p8.generate_multiple_entries_plot),
    ("Clean fixed boundary trajectory", p9.generate_fixed_clean_plots),
]


def find_all_session_dirs(base_path):
    """Find all directories containing a bee_track_*.csv file."""
    session_paths = []
    seen = set()

    if not os.path.exists(base_path):
        return []

    # Check if base_path itself is a session directory
    csvs = [f for f in os.listdir(base_path) if f.startswith("bee_track_") and f.endswith(".csv") and os.path.isfile(os.path.join(base_path, f))]
    if csvs:
        return [base_path]

    for root, dirs, files in os.walk(base_path):
        # Exclude plot output folders from scanning
        dirs[:] = [d for d in dirs if not d.endswith("paper_plots") and d != "plots"]
        for f in files:
            if f.startswith("bee_track_") and f.endswith(".csv"):
                if root not in seen:
                    session_paths.append(root)
                    seen.add(root)
                break

    return sorted(session_paths)


def find_bee_track_csv(session_path):
    for f in os.listdir(session_path):
        if f.startswith("bee_track_") and f.endswith(".csv"):
            return os.path.join(session_path, f)
    return None


def run_analysis_for_sessions(base_path):
    sessions = find_all_session_dirs(base_path)
    if not sessions:
        print(f"No tracking session folders containing 'bee_track_*.csv' found in: {base_path}")
        return

    print(f"\nFound {len(sessions)} tracking session(s) in: {base_path}")
    print("=" * 60)

    success_count = 0
    for idx, session_path in enumerate(sessions, 1):
        session_name = os.path.basename(session_path)
        print(f"\n[{idx}/{len(sessions)}] Processing session: {session_name}")
        
        csv_file = find_bee_track_csv(session_path)
        if not csv_file:
            print("  [SKIP] No bee_track_*.csv found")
            continue

        try:
            df = pd.read_csv(csv_file, low_memory=False)
        except Exception as e:
            print(f"  [ERROR] Failed to read CSV: {e}")
            continue

        # Run primary dataframe-based plots
        for label, fn in PLOTTING_STEPS:
            try:
                fn(df, session_path)
                print(f"  ✓ {label}")
            except Exception as e:
                print(f"  ✗ {label}: {e}")

        # Run specialized gap-bridged / fixed version generators
        try:
            p10.process_single_trial(session_path)
            print("  ✓ Gap-bridged sinusoidal paths (final_trajectories.png)")
        except Exception as e:
            print(f"  ✗ Gap-bridged sinusoidal paths: {e}")

        try:
            p11.process_single_trial(session_path)
            print("  ✓ Gradient fixed trajectory (gradient_fixed_trajectory.png)")
        except Exception as e:
            print(f"  ✗ Gradient fixed trajectory: {e}")

        success_count += 1

    print("\n" + "=" * 60)
    print(f"Individual Analysis Complete: {success_count}/{len(sessions)} sessions processed successfully.")


def main():
    if len(sys.argv) > 1:
        target_path = sys.argv[1]
    else:
        target_path = os.environ.get("TARGET_DIR", os.environ.get("VIDEOS_PATH", "results"))

    run_analysis_for_sessions(target_path)


if __name__ == "__main__":
    main()
