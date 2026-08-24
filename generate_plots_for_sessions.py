#!/usr/bin/env python3
"""
Run the full analysis.sh plot suite (the per-session generators only, not the
cross-dataset paper_plots/notebook export) against a specific list of session
folders under results/, instead of rescanning every session in the dataset.

Usage:
  python generate_plots_for_sessions.py "<session_name_1>" "<session_name_2>" ...
"""
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis import generate_analysis_plots as p1
from analysis import generate_feeder_to_inner_circle_plot as p2
from analysis import generate_feeder_to_final_exit_plot as p3
from analysis import generate_single_colour_plot as p4
from analysis import generate_gradient_coloured_plot as p5
from analysis import generate_60fps_gradient_plots as p6
from analysis import generate_smooth_path_gradient_plots as p7
from analysis import generate_multiple_entries_plot as p8
from analysis import generate_fixed_clean_trajectory_plot as p9

STEPS = [
    ("complete_trajectory / approach_and_home_search", p1.generate_figure),
    ("feeder_to_inner_circle", p2.generate_feeder_to_inner_plot),
    ("feeder_to_final_exit", p3.generate_feeder_to_final_exit_plot),
    ("single_colour", p4.generate_single_colour),
    ("gradient_exit_only / gradient_full_trial", p5.generate_gradient_coloured_plot),
    ("60fps gradient plots", p6.generate_60fps_gradient_plots),
    ("smooth path gradient plots", p7.generate_smooth_path_gradient_plots),
    ("multiple_entries_exits", p8.generate_multiple_entries_plot),
    ("fixed clean trajectory plots", p9.generate_fixed_clean_plots),
]


def find_bee_track_csv(session_path):
    for f in os.listdir(session_path):
        if f.startswith("bee_track_") and f.endswith(".csv"):
            return os.path.join(session_path, f)
    return None


def main():
    session_names = sys.argv[1:]
    if not session_names:
        print("Usage: python generate_plots_for_sessions.py <session_name> [more session names...]")
        sys.exit(1)

    results_dir = "results/m"
    for session_name in session_names:
        session_path = os.path.join(results_dir, session_name)
        if not os.path.isdir(session_path):
            print(f"[{session_name}] SKIP - folder not found")
            continue
        track_csv = find_bee_track_csv(session_path)
        if not track_csv:
            print(f"[{session_name}] SKIP - no bee_track_*.csv found")
            continue

        print(f"=== {session_name} ===")
        df = pd.read_csv(track_csv)
        for label, fn in STEPS:
            try:
                fn(df, session_path)
                print(f"  ok  - {label}")
            except Exception as e:
                print(f"  ERR - {label}: {e}")


if __name__ == "__main__":
    main()
