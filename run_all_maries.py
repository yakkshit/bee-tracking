#!/usr/bin/env python3
"""
Batch-run the TRex->bee_track conversion and the full plot suite for every
Maries session folder, reading names from /tmp/maries_sessions.txt (one per
line) to avoid shell-quoting issues with spaces/parentheses in folder names.
"""
import sys
import os
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import convert_trex_to_bee_track as conv
from analysis import generate_analysis_plots as p1
from analysis import generate_feeder_to_inner_circle_plot as p2
from analysis import generate_feeder_to_final_exit_plot as p3
from analysis import generate_single_colour_plot as p4
from analysis import generate_gradient_coloured_plot as p5
from analysis import generate_60fps_gradient_plots as p6
from analysis import generate_smooth_path_gradient_plots as p7
from analysis import generate_multiple_entries_plot as p8
from analysis import generate_fixed_clean_trajectory_plot as p9

PLOT_STEPS = [
    p1.generate_figure,
    p2.generate_feeder_to_inner_plot,
    p3.generate_feeder_to_final_exit_plot,
    p4.generate_single_colour,
    p5.generate_gradient_coloured_plot,
    p6.generate_60fps_gradient_plots,
    p7.generate_smooth_path_gradient_plots,
    p8.generate_multiple_entries_plot,
    p9.generate_fixed_clean_plots,
]

MARIES_ROOT = Path("/Users/yakkshit/Downloads/project/hiwi2/p1/Videos/BBP2025/working/results/maries/output")
RESULTS_ROOT = Path("/Users/yakkshit/Downloads/project/hiwi2/p1/Videos/BBP2025/working/results/m")


def main():
    default_list = "scratch/maries_sessions.txt" if os.path.exists("scratch/maries_sessions.txt") else "/tmp/maries_sessions.txt"
    list_path = sys.argv[1] if len(sys.argv) > 1 else default_list
    names = [l.rstrip("\n") for l in open(list_path) if l.strip()]

    ok, skipped, errors = [], [], []

    for name in names:
        session_dir = MARIES_ROOT / name
        if not session_dir.exists():
            print(f"[{name}] SKIP - source folder not found")
            skipped.append(name)
            continue

        try:
            conv.convert_session(session_dir, RESULTS_ROOT)
        except Exception as e:
            print(f"[{name}] CONVERT ERROR: {e}")
            errors.append((name, "convert", str(e)))
            continue

        out_dir = RESULTS_ROOT / name
        track_csvs = list(out_dir.glob("bee_track_*.csv"))
        if not track_csvs:
            print(f"[{name}] SKIP plots - no bee_track csv produced")
            skipped.append(name)
            continue

        df = pd.read_csv(track_csvs[0])
        for fn in PLOT_STEPS:
            try:
                fn(df, str(out_dir))
            except Exception as e:
                errors.append((name, fn.__name__, str(e)))

        # Also mirror plots folder and bee_track.csv to session_dir in results/maries/output
        import shutil
        maries_plots = session_dir / "plots"
        os.makedirs(maries_plots, exist_ok=True)
        if (out_dir / "plots").exists():
            shutil.copytree(out_dir / "plots", maries_plots, dirs_exist_ok=True)
        for csv_f in out_dir.glob("bee_track_*.csv"):
            shutil.copy2(csv_f, session_dir / csv_f.name)

        ok.append(name)
        print(f"[{name}] done ({len(df)} rows)")

    print("\n=== SUMMARY ===")
    print(f"Converted + plotted: {len(ok)}")
    print(f"Skipped: {len(skipped)} -> {skipped}")
    print(f"Errors: {len(errors)}")
    for e in errors:
        print("   ", e)


if __name__ == "__main__":
    main()
