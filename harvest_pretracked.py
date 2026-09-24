#!/usr/bin/env python3
"""
Pre-tracked Video Dataset Harvester for Bee Arena Tracker.

Iterates through pre-tracked video sessions, extracts frame images and bounding box
annotations from existing tracking CSVs, and compiles a clean, normalized YOLO dataset (dataset.yaml).

Usage:
  python harvest_pretracked.py
  python harvest_pretracked.py --search-dir results --max-samples 150
"""

import os
import sys

# Windows DLL loading & safety configuration
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

if sys.platform.startswith("win"):
    try:
        import site
        for site_path in site.getsitepackages():
            for lib_folder in ("torch/lib", "scipy.libs", "numpy.libs", "matplotlib.libs"):
                dll_dir = os.path.abspath(os.path.join(site_path, lib_folder))
                if os.path.exists(dll_dir):
                    win_dll = f"\\\\?\\{dll_dir}" if not dll_dir.startswith("\\\\?\\") else dll_dir
                    try:
                        os.add_dll_directory(dll_dir)
                    except Exception:
                        try:
                            os.add_dll_directory(win_dll)
                        except Exception:
                            pass
    except Exception:
        pass

import glob
import random
import argparse
import numpy as np
import pandas as pd
import cv2

DATASET_DIR = os.path.abspath("data/yolo_bee_dataset")
DEFAULT_BOX_SIZE_PX = 38


def find_session_video(session_path):
    """Locates the video file corresponding to a tracked session directory."""
    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
        vids = glob.glob(os.path.join(session_path, ext))
        if vids:
            raw = [v for v in vids if "tracked_preview" not in os.path.basename(v)]
            return raw[0] if raw else vids[0]

    parent = os.path.dirname(session_path)
    sess_name = os.path.basename(session_path)
    for ext in (".mp4", ".avi", ".mov"):
        cand = os.path.join(parent, sess_name + ext)
        if os.path.exists(cand):
            return cand

    return None


def harvest_session(session_path, output_dir=DATASET_DIR, max_samples=100):
    """Harvests frame images and normalized YOLO bounding box labels for a session."""
    csv_files = glob.glob(os.path.join(session_path, "bee_track_*.csv"))
    if not csv_files:
        return 0

    video_path = find_session_video(session_path)
    if not video_path or not os.path.exists(video_path):
        return 0

    try:
        df = pd.read_csv(csv_files[0], low_memory=False)
    except Exception:
        return 0

    if "frame" not in df.columns or "x_pixel" not in df.columns or "y_pixel" not in df.columns:
        return 0

    valid_df = df.dropna(subset=["frame", "x_pixel", "y_pixel"]).copy()
    if len(valid_df) == 0:
        return 0

    total_valid = len(valid_df)
    step = max(1, total_valid // max_samples)
    sampled_indices = list(range(0, total_valid, step))[:max_samples]
    sampled_rows = valid_df.iloc[sampled_indices]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0

    train_img_dir = os.path.join(output_dir, "images", "train")
    val_img_dir = os.path.join(output_dir, "images", "val")
    train_lbl_dir = os.path.join(output_dir, "labels", "train")
    val_lbl_dir = os.path.join(output_dir, "labels", "val")

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    sess_id = os.path.basename(session_path).replace(" ", "_").replace(".", "_")
    harvested = 0

    for idx, row in sampled_rows.iterrows():
        f_idx = int(row["frame"])
        cx_px = float(row["x_pixel"])
        cy_px = float(row["y_pixel"])

        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        h, w = frame.shape[:2]
        if cx_px < 0 or cx_px >= w or cy_px < 0 or cy_px >= h:
            continue

        norm_cx = cx_px / w
        norm_cy = cy_px / h
        norm_w = DEFAULT_BOX_SIZE_PX / w
        norm_h = DEFAULT_BOX_SIZE_PX / h

        split = "val" if (harvested % 5 == 0) else "train"
        img_name = f"{sess_id}_f{f_idx}_{harvested}.jpg"
        lbl_name = f"{sess_id}_f{f_idx}_{harvested}.txt"

        img_out = os.path.join(output_dir, "images", split, img_name)
        lbl_out = os.path.join(output_dir, "labels", split, lbl_name)

        cv2.imwrite(img_out, frame)
        with open(lbl_out, "w") as f:
            f.write(f"0 {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f}\n")

        harvested += 1

    cap.release()
    return harvested


def create_dataset_yaml(output_dir=DATASET_DIR):
    """Generates normalized YOLO dataset.yaml configuration."""
    yaml_path = os.path.join(output_dir, "bee_dataset.yaml")
    yaml_content = f"""# Bee Arena Tracker YOLO Dataset
path: {os.path.abspath(output_dir)}
train: images/train
val: images/val
names:
  0: bee
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="Harvest YOLO dataset from pre-tracked videos")
    parser.add_argument("--search-dirs", nargs="+", default=["results", "calibrations"], help="Directories to scan for session tracking data")
    parser.add_argument("--max-sessions", type=int, default=40, help="Maximum number of sessions to process")
    parser.add_argument("--max-samples", type=int, default=100, help="Maximum frame samples per session")
    args = parser.parse_args()

    print("=================================================")
    print(" 🐝 PRE-TRACKED VIDEO DATASET HARVESTER ")
    print("=================================================")

    session_dirs = []
    seen = set()

    for s_dir in args.search_dirs:
        if not os.path.exists(s_dir):
            continue
        for root, dirs, files in os.walk(s_dir):
            dirs[:] = [d for d in dirs if not d.endswith("paper_plots") and d != "plots"]
            for f in files:
                if f.startswith("bee_track_") and f.endswith(".csv"):
                    if root not in seen:
                        session_dirs.append(root)
                        seen.add(root)
                    break

    print(f"Found {len(session_dirs)} tracked session folders.")
    random.shuffle(session_dirs)
    selected = session_dirs[:args.max_sessions]

    total = 0
    for idx, s_path in enumerate(selected, 1):
        s_name = os.path.basename(s_path)
        print(f" [{idx}/{len(selected)}] Harvesting {s_name}...", end="", flush=True)
        count = harvest_session(s_path, max_samples=args.max_samples)
        print(f" ➔ {count} frame samples harvested")
        total += count

    print("-------------------------------------------------")
    print(f"Total harvested YOLO training frames: {total}")
    yaml_path = create_dataset_yaml()
    print(f"Created dataset configuration: {yaml_path}")
    print("=================================================")


if __name__ == "__main__":
    main()
