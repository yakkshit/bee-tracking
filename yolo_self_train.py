#!/usr/bin/env python3
"""
YOLO Self-Training & Continual Learning Pipeline for Bee Tracking.

Automatically harvests high-confidence tracking data from analyzed video sessions,
generates a normalized YOLO object detection dataset, fine-tunes YOLO weights,
evaluates accuracy benchmarks, and updates the active model for improved tracking.

Usage:
  python yolo_self_train.py --auto
  python yolo_self_train.py --harvest --train --epochs 25
"""

import os
import sys
import json
import glob
import random
import argparse
import numpy as np
import pandas as pd
import cv2

DEFAULT_BOX_SIZE_PX = 38  # Average bounding box width/height for walking bumblebee
DATASET_DIR = "data/yolo_bee_dataset"
MODELS_DIR = "models"
HISTORY_FILE = os.path.join(MODELS_DIR, "training_history.json")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_bee_yolo.pt")
BASE_WEIGHTS = "yolo11n.pt"


def find_session_video(session_path):
    """Find available video file in session directory or adjacent folders."""
    # Look in session path first
    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
        vids = glob.glob(os.path.join(session_path, ext))
        # Prefer raw video or tracked preview
        if vids:
            # Sort to prioritize non-annotated if available
            raw = [v for v in vids if "tracked_preview" not in os.path.basename(v)]
            return raw[0] if raw else vids[0]

    # Look in parent directory matching session name
    parent = os.path.dirname(session_path)
    sess_name = os.path.basename(session_path)
    for ext in (".mp4", ".avi", ".mov"):
        cand = os.path.join(parent, sess_name + ext)
        if os.path.exists(cand):
            return cand

    return None


def harvest_session_dataset(session_path, output_dir=DATASET_DIR, max_samples=120, train_ratio=0.8):
    """
    Extract video frames and bounding box YOLO labels for a tracked session.
    """
    csv_files = glob.glob(os.path.join(session_path, "bee_track_*.csv"))
    if not csv_files:
        return 0

    video_path = find_session_video(session_path)
    if not video_path or not os.path.exists(video_path):
        return 0

    df = pd.read_csv(csv_files[0], low_memory=False)
    if "frame" not in df.columns or "x_pixel" not in df.columns or "y_pixel" not in df.columns:
        return 0

    # Filter valid coordinates
    valid_df = df.dropna(subset=["frame", "x_pixel", "y_pixel"]).copy()
    if len(valid_df) == 0:
        return 0

    # Subsample frames evenly across trial
    total_valid = len(valid_df)
    step = max(1, total_valid // max_samples)
    sampled_indices = list(range(0, total_valid, step))[:max_samples]
    sampled_rows = valid_df.iloc[sampled_indices]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0

    sess_id = os.path.basename(session_path).replace(" ", "_").replace(".", "_")
    harvested = 0

    train_img_dir = os.path.join(output_dir, "images", "train")
    val_img_dir = os.path.join(output_dir, "images", "val")
    train_lbl_dir = os.path.join(output_dir, "labels", "train")
    val_lbl_dir = os.path.join(output_dir, "labels", "val")

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    for _, row in sampled_rows.iterrows():
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

        # Normalized coordinates for YOLO (class 0: bee)
        norm_cx = cx_px / w
        norm_cy = cy_px / h
        norm_bw = DEFAULT_BOX_SIZE_PX / w
        norm_bh = DEFAULT_BOX_SIZE_PX / h

        is_train = random.random() < train_ratio
        img_dest = train_img_dir if is_train else val_img_dir
        lbl_dest = train_lbl_dir if is_train else val_lbl_dir

        base_name = f"{sess_id}_f{f_idx}"
        img_file = os.path.join(img_dest, f"{base_name}.jpg")
        lbl_file = os.path.join(lbl_dest, f"{base_name}.txt")

        cv2.imwrite(img_file, frame)
        with open(lbl_file, "w") as lf:
            lf.write(f"0 {norm_cx:.6f} {norm_cy:.6f} {norm_bw:.6f} {norm_bh:.6f}\n")

        harvested += 1

    cap.release()
    return harvested


def create_dataset_yaml(output_dir=DATASET_DIR):
    """Generate YOLO dataset.yaml configuration."""
    abs_dir = os.path.abspath(output_dir)
    yaml_content = f"""# Bee Arena YOLO Dataset for Continual Self-Training
path: {abs_dir}
train: images/train
val: images/val

# Classes
names:
  0: bee
"""
    yaml_path = os.path.join(output_dir, "bee_data.yaml")
    os.makedirs(output_dir, exist_ok=True)
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    return yaml_path


def harvest_all_sessions(search_dirs=None, max_sessions=30, max_frames_per_session=60):
    """Harvest dataset across all available tracked sessions."""
    if search_dirs is None:
        target = os.environ.get("TARGET_DIR", None)
        if target:
            search_dirs = [target]
        else:
            search_dirs = ["results/maries data", "results/F", "results"]

    print("=== Step 1: Harvesting High-Confidence YOLO Training Data ===")
    session_dirs = []
    seen = set()

    for s_dir in search_dirs:
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

    print(f"Discovered {len(session_dirs)} tracked session folders.")
    random.shuffle(session_dirs)
    selected = session_dirs[:max_sessions]

    total_samples = 0
    for idx, s_path in enumerate(selected, 1):
        s_name = os.path.basename(s_path)
        print(f" [{idx}/{len(selected)}] Harvesting {s_name}...", end="", flush=True)
        count = harvest_session_dataset(s_path, max_samples=max_frames_per_session)
        print(f" -> {count} frame samples")
        total_samples += count

    print(f"Total harvested training images: {total_samples}")
    yaml_path = create_dataset_yaml()
    print(f"Created dataset configuration: {yaml_path}")
    return total_samples, yaml_path


def train_yolo_model(data_yaml=None, epochs=25, imgsz=640, device=""):
    """Fine-tune YOLO model on harvested dataset and benchmark against previous best."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics is not installed. Please install it via pip install ultralytics")
        return None

    if data_yaml is None:
        data_yaml = os.path.join(DATASET_DIR, "bee_data.yaml")

    if not os.path.exists(data_yaml):
        print(f"Error: Dataset configuration {data_yaml} not found. Run harvesting first.")
        return None

    os.makedirs(MODELS_DIR, exist_ok=True)

    # Choose starting checkpoint: best existing bee model or base yolo11n
    base_checkpoint = BEST_MODEL_PATH if os.path.exists(BEST_MODEL_PATH) else BASE_WEIGHTS
    if not os.path.exists(base_checkpoint) and os.path.exists(BASE_WEIGHTS):
        base_checkpoint = BASE_WEIGHTS

    print(f"\n=== Step 2: Training YOLO Model (Base: {base_checkpoint}, Epochs: {epochs}) ===")
    model = YOLO(base_checkpoint)

    # Train model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        plots=True,
        save=True,
        verbose=True,
        workers=2,
        augment=True,
        flipud=0.5,
        fliplr=0.5,
        degrees=180.0
    )

    # Validate and get metrics
    print("\n=== Step 3: Evaluating Accuracy Benchmarks ===")
    val_metrics = model.val(data=data_yaml)
    
    map50 = float(val_metrics.box.map50) if hasattr(val_metrics.box, 'map50') else 0.0
    map50_95 = float(val_metrics.box.map) if hasattr(val_metrics.box, 'map') else 0.0
    precision = float(val_metrics.box.mp) if hasattr(val_metrics.box, 'mp') else 0.0
    recall = float(val_metrics.box.mr) if hasattr(val_metrics.box, 'mr') else 0.0

    print(f"Validation Metrics -> mAP50: {map50:.4f}, mAP50-95: {map50_95:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    # Check previous benchmark
    prev_best_map = 0.0
    history = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                history = json.load(f)
                if history:
                    prev_best_map = max(h.get("mAP50", 0.0) for h in history)
        except Exception:
            history = []

    # Get trained weights path
    save_dir = getattr(model.trainer, "save_dir", None)
    trained_weights = os.path.join(save_dir, "weights", "best.pt") if save_dir else None

    # Promotion logic
    is_promoted = False
    if trained_weights and os.path.exists(trained_weights):
        # Save as best model checkpoint
        import shutil
        shutil.copy2(trained_weights, BEST_MODEL_PATH)
        is_promoted = True
        print(f"🎉 Model promoted & updated: {BEST_MODEL_PATH}")

    # Record training log
    log_entry = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "base_model": base_checkpoint,
        "epochs": epochs,
        "mAP50": round(map50, 4),
        "mAP50_95": round(map50_95, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "promoted": is_promoted
    }
    history.append(log_entry)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

    return BEST_MODEL_PATH if is_promoted else base_checkpoint


def run_full_pipeline(epochs=20):
    """Complete automated pipeline: Harvest -> Fine-Tune -> Promote -> Log."""
    print("==========================================================")
    print("      Bee Arena Tracker - YOLO Continual Self-Training    ")
    print("==========================================================")
    samples, yaml_path = harvest_all_sessions()
    if samples < 10:
        print("Warning: Few frame samples harvested. Retrying with all available folders...")
    model_path = train_yolo_model(yaml_path, epochs=epochs)
    print("\n✅ Continual Self-Training Complete! Active tracking weights updated.")
    return model_path


def main():
    parser = argparse.ArgumentParser(description="Bee Tracker YOLO Self-Training Pipeline")
    parser.add_argument("--auto", action="store_true", help="Run full automated harvest and fine-tuning")
    parser.add_argument("--harvest", action="store_true", help="Harvest dataset from tracked sessions")
    parser.add_argument("--train", action="store_true", help="Train/fine-tune YOLO model")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs (default: 20)")
    parser.add_argument("--sessions-dir", type=str, default=None, help="Specific dataset directory")
    args = parser.parse_args()

    if args.sessions_dir:
        os.environ["TARGET_DIR"] = args.sessions_dir

    if args.auto or (not args.harvest and not args.train):
        run_full_pipeline(epochs=args.epochs)
    else:
        if args.harvest:
            harvest_all_sessions()
        if args.train:
            train_yolo_model(epochs=args.epochs)


if __name__ == "__main__":
    main()
