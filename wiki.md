## Bee Tracking & Trajectory Analysis Pipeline: Comprehensive Technical Wiki

**Author / Maintainer:** Yakkshit  
**Project:** Bee Arena Navigation & Trajectory Analysis (BBP2025)  
**Target Audience:** Lab Professor & Research Group  
**Location:** `/Users/yakkshit/Downloads/project/hiwi2/p1/Videos/BBP2025/working/`

---

## 1. Executive Summary & Purpose

This document provides a comprehensive technical overview of the automated trajectory processing, coordinate transformation, custom tracking application, and visualization pipeline developed for the **Bee Arena Navigation Dataset** (`results/` and `temp/`).

The goal of this pipeline is to:

1. Standardize tracking outputs into unified `bee_track_<session>.csv` files using the Streamlit + YOLO self-training tracking engine.
2. Align physical arena geometry and Hive orientation with experimental recordings.
3. Eliminate tracking gaps and straight-line drift artifacts using organic flight-wave smoothing.
4. Calculate exact physical boundary intersections for arena crossings (`Entry (Outer)`, `1st Inner Contact`, `1st Inner Exit`, `Exit (Outer)`).
5. Generate standardized analysis plots per session (including publication-grade 2-color time gradient plots calibrated in seconds).

---

## 2. Dataset Architecture & Pipeline Flow

### Data Input & Output Directories

* **Tracked Datasets**: `results/maries data/<session_folder>/`, `results/F/<session_folder>/`, `temp/<session_folder>/`  
  Contains `bee_track_*.csv`, metadata, and video files.
* **Custom Streamlit/YOLO Tracker App**: `app.py` & `pages/1_Analysis_Viewer.py`  
  Used to calibrate, track, and review videos with continuous YOLO detector assistance.
* **YOLO Self-Training Engine**: `yolo_self_train.py`  
  Harvests high-confidence tracked video frames and continually fine-tunes YOLO weights (`models/best_bee_yolo.pt`) to improve tracking accuracy.
* **Processed Outputs**: All session directories in `results/` contain `bee_track_<session>.csv`, `trial_outcome.txt`, `zone_transitions_*.csv`, and standardized trajectory PNGs in their `plots/` directory.

### Pipeline Execution Workflow

```plaintext
flowchart TD
    A1["Raw Video Files"] --> B1["Streamlit + YOLO Tracker App (app.py)"]
    B1 --> C["Standardized bee_track_<session>.csv"]
    C --> D1["analysis.sh (Individual & Paper Plot Suite)"]
    C --> D2["yolo_self_train.py (Continual Learning)"]
    D2 -->|"Updates best_bee_yolo.pt"| B1
    D1 --> E["Standardized Trajectory Figures (plots/*.png & paper_plots/)"]
```

---

## 3. Arena Geometry & Coordinate Transformation

### Spatial Reference System

* **Feeder (Center)**: Fixed at Origin $(0, 0)\text{ mm}$.
* **Inner Boundary Circle**: Radius $R_{inner} = 210.0\text{ mm}$ (dashed grey circle).
* **Outer Boundary Circle**: Radius $R_{outer} = 420.0\text{ mm}$ (solid dark boundary).
* **Hive Position**:
  * Fixed at **Bottom Center $(0.0, -450.0)\text{ mm}$** or dynamically extracted from tagged entry coordinates $(x_e, y_e)$ projected to $R = 450.0\text{ mm}$.

### Pixel to Millimeter Coordinate Mapping

1. Pixel coordinates $(x_{px}, y_{px})$ are centered relative to feeder center $(x_0, y_0)$.
2. Scaled using camera scale factor $s \text{ (cm/pixel)} \rightarrow \text{mm}$:  
   $$\begin{aligned}  
   x_{mm} &= (x_{px} - x_0) \times s \times 10 \\  
   y_{mm} &= -(y_{px} - y_0) \times s \times 10  
   \end{aligned}$$
3. Radial distance from feeder: $r = \sqrt{x_{mm}^2 + y_{mm}^2}$.

---

## 4. Key Trajectory Marker & Boundary Intersection Logic

Boundary crossings are calculated directly from the **continuous reconstructed flight path** to ensure markers sit precisely on the trajectory line where it intersects the boundary circles:

| Marker Symbol | Marker Name | Boundary | Logic / Continuous Path Calculation |
| --- | --- | --- | --- |
| **Green Triangle (▲)** | Entry (Outer) | Outer Boundary ($420\text{ mm}$) | First point along the continuous path where distance crosses $R \le 420.0\text{ mm}$ into the arena. |
| **Red Diamond (◆)** | 1st Inner Contact (In) | Inner Boundary ($210\text{ mm}$) | Point along continuous path entering $R \le 210.0\text{ mm}$ that immediately precedes the closest approach to the feeder. |
| **Purple Diamond (◆)** | 1st Inner Exit (Out) | Inner Boundary ($210\text{ mm}$) | Point along continuous path exiting $R > 210.0\text{ mm}$ that immediately follows the closest approach to the feeder. |
| **Blue Circle (●)** | Exit (Outer) | Outer Boundary ($420\text{ mm}$) | Point along continuous path where the bee leaves the outer arena boundary ($R > 420.0\text{ mm}$). |
| **Dark Circle (●)** | Hive | Outer Hive ($450\text{ mm}$) | Physical entrance to the Hive. For `Went back` trials, the flight path smoothly curves directly into this marker. |

```plaintext
                 Outer Circle (R = 420 mm)
                 .-----------------------.
                /        (Top Exit)       \
               /                           \
              /      Inner Circle (210mm)   \
             |       .--------------.        |
             |      /   (Feeder)     \       |
             |     |      (0,0)       |      |
             |      \  ◆ Inner Exit  /       |
             |       '--------------'        |
              \      ◆ Inner Entry          /
               \                           /
                \     ▲ Outer Entry       /
                 '-----------------------'
                       (0, -420 mm)
                            |
                     [Hive Entrance]
                       (0, -450 mm)
```

---

## 5. Continuous Trajectory Reconstruction & Smoothing

### 1. Organic Flight Wave Gap Bridging (No Straight Lines)
When tracking is interrupted or jumps between manual help points, artificial straight chords are eliminated using harmonic sine wave synthesis perpendicular to the flight vector:
$$\mathbf{p}(t) = (1-t)\mathbf{p}_1 + t\mathbf{p}_2 + \mathbf{n}_\perp \cdot A \sin(\pi t) \sin(2\pi \cdot 1.5 t)$$
Where $\mathbf{n}_\perp = (-\Delta y/L, \Delta x/L)$ and amplitude $A = \min(10.0, \max(2.5, 0.08 L))$.

### 2. Multi-Pass Smoothing & Arc-Length Densification
* **Convolution & Savitzky-Golay Filtering**: Trajectory arrays are smoothed using Savitzky-Golay polynomial filters ($w = 35$, polyorder = 2) to preserve authentic flight dynamics.
* **Arc-Length Densification**: Subsampled to 2,000+ equidistant spatial points to produce perfectly smooth 2-color gradient rendering.

### 3. Hive Connection & Return Trajectory
For trials where the outcome is `Went back` / `Returned to Hive`, the path connects seamlessly into the Hive entrance marker without sharp geometric angles or false tracker reflection drift.

---

## 6. Metadata Extraction & Outcome Logic

### 1. Robust Metadata & Bee ID Extraction
* **Bee ID**: Parsed from `trial_outcome.txt`, CSV columns, or directory names via regex `(?:^|[._\s])([RGWBYOP]_\d+|[RGWBYOP]\d+|\d+[wWgGbByYoOpPrR])(?:[._\s]|$)` (e.g. `G_48`, `40W`, `21W`, `W_36`, `R13`).
* **Stimulus & Cue**: Identifies orientation (`LR`, `TB`) and stimulus parameters (e.g. `p8 u1`, `p4.4 u1`, `p0.3 u2`) and polarization degree (`DoP = 0.115`).

### 2. Trial Outcome Determination
* Evaluates `trial_outcome.txt` or final coordinate distance relative to the Hive:
  * `**Went back / Returned to Hive**`: Bee returns to the hive entrance.
  * `**Still in arena**`: Bee remains active inside the arena through the end of the analysis duration.

---

## 7. Complete Time Gradient Plot Suite

Each session generates standardized high-resolution figures in `plots/`:

1. `gradient_complete_trajectory_time.png`: Publication figure with a continuous 2-color gradient (**Bright Yellow** `#FFEE58` at $0.0\text{ s}$ $\rightarrow$ **Dark Royal Blue** `#0D47A1` at trial end), with a horizontal colorbar calibrated directly in elapsed seconds ($0.0\text{s} \to T\text{s}$).
2. `gradient_full_trajectory_smooth.png`: Arc-length smoothed complete gradient representation.
3. `gradient_full_trial_60fps.png` & `gradient_full_60fps_time.png`: Full trial calibrated frame-rate plots.

---

## 8. Summary of Core Scripts & Utilities

* `app.py`: Streamlit + YOLO interactive web app for tracking and arena calibration.
* `yolo_self_train.py`: Continual self-training engine that harvests tracking data and fine-tunes YOLO weights to improve detection accuracy.
* `analysis/generate_complete_hive_trajectory_plot.py`: Master visualization script implementing continuous boundary crossing calculations, organic flight wave smoothing, relative time gradient mapping, and batch processing across all directories.