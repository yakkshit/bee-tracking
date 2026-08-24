## Bee Tracking & Trajectory Analysis Pipeline: Comprehensive Technical Wiki

**Author / Maintainer:** Yakkshit  
**Project:** Bee Arena Navigation & Trajectory Analysis (BBP2025)  
**Target Audience:** Lab Professor & Research Group  
**Location:** `/Users/yakkshit/Downloads/project/hiwi2/p1/Videos/BBP2025/working/`

## 1\. Executive Summary & Purpose

This document provides a comprehensive technical overview of the automated trajectory processing, coordinate transformation, custom tracking application, and visualization pipeline developed for the **Bee Arena Navigation Dataset** (`results/maries/output/`).

The goal of this pipeline is to:

1.  Standardize raw tracking outputs (from both **TRex** and a custom **Streamlit + YOLO/OpenCV interactive tracker**) into unified `bee_track_<session>.csv` files.
2.  Align physical arena geometry and Hive orientation with experimental recordings.
3.  Generate 19 standardized analysis plots per session matching the reference style in `results/F/`.

## 2\. Dataset Architecture & Pipeline Flow

### Data Input & Output Directories

*   **Raw TRex Data Inputs**: `results/maries/output/<session_folder>/`  
    Contains `id0_trex.csv` (frame index, pixel coordinates $X, Y$), metadata, and video files.
*   **Custom Streamlit/YOLO Tracker App**: `app.py` & `pages/1_Analysis_Viewer.py`  
    Used to calibrate, manually track, or review videos where TRex failed or was not run.
*   **Reference Standard Plots**: `results/F/<session_folder>/plots/`  
    Established baseline plot formatting and logic.
*   **Processed TRex Outputs**: `results/m/<session_folder>/` and `results/maries/output/<session_folder>/plots/`  
    Contains converted `bee_track_<session>.csv`, `trial_outcome.txt`, and 19 standardized trajectory plot PNGs.

### Pipeline Execution Workflow

```plaintext
flowchart TD
    A1["TRex Output (id0_trex.csv)"] --&gt; B1["convert_trex_to_bee_track.py"]
    A2["Raw Video Files"] --&gt; B2["Streamlit + YOLO Tracker App (app.py)"]
    B2 --&gt; C["Standardized bee_track_<session>.csv"]
    B1 --&gt; C
    C --&gt; D["generate_plots_for_sessions.py / run_all_maries.py"]
    D --&gt; E["analysis/*.py plot modules (19 PNG figures per session)"]
```

## 3\. Arena Geometry & Coordinate Transformation

### Spatial Reference System

*   **Feeder (Center)**: Fixed at Origin $(0, 0)\\text{ mm}$.
*   **Inner Boundary Circle**: Radius $R\_{inner} = 210.0\\text{ mm}$ (dashed grey circle).
*   **Outer Boundary Circle**: Radius $R\_{outer} = 420.0\\text{ mm}$ (solid dark boundary).
*   **Hive Position**:
    *   **Marie TRex Sessions**: Fixed at **Bottom Center $(0.0, -450.0)\\text{ mm}$** (outside outer boundary at bottom entrance, matching experimental video orientation).

### Pixel to Millimeter Coordinate Mapping

1.  Pixel coordinates $(x\_{px}, y\_{px})$ are centered relative to feeder center $(x\_0, y\_0)$.
2.  Scaled using camera scale factor $s \\text{ (cm/pixel)} \\rightarrow \\text{mm}$:  
    $$\\begin{aligned}  
    x\_{mm} &= (x\_{px} - x\_0) \\times s \\times 10 \\  
    y\_{mm} &= -(y\_{px} - y\_0) \\times s \\times 10  
    \\end{aligned}$$
3.  Radial distance from feeder: $r = \\sqrt{x\_{mm}^2 + y\_{mm}^2}$.

## 4\. Key Trajectory Marker Logic

Each trajectory plot incorporates **4 specific circle crossing markers** that define the critical phases of a bee's trial:

| Marker Symbol | Marker Name | Circle Boundary | Logic / Position |
| --- | --- | --- | --- |
| **Marker ① (green ▲)** | 1st Outer Circle Entry | Outer Boundary ($420\\text{ mm}$) | First point where bee crosses $R \\le 420\\text{ mm}$. If video tracking starts inside the arena, defaults to **$(0.0, -420.0)\\text{ mm}$** (Hive entrance). |
| **Marker ◆ (red diamond)** | 1st Inner Circle Entry | Inner Boundary ($210\\text{ mm}$) | First point where bee crosses $R \\le 210\\text{ mm}$ approaching the feeder. Defaults to $(0.0, -210.0)\\text{ mm}$ if tracking starts inside inner circle. |
| **Marker ◆ (purple diamond)** | 1st Inner Circle Exit | Inner Boundary ($210\\text{ mm}$) | First point where bee crosses $R > 210\\text{ mm}$ after visiting/approaching the feeder. |
| **Marker ① (blue ●)** | 1st Outer Circle Exit | Outer Boundary ($420\\text{ mm}$) | Final point where bee exits the $420\\text{ mm}$ outer boundary or last tracked trajectory point. |

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

## 5\. Continuous Trajectory Reconstruction & Smoothing

### Smooth Motion Interpolation (No Straight Lines)

*   **Savitzky-Golay & B-Spline Smoothing**: Trajectory paths use adaptive polynomial smoothing (window size $w=15\\dots 17$, degree $k=2$) to preserve natural curved bee flight dynamics.
*   **Entry Segment Connection**:
    *   When video tracking starts inside the arena (e.g. near feeder at frame 0), an artificial straight line is **never** drawn.
    *   Instead, a smooth curved spline segment is prepended connecting:  
        $$\\text{Outer Entry }(0, -420) \\longrightarrow \\text{Inner Entry }(0, -210) \\longrightarrow \\text{First Tracked Coordinate } (x\_0, y\_0)$$
    *   This ensures every trajectory connects continuously from outer circle entry to outer circle exit.

## 6\. Metadata Extraction & Outcome Logic

### 1\. Bee ID Parsing

Bee ID is extracted directly from the session folder name using regular expressions:

*   Matches patterns such as `\b(\d+[wWgG])\b` (e.g., `49w` $\\rightarrow$ `**49W**`, `40w` $\\rightarrow$ `**40W**`, `70g` $\\rightarrow$ `**70G**`, `21w` $\\rightarrow$ `**21W**`).
*   Fallbacks: Color tags (`R_1`, `G_2`), `unmarked`, or `Unknown` if unspecified.

### 2\. Trial Outcome Determination

Evaluates the final recorded coordinate $(x\_{last}, y\_{last})$ relative to Hive position $(0, -450)\\text{ mm}$:  
$$\\text{Distance to Hive} = \\sqrt{(x\_{last} - 0)^2 + (y\_{last} - (-450))^2}$$

*   `**"Returned to Hive"**`: If $\\text{Distance to Hive} \< 280\\text{ mm}$ or $y\_{last} \< -150\\text{ mm}$ near bottom wall (near Hive entrance).
*   `**"Still in Arena"**`: If last point is far or opposite from the Hive (e.g. top arena wall $y > 0\\text{ mm}$).

## 7\. Interactive Streamlit + YOLO Web Tracking App (`app.py`)

In addition to automated TRex data conversion, a custom web application was built in Python using **Streamlit**, **YOLO object detection**, **OpenCV**, and `streamlit_drawable_canvas`.

### Core Capabilities Beyond TRex

1.  **Fallback Tracking for Untracked Videos**:
    *   TRex sometimes fails on videos with low lighting, reflections, or occlusions.
    *   The custom app uses a trained **YOLO / OpenCV tracking engine** to track bees in videos that TRex could not process.
2.  **Interactive Visual Calibration**:
    *   Researchers can load any raw MP4 video, draw on key frames to calibrate the Feeder origin $(x\_0, y\_0)$ and boundary radii ($R\_{inner}, R\_{outer}$), and visually inspect tracking quality in real time.
3.  **Analysis Viewer (**`**pages/1_Analysis_Viewer.py**`**)**:
    *   Interactive dashboard allowing researchers to load any session, inspect the 19 generated plots, adjust entry/exit trim sliders, and export cleaned CSV files.

## 8\. Plot Suite Overview (19 Output Figures per Session)

Each session folder generates 19 standardized PNG plots inside `plots/`:

1.  `complete_trajectory.png` / `approach_and_home_search.png`: Full trial trajectory colored with yellow-green-blue time gradient.
2.  `single_colour.png`: Inbound path (grey dotted) vs Outbound path (blue solid) with direction arrows.
3.  `gradient_full_trial.png` & `gradient_exit_only.png`: Smooth two-color (yellow $\\rightarrow$ dark blue) progress gradient.
4.  `gradient_full_trajectory_smooth.png` & `gradient_exit_trajectory_smooth.png`: Arc-length normalized smooth gradient plots.
5.  `fixed_clean_full_trial.png` & `fixed_clean_exit_path.png`: High-contrast publication-ready figures.
6.  `multiple_entries_exits.png`: Multi-pass trajectory decomposition with mean path.
7.  `feeder_to_inner_circle.png` & `feeder_to_final_exit.png`: Targeted phase plots.

## 9\. Summary of Script Files & Command Utilities

*   `**app.py**`: Streamlit + YOLO interactive web app for tracking and arena calibration.
*   `**convert_trex_to_bee_track.py**`: Converts TRex `id0_trex.csv` to `bee_track_<session>.csv`, calculates distances, zone transitions, Bee ID, and Trial Outcome.
*   `**run_all_maries.py**`: Batch processes all 77 Marie sessions in sequence.
*   `**generate_plots_for_sessions.py**`: CLI utility to generate plots for specific session names.
*   `**analysis/*.py**`: Modular visualization scripts (`generate_single_colour_plot.py`, `generate_gradient_coloured_plot.py`, etc.).

docker build -t bee-arena-tracker .  
docker run -p 8501:8501 -v $(pwd):/app bee-arena-tracker