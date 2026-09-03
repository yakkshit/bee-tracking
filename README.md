# Bee Arena Tracker & Trajectory Analysis

A comprehensive automated tracking, trajectory visualization, and neuroethological analysis platform for analyzing walking bumblebee (*Bombus terrestris*) path integration, visual cue polarization, and memory mechanisms (BBP2025).

---

## 🚀 Quick Start & How to Run

### 1. Containerized Web Application (`./webapp.sh`)
Launch the complete interactive web application in Docker with automatic browser opening:
```bash
./webapp.sh
```
- **Automatic Browser Launch**: Opens `http://localhost:8501` in your browser as soon as the container is healthy.
- **Dockerized Environment**: Runs OpenCV, YOLO tracking, interactive canvas calibration, file management, and statistical plotting in an isolated container.

### 2. Interactive Launcher (`./run.sh` or `run.bat`)
Cross-platform terminal launcher:
```bash
./run.sh
```
- **Option 1 (Local / Python virtualenv)**: Runs `app.py` in local `.venv`.
- **Option 2 (Docker)**: Launches containerized app via Docker Compose.

### 3. Plot & Analysis Pipeline (`./analysis.sh`)
Executes either individual trajectory plots or the complete paper figures pipeline:
```bash
./analysis.sh
```
- **Option 1 (Individual Analysis)**: Prompts for the folder where tracking sessions are located (e.g. `results/F` or `results/maries data`), then runs all individual trajectory generators for every session found.
- **Option 2 (Complete Paper Plots)**: Prompts for the tracking dataset folder (e.g. `results/F`, `results/maries data`, or `results`), creates a `paper_plots/` folder directly inside that directory, and outputs all paper figures (Figs 9–26), speed distribution analysis, and CSV statistical summaries there.

### 4. YOLO Continual Self-Training (`python yolo_self_train.py`)
Automatically harvests verified tracking frames from analyzed sessions, generates normalized YOLO bounding boxes, fine-tunes the model, and updates the active detector (`models/best_bee_yolo.pt`) to continually improve tracking accuracy:
```bash
# Automated harvest + fine-tuning cycle (20 epochs)
python yolo_self_train.py --auto

# Or fine-tune with custom epochs / dataset path
python yolo_self_train.py --harvest --train --epochs 25 --sessions-dir "results/F"
```
*(Also accessible directly via the **"🚀 Fine-Tune YOLO Model on Tracked Data"** button in the Streamlit web application).*

---

## 📁 Repository Structure & Organization

```
working/
├── app.py                            # Interactive Streamlit web application
├── yolo_self_train.py               # Continual self-training pipeline for YOLO accuracy improvement
├── webapp.sh                         # Single-command Docker web app launcher
├── run.sh / run.bat                  # Cross-platform interactive CLI script
├── analysis.sh                       # Paper plot & statistical pipeline runner
├── tracking_logic.py                 # BeeYOLODetector & tracking primitives
├── requirements.txt                  # Python package requirements
├── Dockerfile                        # Docker container definition
├── docker-compose.yml                # Docker Compose service definition
├── yolo11n.pt                        # Base YOLO object detection weights
│
├── analysis/                         # Trajectory & paper figure generators
│   ├── generate_analysis_plots.py            # Complete session trajectories
│   ├── generate_feeder_to_inner_circle_plot.py# Inner circle pass plots
│   ├── generate_feeder_to_final_exit_plot.py # Full arena pass plots
│   ├── generate_single_colour_plot.py       # Publication inbound/outbound plots
│   ├── generate_gradient_coloured_plot.py   # Continuous time-gradient plots
│   ├── generate_60fps_gradient_plots.py     # High-fps time-gradient plots
│   ├── generate_smooth_path_gradient_plots.py# Path-length normalized gradient plots
│   ├── generate_multiple_entries_plot.py    # Multi-pass entry/exit plots + mean path
│   ├── generate_fixed_clean_trajectory_plot.py# Clean boundary crossing plots
│   ├── fixed_version.py                      # Sinusoidal gap-bridging model (final_trajectories.png)
│   ├── gradient_fixed.py                     # High-res time gradient gap-bridging (trajectory_generated_paths.png)
│   ├── generate_paper_plots.py               # Paper figures (Figs 9-23, 25-26) & stats CSV
│   ├── generate_speed_plot.py                # Movement speed analysis (Fig 24)
│   └── export_notebook_to_html.py            # HTML analysis report builder
│
└── results/                          # Output directory for data and plots
    ├── F/                                    # Frestas video dataset (85 sessions)
    │   └── F_Paper_plots/                    # Individual paper figures for F dataset
    ├── maries data/                          # Marie Jansen dataset (51 sessions)
    │   └── m_paper_plots/                    # Individual paper figures for Marie's dataset
    ├── paper_plots/                          # Combined dataset paper figures (Figs 9–26)
    └── f+m_paper_plots/                      # Combined dataset analysis output
```

---

## 🐍 Session Folders & Trajectory CSV Specification

Each session folder (e.g. `results/F/2024-11-17_17-04-16.R13.LR.P0U8/`) contains:

| File Name | Description & Contents |
| :--- | :--- |
| **`bee_track_*.csv`** | **Primary Trajectory CSV**:<br>• `frame`: Frame index (30/60 fps)<br>• `time_sec` / `time_s`: Timestamp in seconds<br>• `x_mm`, `y_mm`: Centered coordinates in mm ($0,0$ = Feeder Center)<br>• `distance_from_center_mm`: Radial distance $r = \sqrt{x^2 + y^2}$ in mm<br>• `speed_mm_s`: Instantaneous velocity in mm/s |
| **`session_info.json`** | Metadata: FPS, video resolution, duration, condition (`Training`, `Strong LR`, `Weak LR`, `Strong TB`, `Weak TB`). |
| **`calibration.json`** | Arena pixel-to-mm scaling factor (`scale_px_per_mm`) and transformation matrices. |

---

## 🗺️ Trajectory Plot Types, Primary Figures & Mathematical Foundations

```
                          [ Outer Boundary Exit (R = 420mm) ]
                                         │
                                [ Outbound Trajectory ]
                                         │
                                [ Inner Circle Exit (R = 210mm) ]
                                         │
                                   [ Feeder (0,0) ]
                                         │
                                [ Inner Circle Entry (R = 210mm) ]
                                         │
                                [ Inbound Trajectory ]
                                         │
                          [ Outer Boundary Entry (R = 420mm) ]
```

### ⭐ PRIMARY TRAJECTORY PLOTS (CRUCIAL FINAL FIGURES FOR THESIS & PAPER)

Out of all generated session plots, **two primary trajectory plots** serve as the definitive, gold-standard figures:

#### 1. 🏆 Primary Final Gap-Bridged Trajectory (`trajectory_generated_paths.png` / `final_trajectories.png`)
> **Crucial Primary Figure**: Found in `<session_folder>/plots/trajectory_generated_paths.png` and `final_trajectories.png`.

- **Purpose**: Combines empirical video tracking with mathematical trajectory restoration across tracking gaps ($\Delta \text{frames} > 80$ or spatial jumps $> 80\text{ mm}$) caused by visual occlusions or bee illumination loss.
- **Visual Encoding**:
  - **Tracked Observed Path**: Rendered as a **Solid Line (`-`)** colored by time gradient.
  - **Generated Gap Bridge**: Rendered as a **Dashed Line (`---`)** colored with the identical continuous time gradient.
  - **Hive Entrance**: Marked at fixed physical location $(0, -450\text{ mm})$ with a black circle.
  - **Event Annotations**: Numbered markers with arrow pointers indicating:
    - ① **First Outer Entry** ($R = 420\text{ mm}$, Green `^`)
    - ① **First Inner Entry** ($R = 210\text{ mm}$, Red `D`)
    - ① **First Inner Exit** ($R = 210\text{ mm}$, Purple `D`)
    - ① **First Outer Exit** ($R = 420\text{ mm}$, Orange `^`)
  - **Colorbar**: `Time Gradient (seconds): Start (Yellow) → End (Dark Blue)`.

#### 2. ⚡ Primary High-FPS Smooth Time-Gradient Trajectory (`gradient_full_trajectory_smooth.png`)
> **Crucial Smooth Gradient Figure**: Found in `<session_folder>/plots/gradient_full_trajectory_smooth.png`.

- **Purpose**: Provides a continuous, high-definition time-coded visual representation of the bee's entire walk without gap-bridging overlays.
- **Visual Encoding**:
  - Continuous 60 fps Savitzky-Golay smoothed trajectory path line.
  - Color-coded from start ($t = 0\text{ s}$, Yellow/Gold) to end ($t = t_{\text{end}}$, Deep Blue/Indigo).
  - Annotates exact timestamps ($t = 0.0\text{ s}$, $t = t_{\text{end}}$) and feeder center ($0,0$).

---

### 🧮 Mathematical Foundations & Algorithms Behind Trajectories

#### A. Spatial Distance & Velocity Kinematics
For frame $k$ with raw pixel coordinates $(x_k^{\text{px}}, y_k^{\text{px}})$ and pixel scale $S \text{ (px/mm)}$:
$$x_k = \frac{x_k^{\text{px}} - x_{\text{center}}^{\text{px}}}{S}, \quad y_k = \frac{y_k^{\text{px}} - y_{\text{center}}^{\text{px}}}{S}$$

Radial distance $r_k$ from feeder center $(0,0)$ and instantaneous walking speed $v_k$:
$$r_k = \sqrt{x_k^2 + y_k^2}$$
$$v_k = \frac{\sqrt{(x_k - x_{k-1})^2 + (y_k - y_{k-1})^2}}{\Delta t_k} \quad (\text{mm/s})$$

#### B. Sinusoidal Gap-Bridging Interpolation Algorithm (`fixed_version.py`)
When a tracking loss occurs between point $\mathbf{P}_1 = (x_1, y_1)$ at time $t_1$ and $\mathbf{P}_2 = (x_2, y_2)$ at time $t_2$ where $\Delta \text{frame} > 80$ or $\Delta d > 80\text{ mm}$:

1. **Linear Baseline Parameterization** ($s \in [0, 1]$):
   $$\mathbf{L}(s) = (1 - s)\mathbf{P}_1 + s\mathbf{P}_2, \quad t(s) = (1 - s)t_1 + s t_2$$

2. **Perpendicular Normal Unit Vector**:
   $$\Delta x = x_2 - x_1, \quad \Delta y = y_2 - y_1, \quad L = \sqrt{\Delta x^2 + \Delta y^2}$$
   $$\mathbf{N}_{\perp} = \left(-\frac{\Delta y}{L}, \frac{\Delta x}{L}\right)$$

3. **Parametric Oscillatory Wiggle Function**:
   To mimic realistic insect searching meanders rather than unnatural straight lines, an amplitude-envelope sine wave is superimposed:
   $$E(s) = \sin(\pi s) \quad \text{(Envelope pinning endpoints to zero deviation)}$$
   $$W(s) = A \cdot E(s) \cdot \sin(2\pi \cdot f \cdot s) \quad (A = 20.0\text{ mm}, \; f = 2.5\text{ cycles})$$

4. **Reconstructed Gap Coordinates**:
   $$\mathbf{P}_{\text{gen}}(s) = \mathbf{L}(s) + W(s) \cdot \mathbf{N}_{\perp}$$

#### C. Savitzky-Golay Trajectory Smoothing Filter
To remove single-frame pixel jitter while preserving physical turning dynamics, a 3rd-order polynomial Savitzky-Golay filter over an 11-frame window ($N = 11, M = 3$) is applied across the full coordinate series:
$$\hat{x}_k = \sum_{j=-(N-1)/2}^{(N-1)/2} c_j \, x_{k+j}$$

---

### 📋 Full Catalog of Session Trajectory Plots (`<session_folder>/plots/`)

| Plot File Name | Description & Purpose | Line Style & Color Encoding |
| :--- | :--- | :--- |
| **`trajectory_generated_paths.png`** <br> **`final_trajectories.png`** | ⭐ **PRIMARY DEFINITIVE TRACKING PLOT**. Gap-bridged full trajectory with event markers & hive location. | **Solid Line (`-`)**: Tracked.<br>**Dashed Line (`---`)**: Gap bridge.<br>**Time Gradient**: Yellow $\rightarrow$ Blue. |
| **`gradient_full_trajectory_smooth.png`** | ⭐ **PRIMARY TIME-GRADIENT PLOT**. Continuous 60 fps smooth trajectory without gap overlays. | **Solid Line (`-`)**: Time gradient (Yellow $\rightarrow$ Blue). |
| **`complete_trajectory.png`** | Full unsegmented trajectory across all recorded video frames. | Solid Blue Line (`#1976D2`). |
| **`feeder_to_inner_circle.png`** | Segmented inner circle pass ($R = 210\text{ mm}$) to feeder and exit. | Inbound (Blue) / Outbound (Red). |
| **`feeder_to_final_exit.png`** | Segmented full arena pass from outer entry ($R = 420\text{ mm}$) to feeder to outer exit. | Inbound (Blue) / Outbound (Red). |
| **`single_colour.png`** | Clean publication view of inbound vs outbound path. | Inbound: Solid Blue (`#2980B9`).<br>Outbound: Solid Red (`#C0392B`). |
| **`gradient_coloured.png`** | Standard time-gradient plot across raw recorded points. | Color-coded by timestamp $t$. |
| **`60fps_gradient.png`** | High-fps timestamp gradient plot. | Color-coded by timestamp $t$. |
| **`smooth_path_gradient.png`** | Path-length normalized gradient plot. | Color-coded by cumulative path length (0 mm $\rightarrow$ End). |
| **`multiple_entries_exits.png`** | Multi-pass entry/exit trajectory analysis with calculated mean path line. | Individual passes: Numbered dashed lines.<br>Mean path: Bold Black Line. |
| **`fixed_clean_trajectory.png`** | Minimalist clean trajectory plot with boundary crossing markers. | Clean solid lines with boundary icons. |

---

## 📐 Arena Geometry, Angles & Polar Coordinates Explanation

To ensure non-technical readers and reviewers can interpret all circular and spatial plots without needing verbal explanation, the key angular conventions and geometry are standardized as follows:

```
                  0° / 2π (Fictive North)
                           │
                           │
    270° / 3π/2 ───────── (0,0) ───────── 90° / π/2
   (West / Left)        Feeder           (East / Right)
                           │
                           │
                180° / π (Real Nest Entrance)
```

1. **Center $(0,0)$**: The **Conical Feeder** located in the middle of the circular arena where bumblebees drink sucrose solution.
2. **Nest Entrance ($180^\circ$ or $\pi$)**: The physical home entrance from which bees enter and return to the colony. Located at the bottom ($y = -450\text{ mm}$).
3. **Angles & Radians**:
   - $0^\circ$ (or $0 \text{ rad}$): Top / Fictive North (opposite of nest).
   - $90^\circ$ (or $\pi/2 \text{ rad} \approx 1.57 \text{ rad}$): Right / East sector.
   - $180^\circ$ (or $\pi \text{ rad} \approx 3.14 \text{ rad}$): Bottom / Real Nest Direction.
   - $270^\circ$ (or $3\pi/2 \text{ rad} \approx 4.71 \text{ rad}$): Left / West sector.
4. **Mean Direction Arrow ($\mu$)**: In circular plots, the solid arrow starting from center $(0,0)$ points towards the mean direction ($\mu$) of all exit bearings.
5. **Arrow Length ($r$)**: Proportional to the Rayleigh mean vector length ($0 \le r \le 1$). A long arrow extending near the circle boundary indicates strong directional clustering; a short arrow near center indicates random/dispersed directions.
6. **Inner Circle ($R = 210\text{ mm}$)**: The illuminated central arena zone where polarized UV light stimulus is actively emitted.
7. **Outer Circle ($R = 420\text{ mm}$)**: The arena outer boundary curtain exit ring.

---

## 🧠 Short-Term Memory (STM) vs Long-Term Memory (LTM) Analysis

A key neurobiological question in bee navigation (Jansen 2025; Patel et al. 2024) is whether bumblebees rely on **Short-Term Memory (STM)** (following the rotated visual light stimulus) or **Long-Term Memory (LTM)** (relying on memorized home vectors towards the real nest).

```
                      [ Long-Term Memory (LTM) ]
                            315° to 45°
                                 │
 [ Short-Term Memory ]           │           [ Short-Term Memory ]
    225° to 315°       ───────── + ─────────      45° to 135°
        (STM)                    │                  (STM)
                                 │
                      [ Long-Term Memory (LTM) ]
                           135° to 225°
```

### Memory Sector Definition:
- **Long-Term Memory (LTM) Sectors** (Dark Green `#2E7D32`): $315^\circ \text{ to } 45^\circ$ and $135^\circ \text{ to } 225^\circ$ (Aligned with the familiar training axis pointing to real nest or anti-nest).
- **Short-Term Memory (STM) Sectors** (Light Blue `#A0D7E6`): $45^\circ \text{ to } 135^\circ$ and $225^\circ \text{ to } 315^\circ$ (Aligned with $90^\circ$ rotated stimulus axis / fictive nest).

---

## 📁 Publication Figures & Data Breakdown (Figures 9 – 26)

All paper figures are systematically organized in **3 separate analysis directories**:
1. **`results/maries data/m_paper_plots/`**: Marie Jansen Thesis Dataset analysis (51 tracked sessions).
2. **`results/F/F_Paper_plots/`**: Frestas Video Dataset analysis (85 tracked sessions).
3. **`results/paper_plots/`** / **`results/f+m_paper_plots/`**: **Combined Overall Dataset analysis** (136 total sessions).

| Figure | Image File Path | Title & Description | Key Metrics & Data |
| :--- | :--- | :--- | :--- |
| **Fig 9** | [`fig09_training_inner_circle.png`](file:///Users/yakkshit/Downloads/project/hiwi2/p1/Videos/BBP2025/working/results/paper_plots/fig09_training_inner_circle.png) | **Training Control Trials (Raw & Circ-MLE Bimodal Fit)**<br>Left: Raw scatter of exit bearings during training. Right: Bimodal von Mises model fit. | Bimodal distribution clustered at $115.67^\circ$ ($\kappa_2 = 4.06$) and $295.70^\circ$ ($\kappa_1 = 4.06$). Rayleigh $r = 0.15, p = 0.87$. |
| **Fig 10** | [`fig10_strong_polarization_lr.png`](file:///Users/yakkshit/Downloads/project/hiwi2/p1/Videos/BBP2025/working/results/paper_plots/fig10_strong_polarization_lr.png) | **Strong Polarization (LR Condition)**<br>Inner circle exit bearings under strong $e$-vector polarization in Left-Right rotation. | Mean vector $\mu = 207.33^\circ$, $r = 0.36, p = 0.33, \kappa = 0.76$. |
| **Fig 11** | [`fig11_weak_polarization_lr.png`](file:///Users/yakkshit/Downloads/project/hiwi2/p1/Videos/BBP2025/working/results/paper_plots/fig11_weak_polarization_lr.png) | **Weak Polarization (LR Condition)**<br>Inner circle exit bearings under weak polarization. | Mean vector $\mu = 160.0^\circ$, $r = 0.31, p = 0.70, \kappa = 0.67$. |
| **Fig 12** | [`fig12_strong_polarization_tb.png`](file:///Users/yakkshit/Downloads/project/hiwi2/p1/Videos/BBP2025/working/results/paper_plots/fig12_strong_polarization_tb.png) | **Strong Polarization (TB Condition)**<br>Inner circle exit bearings under strong $e$-vector in Top-Bottom rotation. | Mean vector $\mu = 24.78^\circ$, $r = 0.15, p = 0.71, \kappa = 0.31$. |
| **Fig 13** | [`fig13_weak_polarization_tb.png`](file:///Users/yakkshit/Downloads/project/hiwi2/p1/Videos/BBP2025/working/results/paper_plots/fig13_weak_polarization_tb.png) | **Weak Polarization (TB Condition)**<br>Inner circle exit bearings under weak $e$-vector in Top-Bottom rotation. | Mean vector $\mu = 241.79^\circ$, $r = 0.45, p = 0.31, \kappa = 1.01$. |
| **Fig 14** | [`fig14_home_vs_fictive_lr.png`](file:///Users/yakkshit/Downloads/project/hiwi2/p1/Videos/BBP2025/working/results/paper_plots/fig14_home_vs_fictive_lr.png) | **Home vs Fictive Distribution (LR Condition)**<br>Percentage bar chart comparing chosen direction across strong & weak DOP. | Percentage (%) y-axis with bold percentage annotations on top of each bar. |
| **Fig 15** | [`fig15_home_vs_fictive_tb.png`](file:///Users/yakkshit/Downloads/project/hiwi2/p1/Videos/BBP2025/working/results/paper_plots/fig15_home_vs_fictive_tb.png) | **Home vs Fictive Distribution (TB Condition)**<br>Percentage bar chart of chosen paths in $90^\circ$ shifted TB trials. | Percentage (%) y-axis with bold percentage annotations on top of each bar. |
| **Fig 16** | [`fig16_circular_plots_lr_tb.png`](file:///Users/yakkshit/Downloads/project/hiwi2/p1/Videos/BBP2025/working/results/paper_plots/fig16_circular_plots_lr_tb.png) | **2x2 Circular Plots Grid across Conditions**<br>Side-by-side circular bearings comparing Weak LR, Weak TB, Strong LR, and Strong TB. | Visual distribution overview across all 4 main experimental quadrants. |
| **Fig 17** | [`fig17_average_homing_accuracy.png`](file:///Users/yakkshit/Downloads/project/hiwi2/p1/Videos/BBP2025/working/results/paper_plots/fig17_average_homing_accuracy.png) | **Average Homing Accuracy (Mean Angular Deviation)**<br>Bar plot of mean angular deviation from nest entrance ($180^\circ$). | Strong LR: $68.36^\circ$, Weak LR: $105.79^\circ$, Strong TB: $98.71^\circ$, Weak TB: $75.03^\circ$. |
| **Fig 18** | [`fig18_boxplot_deviation_from_nest.png`](file:///Users/yakkshit/Downloads/project/hiwi2/p1/Videos/BBP2025/working/results/paper_plots/fig18_boxplot_deviation_from_nest.png) | **Boxplot of Angular Deviation from Nest**<br>Interquartile range and median angular deviation per condition. | Boxplot with median line and IQR bounds (no dataset-specific text overlay). |
| **Fig 19** | [`fig19_heatmap_strong_vs_weak.png`](file:///Users/yakkshit/Downloads/project/hiwi2/p1/Videos/BBP2025/working/results/paper_plots/fig19_heatmap_strong_vs_weak.png) | **Spatial Density Heatmap (Strong vs Weak DOP)**<br>2D Gaussian smoothed arena occupancy heatmaps. | Highlights higher spatial dispersion in weak DOP compared to focused home paths in strong DOP. |
| **Fig 20** | [`fig20_heatmap_conditions_grid.png`](file:///Users/yakkshit/Downloads/project/hiwi2/p1/Videos/BBP2025/working/results/paper_plots/fig20_heatmap_conditions_grid.png) | **2x2 Trajectory Spatial Occupancy Grid**<br>Spatial heatmaps for Strong LR, Strong TB, Weak LR, Weak TB (`fig20a-d` also standalone). | Full arena spatial density comparison across rotation and polarization levels. |
| **Fig 21** | [`fig21_heatmap_overall_occupancy.png`](file:///Users/yakkshit/Downloads/project/hiwi2/p1/Videos/BBP2025/working/results/paper_plots/fig21_heatmap_overall_occupancy.png) | **Overall Cumulative Arena Occupancy Heatmap**<br>Aggregated spatial heatmap across all trial trajectories. | Identifies global arena high-density foraging corridors and feeder search zones. |
| **Fig 22** | [`fig22_exit_angle_polar_heatmaps.png`](file:///Users/yakkshit/Downloads/project/hiwi2/p1/Videos/BBP2025/working/results/paper_plots/fig22_exit_angle_polar_heatmaps.png) | **Dual Polar Exit Angle Donut Heatmaps**<br>Polar exit angle density distributions (`fig22a-d` also standalone). | Clean `LR` (outer ring) vs `TB` (inner ring) polar exit angle relative density distributions. |
| **Fig 23** | [`fig23_exit_angle_matrix_heatmap.png`](file:///Users/yakkshit/Downloads/project/hiwi2/p1/Videos/BBP2025/working/results/paper_plots/fig23_exit_angle_matrix_heatmap.png) | **Sector Matrix Heatmap (Inner vs Outer Circle)**<br>12-sector ($30^\circ$ bins) exit frequency heatmap across 5 conditions. | Quantitative percentage exit matrix for Inner ($R=210\text{mm}$) and Outer ($R=420\text{mm}$) circles. |
| **Fig 24** | [`fig24_speed_of_movement_analysis.png`](file:///Users/yakkshit/Downloads/project/hiwi2/p1/Videos/BBP2025/working/results/paper_plots/fig24_speed_of_movement_analysis.png) | **Speed of Movement Analysis (4 Subplots)**<br>A) Speed KDE, B) Session Mean Speeds (4 conditions), C) Zone Speeds, D) Condition Speeds. | Marie Data in Green (`#2E7D32`), F Dataset in Blue (`#1976D2`) (`fig24a-d` also standalone). |
| **Fig 25** | [`fig25_short_vs_long_term_memory.png`](file:///Users/yakkshit/Downloads/project/hiwi2/p1/Videos/BBP2025/working/results/paper_plots/fig25_short_vs_long_term_memory.png) | **100% Stacked Memory Bar Chart (STM vs LTM)**<br>Panel A: Combined LR vs TB; Panel B: Strong/Weak LR & TB. | STM bottom in light blue (`#A0D7E6`), LTM top in dark green (`#2E7D32`) with Chi-squared stats. |
| **Fig 26** | [`fig26_feeder_visit_percentage.png`](file:///Users/yakkshit/Downloads/project/hiwi2/p1/Videos/BBP2025/working/results/paper_plots/fig26_feeder_visit_percentage.png) | **Feeder Visit Percentage Bar Chart**<br>Return trajectory classification: Returned Back vs Still in Arena / Exited. | Percentage (%) y-axis without raw sample numbers on bars per specification. |
| **Fig 26** | [`fig26_feeder_visit_percentage.png`](file:///Users/yakkshit/Downloads/project/hiwi2/p1/Videos/BBP2025/working/results/paper_plots/fig26_feeder_visit_percentage.png) | **Feeder Visit Percentage Bar Chart**<br>Return trajectory classification: Returned Back vs Still in Arena / Exited. | Percentage (%) y-axis without raw sample numbers on bars per specification. |