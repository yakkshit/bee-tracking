# 🐝 Bee Arena Tracker — Complete User Guide & Operating Instructions

Welcome to the **Bee Arena Tracker** documentation. This step-by-step guide explains how to install, launch, run, and operate the tracking software on both **Windows** and **macOS** computers.

---

## 📋 Table of Contents
1. [💻 macOS Operating Guide (Step-by-Step)](#-macos-operating-guide-step-by-step)
2. [🪟 Windows Operating Guide (Step-by-Step)](#-windows-operating-guide-step-by-step)
3. [🚀 How to Use the Application Workflow](#-how-to-use-the-application-workflow)
   - [Step 1: Setup & Load Video](#step-1-setup--load-video)
   - [Step 2: Calibrate Arena Circles](#step-2-calibrate-arena-circles)
   - [Step 3: Tracking & Tagging Room](#step-3-tracking--tagging-room)
   - [Step 4: Analysis & Data Export](#step-4-analysis--data-export)
4. [🛠 Troubleshooting & FAQ](#-troubleshooting--faq)

---

## 💻 macOS Operating Guide (Step-by-Step)

Follow these steps to run the software on a Mac laptop or desktop:

### **Step 1: Open the Terminal**
- Press `Cmd + Space` to open Spotlight Search.
- Type `Terminal` and press `Enter`.

### **Step 2: Navigate to Project Folder**
In the Terminal window, navigate to the folder where the project is located:
```bash
cd /path/to/bee-tracking
```
*(Tip: You can type `cd ` and drag-and-drop the project folder from Finder directly into the Terminal window).*

### **Step 3: Check Python Installation**
Verify that Python 3 is installed on your Mac:
```bash
python3 --version
```
If Python is installed, it will display a version number like `Python 3.11.x`.

### **Step 4: Launch the Application**
Run the launcher script:
```bash
./run.sh
```
*(If permission is denied, grant execute permission first: `chmod +x run.sh`).*

When prompted:
```text
Select option (1, 2, or 3) [Default: 1]:
```
- **Option 1 (Default)**: Normal launch using existing virtual environment (`.venv`).
- **Option 2**: Recreates a fresh virtual environment if libraries were broken or updated.
- **Option 3**: Launches inside a Docker container.

### **Step 5: Access in Web Browser**
The app will automatically launch in your browser at:
👉 **`http://localhost:8501`**

---

## 🪟 Windows Operating Guide (Step-by-Step)

Follow these steps to run the software on a Windows laptop or desktop:

### **Step 1: Open Command Prompt or PowerShell**
- Press the `Windows Key` on your keyboard.
- Type `cmd` or `PowerShell` and press `Enter`.

### **Step 2: Navigate to Project Directory**
Change directory to the project folder:
copy the path and 
```
cd paste the copied path here

```
it should look some thing like this 
```
cd "C:\user path\p1\Videos\BBP2025\working"
```

### **Step 3: Check Python Environment**
Check if Python is added to your Windows PATH:
```cmd
python --version
```
If Python is not recognized, install Python 3.10 or 3.11 from [python.org](https://www.python.org/) and ensure you check **"Add Python to PATH"** during installation.

### **Step 4: Launch using `run.bat`**
Double-click `run.bat` in File Explorer, or execute it in Command Prompt:
```cmd
run.bat
```

Select your launch option:
- **`1` (Default)**: Normal launch.
- **`2` (Clean Reinstall)**: Automatically deletes and recreates the `.venv` folder to fix DLL errors.
- **`3` (Docker)**: Containerized launch.

### **Step 5: Access in Web Browser**
Open your web browser (Chrome, Edge, Firefox) and go to:
👉 **`http://localhost:8501`**

---

## 🚀 How to Use the Application Workflow

The application consists of 4 main step-by-step workflow rooms displayed in the top navigation bar:

```
[ 1. Load Video ]  ➔  [ 2. Calibrate ]  ➔  [ 3. Track Room ]  ➔  [ 4. Analysis ]
```

---

### Step 1: Setup & Load Video
1. Select how many video slots you want to track concurrently (1 to 4 videos).
2. Choose your input source:
   - **Upload Video File**: Upload an `.mp4`, `.avi`, or `.mov` file.
   - **Local Directory / Default Video**: Select pre-existing arena video files.
   - **Live Camera Feed**: Check "📷 Use Live Camera Feed" if connected to a real-time arena camera.
3. Click **"Next: Calibrate arena →"**.

---

### Step 2: Calibrate Arena Circles
To measure real-world distances in millimeters ($mm$), you need to mark 9 calibration points:

1. **Outer Rim (Points 1 to 4)**: Click 4 points along the outer circle rim of the arena.
2. **Inner Rim (Points 5 to 8)**: Click 4 points along the inner feeder circle rim.
3. **Hive Entry (Point 9)**: Click 1 final point where the bee enters/exits to the hive.

*Once 9 points are clicked, yellow and cyan fitted arena overlay circles automatically appear on the canvas.*
Click **"Next: Start tracking room →"**.

---

### Step 3: Tracking & Tagging Room
The tracking room displays interactive video controls, full arena auto-tracking, and manual tagging tools:

#### **🎯 Full Arena Auto-Tracking Mode (Toggle)**:
- Enable **"🎯 Auto-Track Full Arena"** in `⚙️ Tracking Config`:
  - Automatically detects and tracks the bee across the entire arena right away on live feeds or recorded videos without requiring an initial Entry tag first.
  - Users can retroactively set an **Entry tag**, **Exit tag**, or **End tag** at any frame afterwards.

#### **🏷️ Manual Tagging Tools**:
- 🟢 **Entry tag** (`E` key): Mark where the bee enters the arena.
- 🟡 **Help tag** (`H` key): If the bee is hidden, flies fast, or tracker loses focus, press `H` (or click Help tag) and click directly on the bee to re-acquire instant 0-ms tracking.
- 🔴 **Exit tag** (`X` key): Mark where the bee exits the arena.
- ⏹ **End tag** (`S` key): Terminate analysis tracking boundary.

#### **⌨️ Cross-Platform Keyboard Shortcuts (Mac & Windows)**:
| Action | Key / Hotkey | Description |
| :--- | :--- | :--- |
| **Play / Pause** | `Spacebar` | Start or pause video playback / live tracking |
| **Entry Tag** | `E` | Activate Entry tag mode |
| **Help Tag** | `H` | Activate Help tag mode (re-acquire bee) |
| **Exit Tag** | `X` | Activate Exit tag mode |
| **End Tag** | `S` | Activate End tag mode |
| **Step Backward** | `Cmd + ◀` / `Ctrl + ◀` / `◀` | Step backward 1 stride / 5 seconds |
| **Step Forward** | `Cmd + ▶` / `Ctrl + ▶` / `▶` / `Shift + D` | Step forward 1 stride / 5 seconds |

---

### Step 4: Analysis & Data Export
Once tracking is complete, switch to **Tab 4 (Analysis)**:

1. **Trajectory & Path Plots**: View spatial heatmap plots, distance from feeder, velocity, and exit bearing angles.
2. **Outcome Summary**: Review total time on feeder, path length ($cm$), and trial outcome ("Went back" vs "Still in arena").
3. **Data Downloads**:
   - 📥 **Download Results Folder (ZIP)**: Download all session CSVs, high-res trajectory plots, and raw data archives.
   - 📄 **Download CSV**: Export raw per-frame coordinate table containing frame number, timestamp ($s$), $x/y$ coordinates ($mm$), velocity ($mm/s$), and feeder status.

---

## 🛠 Troubleshooting & FAQ

### ❓ **Help Tag or Interface Freezing on Laptops**
- **Cause**: On some laptops, background PyTorch model training spammed CPU thread pools during interactive tagging.
- **Fix**: Automatic background PyTorch training has been isolated. Help tag clicks now execute instant (0 ms) OpenCV re-acquisition without thread locking. If you wish to fine-tune YOLO models, use the dedicated **"🚀 Fine-Tune YOLO Model"** button in Tab 4.

### 3. ❓ **Windows Path Length Error (`[WinError 206] The filename or extension is too long`)**
- **Cause**: Windows legacy limits file path lengths to 260 characters (`MAX_PATH`). When the project is stored inside deep nested folders (e.g. `C:\Users\Name\Downloads\Folder\Subfolder\bee-tracking\.venv\Lib\site-packages\torch\lib`), Windows blocks C-extension DLL loading.
- **Fix (Option A - Recommended)**: Move or copy the project folder to a shorter path (e.g., `C:\bee-tracking` or `D:\bee-tracking`).
- **Fix (Option B)**: Enable Windows 10/11 Long Paths:
  1. Press `Win + R`, type `regedit` and press `Enter`.
  2. Navigate to: `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem`
  3. Set `LongPathsEnabled` to `1`.

### 4. ❓ **Windows DLL Initialization Error (`c10.dll` / `WinError 1114`)**
- **Cause**: Missing Microsoft Visual C++ Redistributable on Windows.
- **Fix**:
  1. Download and install **Visual C++ Redistributable 2015–2022** from Microsoft:  
     👉 [https://aka.ms/vs/17/release/vc_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe)
  2. Launch `run.bat` and select **Option 2** (Clean Reinstall Environment).

### 5. ❓ **Camera Feed Warning in macOS Terminal**
- **Notice**: `AVCaptureDeviceTypeExternal is deprecated for Continuity Cameras`.
- **Explanation**: This is a harmless system log notice from macOS AVFoundation. OpenCV logging levels have been set to `OFF` in `app.py` to prevent stdout clutter.

---
*Bee Arena Tracker — Built with Streamlit, OpenCV, and Python.*
