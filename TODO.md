# TODO

## Fix preview video truncation & improve trajectory plot

### Step 1: Fix `generate_tracked_video()` call in Analysis tab
- Change the call to pass `coords_max` (last tracked frame) instead of `exit_frame`
- ✅ Done

### Step 2: Improve `plot_trajectory()` with clearer progressive gradient
- Change colormap from `viridis` to `turbo` for better visual progression
- Add time progress markers at regular intervals (every 25% of duration)
- Add a direction arrow at the path endpoint
- Make the line slightly thicker for better visibility
- ✅ Done

### Step 3: Add hive entry location to trajectory plot
- Accept `hive_entry_mm` parameter in `plot_trajectory()`
- Plot hive as green star marker with "Hive" label
- Convert hive_entry_point from pixel to mm in Analysis tab before calling plot
- ✅ Done

