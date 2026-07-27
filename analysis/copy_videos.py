"""
Copy 49 video files from important/11 into their respective session folders under results/.
"""
import os
import glob
import shutil

video_dir = '/Users/yakkshit/Downloads/project/hiwi2/p1/Videos/BBP2025/important/11'
results_dir = 'results'

v_files = sorted(glob.glob(os.path.join(video_dir, '*.mp4')))
s_dirs = sorted([d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d)) and d != 'paper_plots'])

print(f"Starting video copy for {len(v_files)} files...")

copied_count = 0
for v in v_files:
    fname = os.path.basename(v)
    parts = fname.split('.')
    ts_key = parts[0].replace(' ', '_')
    
    matched_dir = None
    for sd in s_dirs:
        if sd.startswith(ts_key):
            matched_dir = sd
            break
            
    if matched_dir:
        dest_dir = os.path.join(results_dir, matched_dir)
        dest_file = os.path.join(dest_dir, fname)
        print(f"Copying {fname} -> {matched_dir}/")
        shutil.copy2(v, dest_file)
        copied_count += 1
    else:
        print(f"WARNING: Could not match video {fname}")

print(f"Successfully copied {copied_count}/{len(v_files)} video files into session folders!")
