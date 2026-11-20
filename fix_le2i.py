import subprocess
import re
from pathlib import Path

# Path.cwd() automatically gets the folder where this script is running
PROJECT_ROOT = Path.cwd()
TARGET_ROOT = PROJECT_ROOT / "le2i_formatted"

TARGET_ROOT.mkdir(parents=True, exist_ok=True)

# Find all .avi files in all subfolders (case-insensitive)
avi_files = [f for f in PROJECT_ROOT.rglob("*") if f.is_file() and f.suffix.lower() == '.avi']
print(f"Found {len(avi_files)} .avi files in {PROJECT_ROOT}.")

if len(avi_files) == 0:
    print("\nStill finding 0 files. Please check:")
    print("1. Did you extract the Kaggle ZIP file?")
    print("2. Are the files actually .avi? (If they are .mp4, tell me and I will change the script)")

for avi in avi_files:
    # Figure out the scene name
    if "Videos" in avi.parts:
        idx = avi.parts.index("Videos")
        scene_name = avi.parts[idx - 1]
    else:
        scene_name = avi.parent.name

    # Extract the number from "video (1).avi"
    m = re.search(r'\((\d+)\)', avi.name)
    if not m:
        # If there are no parentheses, maybe it's just "video1.avi"
        m = re.search(r'(\d+)', avi.name)
        
    if not m:
        continue
    num = m.group(1)

    out_dir = TARGET_ROOT / scene_name
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"video_{num}.mp4"

    if out_path.exists():
        continue 

    print(f"Converting {scene_name}/{avi.name} -> video_{num}.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-i", str(avi),
        "-c:v", "libx264", "-crf", "23",
        "-c:a", "aac", str(out_path)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print(f"\nDone! Files are formatted in {TARGET_ROOT}")