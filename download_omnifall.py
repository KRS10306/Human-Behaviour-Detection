import omnifall
import os
import torch
import imageio
import cv2

# --- ADD THIS LINE ---
os.environ["OMNIFALL_VIDEO_ROOT__le2i"] = r"C:\Users\Rishit\OneDrive\Desktop\Project_college111\le2i_formatted"
# ---------------------

# 1. Load annotations
parts = omnifall.load("le2i-cs", split="train")
print(f"Total video segments available: {len(parts)}")

# 2. Subset (Load all segments)
subset = parts.select(range(len(parts)))

# 3. Resolve videos
dataset_with_videos = omnifall.add_video(subset)
print(f"Successfully loaded {len(dataset_with_videos)} video segments.")

# 4. Mapping for OMNIFALL (0-15) -> Project Folder Names
# Note: We do NOT use CLASS_MAP here because OmniFall outputs numbers, not folder names.
OMNIFALL_TO_PROJECT = {
    0: "Walking",
    1: "Fall_Down",
    2: "Fall_Down",   # fallen
    3: "Sit_down",
    4: "Sitting",
    5: "Lying_Down",  # lie_down
    6: "Lying_Down",  # lying
    7: "Stand_up",
    8: "Standing",
}

OUTPUT_DIR = "omnifall_le2i"
os.makedirs(OUTPUT_DIR, exist_ok=True)

saved, skipped, failed = 0, 0, 0

for i, example in enumerate(dataset_with_videos):
    video_data = example.get("video")
    
    if video_data is None:
        failed += 1
        continue

    label_id = example["label"] # This is an integer 0-15
    
    # Skip OmniFall labels that don't map to our 7 classes (e.g., 9-15)
    if label_id not in OMNIFALL_TO_PROJECT:
        skipped += 1
        continue

    class_name = OMNIFALL_TO_PROJECT[label_id]
    class_dir = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(class_dir, exist_ok=True)
    out_path = os.path.join(class_dir, f"clip_{i:05d}.mp4")

    # --- CASE 1: The data is a string (file path) ---
    if isinstance(video_data, str):
        cap = cv2.VideoCapture(video_data)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: 
            fps = 30
            
        start_sec = example.get("start", 0)
        end_sec = example.get("end", float('inf'))
        
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8)
        
        curr_frame = start_frame
        while cap.isOpened() and curr_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            writer.append_data(frame_rgb)
            curr_frame += 1
            
        writer.close()
        cap.release()
        saved += 1

    # --- CASE 2: The data is a PyTorch Tensor ---
    elif isinstance(video_data, torch.Tensor):
        video_tensor = video_data.permute(0, 2, 3, 1).to(torch.uint8)
        frames_np = video_tensor.cpu().numpy()
        writer = imageio.get_writer(out_path, fps=30, codec="libx264", quality=8)
        for frame in frames_np:
            writer.append_data(frame)
        writer.close()
        saved += 1
        
    else:
        failed += 1

    if (i + 1) % 50 == 0:
        print(f"Processed {i + 1}/{len(dataset_with_videos)} | saved={saved}, skipped={skipped}, failed={failed}")

print(f"\nDone. Saved {saved} clips to '{OUTPUT_DIR}'. Skipped {skipped}, failed {failed}.")