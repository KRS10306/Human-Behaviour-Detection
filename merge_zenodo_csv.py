import pandas as pd
import numpy as np
from feature_engine import normalize_skeleton, make_windows, extract_tsfel_features

# 1. Load your existing dataset
print("Loading existing dataset...")
df_main = pd.read_csv("pose_dataset_yolo26.csv")

# 2. Load the Zenodo dataset
print("Loading Zenodo data.csv...")
df_zenodo = pd.read_csv("data.csv") 

# 3. Filter for leaning classes
leaning_data = df_zenodo[df_zenodo['upperbody_label'].isin(['TLB', 'TLF', 'TLR', 'TLL'])]
print(f"Found {len(leaning_data)} leaning frames in Zenodo dataset.")

# Mapping from MediaPipe (Zenodo) to COCO (YOLO)
COCO_MAPPING = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle"
}

new_rows = []

# 4. Group by subject to keep consecutive frames together for motion analysis
for (subject, label), group in leaning_data.groupby(['subject', 'upperbody_label']):
    # Convert this group's frames into normalized skeletons
    normalized_skeletons = []
    
    for index, row in group.iterrows():
        skeleton = np.full((17, 3), np.nan)
        
        for coco_idx, mp_name in COCO_MAPPING.items():
            x_col = f"{mp_name}_x"
            y_col = f"{mp_name}_y"
            
            if x_col in row and y_col in row:
                skeleton[coco_idx, 0] = row[x_col]
                skeleton[coco_idx, 1] = row[y_col]
                skeleton[coco_idx, 2] = 1.0 # Confidence is 1.0 for ground truth
                
        norm = normalize_skeleton(skeleton)
        if norm is not None:
            normalized_skeletons.append(norm)
            
    # 5. Create 30-frame sliding windows (Stride 15)
    window_size = 30
    stride = 15
    
    if len(normalized_skeletons) >= window_size:
        for i in range(0, len(normalized_skeletons) - window_size + 1, stride):
            window = normalized_skeletons[i:i+window_size]
            
            # Extract TSFEL features
            window_df = make_windows(window, window=window_size, stride=window_size)[0]
            feat = extract_tsfel_features(window_df)
            
            # Add labels
            feat["label"] = 11  # 11 is "Leaning"
            feat["video_id"] = f"zenodo_{subject}_{label}_{i}"
            new_rows.append(feat)

# 6. Merge and save
if new_rows:
    df_new = pd.concat(new_rows, ignore_index=True)
    df_final = pd.concat([df_main, df_new], ignore_index=True)
    df_final = df_final.fillna(0)
    
    df_final.to_csv("pose_dataset_yolo26_updated.csv", index=False)
    print(f"\nSuccessfully added {len(new_rows)} leaning windows to the dataset!")
    print("Saved to: pose_dataset_yolo26_updated.csv")
else:
    print("\nNo leaning data was processed. Not enough consecutive frames.")