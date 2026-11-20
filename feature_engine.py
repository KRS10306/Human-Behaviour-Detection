import numpy as np
import pandas as pd
import tsfel

WINDOW = 30
STRIDE = 15
FPS = 30

def normalize_skeleton(kpts):
    """kpts: (17, 3) = [x, y, confidence]"""
    kpts = kpts.copy()
    shoulders = kpts[[5, 6], :2]   # left/right shoulder
    hips      = kpts[[11, 12], :2] # left/right hip

    if np.all(np.isnan(shoulders)) or np.all(np.isnan(hips)):
        return None

    hip_center = np.nanmean(hips, axis=0)
    shoulder_center = np.nanmean(shoulders, axis=0)
    torso = np.linalg.norm(shoulder_center - hip_center)
    if torso < 1e-6 or np.isnan(torso):
        return None

    kpts[:, 0] = (kpts[:, 0] - hip_center[0]) / torso
    kpts[:, 1] = (kpts[:, 1] - hip_center[1]) / torso
    return kpts

def make_windows(normalized_frames, window=WINDOW, stride=STRIDE):
    """Convert a list of (17,3) frames into a list of pandas DataFrames (one per window)."""
    windows = []
    for start in range(0, len(normalized_frames) - window + 1, stride):
        chunk = np.stack(normalized_frames[start:start + window], axis=0)
        flat = chunk.reshape(window, -1)  # (150, 51) -> 17 points * 3 values

        cols = []
        for i in range(17):
            cols += [f"kp{i}_x", f"kp{i}_y", f"kp{i}_c"]
        df = pd.DataFrame(flat, columns=cols)
        df = df.interpolate(limit_direction="both").fillna(0)
        windows.append(df)
    return windows

def extract_tsfel_features(window_df):
    """Extracts all features from a single window DataFrame."""
    cfg = tsfel.get_features_by_domain("all")
    return tsfel.time_series_features_extractor(cfg, window_df, fs=FPS, verbose=0)