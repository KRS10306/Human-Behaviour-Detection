import numpy as np
import torch
from ultralytics import YOLO

class YOLO26PoseExtractor:
    # Change conf=0.25 to conf=0.60
    def __init__(self, model_path="yolo26n-pose.pt", conf=0.40, kpt_conf=0.4, device=None):
        self.model = YOLO(model_path)
        self.conf = conf
        self.kpt_conf = kpt_conf
        if device is None:
            device = 0 if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)
        print(f"[pose_extractor] Using device: {self.device}")

    def extract(self, frame):
        """Single-frame API (used by live inference)."""
        return self.extract_batch([frame])[0]

    def extract_batch(self, frames, batch_size=16):
        """Batched inference for offline dataset building."""
        if not frames:
            return []

        results = self.model.predict(
            frames,
            conf=self.conf,
            verbose=False,
            device=self.device,
            batch=batch_size,
        )

        out = []
        for r in results:
            if r.keypoints is None or r.keypoints.data is None:
                out.append([])
                continue
            kpts_data = r.keypoints.data.cpu().numpy()
            persons = []
            for kpts in kpts_data:
                kpts = kpts.copy()
                low = kpts[:, 2] < self.kpt_conf
                kpts[low, 0] = np.nan
                kpts[low, 1] = np.nan
                persons.append(kpts)
            out.append(persons)
        return out


def select_main_person(persons):
    if not persons:
        return None
    return max(persons, key=lambda k: np.nanmean(k[:, 2]))