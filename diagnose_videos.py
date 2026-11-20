import cv2
import numpy as np
from pathlib import Path
from pose_extractor_yolo26 import YOLO26PoseExtractor, select_main_person
from feature_engine import normalize_skeleton

extractor = YOLO26PoseExtractor("yolo26n-pose.pt")

for video in Path("dataset").rglob("*.avi"):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps

    detected = normalized = sampled = idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % 10 == 0:          # sample every 10th frame for speed
            sampled += 1
            persons = extractor.extract(frame)
            if persons:
                detected += 1
                person = select_main_person(persons)
                if person is not None and normalize_skeleton(person) is not None:
                    normalized += 1
        idx += 1
    cap.release()

    pct_det = 100 * detected / max(sampled, 1)
    pct_norm = 100 * normalized / max(sampled, 1)
    print(f"{video.parent.name}/{video.name}: "
          f"{duration:5.1f}s @ {fps:4.1f}fps, {total} frames, "
          f"detected {pct_det:5.1f}%, normalised {pct_norm:5.1f}%")