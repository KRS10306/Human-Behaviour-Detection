import time
import cv2
import pandas as pd
from pathlib import Path
from multiprocessing import Pool
from pose_extractor_yolo26 import YOLO26PoseExtractor, select_main_person
from feature_engine import normalize_skeleton, make_windows, extract_tsfel_features

CLASS_MAP = {
    "Lying_Down": 0,
    "Sit_down": 1,
    "Standing": 2,
    "Stand_up": 3,
    "Walking": 4,
    "Sitting": 5,
    "Fall_Down": 6,
    "Blowing_Nose": 7,
    "Cleaning": 8,
    "Eating": 9,
    "Watching_TV": 10,
    "No_Action": 11,
}

ROOT         = "dataset"
CACHE_DIR    = "cache_per_video"
FINAL_CSV    = "pose_dataset_yolo26.csv"
NUM_WORKERS  = 4          # lower to 2 if you have < 8 GB VRAM
BATCH_FRAMES = 16         # lower to 8 if you get out-of-memory errors


def process_video(args):
    video_path, label, cache_dir = args
    video_path = Path(video_path)
    cache_path = Path(cache_dir) / f"{video_path.stem}.csv"

    if cache_path.exists():
        df = pd.read_csv(cache_path)
        return (video_path.stem, str(cache_path), len(df), 0.0)

    t0 = time.time()
    extractor = YOLO26PoseExtractor("yolo26n-pose.pt")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return (video_path.stem, None, 0, time.time() - t0)

    normalized_frames = []
    frame_buf = []

    def flush():
        if not frame_buf:
            return
        batch_out = extractor.extract_batch(frame_buf, batch_size=BATCH_FRAMES)
        for persons in batch_out:
            person = select_main_person(persons)
            if person is None:
                continue
            norm = normalize_skeleton(person)
            if norm is not None:
                normalized_frames.append(norm)
        frame_buf.clear()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_buf.append(frame)
        if len(frame_buf) >= BATCH_FRAMES:
            flush()
    flush()
    cap.release()

    rows = []
    for wdf in make_windows(normalized_frames):
        feat = extract_tsfel_features(wdf)
        feat["label"] = label
        feat["video_id"] = video_path.stem
        rows.append(feat)

    if rows:
        df = pd.concat(rows, ignore_index=True)
        df.to_csv(cache_path, index=False)
        return (video_path.stem, str(cache_path), len(df), time.time() - t0)
    return (video_path.stem, None, 0, time.time() - t0)


def build_dataset():
    Path(CACHE_DIR).mkdir(exist_ok=True)

    tasks = []
    for class_dir in Path(ROOT).iterdir():
        if not class_dir.is_dir():
            continue
        label = CLASS_MAP.get(class_dir.name)
        if label is None:
            continue
        for video in class_dir.iterdir():
            if video.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
                tasks.append((str(video), label, CACHE_DIR))

    if not tasks:
        print("No videos found in dataset/")
        return

    cached = sum(Path(CACHE_DIR, f"{Path(t[0]).stem}.csv").exists() for t in tasks)
    print(f"Total videos: {len(tasks)} | Already cached: {cached}")

    t_start = time.time()
    with Pool(NUM_WORKERS) as pool:
        for i, result in enumerate(pool.imap_unordered(process_video, tasks), 1):
            vid, path, n, elapsed = result
            done, total = i, len(tasks)
            eta = (time.time() - t_start) / done * (total - done)
            print(f"[{done}/{total}] {vid}: {n} rows, {elapsed:.1f}s | ETA {eta/60:.1f} min")

    all_csvs = list(Path(CACHE_DIR).glob("*.csv"))
    if not all_csvs:
        print("No features extracted.")
        return
    combined = pd.concat([pd.read_csv(f) for f in all_csvs], ignore_index=True)
    combined.to_csv(FINAL_CSV, index=False)
    print(f"\nMerged {len(all_csvs)} videos -> {FINAL_CSV} ({combined.shape[0]} rows)")
    print(f"Total time: {(time.time() - t_start)/60:.1f} min")


if __name__ == "__main__":
    build_dataset()