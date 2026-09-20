"""
TASK 4: Real-Time Object Detection and Tracking
-------------------------------------------------
Pipeline:
    1. Real-time video input (webcam or video file) via OpenCV
    2. Object detection using a pre-trained YOLOv8 model
    3. Bounding boxes + labels drawn on each frame
    4. Multi-object tracking using DeepSORT (assigns persistent IDs)
    5. Live display of detections + tracking IDs

Usage:
    python object_tracking.py --source 0                # webcam
    python object_tracking.py --source video.mp4         # video file
    python object_tracking.py --source 0 --model yolov8n.pt --conf 0.4
    python object_tracking.py --source 0 --classes person car
    python object_tracking.py --source 0 --save output.mp4
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time Object Detection and Tracking")
    parser.add_argument(
        "--source", type=str, default="0",
        help="Video source: webcam index (e.g. 0) or path to a video file"
    )
    parser.add_argument(
        "--model", type=str, default="yolov8n.pt",
        help="Path/name of the YOLO model (yolov8n/s/m/l/x.pt). "
             "'n' (nano) is fastest and downloads automatically on first run."
    )
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument(
        "--classes", type=str, nargs="*", default=None,
        help="Optional list of class names to keep (e.g. --classes person car). "
             "Default: detect all classes the model knows."
    )
    parser.add_argument(
        "--max-age", type=int, default=30,
        help="DeepSORT: frames to keep a lost track alive before deleting it"
    )
    parser.add_argument("--save", type=str, default=None, help="Optional path to save the output video")
    parser.add_argument("--no-display", action="store_true", help="Disable the live preview window")
    return parser.parse_args()


def get_video_source(source_str: str):
    """Allow --source to be either an integer webcam index or a file path."""
    if source_str.isdigit():
        return int(source_str)
    return source_str


def color_for_id(track_id: int):
    """Deterministic, visually distinct BGR color per tracking ID."""
    np.random.seed(track_id * 37 + 11)
    return tuple(int(c) for c in np.random.randint(60, 255, size=3))


def main():
    args = parse_args()

    # ---------------------------------------------------------------
    # 1. Load pre-trained detector (YOLOv8) and DeepSORT tracker
    # ---------------------------------------------------------------
    print(f"[INFO] Loading YOLO model: {args.model}")
    model = YOLO(args.model)
    class_names = model.names  # dict: {class_id: class_name}

    wanted_class_ids = None
    if args.classes:
        wanted_class_ids = {
            cid for cid, name in class_names.items() if name.lower() in {c.lower() for c in args.classes}
        }
        if not wanted_class_ids:
            print(f"[WARN] None of {args.classes} matched model classes. Detecting all classes instead.")
            wanted_class_ids = None

    tracker = DeepSort(max_age=args.max_age, n_init=3, nms_max_overlap=1.0)

    # ---------------------------------------------------------------
    # 2. Set up real-time video input
    # ---------------------------------------------------------------
    source = get_video_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps_in, (width, height))
        print(f"[INFO] Saving output to: {args.save}")

    print("[INFO] Starting stream. Press 'q' to quit.")
    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of stream / cannot read frame.")
                break

            # -----------------------------------------------------
            # 3. Run detection on the frame
            # -----------------------------------------------------
            results = model.predict(frame, conf=args.conf, verbose=False)[0]

            detections = []  # format for DeepSORT: ([x, y, w, h], confidence, class_name)
            for box in results.boxes:
                cls_id = int(box.cls[0])
                if wanted_class_ids is not None and cls_id not in wanted_class_ids:
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w, h = x2 - x1, y2 - y1
                detections.append(([x1, y1, w, h], conf, class_names[cls_id]))

            # -----------------------------------------------------
            # 4. Update tracker with this frame's detections
            # -----------------------------------------------------
            tracks = tracker.update_tracks(detections, frame=frame)

            # -----------------------------------------------------
            # 5. Draw bounding boxes + labels + tracking IDs
            # -----------------------------------------------------
            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                class_name = track.get_det_class() or "object"
                l, t, r, b = map(int, track.to_ltrb())
                color = color_for_id(int(track_id) if str(track_id).isdigit() else hash(track_id) % 1000)

                cv2.rectangle(frame, (l, t), (r, b), color, 2)
                label = f"ID {track_id} | {class_name}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(frame, (l, t - th - 10), (l + tw + 6, t), color, -1)
                cv2.putText(frame, label, (l + 3, t - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (255, 255, 255), 2, cv2.LINE_AA)

            # -----------------------------------------------------
            # FPS overlay
            # -----------------------------------------------------
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2, cv2.LINE_AA)

            if writer:
                writer.write(frame)

            if not args.no_display:
                cv2.imshow("Object Detection & Tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] Quit requested by user.")
                    break

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print("[INFO] Stream closed. Resources released.")


if __name__ == "__main__":
    main()
