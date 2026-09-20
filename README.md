# Task 4: Real-Time Object Detection and Tracking

A complete pipeline for detecting and tracking objects in real time using a
webcam or a video file.

## Pipeline

| Step | Technology |
|---|---|
| Video input | OpenCV (`cv2.VideoCapture`) — webcam or video file |
| Object detection | Pre-trained **YOLOv8** (via `ultralytics`) |
| Bounding boxes + labels | Drawn per-frame with OpenCV |
| Object tracking | **DeepSORT** (via `deep-sort-realtime`) — assigns persistent IDs across frames |
| Output | Live window display, optional saved video file |

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

> The first time you run the script, `ultralytics` will automatically
> download the YOLOv8 weights (e.g. `yolov8n.pt`, ~6 MB) — no manual setup
> needed. Requires an internet connection on first run only.

## 2. Run it

**Webcam (default camera):**
```bash
python object_tracking.py --source 0
```

**Video file:**
```bash
python object_tracking.py --source path/to/video.mp4
```

**Save the output while viewing it:**
```bash
python object_tracking.py --source 0 --save output/tracked.mp4
```

**Only track specific classes (e.g. people and cars):**
```bash
python object_tracking.py --source 0 --classes person car
```

**Adjust detection confidence / tracker persistence:**
```bash
python object_tracking.py --source 0 --conf 0.5 --max-age 50
```

Press **`q`** at any time to quit the live window.

## Command-line options

| Flag | Default | Description |
|---|---|---|
| `--source` | `0` | Webcam index or path to a video file |
| `--model` | `yolov8n.pt` | YOLOv8 checkpoint (`n`=nano/fastest → `x`=largest/most accurate) |
| `--conf` | `0.35` | Minimum detection confidence to keep |
| `--classes` | *(all)* | Space-separated list of class names to detect (e.g. `person car dog`) |
| `--max-age` | `30` | Frames a track survives without a matching detection before being dropped |
| `--save` | *(none)* | Path to write the annotated output video (`.mp4`) |
| `--no-display` | off | Skip the live preview window (useful for headless/server runs) |

## How it works (matching the task steps)

1. **Video input** — `cv2.VideoCapture(source)` reads frames from a webcam
   index or a video file path.
2. **Pre-trained detection model** — `YOLO(args.model)` loads YOLOv8 weights
   pre-trained on the COCO dataset (80 common object classes: person, car,
   dog, chair, etc.).
3. **Per-frame processing** — every frame is passed to `model.predict()`,
   which returns bounding boxes, confidence scores, and class IDs.
4. **Tracking** — the raw detections are handed to `DeepSort.update_tracks()`,
   which uses motion prediction + an appearance embedding to match
   detections across frames and assign a stable `track_id` to each object,
   even through brief occlusions.
5. **Display** — boxes, class labels, and tracking IDs are drawn with
   OpenCV and shown live (`cv2.imshow`) with an FPS counter; optionally the
   annotated stream is also written to a video file.

## Notes / tips

- **CPU vs GPU:** Works on CPU out of the box (slower FPS). If you have a
  CUDA GPU with PyTorch installed with CUDA support, `ultralytics` will use
  it automatically for a large speed boost.
- **Bigger/more accurate model:** swap `--model yolov8n.pt` for
  `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, or `yolov8x.pt` (slower, more
  accurate).
- **Custom-trained YOLO model:** point `--model` at your own `best.pt` from
  a custom training run — the rest of the pipeline works unchanged.
- **SORT vs DeepSORT:** this script uses DeepSORT because it's more robust
  (uses appearance features, not just motion), but plain SORT is simpler if
  you want to implement it yourself — the tracker is the only part of the
  script you'd need to swap out (the detection loop stays identical).
