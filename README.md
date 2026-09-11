# NASA Video Pose Extraction Pipeline 🚀

Turns ordinary video clips into **structured human motion data**. For every frame of every video, the pipeline detects a person's 33 body joints (Google's BlazePose model), drops unreliable ones, smooths them over time, and saves the result as clean JSON — plus an MP4 with the skeleton drawn over the original footage.

It uses MediaPipe's current **Tasks API** (`mediapipe.tasks.python.vision.PoseLandmarker`) instead of the legacy `mp.solutions.pose`, which is broken/inconsistent across recent installs — so this works reliably on modern mediapipe + Python versions.

## Project structure

```
nasa videos/
├── videos/                     # Input clips (.mp4 / .mov / .avi / .mkv)
│   ├── clip1.mp4 ... clip5.mp4
├── outputs/                    # Results: *_keypoints.json + *_skeleton.mp4
├── logs/                       # Per-video logs from the last batch run
├── extract_motion_data.py      # Core: pose extraction from a single video
├── batch_extract.py            # Convenience: runs extraction on a whole folder
├── pose_landmarker_lite.task   # MediaPipe model — fast (complexity 0)
├── pose_landmarker_full.task   # MediaPipe model — balanced (complexity 1, default)
├── requirements.txt            # mediapipe, opencv-python, numpy
└── venv/                       # Python virtual environment
```

## How it works (the logic)

`extract_motion_data.py` runs this pipeline on every frame of the input video:

1. **Read & convert** — OpenCV (`VideoCapture`) reads the frame; it's converted BGR → RGB and wrapped as a MediaPipe `mp.Image` (SRGB). Frame timestamps are derived from `frame_id / fps`, which keeps the VIDEO running mode happy.

2. **Detect pose** — `PoseLandmarker` runs in `RunningMode.VIDEO` with `detect_for_video(...)`. Because it's video mode (not image mode), MediaPipe **tracks** the person between frames (`min_tracking_confidence`) instead of re-detecting from scratch every frame, giving temporal coherence.

3. **Filter weak keypoints** — any joint with `visibility < 0.5` (occluded, off-screen, or low confidence) is **dropped entirely** rather than kept as a noisy guess. That's why frames often contain fewer than all 33 joints (e.g. joints hidden behind equipment).

4. **Temporal smoothing** — `TemporalSmoother` keeps a sliding window (default 5 frames) of each joint's (x, y, z) and outputs the moving average, removing frame-to-frame jitter.

5. **Overlay drawing** — if `--overlay` is given, the skeleton (bone connections + joint dots) is drawn on the frame with OpenCV and encoded into a parallel MP4 via `VideoWriter`.

6. **Save & report** — all frame entries are dumped to JSON at the end, then stats are printed: frames saved, pose detection rate, overlay path.

### Coordinate system used in the output

| Field | Meaning |
|-------|---------|
| `x`, `y` | Normalized 0–1 coordinates relative to the frame width/height (multiply by pixel size to get pixels) |
| `z` | Depth relative to the hips — smaller (more negative) = closer to the camera |
| `visibility` | 0–1 confidence that the joint is actually visible in the frame |

The 33 BlazePose landmarks follow MediaPipe's fixed order: `nose`, eyes (inner/outer ×2), `ears`, `mouth`, `shoulders`, `elbows`, `wrists`, `pinky/index/thumb` ×2, `hips`, `knees`, `ankles`, `heels`, `foot_index`.

## Getting started

```powershell
# one-time setup
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

First run auto-downloads the pose model (a few MB, one-time — but the `lite` and `full` models are already included in this repo, and the default complexity is 1 = full, so no download is needed).

### Run a single video

```powershell
python extract_motion_data.py --video videos/clip2.mp4 --out outputs/clip2_keypoints.json --overlay outputs/clip2_skeleton.mp4
```

### Run a whole folder

`batch_extract.py` loops over every video in a folder and runs the extraction on each:

```powershell
python batch_extract.py --video-dir videos --out-dir outputs
```

### Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--frame-skip` | 1 | Process every Nth frame (2 = half the work, half the frame rate) |
| `--model-complexity` | 1 | 0 = lite/fast, 1 = full/balanced, 2 = heavy/most accurate (heavy downloads on first use) |
| `--min-detection-confidence` | 0.5 | Pose detection threshold |
| `--min-tracking-confidence` | 0.5 | Tracking threshold between frames |
| `--visibility-threshold` | 0.5 | Drop keypoints below this visibility score |
| `--smooth-window` | 5 | Frames averaged for temporal smoothing |

## Output format

`*_keypoints.json` — a list, one entry per frame (real sample from `clip2_keypoints.json`):

```json
[
  {
    "frame_id": 240,
    "timestamp": 8.008,
    "keypoints": [
      { "joint": "nose",           "x": 0.5253, "y": 0.4294, "z": -0.0734, "visibility": 0.9953 },
      { "joint": "left_eye_inner", "x": 0.5278, "y": 0.4295, "z": -0.0982, "visibility": 0.9974 },
      { "joint": "left_eye",       "x": 0.5301, "y": 0.4284, "z": -0.0982, "visibility": 0.9976 }
      // ... 25 joints in this frame
    ]
  }
  // ...
]
```

`*_skeleton.mp4` — the original video with the skeleton (green bones, red joints) drawn on top, encoded at `fps / frame_skip`.

## Results from the latest full run

| Clip | Resolution | FPS | Frames | Duration | Pose detected | JSON size |
|------|-----------|----:|-------:|---------:|--------------:|----------:|
| clip1 | 854×480   | 30  | 15,622 | 521 s | — (overlay only, JSON missing) | — |
| clip2 | 640×480   | 30  | 8,544  | 285 s | 5,807 frames (68.0%) | ~105 MB |
| clip3 | 1920×1080 | 30  | 2,852  | 95 s  | 1,180 frames (41.4%) | ~40 MB |
| clip4 | 1920×1080 | 59.9| 6,088  | 102 s | 5,851 frames (96.1%) | ~31 MB |
| clip5 | 640×478   | 30  | 10,767 | 359 s | 8,449 frames (78.5%) | ~62 MB |

- **clip4** is the cleanest subject (96.1% of frames had a detected pose).
- **clip3** is the weakest (41.4%) — typically caused by the subject being small/far from the camera, occlusion, or partially out of frame. Check `clip3_skeleton.mp4` to see which sections produced no skeleton.
- Frames with no detected pose still appear in the JSON (with empty `keypoints`), so frame indices/timestamps stay aligned across all clips.

## Notes & troubleshooting

- **Regenerate clip1's JSON** (its overlay exists but the JSON was missing after an earlier run):
  ```powershell
  python extract_motion_data.py --video videos/clip1.mp4 --out outputs/clip1_keypoints.json --overlay outputs/clip1_skeleton.mp4
  ```
- **Low detection rate?** Loosen the thresholds, e.g. `--min-detection-confidence 0.3 --visibility-threshold 0.3`, or use `--model-complexity 2` for the most accurate (slowest) model.
- **Mediapipe warnings** in the console (`inference_feedback_manager`, `landmark_projection_calculator ... NORM_RECT`) are harmless internal messages, not errors.
- Faster runs: `--frame-skip 2 --model-complexity 0` trades accuracy for speed (note the overlay is then encoded at `fps / frame_skip`, so it plays back at the same speed but with fewer analyzed frames).

