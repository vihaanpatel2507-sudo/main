# 🚀 Space Station Monitor

ML project that watches space-station camera videos and detects **objects**
(astronauts, equipment) and **motion** (who is moving, where, how fast).
Built to be trained on your own videos of astronauts.

```
video ──► motion detection ──► tracking + speed ──► annotated video
      └─► YOLO object detection ─┘                 + events.jsonl
                                                   + summary.json
```

## ⚡ Fast start (2 minutes)

```powershell
cd space-station-monitor
.venv\Scripts\activate              # activate the project venv
python main.py demo                 # synthetic space clip -> tests everything
```

You get `outputs/demo_spacewalk_annotated.mp4` (green = calm motion,
red = fast motion, yellow trail = path traveled), plus JSON event logs.

## 📹 Analyze your real videos

Put your space-station videos into `data/videos/`, then:

```powershell
# full analysis: motion + objects (astronaut = "person" in the pretrained model)
python main.py analyze data/videos/your_video.mp4

# motion-only (fast, no PyTorch)
python main.py analyze data/videos/your_video.mp4 --no-objects

# watch it live while processing (press q to quit)
python main.py analyze data/videos/your_video.mp4 --show

# use a model you trained (see below)
python main.py analyze data/videos/your_video.mp4 --weights models/astronaut_yolo/weights/best.pt
```

Useful knobs: `--conf 0.4` (detection threshold), `--min-area 1500`
(ignore small motions), `--every 3` (object detection every Nth frame = faster),
`--max-frames 500` (quick test on a clip).

## 🏋️ Train YOLO on YOUR videos (~20 min of work)

**Step 1 — extract frames**
```powershell
python main.py frames data/videos/your_video.mp4 --every 20
```

**Step 2 — label them** (draw boxes around astronauts/equipment)
- Easiest: [Roboflow](https://roboflow.com) (web, free, exports YOLO format)
- Also good: [CVAT](https://www.cvat.ai), or LabelImg (desktop: `pip install labelImg`)

Export in **YOLO format**. Put the `.txt` label files into `data/labels/`
(one `<imagename>.txt` per image) and a `classes.txt` listing your class
names, one per line, e.g.:
```
astronaut
equipment
tool
```

**Step 3 — build the dataset**
```powershell
python main.py prepare            # creates data/dataset/ + data.yaml (80/20 split)
```

**Step 4 — train** (CPU: ~1-2 h for a few hundred images; on Google Colab's
free GPU: ~10 min)
```powershell
python main.py train --epochs 40 --imgsz 480
# Colab alternative: upload data/dataset/, then:
#   !yolo detect train data=data.yaml model=yolo11n.pt epochs=40 imgsz=480
```

**Step 5 — use your model**
```powershell
python main.py analyze data/videos/your_video.mp4 --weights models/astronaut_yolo/weights/best.pt
```

## 📦 What the outputs mean

| File | Content |
|---|---|
| `*_annotated.mp4` | video with boxes, IDs, speed labels, trails, HUD |
| `*_events.jsonl` | one JSON line per event: `{"type": "motion", "id": 3, "t": 12.5, "speed_px_s": 41.2, "bbox": [...]}` |
| `*_summary.json` | duration, avg activity, % time active, unique tracks, max speed |

Motion events are logged for every tracked blob moving faster than
`8 px/s` (tune in `SpaceStationMonitor.__init__` or `--min-area`).

## 🔧 Troubleshooting

- **`import torch` → DLL load failed**: your Visual C++ runtime is too old.
  Install the latest: https://aka.ms/vs/17/release/vc_redist.x64.exe
- **Too many false motion detections**: raise `--min-area` (e.g. 2000),
  or raise `var_threshold` in `src/motion.py` (default 40).
- **Astronaut not detected**: pretrained YOLO sees astronauts as `person`
  (class 0). It works best when the astronaut fills a decent part of the
  frame — train on your own videos for reliability (steps above).
- **CPU too slow for training**: use Google Colab (free GPU) —
  Runtime → Change runtime type → T4 GPU.

## 📁 Project structure

```
space-station-monitor/
├── main.py                  # CLI: demo / analyze / frames / prepare / train
├── src/
│   ├── motion.py            # MOG2 background subtraction motion detector
│   ├── detector.py          # YOLO object detector wrapper
│   ├── tracker.py           # centroid tracker (stable IDs + trails)
│   └── pipeline.py          # full pipeline: detect → track → log → render
├── data/
│   ├── videos/              # ← put your space station videos here
│   ├── frames/              # extracted frames for labeling
│   ├── labels/              # your YOLO .txt labels + classes.txt
│   └── dataset/             # generated YOLO dataset (train/val)
├── models/                  # trained weights land here
└── outputs/                 # annotated videos + events + summaries
```

## 🛰️ Ideas to extend

- Pose estimation (MediaPipe is already installed) for astronaut activity
  recognition (floating / working / exercising / sleeping)
- Alert rules: "no motion in module for X minutes", "person near hatch"
- Multiple camera support with per-camera configs
- Stream input instead of file (cv2.VideoCapture RTSP URL)

