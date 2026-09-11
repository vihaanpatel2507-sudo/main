# 📘 How This Project Works — Full Explanation

This document explains what the code does, how it works inside, and what
the results of `clip1.mp4` mean.

---

## 1. The Big Picture

The project watches a video the same way a human operator would:

```
                 ┌────────────────────────────────────────────────┐
 VIDEO ──frames──►                                                │
                 │  1) MOTION DETECTION   (every frame)           │
                 │     "Something moved here" → box + position    │
                 │                                                │
                 │  2) OBJECT DETECTION   (every 3rd frame)       │
                 │     "That thing is a PERSON (astronaut)" → box │
                 │                                                │
                 │  3) TRACKING           (every frame)           │
                 │     "It's the same one as before" → ID + trail │
                 │     "It moved X pixels since last frame"       │
                 │                  → SPEED                       │
                 │                                                │
                 │  4) OUTPUT                                     │
                 │     • annotated video  (boxes, IDs, trails)    │
                 │     • events.jsonl     (machine-readable log)  │
                 │     • summary.json     (statistics)            │
                 └────────────────────────────────────────────────┘
```

Motion detection finds **that** something moved. Object detection tells you
**what** it is. Tracking connects frames together so you get **who, where,
how fast**.

---

## 2. The Files and What Each One Does

### `src/motion.py` — MotionDetector (finds moving pixels)
**Method:** MOG2 Background Subtraction (Mixture of Gaussians).

- The camera on a space station is **fixed**, so most of the picture never
  changes (walls, panels, racks). The algorithm builds a statistical model
  of this "background" from the last ~300 frames (`history=300`).
- Every new frame is compared to that model, pixel by pixel. If a pixel is
  too different from the background (`varThreshold=40`), it is marked as
  **foreground** = motion.
- Because lighting slowly changes on the ISS, the background model keeps
  **adapting** — that is why a light turning on does not create permanent
  false motion.
- MOG2 also marks **shadows** (value 127). We throw those away and keep
  only solid motion (value 255).
- Then morphological cleanup (open/close/dilate) removes salt-and-pepper
  noise and fills the moving shape into one solid blob.
- Finally, contours are found. Blobs smaller than `min_area` (800 px) are
  ignored as noise. Result: a list of **motion blobs**, each with a box
  `(x, y, w, h)`, a pixel `area`, and a `centroid` (center point).

### `src/detector.py` — ObjectDetector (names the objects)
**Method:** YOLO11-nano neural network (Ultralytics), running on CPU.

- Every 3rd frame, the frame is passed to YOLO, which slides a neural net
  over the image and returns boxes + class names + confidence.
- The pretrained model was trained on everyday photos (COCO dataset), so it
  knows 80 classes like `person`, `bottle`, `laptop`. An astronaut in a
  white suit is detected as `person`.
- `conf=0.30` = only keep detections that are at least 30% sure.
- `imgsz=480` = the frame is shrunk to 480 px before the network (much
  faster on CPU, still accurate enough).

### `src/tracker.py` — CentroidTracker (gives IDs, connects frames)
**Method:** nearest-neighbour matching of center points.

- YOLO/motion detection alone has no memory — every frame it would say
  "person, person, person..." with no idea if it's the same person.
- The tracker remembers each object's center from the last frame. A new
  detection is matched to the closest remembered center (max 90 px away).
  Match → same ID continues. No match for 15–40 frames → ID is retired.
- Each track keeps a **trail** (last 40 positions) — that is the yellow
  path you see drawn in the output video.

### `src/pipeline.py` — SpaceStationMonitor (the conductor)
Runs every frame of the video in this order:

1. `MotionDetector.detect(frame)` → blobs + a motion mask.
   **Activity level** = moving pixels ÷ total pixels.
2. `motion_tracker.update(...)` → stable IDs for motion blobs.
3. Speed: distance between the same ID's center now vs. last frame,
   divided by frame time (1/fps) → **pixels/second**, smoothed with a
   moving average (70% old + 30% new) so it doesn't jitter.
4. Every 3rd frame: `ObjectDetector.detect(frame)` → named objects,
   tracked with a second tracker (IDs last longer: 40 frames).
5. Drawing: green box = motion (calm), **red box = motion faster than
   8 px/s**, cyan box = YOLO object, yellow line = path traveled, black
   HUD bar on top with time / activity level / counts.
6. Writing: annotated frame → video, events → `events.jsonl`.


### `main.py` — the command menu
| Command | What it does |
|---|---|
| `demo` | builds a fake space clip and tests the whole pipeline |
| `analyze <video>` | runs the full pipeline on one video |
| `frames <video>` | saves every Nth frame as JPG for labeling |
| `prepare` | builds the YOLO training dataset from frames + labels |
| `train` | fine-tunes YOLO on YOUR labeled frames |

### `run_all.py` — batch mode
Finds every `clip*.mp4` in `data/videos/`, loads YOLO **once** (loading the
model per video would waste ~5 s each), then processes the videos one after
another. One video failing does not stop the rest.

---

## 3. The Output Files Explained

### `<name>_annotated.mp4`
| What you see | Meaning |
|---|---|
| 🟩 green box `M12 34px/s` | motion blob, ID 12, currently slow |
| 🟥 red box | motion blob moving fast (> 8 px/s) |
| 🟦 cyan box `person #7 0.87` | YOLO detected a person, 87% sure |
| 🟨 yellow line | the path that object traveled |
| top bar `t= 45.2s activity=HIGH motion=3 objects=2` | video time, how much of the frame is moving (LOW < 0.5%, MODERATE < 3%, HIGH ≥ 3%), how many tracked blobs/objects exist right now |

### `<name>_events.jsonl` — one JSON line per event
```json
{"type": "motion", "id": 1, "t": 0.33, "frame": 10,
 "bbox": [375, 209, 44, 141], "speed_px_s": 295.6, "area_px": 6204}
{"type": "object", "id": 7, "cls": "person", "conf": 0.925,
 "t": 1.5, "frame": 45, "bbox": [210, 88, 180, 320]}
```
- `motion` lines are written for every fast-moving tracked blob,
  every 5th frame (keeps the file small but complete).
- `object` lines are written the first time a new object ID appears.

### `<name>_summary.json` — the statistics

---

## 4. 📊 Results of clip1.mp4 — What Happened

**The video:** 521 seconds (8.7 min), 854×480 @ 30 fps → 15,622 frames.
**Processing time:** 689 s (~1.3× real time on your CPU).

| Metric | Value | What it means |
|---|---|---|
| `frames_processed` | 15,622 | whole video, no frames skipped |
| `avg_activity` | 0.1133 | on average **11.3% of the frame was moving** |
| `max_activity` | 1.0 | at some moment **the entire frame moved** → camera pan / scene cut / big lighting change |
| `active_time_pct` | 99.1% | motion existed **almost the entire video** |
| `unique_motion_tracks` | 4,155 | 4,155 tracked motion segments (see below ⚠️) |
| `unique_objects` | 43 classes | YOLO named many things (see below) |
| `max_speed_px_s` | 326.9 (summary) | fastest *sustained* motion; single spikes reached 1,957 px/s |

### Motion events (from events.jsonl)
- **38,827 motion events** logged.
- Average speed **167 px/s**, typical (median) **95 px/s** — people drifting
  and working, not statues and not flying.
- Average moving blob size **≈ 7,558 px** — roughly a torso-sized region at
  this resolution.

### ⚠️ Why 4,155 motion tracks (not ~4 astronauts)?
This number counts **track segments**, not people. On this footage almost
something is always moving (99% active time), so blobs constantly:
- **split** (an arm drifts away from the body → 1 blob becomes 2),
- **merge** (two astronauts overlap → 2 blobs become 1),
- **re-appear** after touching/overlapping.

Every split/merge/overlap gives the blob a new ID. That is normal for
centroid tracking on busy scenes. If you want fewer, longer-lived IDs:
- `--min-area 2000` (ignore small fragments)
- raise `max_lost` and `max_distance` in `pipeline.py`
- best fix: per-pixel object tracking on the YOLO boxes (upgrade idea)

### Why did YOLO see "refrigerator", "suitcase", "train"...?
The pretrained YOLO only knows 80 everyday-COCO classes. Space station
equipment looks *similar* to everyday things: white panels ↔ `refrigerator`
(48 hits), gear bags ↔ `suitcase` (81), cables/panels ↔ `tv` (12).
**91 `person` detections** (avg confidence 0.39, best 0.925) = real
astronauts found, but confidence is low because astronauts in white suits
at odd zero-g angles are outside the model's training data.

**This is exactly why training on YOUR videos matters** — after labeling
~200 frames and running `train`, the model gets classes like `astronaut`
and `equipment` with high confidence, and all the "train/sink/banana"
noise disappears.

### About the speed spikes (1,957 px/s)
When two blobs merge or a blob jumps (ID switch), the tracker's center
"teleports" for one frame → one huge speed sample. The summary's
`max_speed_px_s` (326.9) uses the smoothed speed and is trustworthy; the
raw 1,957 px/s spikes in events are artifacts, not real astronauts
(≈ 65 m/s at this scale!).


---

## 5. Verdict on clip1

✅ The pipeline worked correctly end to end:
- found motion almost everywhere (this clip is very busy),
- detected astronauts (`person`) at up to 92.5% confidence,
- measured sensible drift speeds (typical 95 px/s),
- logged 38,827 events you can query later.

⚠️ Two things to improve (both expected, both fixable):
1. **Track fragmentation** (4,155 IDs) — inherent to blob tracking on a
   busy scene; tune `min-area` / tracker distances, or upgrade to
   YOLO-box-based tracking after custom training.
2. **COCO class noise** (train, banana, hot dog...) — pretrained model
   mismatch; disappears after you train on your own labeled frames.

---

# 6. Tuning Cheat Sheet

| Problem | Fix |
|---|---|
| Too many small/false motion boxes | `--min-area 2000` (or 3000) |
| Missing slow/subtle motion | lower `var_threshold` to 25 in `pipeline.py` |
| Camera pan ruins everything | fixed cameras only; or lower `--min-area` + ignore high-activity frames |
| YOLO misses astronauts | train custom model (README §Training), then `--weights models/astronaut_yolo/weights/best.pt` |
| YOLO too slow | `--every 5` (detection every 5th frame) |
| Too many IDs | raise `max_lost` (tracker survives longer) and `max_distance` |
| Speeds look crazy | they are single-frame merge artifacts; filter `speed_px_s > 800` in analysis |

## 7. Quick Analysis Recipes

```python
# count how long each astronaut (object ID) was visible
import json, collections
ev = [json.loads(l) for l in open("outputs/clip1_events.jsonl")]
obj = [e for e in ev if e["type"] == "object"]
times = collections.defaultdict(list)
for e in obj:
    times[e["id"]].append(e["t"])
for oid, ts in times.items():
    print(f"object #{oid}: seen {min(ts):.1f}s → {max(ts):.1f}s ({max(ts)-min(ts):.1f}s)")
```

```python
# find the calmest moments (good for screenshots / inspection)
import json
ev = [json.loads(l) for l in open("outputs/clip1_events.jsonl")]
calm = sorted([e for e in ev if e["type"] == "motion"], key=lambda e: e["speed_px_s"])
print("calmest moments:", [(e["t"], e["speed_px_s"]) for e in calm[:5]])
```


