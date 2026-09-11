"""Full monitoring pipeline: motion detection + object detection + tracking.

Outputs (in outputs/):
  <name>_annotated.mp4  - video with boxes, IDs, motion trails and a HUD
  <name>_events.jsonl   - one JSON event per line (motion + object events)
  <name>_summary.json   - per-video statistics
"""
from __future__ import annotations

import cv2
import json
import os
import time

import numpy as np

from .motion import MotionDetector
from .tracker import CentroidTracker

GREEN = (60, 200, 60)
CYAN = (220, 220, 60)
YELLOW = (60, 200, 240)
RED = (60, 60, 240)
WHITE = (240, 240, 240)


class SpaceStationMonitor:
    """Combines motion detection, object detection and tracking into one pipeline."""

    def __init__(self, detector=None, min_area=800, var_threshold=40,
                 speed_threshold=8.0):
        """
        detector        : ObjectDetector instance, or None for motion-only mode
        speed_threshold : px/second above which a motion track counts as "moving"
        """
        self.detector = detector
        self.motion = MotionDetector(min_area=min_area, var_threshold=var_threshold)
        self.motion_tracker = CentroidTracker(max_distance=90, max_lost=15)
        self.obj_tracker = CentroidTracker(max_distance=110, max_lost=40)
        self.speed_threshold = speed_threshold

    # ------------------------------------------------------------------ #
    def process(self, video_path, out_dir="outputs", show=False,
                detect_every=2, max_frames=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        stem = os.path.splitext(os.path.basename(video_path))[0]
        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, f"{stem}_annotated.mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (width, height))
        if not writer.isOpened():  # codec fallback
            out_path = os.path.join(out_dir, f"{stem}_annotated.avi")
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"XVID"),
                                     fps, (width, height))

        events_path = os.path.join(out_dir, f"{stem}_events.jsonl")
        events = open(events_path, "w", encoding="utf-8")

        prev_pos = {}       # motion track id -> previous (cx, cy)
        smooth_speed = {}   # motion track id -> smoothed speed (px/s)
        activity = []       # per-frame moving-pixel fraction
        seen_motion = set()
        seen_objects = {}   # object track id -> class name
        n_frames = 0
        last_objects = []
        t0 = time.time()

        while True:
            ok, frame = cap.read()
            if not ok or (max_frames and n_frames >= max_frames):
                break

            # ---- 1) motion detection + tracking --------------------------
            blobs, fg = self.motion.detect(frame)
            act = float(np.count_nonzero(fg)) / fg.size
            activity.append(act)

            det = [(b["centroid"][0], b["centroid"][1], b["bbox"]) for b in blobs]
            tracks = self.motion_tracker.update(det)

            dt = 1.0 / fps
            for oid, tr in tracks.items():
                if oid in prev_pos:
                    d = float(np.hypot(tr["cx"] - prev_pos[oid][0],
                                       tr["cy"] - prev_pos[oid][1]))
                    inst = d / dt
                    s = smooth_speed.get(oid, inst)
                    smooth_speed[oid] = 0.7 * s + 0.3 * inst
                prev_pos[oid] = (tr["cx"], tr["cy"])
                seen_motion.add(oid)
                speed = smooth_speed.get(oid, 0.0)
                if speed > self.speed_threshold and n_frames % 5 == 0:
                    x, y, w, h = tr["bbox"]
                    events.write(json.dumps({
                        "type": "motion", "id": oid,
                        "t": round(n_frames / fps, 2), "frame": n_frames,
                        "bbox": [int(x), int(y), int(w), int(h)],
                        "speed_px_s": round(speed, 1), "area_px": int(w * h),
                    }) + "\n")

            # ---- 2) object detection (every k-th frame) ------------------
            otracks = {}
            if self.detector is not None and n_frames % max(1, detect_every) == 0:
                last_objects = self.detector.detect(frame)
            if last_objects:
                odet = [(o["bbox"][0] + o["bbox"][2] // 2,
                         o["bbox"][1] + o["bbox"][3] // 2,
                         o["bbox"], {"cls": o["cls"], "conf": o["conf"]})
                        for o in last_objects]
                otracks = self.obj_tracker.update(odet)
                for oid, tr in otracks.items():
                    if oid not in seen_objects:
                        seen_objects[oid] = tr.get("cls", "obj")
                        events.write(json.dumps({
                            "type": "object", "id": oid, "cls": tr.get("cls"),
                            "conf": round(tr.get("conf", 0.0), 3),
                            "t": round(n_frames / fps, 2), "frame": n_frames,
                            "bbox": [int(v) for v in tr["bbox"]],
                        }) + "\n")

            # ---- 3) annotate + write -------------------------------------
            self._draw(frame, tracks, smooth_speed, otracks, n_frames, fps, act)
            writer.write(frame)

            if show:
                cv2.imshow("Space Station Monitor", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            n_frames += 1
            if n_frames % 100 == 0:
                print(f"  ... {n_frames} frames  (activity={act:.4f})")

        cap.release()
        writer.release()
        events.close()
        if show:
            cv2.destroyAllWindows()

        summary = self._summary(video_path, n_frames, fps, activity,
                                seen_motion, seen_objects, smooth_speed)
        summary_path = os.path.join(out_dir, f"{stem}_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"\n[done] processed {n_frames} frames in {time.time() - t0:.1f}s")
        print(f"  annotated video : {out_path}")
        print(f"  events log      : {events_path}")
        print(f"  summary         : {summary_path}")
        print(json.dumps(summary, indent=2))
        return summary

    # ------------------------------------------------------------------ #
    def _summary(self, video_path, n_frames, fps, activity, seen_motion,
                 seen_objects, smooth_speed):
        act_arr = np.array(activity) if activity else np.zeros(1)
        class_counts = {}
        for cls in seen_objects.values():
            class_counts[cls] = class_counts.get(cls, 0) + 1
        return {
            "video": os.path.abspath(video_path),
            "frames_processed": n_frames,
            "duration_s": round(n_frames / fps, 1),
            "avg_activity": round(float(act_arr.mean()), 5),
            "max_activity": round(float(act_arr.max()), 5),
            "active_time_pct": round(100.0 * float((act_arr > 0.005).mean()), 1),
            "unique_motion_tracks": len(seen_motion),
            "unique_objects": class_counts,
            "max_speed_px_s": round(max(smooth_speed.values()), 1) if smooth_speed else 0.0,
        }

    # ------------------------------------------------------------------ #
    def _draw(self, frame, tracks, speeds, otracks, n, fps, act):
        # motion blobs
        for oid, tr in tracks.items():
            x, y, w, h = (int(v) for v in tr["bbox"])
            sp = speeds.get(oid, 0.0)
            color = RED if sp > self.speed_threshold else GREEN
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"M{oid} {sp:.0f}px/s", (x, max(15, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            self._trail(frame, tr, color)

        # detected objects
        for oid, tr in otracks.items():
            x, y, w, h = (int(v) for v in tr["bbox"])
            cv2.rectangle(frame, (x, y), (x + w, y + h), CYAN, 2)
            label = f"{tr.get('cls', 'obj')} #{oid}"
            if "conf" in tr:
                label += f" {tr['conf']:.2f}"
            cv2.putText(frame, label, (x, min(frame.shape[0] - 5, y + h + 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1, cv2.LINE_AA)
            self._trail(frame, tr, YELLOW)

        # HUD
        level = "LOW" if act < 0.005 else ("MODERATE" if act < 0.03 else "HIGH")
        hud = (f"t={n / fps:6.1f}s  frame={n}  activity={level:<8} "
               f"motion={len(tracks)}  objects={len(otracks)}")
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(frame, hud, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    WHITE, 1, cv2.LINE_AA)

    @staticmethod
    def _trail(frame, tr, color):
        pts = tr.get("trail") or []
        if len(pts) > 1:
            cv2.polylines(frame, [np.array(pts, np.int32)], False, color, 1,
                          cv2.LINE_AA)


