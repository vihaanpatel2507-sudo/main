"""Run the monitoring pipeline on ALL videos in data/videos/ (skips *_annotated).

Loads YOLO once and reuses it for every video (much faster than running
main.py repeatedly). Usage:

    .venv\\Scripts\\python.exe run_all.py [--every 2] [--min-area 800]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import SpaceStationMonitor  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folder", default="data/videos")
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--every", type=int, default=2, help="object detection every Nth frame")
    ap.add_argument("--min-area", type=int, default=800)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    videos = [v for v in sorted(glob.glob(os.path.join(args.folder, "*.mp4"))
                                + glob.glob(os.path.join(args.folder, "*.avi"))
                                + glob.glob(os.path.join(args.folder, "*.mov")))
              if "_annotated" not in v
              and not os.path.basename(v).lower().startswith("demo")]
    if not videos:
        raise SystemExit(f"No videos found in {args.folder}")

    print(f"[run_all] {len(videos)} videos found:")
    for v in videos:
        print(f"  - {v}")

    # load YOLO once for all videos
    detector = None
    try:
        from src.detector import ObjectDetector
        detector = ObjectDetector(weights=args.weights, conf=args.conf,
                                  device=args.device)
        print("[run_all] object detection enabled (person/astronaut)")
    except SystemExit as e:
        print(f"[run_all] object detection disabled: {e}")

    results = []
    for i, video in enumerate(videos, 1):
        print(f"\n{'=' * 60}\n[{i}/{len(videos)}] {video}\n{'=' * 60}")
        try:
            monitor = SpaceStationMonitor(detector=detector, min_area=args.min_area)
            summary = monitor.process(video, out_dir="outputs", detect_every=args.every)
            results.append(summary)
        except Exception as exc:  # keep processing the remaining videos
            print(f"[run_all] FAILED {video}: {exc}")
            results.append({"video": os.path.abspath(video), "error": str(exc)})

    print(f"\n{'=' * 60}\nBATCH SUMMARY\n{'=' * 60}")
    for r in results:
        name = os.path.basename(r.get("video", "?"))
        if "error" in r:
            print(f"  {name:<28} ERROR: {r['error'][:50]}")
        else:
            print(f"  {name:<28} {r['duration_s']:>6}s  active {r['active_time_pct']:>5}%  "
                  f"tracks {r['unique_motion_tracks']:>3}  objects {r['unique_objects']}  "
                  f"max {r['max_speed_px_s']}px/s")


if __name__ == "__main__":
    main()
