"""Space Station Monitor - object & motion detection for astronaut cameras.

Quick start:
  python main.py demo                          # synthetic test + pipeline (motion-only, instant)
  python main.py analyze data/videos/my.mp4    # full analysis (needs ultralytics installed)
  python main.py analyze my.mp4 --no-objects   # motion-only, no PyTorch needed

Training workflow (train on YOUR videos):
  1) python main.py frames data/videos/my.mp4 --every 20    # extract frames
  2) label frames (Roboflow / CVAT / LabelImg) -> YOLO .txt in data/labels
  3) python main.py prepare                                 # build YOLO dataset
  4) python main.py train --epochs 40                       # fine-tune YOLO
  5) python main.py analyze my.mp4 --weights models/astronaut_yolo/weights/best.pt
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import random
import shutil
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import SpaceStationMonitor  # noqa: E402

DEFAULT_WEIGHTS = "yolo11n.pt"  # auto-downloads on first use (~6 MB)
DATASET_YAML = os.path.join("data", "dataset", "data.yaml")


# --------------------------------------------------------------------------- #
# demo
# --------------------------------------------------------------------------- #
def make_demo_video(path, seconds=12, fps=30, size=(640, 400)):
    """Synthetic clip: static equipment panels + one drifting astronaut + one tumbling tool."""
    W, H = size
    n = int(seconds * fps)
    rng = np.random.default_rng(7)

    bg = np.full((H, W, 3), (16, 20, 28), np.uint8)
    for k in range(6):  # static equipment panels
        x = 40 + k * 100
        cv2.rectangle(bg, (x, 40), (x + 70, 150), (40, 46, 60), -1)
        cv2.rectangle(bg, (x, 40), (x + 70, 150), (70, 80, 100), 1)
        for r in range(3):
            cv2.circle(bg, (x + 15 + r * 20, 95), 6, (90, 100, 120), -1)
    cv2.putText(bg, "ISS MODULE (SIM)", (12, H - 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (120, 130, 150), 1, cv2.LINE_AA)

    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(n):
        frame = np.clip(bg.astype(np.int16) + rng.normal(0, 3, bg.shape).astype(np.int16),
                        0, 255).astype(np.uint8)
        p = i / n
        speed = 3.0 if 0.33 < p < 0.66 else 1.0  # astronaut moves fast mid-clip
        cx = int(W / 2 + W * 0.34 * math.sin(2 * math.pi * p * speed))
        cy = int(H / 2 + H * 0.22 * math.sin(2 * math.pi * p * 0.7 + 1.2))
        angle = int(25 * math.sin(2 * math.pi * p * 0.5))
        cv2.ellipse(frame, (cx, cy), (30, 62), angle, 0, 360, (225, 230, 240), -1)
        cv2.circle(frame, (cx, cy - 58), 18, (245, 248, 252), -1)  # helmet
        cv2.rectangle(frame, (cx - 40, cy - 10), (cx - 22, cy + 30), (210, 215, 225), -1)
        cv2.rectangle(frame, (cx + 22, cy - 10), (cx + 40, cy + 30), (210, 215, 225), -1)
        tx, ty = W - 90, int(H - 120 + 18 * math.sin(2 * math.pi * p * 2.0))
        pts = cv2.boxPoints(((tx, ty), (26, 14), int(360 * p * 2))).astype(np.int32)
        cv2.fillPoly(frame, [pts], (160, 190, 220))  # tumbling tool
        writer.write(frame)
    writer.release()
    print(f"[demo] wrote {n} frames -> {path}")
    return path


def _load_detector(weights, conf, classes_arg, device):
    if os.path.exists(weights):
        class_ids = None  # custom-trained model: use all its classes
    else:
        class_ids = [0] if not classes_arg else classes_arg  # COCO class 0 = person
    from src.detector import ObjectDetector
    return ObjectDetector(weights=weights, conf=conf, class_ids=class_ids, device=device)


def cmd_demo(args):
    video = make_demo_video(os.path.join("data", "videos", "demo_spacewalk.mp4"),
                            seconds=args.seconds)
    detector = None
    if args.objects:
        detector = _load_detector(args.weights, args.conf, args.classes, args.device)
    SpaceStationMonitor(detector=detector, min_area=args.min_area).process(
        video, out_dir=args.out, show=args.show, detect_every=args.every)


def cmd_analyze(args):
    detector = None
    if not args.no_objects:
        detector = _load_detector(args.weights, args.conf, args.classes, args.device)
    monitor = SpaceStationMonitor(detector=detector, min_area=args.min_area)
    monitor.process(args.video, out_dir=args.out, show=args.show,
                    detect_every=args.every, max_frames=args.max_frames)


def cmd_frames(args):
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")
    stem = os.path.splitext(os.path.basename(args.video))[0]
    out = os.path.join(args.out, stem)
    os.makedirs(out, exist_ok=True)
    i = saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % args.every == 0:
            cv2.imwrite(os.path.join(out, f"{stem}_{i:06d}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            saved += 1
        i += 1
    print(f"[frames] saved {saved} frames -> {out}")
    print("Label them (Roboflow / CVAT / LabelImg), export YOLO format, and put")
    print("the .txt files into data/labels/ (classes.txt optional).")


# --------------------------------------------------------------------------- #
# prepare: build a YOLO-format dataset from frames + labels
# --------------------------------------------------------------------------- #
def cmd_prepare(args):
    frames = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        frames += glob.glob(os.path.join(args.frames, "**", ext), recursive=True)
    pairs = []
    for img in sorted(frames):
        stem = os.path.splitext(os.path.basename(img))[0]
        lbl = os.path.join(args.labels, stem + ".txt")
        if os.path.exists(lbl):
            pairs.append((img, lbl))
        else:
            print(f"[prepare] WARNING no label for {os.path.basename(img)} - skipped")
    if not pairs:
        raise SystemExit("No image+label pairs found. Label frames first (README step 2).")

    classes_file = os.path.join(args.labels, "classes.txt")
    if os.path.exists(classes_file):
        with open(classes_file, encoding="utf-8") as f:
            names = [l.strip() for l in f if l.strip()]
    else:
        names = ["astronaut", "equipment"]
        print("[prepare] no data/labels/classes.txt - using default names:", names)

    random.seed(42)
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * args.val_split))
    splits = {"val": pairs[:n_val], "train": pairs[n_val:]}

    for split, items in splits.items():
        img_dir = os.path.join(args.out, "images", split)
        lbl_dir = os.path.join(args.out, "labels", split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        for img, lbl in items:
            shutil.copy(img, os.path.join(img_dir, os.path.basename(img)))
            shutil.copy(lbl, os.path.join(lbl_dir, os.path.basename(lbl)))
        print(f"[prepare] {split}: {len(items)} images")

    with open(DATASET_YAML, "w", encoding="utf-8") as f:
        f.write(f"path: {os.path.abspath(args.out)}\n")
        f.write("train: images/train\nval: images/val\nnames:\n")
        for i, nm in enumerate(names):
            f.write(f"  {i}: {nm}\n")
    print(f"[prepare] wrote {DATASET_YAML}")
    print("[prepare] ready to train:  python main.py train")


# --------------------------------------------------------------------------- #
# train: fine-tune YOLO on your labeled videos
# --------------------------------------------------------------------------- #
def cmd_train(args):
    import json
    from ultralytics import YOLO
    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=args.device, workers=2,
                project="models", name="astronaut_yolo")
    best = os.path.join("models", "astronaut_yolo", "weights", "best.pt")

    # ---- accuracy report on the validation set ----
    print("\n[train] validating best model ...")
    metrics = model.val(data=args.data, imgsz=args.imgsz, device=args.device)
    m = metrics.results_dict
    report = {
        "model": os.path.abspath(best),
        "dataset": os.path.abspath(args.data),
        "epochs": args.epochs,
        "mAP50": round(float(m.get("metrics/mAP50(B)", 0)), 4),
        "mAP50-95": round(float(m.get("metrics/mAP50-95(B)", 0)), 4),
        "precision": round(float(m.get("metrics/precision(B)", 0)), 4),
        "recall": round(float(m.get("metrics/recall(B)", 0)), 4),
        "per_class": {},
    }
    try:
        names = metrics.names
        for i, cls_name in names.items():
            report["per_class"][cls_name] = {
                "mAP50": round(float(metrics.box.maps50[i]) if hasattr(metrics.box, "maps50") else 0, 4),
                "precision": round(float(metrics.box.p[i]), 4),
                "recall": round(float(metrics.box.r[i]), 4),
            }
    except Exception:
        pass

    rep_path = os.path.join("models", "accuracy_report.json")
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== TRAINING ACCURACY (validation set) ===")
    print(f"  mAP50        : {report['mAP50'] * 100:.1f}%   <- overall detection accuracy @ IoU 0.5")
    print(f"  mAP50-95     : {report['mAP50-95'] * 100:.1f}%   <- stricter accuracy (IoU 0.5..0.95)")
    print(f"  precision    : {report['precision'] * 100:.1f}%   <- of detections, how many were correct")
    print(f"  recall       : {report['recall'] * 100:.1f}%   <- of real objects, how many were found")
    for cls, v in report["per_class"].items():
        print(f"    - {cls:<12} mAP50 {v['mAP50'] * 100:5.1f}%  P {v['precision'] * 100:5.1f}%  R {v['recall'] * 100:5.1f}%")
    print(f"  full report  : {rep_path}")
    print("\n[train] done. Run your custom model with:")
    print(f"  python main.py analyze data/videos/<your video>.mp4 --weights {best}")


def cmd_eval(args):
    """Measure accuracy of a trained model on the validation dataset."""
    import json
    from ultralytics import YOLO
    model = YOLO(args.weights)
    metrics = model.val(data=args.data, imgsz=args.imgsz, device=args.device)
    m = metrics.results_dict
    report = {
        "model": os.path.abspath(args.weights),
        "dataset": os.path.abspath(args.data),
        "mAP50": round(float(m.get("metrics/mAP50(B)", 0)), 4),
        "mAP50-95": round(float(m.get("metrics/mAP50-95(B)", 0)), 4),
        "precision": round(float(m.get("metrics/precision(B)", 0)), 4),
        "recall": round(float(m.get("metrics/recall(B)", 0)), 4),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("\n=== MODEL ACCURACY (validation set) ===")
    print(f"  mAP50     : {report['mAP50'] * 100:.1f}%")
    print(f"  mAP50-95  : {report['mAP50-95'] * 100:.1f}%")
    print(f"  precision : {report['precision'] * 100:.1f}%")
    print(f"  recall    : {report['recall'] * 100:.1f}%")
    print(f"  saved to  : {args.out}")



# --------------------------------------------------------------------------- #
def _common_args(p):
    p.add_argument("--weights", default=DEFAULT_WEIGHTS)
    p.add_argument("--conf", type=float, default=0.30)
    p.add_argument("--classes", default=None, help="comma-separated class ids, e.g. 0")
    p.add_argument("--device", default="cpu", help="cpu or 0 for GPU")
    p.add_argument("--min-area", type=int, default=800)
    p.add_argument("--every", type=int, default=2, help="object detection every Nth frame")
    p.add_argument("--show", action="store_true", help="live preview window (q quits)")
    p.add_argument("--out", default="outputs")
    p.add_argument("--max-frames", type=int, default=None)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="generate a synthetic space clip and test the pipeline")
    d.add_argument("--seconds", type=int, default=12)
    d.add_argument("--objects", action="store_true", help="also run YOLO object detection")
    _common_args(d)
    d.set_defaults(func=cmd_demo)

    a = sub.add_parser("analyze", help="analyze a video (motion + objects)")
    a.add_argument("video")
    a.add_argument("--no-objects", action="store_true", help="motion-only (no PyTorch needed)")
    _common_args(a)
    a.set_defaults(func=cmd_analyze)

    f = sub.add_parser("frames", help="extract frames from a video for labeling")
    f.add_argument("video")
    f.add_argument("--every", type=int, default=20, help="save every Nth frame")
    f.add_argument("--out", default="data/frames")
    f.set_defaults(func=cmd_frames)

    p = sub.add_parser("prepare", help="build YOLO dataset from labeled frames")
    p.add_argument("--frames", default="data/frames")
    p.add_argument("--labels", default="data/labels")
    p.add_argument("--out", default="data/dataset")
    p.add_argument("--val-split", type=float, default=0.2)
    p.set_defaults(func=cmd_prepare)

    t = sub.add_parser("train", help="fine-tune YOLO on your dataset")
    t.add_argument("--data", default=DATASET_YAML)
    t.add_argument("--model", default=DEFAULT_WEIGHTS)
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--imgsz", type=int, default=480)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--device", default="cpu")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("eval", help="measure accuracy of a trained model")
    e.add_argument("--weights", default=os.path.join("models", "astronaut_yolo", "weights", "best.pt"))
    e.add_argument("--data", default=DATASET_YAML)
    e.add_argument("--imgsz", type=int, default=480)
    e.add_argument("--device", default="cpu")
    e.add_argument("--out", default=os.path.join("models", "accuracy_report.json"))
    e.set_defaults(func=cmd_eval)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

