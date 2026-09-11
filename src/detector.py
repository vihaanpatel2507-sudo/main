"""YOLO-based object detection (astronauts, equipment, ...).

Lazy-imports ultralytics so motion-only mode works even without PyTorch.
"""
from __future__ import annotations


class ObjectDetector:
    """Thin wrapper around Ultralytics YOLO for per-frame detection."""

    def __init__(self, weights="yolo11n.pt", conf=0.30, imgsz=480,
                 class_ids=None, device="cpu"):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise SystemExit(
                "ultralytics is not installed.\n"
                "  Install it:        py -m pip install ultralytics\n"
                "  Or run motion-only: python main.py analyze <video> --no-objects")
        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.class_ids = class_ids  # None = all classes in the model

    def detect(self, frame):
        """Return a list of dicts: bbox=(x, y, w, h), conf, cls (class name)."""
        results = self.model.predict(
            frame, conf=self.conf, imgsz=self.imgsz, device=self.device,
            classes=self.class_ids, verbose=False)
        out = []
        r = results[0]
        names = r.names
        for b in r.boxes:
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            out.append({
                "bbox": (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                "conf": float(b.conf[0]),
                "cls": names[int(b.cls[0])],
            })
        return out
