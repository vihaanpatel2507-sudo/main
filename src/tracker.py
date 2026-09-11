"""Simple nearest-neighbour centroid tracker.

Assigns stable IDs to detections across frames and keeps a motion trail for
each tracked object. Good enough for monitoring cameras with moderate crowd
levels (a handful of astronauts per camera view).
"""
from __future__ import annotations

import numpy as np


class CentroidTracker:
    def __init__(self, max_distance=90, max_lost=20, trail_len=40):
        """
        max_distance : max centroid distance (px) to match a detection to an existing track
        max_lost     : frames a track survives without a matching detection
        trail_len    : length of the stored motion trail per track
        """
        self.next_id = 1
        self.max_distance = max_distance
        self.max_lost = max_lost
        self.trail_len = trail_len
        self.objects = {}  # id -> {"cx","cy","bbox","lost","trail", ...extra}

    def update(self, detections):
        """
        detections: list of (cx, cy, bbox) or (cx, cy, bbox, extra_dict)
        returns   : {track_id: track_dict}
        """
        used = set()

        # 1) match existing tracks to the nearest detection
        for oid, obj in list(self.objects.items()):
            best, best_d = None, self.max_distance
            for i, d in enumerate(detections):
                if i in used:
                    continue
                dist = float(np.hypot(d[0] - obj["cx"], d[1] - obj["cy"]))
                if dist < best_d:
                    best, best_d = i, dist
            if best is None:
                obj["lost"] += 1
            else:
                used.add(best)
                d = detections[best]
                obj["cx"], obj["cy"], obj["bbox"] = d[0], d[1], d[2]
                obj["lost"] = 0
                if len(d) > 3 and d[3]:
                    obj.update(d[3])
                obj["trail"].append((int(obj["cx"]), int(obj["cy"])))
                if len(obj["trail"]) > self.trail_len:
                    obj["trail"].pop(0)

        # 2) unmatched detections -> new tracks
        for i, d in enumerate(detections):
            if i in used:
                continue
            extra = d[3] if len(d) > 3 and d[3] else {}
            self.objects[self.next_id] = {
                "cx": d[0], "cy": d[1], "bbox": d[2], "lost": 0,
                "trail": [(int(d[0]), int(d[1]))], **extra}
            self.next_id += 1

        # 3) delete stale tracks
        stale = [o for o, v in self.objects.items() if v["lost"] > self.max_lost]
        for oid in stale:
            del self.objects[oid]

        return self.objects
