"""Motion detection for fixed space-station cameras.

Uses MOG2 background subtraction (adapts to slow lighting changes), followed
by morphological cleanup and contour analysis to find moving blobs.
Works out of the box - no training required.
"""
from __future__ import annotations

import cv2
import numpy as np


class MotionDetector:
    """Finds moving regions (astronauts, tools, debris) in a fixed camera view."""

    def __init__(self, min_area=800, history=300, var_threshold=40,
                 learning_rate=None, shadow_value=200):
        """
        min_area      : ignore blobs smaller than this many pixels (noise filter)
        history       : number of frames used to model the background
        var_threshold : sensitivity of the Gaussian-mixture model (lower = more sensitive)
        """
        self.min_area = min_area
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=True)
        self.learning_rate = learning_rate
        self.shadow_value = shadow_value
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, frame: np.ndarray):
        """Return (blobs, fgmask).

        Each blob is a dict with:
          bbox     -> (x, y, w, h)
          area     -> pixel area of the moving region
          centroid -> (cx, cy)
        """
        fg = self.bg.apply(frame, learningRate=self.learning_rate)
        # Keep only definite foreground; drop shadow pixels (value 127)
        _, fg = cv2.threshold(fg, self.shadow_value - 1, 255, cv2.THRESH_BINARY)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.kernel)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        fg = cv2.dilate(fg, self.kernel, iterations=1)

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = fg.shape[:2]
        blobs = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            # Skip blobs covering (almost) the whole frame -> exposure flash / camera jump
            if bw > 0.95 * w and bh > 0.95 * h:
                continue
            m = cv2.moments(c)
            cx = int(m["m10"] / m["m00"]) if m["m00"] else x + bw // 2
            cy = int(m["m01"] / m["m00"]) if m["m00"] else y + bh // 2
            blobs.append({"bbox": (x, y, bw, bh), "area": float(area),
                          "centroid": (cx, cy)})
        return blobs, fg
