

import argparse
import json
import os
import urllib.request
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

# Standard BlazePose 33-landmark order (fixed, documented format -- does not

LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner",
    "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left",
    "mouth_right", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index",
    "right_index", "left_thumb", "right_thumb", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel",
    "right_heel", "left_foot_index", "right_foot_index"
]

# Skeleton connections for drawing the overlay (index pairs into LANDMARK_NAMES)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32)
]

MODEL_URLS = {
    0: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    1: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    2: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}

#jitter 
class TemporalSmoother:
   
    def __init__(self, window=5):
        self.window = window
        self.history = {}

    def smooth(self, keypoints):
        smoothed = []
        for kp in keypoints:
            name = kp["joint"]
            if name not in self.history:
                self.history[name] = deque(maxlen=self.window)
            self.history[name].append((kp["x"], kp["y"], kp["z"]))
            arr = np.array(self.history[name])
            avg = arr.mean(axis=0)
            smoothed.append({
                "joint": name,
                "x": round(float(avg[0]), 4),
                "y": round(float(avg[1]), 4),
                "z": round(float(avg[2]), 4),
                "visibility": kp["visibility"]
            })
        return smoothed

#model download
def ensure_model(model_complexity):
    model_path = f"pose_landmarker_{['lite','full','heavy'][model_complexity]}.task"
    if not os.path.exists(model_path):
        print(f"Downloading pose model ({['lite','full','heavy'][model_complexity]}, one-time, needs internet)...")
        urllib.request.urlretrieve(MODEL_URLS[model_complexity], model_path)
        print("Model downloaded.")
    return model_path


def draw_skeleton(frame, landmarks_px):
    for a, b in POSE_CONNECTIONS:
        if a < len(landmarks_px) and b < len(landmarks_px):
            cv2.line(frame, landmarks_px[a], landmarks_px[b], (0, 255, 0), 2)
    for pt in landmarks_px:
        cv2.circle(frame, pt, 3, (0, 0, 255), -1)

# ai model inference
def extract_keypoints(video_path, out_path, frame_skip=1, overlay_video_path=None,
                       model_complexity=1, min_detection_confidence=0.5,
                       min_tracking_confidence=0.5, visibility_threshold=0.5,
                       smooth_window=5, use_cropping=True):
    model_path = ensure_model(model_complexity)

    base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    smoother = TemporalSmoother(window=smooth_window)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = None
    if overlay_video_path:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(overlay_video_path, fourcc, fps / frame_skip, (width, height))

    all_frames_data = []
    frame_id = 0
    saved_frame_id = 0
    low_confidence_frames = 0
#fram cut
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % frame_skip == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int((frame_id / fps) * 1000)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            frame_entry = {
                "frame_id": saved_frame_id,
                "timestamp": round(frame_id / fps, 3),
                "keypoints": []
            }

            if result.pose_landmarks:
                pose = result.pose_landmarks[0]  # first detected person
                raw_keypoints = []
                landmarks_px = []
                h, w = frame.shape[:2]

                for name, lm in zip(LANDMARK_NAMES, pose):
                    visibility = getattr(lm, "visibility", 1.0) or 1.0
                    if visibility < visibility_threshold:
                        continue
                    raw_keypoints.append({
                        "joint": name,
                        "x": round(lm.x, 4),
                        "y": round(lm.y, 4),
                        "z": round(lm.z, 4),
                        "visibility": round(visibility, 4)
                    })
                    landmarks_px.append((int(lm.x * w), int(lm.y * h)))

                frame_entry["keypoints"] = smoother.smooth(raw_keypoints)

                if writer:
                    draw_skeleton(frame, landmarks_px)
            else:
                low_confidence_frames += 1

            all_frames_data.append(frame_entry)
            saved_frame_id += 1

            if saved_frame_id % 30 == 0:
                print(f"  ...processed {saved_frame_id} frames so far "
                      f"(timestamp {round(frame_id / fps, 1)}s)", flush=True)

            if writer:
                writer.write(frame)

        frame_id += 1

    cap.release()
    if writer:
        writer.release()

    with open(out_path, "w") as f:
        json.dump(all_frames_data, f, indent=2)

    detected = saved_frame_id - low_confidence_frames
    print(f"Saved {len(all_frames_data)} frames of keypoint data to {out_path}")
    print(f"Pose detected in {detected}/{saved_frame_id} frames "
          f"({100 * detected / max(saved_frame_id,1):.1f}% detection rate)")
    if overlay_video_path:
        print(f"Saved skeleton overlay video to {overlay_video_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--out", default="output_keypoints.json", help="Path to save keypoint JSON")
    parser.add_argument("--overlay", default=None, help="Optional: path to save a video with skeleton drawn on top")
    parser.add_argument("--frame-skip", type=int, default=1, help="Process every Nth frame (1 = every frame, most accurate)")
    parser.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2],
                         help="0=lite/fast, 1=full/balanced, 2=heavy/most accurate")
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--visibility-threshold", type=float, default=0.5,
                         help="Drop keypoints below this visibility score instead of keeping noisy guesses")
    parser.add_argument("--smooth-window", type=int, default=5, help="Frames to average for temporal smoothing")
    parser.add_argument("--no-crop", action="store_true", help="(kept for compatibility, unused in Tasks API version)")
    args = parser.parse_args()

    extract_keypoints(
        args.video, args.out,
        frame_skip=args.frame_skip,
        overlay_video_path=args.overlay,
        model_complexity=args.model_complexity,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        visibility_threshold=args.visibility_threshold,
        smooth_window=args.smooth_window,
    )