import argparse
import csv
from pathlib import Path

import cv2

from config import (
    DETECTION_MODEL_PATH,
    DEVICE,
    ALERT_PIN_SECONDS,
    LOG_PANEL_WIDTH,
    LOG_VISIBLE_LINES,
    POSE_MODEL_PATH,
    REASONING_INTERVAL_FRAMES,
)
from cosmos_client import CosmosReasonerClient
from movement_logger import MovementLogger
from object_detection import ObjectDetector
from overlay import draw_log_panel
from pose_estimation import PoseEstimator
from text_utils import format_timestamp
from video_store import register_video


def export_telemetry(entries, output_path):
    csv_path = Path(output_path).with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "seconds", "event", "kind"])
        for entry in entries:
            writer.writerow(
                [
                    format_timestamp(entry["time"]),
                    f"{entry['time']:.2f}",
                    entry["text"],
                    entry.get("kind", "event"),
                ]
            )
    return csv_path


def run_pipeline(input_path, output_path, use_cosmos=True, mission="Unclassified"):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    pose_estimator = PoseEstimator(POSE_MODEL_PATH, device=DEVICE)
    object_detector = ObjectDetector(DETECTION_MODEL_PATH, device=DEVICE)
    cosmos_client = CosmosReasonerClient() if use_cosmos else None
    movement_logger = MovementLogger(fps)

    frame_idx = 0
    pinned_alert = None
    pinned_until_frame = -1
    seen_alert_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        pose_result = pose_estimator.infer(frame)
        keypoints_with_ids = pose_estimator.get_keypoints_with_ids(pose_result)
        frame = pose_estimator.draw(frame, keypoints_with_ids)

        for track_id, keypoints in keypoints_with_ids:
            movement_logger.update(frame_idx, track_id, keypoints)

        if len(movement_logger.alerts) > seen_alert_count:
            pinned_alert = movement_logger.latest_alert()
            pinned_until_frame = frame_idx + int(ALERT_PIN_SECONDS * fps)
            seen_alert_count = len(movement_logger.alerts)

        detection_result = object_detector.infer(frame)
        frame = object_detector.draw(frame, detection_result)

        if cosmos_client and frame_idx % REASONING_INTERVAL_FRAMES == 0:
            try:
                summary = cosmos_client.describe_interaction(frame)
            except Exception as exc:
                summary = f"Cosmos reasoning unavailable: {exc}"
            movement_logger.log_ai_summary(frame_idx, summary)

        active_banner = pinned_alert if frame_idx <= pinned_until_frame else None
        frame = draw_log_panel(
            frame,
            movement_logger.recent(LOG_VISIBLE_LINES),
            LOG_PANEL_WIDTH,
            pinned_alert=active_banner,
        )

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    duration_seconds = frame_idx / fps

    csv_path = export_telemetry(movement_logger.entries, output_path)
    video_name = Path(output_path).stem
    register_video(video_name, output_path, csv_path, fps, mission=mission, duration_seconds=duration_seconds)

    print(f"Telemetry log written to {csv_path}")
    print(f"Registered as '{video_name}' under mission '{mission}' - open the showcase site to view it.")


def main():
    parser = argparse.ArgumentParser(description="Astronaut movement/orientation detector")
    parser.add_argument("input", help="Path to input video")
    parser.add_argument("output", help="Path to write annotated output video")
    parser.add_argument("--no-cosmos", action="store_true", help="Skip Cosmos reasoning calls")
    parser.add_argument("--mission", default="Unclassified", help="Mission name to group this session under")
    args = parser.parse_args()

    run_pipeline(args.input, args.output, use_cosmos=not args.no_cosmos, mission=args.mission)


if __name__ == "__main__":
    main()