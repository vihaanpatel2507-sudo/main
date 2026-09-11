"""
batch_extract.py

Runs extract_motion_data.py on every video in a folder automatically.
Drop as many videos as you want into --video-dir, run this once.

Usage:
    python batch_extract.py --video-dir videos --out-dir outputs
"""

import argparse
import os
from extract_motion_data import extract_keypoints

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv")

def batch_process(video_dir, out_dir, **kwargs):
    os.makedirs(out_dir, exist_ok=True)

    videos = [f for f in os.listdir(video_dir) if f.lower().endswith(VIDEO_EXTENSIONS)]
    if not videos:
        print(f"No video files found in {video_dir}")
        return

    print(f"Found {len(videos)} video(s): {videos}")

    for video_file in videos:
        name = os.path.splitext(video_file)[0]
        video_path = os.path.join(video_dir, video_file)
        out_path = os.path.join(out_dir, f"{name}_keypoints.json")
        overlay_path = os.path.join(out_dir, f"{name}_skeleton.mp4")

        print(f"\n--- Processing {video_file} ---")
        extract_keypoints(
            video_path, out_path,
            overlay_video_path=overlay_path,
            **kwargs
        )

    print(f"\nDone. Processed {len(videos)} videos -> results in {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", default="videos", help="Folder containing input videos")
    parser.add_argument("--out-dir", default="outputs", help="Folder to save keypoint JSON + overlay videos")
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2])
    args = parser.parse_args()

    batch_process(
        args.video_dir, args.out_dir,
        frame_skip=args.frame_skip,
        model_complexity=args.model_complexity
    )
