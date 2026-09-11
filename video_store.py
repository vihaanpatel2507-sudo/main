import json
from pathlib import Path

from config import VIDEO_INDEX_PATH


def _load_index():
    path = Path(VIDEO_INDEX_PATH)
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def _save_index(index):
    with open(VIDEO_INDEX_PATH, "w") as f:
        json.dump(index, f, indent=2)


def register_video(name, video_path, csv_path, fps):
    index = _load_index()
    index[name] = {
        "video_path": str(video_path),
        "csv_path": str(csv_path),
        "fps": fps,
    }
    _save_index(index)


def list_videos():
    return _load_index()


def get_video(name):
    return _load_index().get(name)


def latest_video():
    index = _load_index()
    if not index:
        return None, None
    name = list(index.keys())[-1]
    return name, index[name]
