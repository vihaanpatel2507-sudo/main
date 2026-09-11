import cv2
from ultralytics import YOLO

COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6), (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


class PoseEstimator:
    def __init__(self, model_path, device="mps", conf=0.4):
        self.model = YOLO(model_path)
        self.device = device
        self.conf = conf

    def infer(self, frame):
        results = self.model.track(frame, device=self.device, conf=self.conf, persist=True, verbose=False)
        return results[0]

    def get_keypoints_with_ids(self, result):
        if result.keypoints is None or result.boxes is None or len(result.boxes) == 0:
            return []
        keypoints = result.keypoints.xy.cpu().numpy()
        ids = result.boxes.id
        if ids is None:
            ids = list(range(len(keypoints)))
        else:
            ids = ids.cpu().numpy().astype(int).tolist()
        return list(zip(ids, keypoints))

    def draw(self, frame, keypoints_with_ids, color=(0, 220, 255)):
        for track_id, keypoints in keypoints_with_ids:
            for x, y in keypoints:
                if x > 0 and y > 0:
                    cv2.circle(frame, (int(x), int(y)), 3, color, -1)
            for a, b in COCO_SKELETON:
                xa, ya = keypoints[a]
                xb, yb = keypoints[b]
                if xa > 0 and ya > 0 and xb > 0 and yb > 0:
                    cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), color, 2)
            nose = keypoints[0]
            if nose[0] > 0 and nose[1] > 0:
                cv2.putText(
                    frame,
                    f"ID {track_id}",
                    (int(nose[0]) - 10, int(nose[1]) - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )
        return frame
