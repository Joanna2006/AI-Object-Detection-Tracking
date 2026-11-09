"""YOLOv8 detector wrapper (Ultralytics)."""
import numpy as np
from ultralytics import YOLO


class Detector:
    def __init__(self, weights="yolov8n.pt"):
        self.model = YOLO(weights)  # auto-downloads on first run
        self.names = self.model.names

    def detect(self, frame, conf=0.4, classes=None):
        """Returns Nx6 array: [x1, y1, x2, y2, confidence, class_id]."""
        res = self.model(frame, conf=conf, classes=classes, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return np.empty((0, 6))
        b = res.boxes
        return np.hstack([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None],
                          b.cls.cpu().numpy()[:, None]])
