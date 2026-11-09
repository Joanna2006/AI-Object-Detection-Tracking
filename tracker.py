"""Self-contained SORT tracker (Kalman filter + Hungarian matching on IoU)."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_batch(a, b):
    xx1 = np.maximum(a[:, None, 0], b[None, :, 0]); yy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    xx2 = np.minimum(a[:, None, 2], b[None, :, 2]); yy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def to_z(b):
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w * h, w / (h + 1e-9)]).reshape(4, 1)


def to_box(x):
    s, r = x[2, 0], x[3, 0]
    w = np.sqrt(max(s * r, 1e-9)); h = s / (w + 1e-9)
    return np.array([x[0, 0] - w / 2, x[1, 0] - h / 2, x[0, 0] + w / 2, x[1, 0] + h / 2])


class KalmanBoxTracker:
    count = 0

    def __init__(self, det):
        self.F = np.eye(7); self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = 1
        self.H = np.eye(4, 7)
        self.R = np.diag([1., 1., 10., 10.])
        self.P = np.eye(7) * 10; self.P[4:, 4:] *= 1000
        self.Q = np.eye(7); self.Q[-1, -1] *= 0.01; self.Q[4:, 4:] *= 0.01
        self.x = np.zeros((7, 1)); self.x[:4] = to_z(det[:4])
        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        self.hits, self.since = 1, 0
        self.cls, self.conf = int(det[5]), float(det[4])

    def predict(self):
        if self.x[6, 0] + self.x[2, 0] <= 0:
            self.x[6, 0] = 0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.since += 1
        return to_box(self.x)

    def update(self, det):
        self.since, self.hits = 0, self.hits + 1
        self.cls, self.conf = int(det[5]), float(det[4])
        y = to_z(det[:4]) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P


class Sort:
    def __init__(self, max_age=20, min_hits=3, iou_threshold=0.3):
        self.max_age, self.min_hits, self.iou_thr = max_age, min_hits, iou_threshold
        self.trackers, self.frame = [], 0
        KalmanBoxTracker.count = 0

    def update(self, dets):
        """dets: Nx6 [x1,y1,x2,y2,conf,cls]. Returns Mx7 [x1,y1,x2,y2,id,cls,conf]."""
        self.frame += 1
        preds = np.array([t.predict() for t in self.trackers]).reshape(-1, 4)
        matched, unmatched = set(), set(range(len(dets)))
        if len(preds) and len(dets):
            iou = iou_batch(dets[:, :4], preds)
            rows, cols = linear_sum_assignment(-iou)
            for d, t in zip(rows, cols):
                if iou[d, t] >= self.iou_thr:
                    self.trackers[t].update(dets[d]); matched.add(d); unmatched.discard(d)
        for d in unmatched:
            self.trackers.append(KalmanBoxTracker(dets[d]))
        out = []
        for t in self.trackers:
            if t.since < 1 and (t.hits >= self.min_hits or self.frame <= self.min_hits):
                out.append([*to_box(t.x), t.id, t.cls, t.conf])
        self.trackers = [t for t in self.trackers if t.since <= self.max_age]
        return np.array(out).reshape(-1, 7)
