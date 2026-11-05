"""Object detection with OpenCV's DNN module and a pretrained SSD MobileNet v3 (COCO).

The model returns COCO class ids 1-90. That id space has gaps (for example there is
no id 12), so labels.txt has 90 lines with "N/A" in the unused slots. Indexing an
80-name list instead shifts every label after id 11 (a cat comes out as "dog").
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
MODEL_FILE = HERE / "frozen_inference_graph.pb"
CONFIG_FILE = HERE / "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
LABELS_FILE = HERE / "labels.txt"

INPUT_SIZE = (320, 320)  # what SSD MobileNet v3 was trained on
BOX_COLOR = (229, 70, 79)  # BGR
TEXT_COLOR = (255, 255, 255)


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]  # x, y, width, height in pixels

    def to_dict(self) -> dict:
        return {"label": self.label, "confidence": round(self.confidence, 3), "box": list(self.box)}


def load_labels(path: Path = LABELS_FILE) -> list[str]:
    return path.read_text().strip().split("\n")


class ObjectDetector:
    def __init__(self, model: Path = MODEL_FILE, config: Path = CONFIG_FILE, labels: Path = LABELS_FILE):
        self.labels = load_labels(labels)
        self.model = cv2.dnn_DetectionModel(str(model), str(config))
        self.model.setInputSize(*INPUT_SIZE)
        self.model.setInputScale(1.0 / 127.5)           # scale pixels to [-1, 1] ...
        self.model.setInputMean((127.5, 127.5, 127.5))  # ... after subtracting the mean
        self.model.setInputSwapRB(True)                 # OpenCV images are BGR, the model expects RGB

    def label_for(self, class_id: int) -> str:
        if 1 <= class_id <= len(self.labels):
            return self.labels[class_id - 1]
        return "unknown"

    def detect(self, image: np.ndarray, min_confidence: float = 0.5, nms_threshold: float = 0.4) -> list[Detection]:
        """Return detections above min_confidence, highest confidence first."""
        if image is None or image.size == 0:
            raise ValueError("Empty image")
        class_ids, confidences, boxes = self.model.detect(image, confThreshold=min_confidence, nmsThreshold=nms_threshold)
        if len(class_ids) == 0:
            return []
        results = [
            Detection(self.label_for(int(cid)), float(conf), tuple(int(v) for v in box))
            for cid, conf, box in zip(np.array(class_ids).flatten(), np.array(confidences).flatten(), boxes)
        ]
        results = [d for d in results if d.label != "N/A"]
        return sorted(results, key=lambda d: d.confidence, reverse=True)


def draw(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Return a copy of the image with a labeled box for each detection."""
    out = image.copy()
    for d in detections:
        x, y, w, h = d.box
        cv2.rectangle(out, (x, y), (x + w, y + h), BOX_COLOR, 2)
        text = f"{d.label} {d.confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        top = max(y, th + 8)
        cv2.rectangle(out, (x, top - th - 8), (x + tw + 8, top), BOX_COLOR, -1)
        cv2.putText(out, text, (x + 4, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2, cv2.LINE_AA)
    return out
