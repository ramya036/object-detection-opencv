"""Object detection with OpenCV DNN and a pretrained SSD MobileNet v3 (COCO).

Usage:
    python code.py path/to/image.jpg   # detect objects in an image
    python code.py                      # live detection from the webcam (press q to quit)
"""
import sys
from pathlib import Path

import cv2

HERE = Path(__file__).parent
CONFIG_FILE = HERE / "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
MODEL_FILE = HERE / "frozen_inference_graph.pb"
LABELS_FILE = HERE / "labels.txt"
CONFIDENCE = 0.5


def load_model():
    model = cv2.dnn_DetectionModel(str(MODEL_FILE), str(CONFIG_FILE))
    model.setInputSize(320, 320)
    model.setInputScale(1.0 / 127.5)          # scale pixels to [-1, 1] ...
    model.setInputMean((127.5, 127.5, 127.5))  # ... after subtracting the mean
    model.setInputSwapRB(True)                 # OpenCV is BGR, the model expects RGB
    labels = LABELS_FILE.read_text().strip().split("\n")
    return model, labels


def draw_detections(model, labels, frame):
    class_ids, confidences, boxes = model.detect(frame, confThreshold=CONFIDENCE)
    if len(class_ids) == 0:
        return frame
    for class_id, conf, box in zip(class_ids.flatten(), confidences.flatten(), boxes):
        if 1 <= class_id <= len(labels):
            label = f"{labels[class_id - 1]} {conf:.0%}"
            cv2.rectangle(frame, box, (255, 0, 0), 2)
            cv2.putText(frame, label, (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    return frame


def detect_image(model, labels, path):
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"Could not read image: {path}")
    cv2.imshow("Object Detection", draw_detections(model, labels, img))
    cv2.waitKey(0)


def detect_webcam(model, labels):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("Could not open the webcam")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imshow("Object Detection (press q to quit)", draw_detections(model, labels, frame))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()


if __name__ == "__main__":
    model, labels = load_model()
    if len(sys.argv) > 1:
        detect_image(model, labels, sys.argv[1])
    else:
        detect_webcam(model, labels)
    cv2.destroyAllWindows()
