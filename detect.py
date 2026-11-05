"""Detect objects in an image or live webcam video.

Usage:
    python detect.py samples/cat.jpg                    # show the image with boxes
    python detect.py samples/cat.jpg --save out.jpg     # save instead of showing
    python detect.py samples/cat.jpg --json             # print detections as JSON
    python detect.py                                    # live webcam (press q to quit)
    python detect.py --confidence 0.6                   # only show confident detections
"""
from __future__ import annotations

import argparse
import json
import sys

import cv2

from detector import ObjectDetector, draw


def run_image(detector: ObjectDetector, path: str, confidence: float, save: str | None, as_json: bool) -> None:
    image = cv2.imread(path)
    if image is None:
        sys.exit(f"Could not read image: {path}")
    detections = detector.detect(image, confidence)
    if as_json:
        print(json.dumps([d.to_dict() for d in detections], indent=2))
        return
    for d in detections:
        print(f"{d.label:<15} {d.confidence:.0%}")
    annotated = draw(image, detections)
    if save:
        cv2.imwrite(save, annotated)
        print(f"Saved {save}")
    else:
        cv2.imshow("Object Detection", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_webcam(detector: ObjectDetector, confidence: float, camera: int) -> None:
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        sys.exit("Could not open the webcam")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imshow("Object Detection (press q to quit)", draw(frame, detector.detect(frame, confidence)))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Detect objects with SSD MobileNet v3 (COCO).")
    parser.add_argument("image", nargs="?", help="Image file. Leave out to use the webcam.")
    parser.add_argument("--confidence", type=float, default=0.5, help="Minimum confidence, 0 to 1 (default 0.5)")
    parser.add_argument("--save", help="Save the annotated image to this path instead of showing it")
    parser.add_argument("--json", action="store_true", help="Print detections as JSON")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default 0)")
    args = parser.parse_args(argv)

    detector = ObjectDetector()
    if args.image:
        run_image(detector, args.image, args.confidence, args.save, args.json)
    else:
        run_webcam(detector, args.confidence, args.camera)


if __name__ == "__main__":
    main()
