# Object Detection with OpenCV and SSD MobileNet v3

Detects and labels everyday objects (people, cars, dogs, cups and 76 other COCO classes) in **images and live webcam video**, using a pretrained **SSD MobileNet v3** model loaded through OpenCV's DNN module.

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)

## How it works

1. Loads the frozen TensorFlow graph (`frozen_inference_graph.pb`) and its config (`.pbtxt`) into `cv2.dnn_DetectionModel`.
2. Resizes each frame to 320×320, normalizes pixels to [-1, 1] and swaps BGR→RGB to match the model's training input.
3. Keeps detections above a 50% confidence threshold and draws a bounding box and COCO label for each one.

## Results

![Detections on a street scene: cars, a bus, people and a bicycle](https://github.com/user-attachments/assets/77826d02-d01c-4e37-b276-245aaa77fd75)

The same detection runs frame by frame on live webcam video (`python code.py`).

## Run it

```bash
pip install -r requirements.txt

python code.py path/to/photo.jpg   # detect objects in an image
python code.py                      # live detection from your webcam (press q to quit)
```

## Files

| File | Description |
|---|---|
| `code.py` | Detection script for images and webcam |
| `frozen_inference_graph.pb` | Pretrained SSD MobileNet v3 weights (COCO) |
| `ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt` | Model configuration |
| `labels.txt` | The 80 COCO class names |
