import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import detect as cli
from detector import Detection, ObjectDetector, draw, load_labels

ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def detector():
    return ObjectDetector()


def test_labels_follow_coco_ids_with_gaps():
    labels = load_labels()
    assert len(labels) == 90
    assert labels[0] == "person"       # id 1
    assert labels[11] == "N/A"         # id 12 is unused in COCO
    assert labels[16] == "cat"         # id 17
    assert labels[46] == "cup"         # id 47
    assert labels[89] == "toothbrush"  # id 90
    assert len([l for l in labels if l != "N/A"]) == 80


def test_finds_the_cat(detector):
    found = detector.detect(cv2.imread(str(ROOT / "samples/cat.jpg")))
    assert found and found[0].label == "cat"
    assert found[0].confidence > 0.6


def test_finds_the_cup(detector):
    found = detector.detect(cv2.imread(str(ROOT / "samples/coffee.jpg")))
    assert "cup" in [d.label for d in found]


def test_blank_image_has_no_detections(detector):
    assert detector.detect(np.full((320, 320, 3), 127, np.uint8)) == []


def test_results_are_sorted_and_inside_the_image(detector):
    image = cv2.imread(str(ROOT / "samples/coffee.jpg"))
    found = detector.detect(image, min_confidence=0.3)
    assert [d.confidence for d in found] == sorted((d.confidence for d in found), reverse=True)
    h, w = image.shape[:2]
    for d in found:
        x, y, bw, bh = d.box
        assert 0 <= x < w and 0 <= y < h and bw > 0 and bh > 0


def test_higher_threshold_returns_fewer_results(detector):
    image = cv2.imread(str(ROOT / "samples/coffee.jpg"))
    assert len(detector.detect(image, 0.8)) <= len(detector.detect(image, 0.3))


def test_empty_image_is_rejected(detector):
    with pytest.raises(ValueError):
        detector.detect(np.zeros((0, 0, 3), np.uint8))


def test_draw_does_not_modify_input():
    image = np.zeros((100, 100, 3), np.uint8)
    out = draw(image, [Detection("cat", 0.9, (10, 10, 50, 50))])
    assert image.sum() == 0 and out.sum() > 0


def test_cli_json_output(capsys):
    cli.main([str(ROOT / "samples/cat.jpg"), "--json"])
    data = json.loads(capsys.readouterr().out)
    assert data[0]["label"] == "cat" and set(data[0]) == {"label", "confidence", "box"}


def test_cli_saves_annotated_image(tmp_path):
    out = tmp_path / "out.jpg"
    cli.main([str(ROOT / "samples/cat.jpg"), "--save", str(out)])
    assert cv2.imread(str(out)) is not None
