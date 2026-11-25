"""Test cases for the road-defect pipeline (run: pytest -v).
Uses the trained weights and two images from the validation split."""
import json
import os
import sys

import cv2
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from detector import RoadDefectDetector, classify_severity, confidence_reason, REVIEW_THRESHOLD  # noqa: E402
from ethics import generate_ethics_report  # noqa: E402
from explainability import generate_explanation  # noqa: E402
from report import generate_report  # noqa: E402

VAL = os.path.join(ROOT, "unified_dataset", "images", "val")
CRACK_IMG = os.path.join(VAL, "crack_seg_3541.rf.24cce469bb79e1afaa60e6bbf15e6bbe.jpg")
POTHOLE_IMG = os.path.join(VAL, "potholes_yolo_img-386.jpg")
WEIGHTS = os.path.join(ROOT, "runs", "segment", "runs", "road_defect_seg6", "weights", "best.pt")


@pytest.fixture(scope="module")
def detector(tmp_path_factory):
    log = tmp_path_factory.mktemp("audit") / "audit_log.jsonl"
    return RoadDefectDetector(model_path=WEIGHTS, audit_log_path=str(log))


# ---------- severity & reasoning rules ----------
@pytest.mark.parametrize("cls,area,expected", [
    ("crack", 100, "low"), ("crack", 300, "moderate"), ("crack", 1499, "moderate"), ("crack", 1500, "high"),
    ("pothole", 799, "low"), ("pothole", 2000, "moderate"), ("pothole", 5000, "high"),
])
def test_tc01_severity_thresholds(cls, area, expected):
    assert classify_severity(cls, area, 0.9)[0] == expected


def test_tc02_low_confidence_high_severity_needs_review():
    sev, reason = classify_severity("pothole", 9000, REVIEW_THRESHOLD - 0.1)
    assert sev == "high" and "human verification" in reason


def test_tc03_confidence_explanations():
    assert "(high)" in confidence_reason("crack", 0.9, 100)
    assert "(moderate)" in confidence_reason("crack", 0.6, 100)
    assert "(low)" in confidence_reason("crack", 0.3, 100)


# ---------- model inference ----------
def test_tc04_detects_crack(detector):
    dets = detector.detect(cv2.imread(CRACK_IMG), image_path=CRACK_IMG)
    assert dets and dets[0].class_name == "crack" and dets[0].confidence > 0.5
    assert dets[0].area_px > 0 and len(dets[0].mask_polygon) >= 3


def test_tc05_detects_pothole(detector):
    dets = detector.detect(cv2.imread(POTHOLE_IMG), image_path=POTHOLE_IMG)
    assert any(d.class_name == "pothole" and d.confidence > 0.5 for d in dets)


def test_tc06_blank_road_gives_no_detections(detector):
    blank = np.full((416, 416, 3), 128, dtype=np.uint8)
    assert detector.detect(blank, image_path="blank.png") == []


def test_tc07_area_in_cm2_when_scale_given(detector):
    detector.px_per_cm = 10.0
    try:
        d = detector.detect(cv2.imread(POTHOLE_IMG))[0]
        assert d.area_cm2 == pytest.approx(d.area_px / 100.0)
    finally:
        detector.px_per_cm = None


def test_tc08_higher_threshold_never_adds_detections(detector):
    img = cv2.imread(POTHOLE_IMG)
    detector.conf_threshold = 0.10
    loose = len(detector.detect(img))
    detector.conf_threshold = 0.80
    strict = len(detector.detect(img))
    detector.conf_threshold = 0.25
    assert strict <= loose


def test_tc09_every_inference_is_audited(detector):
    lines = open(detector.audit_log_path).read().strip().splitlines()
    assert len(lines) >= 3
    assert {"timestamp", "conf_threshold", "total", "detections"} <= set(json.loads(lines[-1]))


def test_tc10_annotation_keeps_image_size(detector):
    img = cv2.imread(CRACK_IMG)
    out = detector.annotate_image(img, detector.detect(img))
    assert out.shape == img.shape and not np.array_equal(out, img)


# ---------- explainability, ethics, reports ----------
def test_tc11_eigencam_heatmap(detector, tmp_path):
    img = cv2.imread(POTHOLE_IMG)
    xai = generate_explanation(detector.model, img, detector.detect(img), save_dir=str(tmp_path))
    assert xai["heatmap_bgr"].shape == img.shape and xai["summary"]


def test_tc12_ethics_flags_low_confidence(detector):
    img = cv2.imread(POTHOLE_IMG)
    dets = detector.detect(img)
    dets[0].flagged_for_review = True
    rep = generate_ethics_report(dets)
    assert rep.flagged_for_review == 1 and "MANDATORY HUMAN REVIEW" in rep.disclaimer


def test_tc13_report_files_written(detector, tmp_path):
    img = cv2.imread(CRACK_IMG)
    dets = detector.detect(img)
    paths = generate_report("crack.jpg", detector.annotate_image(img, dets), img, dets,
                            generate_ethics_report(dets), "n/a", output_dir=str(tmp_path))
    assert set(paths) == {"annotated", "heatmap", "json", "csv", "ethics"}
    assert all(os.path.exists(p) for p in paths.values())
    assert json.load(open(paths["json"]))["total_defects"] == len(dets)
