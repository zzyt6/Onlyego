from object_model.model import MAX_POINT_TO_SURFACE_MM, MIN_IOU, quality


def _fit(iou, dist):
    return {"silhouette_iou": iou, "point_to_surface_mm": {"median": dist}}


def test_defaults():
    assert MIN_IOU == 0.95
    assert MAX_POINT_TO_SURFACE_MM == 3.0


def test_passes_demo01_values():
    q = quality(_fit(0.9801, 0.68))
    assert q["passed"] and q["failures"] == []


def test_low_iou_fails():
    q = quality(_fit(0.94, 0.68))
    assert not q["passed"]
    assert len(q["failures"]) == 1 and "IoU" in q["failures"][0]


def test_far_points_fail():
    q = quality(_fit(0.98, 3.5))
    assert not q["passed"] and "mm" in q["failures"][0]


def test_both_fail_and_custom_thresholds():
    assert len(quality(_fit(0.5, 10.0))["failures"]) == 2
    assert quality(_fit(0.96, 1.0), min_iou=0.97)["passed"] is False
    assert quality(_fit(0.96, 1.0), min_iou=0.9, max_point_mm=0.5)["passed"] is False
