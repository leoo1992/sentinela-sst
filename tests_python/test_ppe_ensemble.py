import numpy as np

from backend.ppe_ensemble import (
    SH17_TARGET_CLASSES,
    _as_rows,
    _decode_rows,
    _fuse_models,
    Detection,
)


def test_sh17_maps_only_supported_ppe_classes():
    assert SH17_TARGET_CLASSES == {
        8: "oculos",
        9: "luvas",
        10: "capacete",
    }


def test_decode_yolov8_output_filters_irrelevant_classes():
    rows = np.zeros((1, 21, 100), dtype=np.float32)
    rows[0, 0:4, 0] = [320, 160, 120, 80]
    rows[0, 4 + 10, 0] = 0.91
    rows[0, 0:4, 1] = [320, 320, 200, 300]
    rows[0, 4 + 16, 1] = 0.95

    decoded = _decode_rows(_as_rows(rows), 1.0, 0, 0, 640, 640, "test")
    assert len(decoded) == 1
    assert decoded[0].key == "capacete"
    assert decoded[0].score > 0.9


def test_wbf_combines_two_models():
    first = [Detection("capacete", 0.80, (0.10, 0.10, 0.30, 0.30), "a")]
    second = [Detection("capacete", 0.90, (0.11, 0.10, 0.31, 0.30), "b")]
    fused = _fuse_models([first, second], [0.55, 0.45])
    assert fused
    assert fused[0].key == "capacete"
    assert 0.79 <= fused[0].score <= 0.91
