from __future__ import annotations

import base64
import binascii
import os
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2
import numpy as np

try:
    from ensemble_boxes import weighted_boxes_fusion as _weighted_boxes_fusion
except Exception:
    _weighted_boxes_fusion = None


MODEL_SIZE = 640
MODEL_CACHE_DIR = Path(os.environ.get("SENTINELA_MODEL_CACHE", "/tmp/sentinela-sst-models"))
DEFAULT_MODEL_BASE_URL = "https://github.com/leoo1992/sentinela-sst/releases/download/ppe-models-v1"
MODEL_BASE_URL = os.environ.get("SENTINELA_MODEL_BASE_URL", DEFAULT_MODEL_BASE_URL).rstrip("/")

# Ordem canônica de sh17.yaml. Só as três classes usadas pelo Sentinela são aceitas.
SH17_TARGET_CLASSES: dict[int, str] = {
    8: "oculos",
    9: "luvas",
    10: "capacete",
}

EPI_LABELS = {
    "oculos": "Óculos",
    "capacete": "Capacete",
    "luvas": "Luvas",
}

@dataclass(frozen=True)
class Detection:
    key: str
    score: float
    box: tuple[float, float, float, float]
    model: str


@dataclass(frozen=True)
class ModelSpec:
    name: str
    filename: str
    weight: float

    @property
    def url(self) -> str:
        return f"{MODEL_BASE_URL}/{self.filename}"


MODEL_SPECS = (
    ModelSpec("sh17-yolo8n", "sh17-yolo8n.onnx", 0.55),
    ModelSpec("sh17-yolo10n", "sh17-yolo10n.onnx", 0.45),
)

_MODEL_LOCK = threading.Lock()
_MODEL_NETS: dict[str, cv2.dnn.Net] = {}
_MODEL_ERRORS: dict[str, str] = {}


def _decode_image(image_base64: str) -> np.ndarray:
    try:
        raw = base64.b64decode(image_base64, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("Imagem base64 inválida.") from error
    data = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError("Não foi possível decodificar a imagem.")
    return image


def _download_model(spec: ModelSpec) -> Path:
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    destination = MODEL_CACHE_DIR / spec.filename
    if destination.exists() and destination.stat().st_size > 1_000_000:
        return destination

    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(spec.url, headers={"User-Agent": "Sentinela-SST/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response, partial.open("wb") as file:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                file.write(chunk)
        partial.replace(destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return destination


def _load_model(spec: ModelSpec) -> Optional[cv2.dnn.Net]:
    if spec.name in _MODEL_NETS:
        return _MODEL_NETS[spec.name]
    if spec.name in _MODEL_ERRORS:
        return None

    with _MODEL_LOCK:
        if spec.name in _MODEL_NETS:
            return _MODEL_NETS[spec.name]
        if spec.name in _MODEL_ERRORS:
            return None
        try:
            path = _download_model(spec)
            net = cv2.dnn.readNetFromONNX(str(path))
            try:
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            except cv2.error:
                pass
            _MODEL_NETS[spec.name] = net
            return net
        except Exception as error:
            _MODEL_ERRORS[spec.name] = f"{error.__class__.__name__}: {error}"
            return None


def reset_model_cache_for_tests() -> None:
    _MODEL_NETS.clear()
    _MODEL_ERRORS.clear()


def model_runtime_status() -> dict[str, Any]:
    return {
        spec.name: {
            "loaded": spec.name in _MODEL_NETS,
            "error": _MODEL_ERRORS.get(spec.name),
            "url": spec.url,
        }
        for spec in MODEL_SPECS
    }


def _letterbox(image: np.ndarray, size: int = MODEL_SIZE) -> tuple[np.ndarray, float, int, int]:
    height, width = image.shape[:2]
    ratio = min(size / max(width, 1), size / max(height, 1))
    resized_w = max(1, int(round(width * ratio)))
    resized_h = max(1, int(round(height * ratio)))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    pad_x = (size - resized_w) // 2
    pad_y = (size - resized_h) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w] = resized
    return canvas, ratio, pad_x, pad_y


def _prepare_blob(image: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    padded, ratio, pad_x, pad_y = _letterbox(image)
    blob = cv2.dnn.blobFromImage(
        padded, scalefactor=1.0 / 255.0, size=(MODEL_SIZE, MODEL_SIZE),
        swapRB=True, crop=False,
    )
    return blob, ratio, pad_x, pad_y


def _as_rows(output: Any) -> np.ndarray:
    if isinstance(output, (list, tuple)):
        output = output[0]
    rows = np.asarray(output)
    rows = np.squeeze(rows)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    if rows.ndim != 2:
        return np.empty((0, 0), dtype=np.float32)
    if rows.shape[0] <= 64 and rows.shape[1] > rows.shape[0]:
        rows = rows.T
    return rows.astype(np.float32, copy=False)


def _normalize_box(box: tuple[float, float, float, float], width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return (
        max(0.0, min(1.0, x1 / max(width, 1))),
        max(0.0, min(1.0, y1 / max(height, 1))),
        max(0.0, min(1.0, x2 / max(width, 1))),
        max(0.0, min(1.0, y2 / max(height, 1))),
    )


def _unletterbox_xyxy(
    box: tuple[float, float, float, float], ratio: float, pad_x: int, pad_y: int,
    width: int, height: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    x1 = (x1 - pad_x) / max(ratio, 1e-9)
    y1 = (y1 - pad_y) / max(ratio, 1e-9)
    x2 = (x2 - pad_x) / max(ratio, 1e-9)
    y2 = (y2 - pad_y) / max(ratio, 1e-9)
    return (
        float(max(0, min(width, x1))), float(max(0, min(height, y1))),
        float(max(0, min(width, x2))), float(max(0, min(height, y2))),
    )


def _decode_rows(
    rows: np.ndarray, ratio: float, pad_x: int, pad_y: int,
    width: int, height: int, model_name: str, score_threshold: float = 0.12,
) -> list[Detection]:
    if rows.size == 0:
        return []

    detections: list[Detection] = []
    columns = rows.shape[1]

    if columns == 6:
        for row in rows:
            x1, y1, x2, y2, score, class_id = row.tolist()
            key = SH17_TARGET_CLASSES.get(int(round(class_id)))
            if key is None or float(score) < score_threshold:
                continue
            box = _unletterbox_xyxy((x1, y1, x2, y2), ratio, pad_x, pad_y, width, height)
            detections.append(Detection(key, float(score), _normalize_box(box, width, height), model_name))
        return detections

    class_count = 17
    if columns < 4 + class_count:
        return detections

    has_objectness = columns >= 5 + class_count
    class_start = 5 if has_objectness else 4
    for row in rows:
        class_scores = row[class_start:class_start + class_count]
        class_id = int(np.argmax(class_scores))
        key = SH17_TARGET_CLASSES.get(class_id)
        if key is None:
            continue
        class_score = float(class_scores[class_id])
        score = class_score * float(row[4]) if has_objectness else class_score
        if score < score_threshold:
            continue
        center_x, center_y, box_w, box_h = [float(v) for v in row[:4]]
        raw_box = (
            center_x - box_w / 2, center_y - box_h / 2,
            center_x + box_w / 2, center_y + box_h / 2,
        )
        box = _unletterbox_xyxy(raw_box, ratio, pad_x, pad_y, width, height)
        detections.append(Detection(key, score, _normalize_box(box, width, height), model_name))
    return detections


def _box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    union = max(0.0, (ax2 - ax1) * (ay2 - ay1)) + max(0.0, (bx2 - bx1) * (by2 - by1)) - intersection
    return intersection / union if union > 0 else 0.0


def _nms(detections: Iterable[Detection], iou_threshold: float = 0.50) -> list[Detection]:
    output: list[Detection] = []
    by_key: dict[str, list[Detection]] = {}
    for detection in detections:
        by_key.setdefault(detection.key, []).append(detection)
    for group in by_key.values():
        pending = sorted(group, key=lambda value: value.score, reverse=True)
        while pending:
            best = pending.pop(0)
            output.append(best)
            pending = [value for value in pending if _box_iou(best.box, value.box) < iou_threshold]
    return output


def _tile_origins(length: int, tile_size: int, overlap: float) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = max(1, int(round(tile_size * (1.0 - overlap))))
    origins = list(range(0, max(1, length - tile_size + 1), stride))
    last = max(0, length - tile_size)
    if not origins or origins[-1] != last:
        origins.append(last)
    return origins


def _iter_views(image: np.ndarray) -> Iterable[tuple[np.ndarray, int, int, bool]]:
    yield image, 0, 0, False
    height, width = image.shape[:2]
    if max(height, width) < 760:
        return
    tile_size = min(640, max(320, min(height, width)))
    emitted = 0
    for y in _tile_origins(height, tile_size, 0.25):
        for x in _tile_origins(width, tile_size, 0.25):
            if emitted >= 12:
                return
            tile = image[y:min(height, y + tile_size), x:min(width, x + tile_size)]
            if tile.size:
                yield tile, x, y, True
                emitted += 1


def _remap_tile_detection(
    detection: Detection, tile_width: int, tile_height: int,
    offset_x: int, offset_y: int, full_width: int, full_height: int,
) -> Detection:
    x1, y1, x2, y2 = detection.box
    global_box = (
        (x1 * tile_width + offset_x) / max(full_width, 1),
        (y1 * tile_height + offset_y) / max(full_height, 1),
        (x2 * tile_width + offset_x) / max(full_width, 1),
        (y2 * tile_height + offset_y) / max(full_height, 1),
    )
    return Detection(detection.key, detection.score, global_box, detection.model)


def _run_model(spec: ModelSpec, image: np.ndarray) -> tuple[list[Detection], bool]:
    net = _load_model(spec)
    if net is None:
        return [], False
    full_height, full_width = image.shape[:2]
    detections: list[Detection] = []
    for view, offset_x, offset_y, sliced in _iter_views(image):
        view_height, view_width = view.shape[:2]
        blob, ratio, pad_x, pad_y = _prepare_blob(view)
        net.setInput(blob)
        rows = _as_rows(net.forward())
        current = _decode_rows(
            rows, ratio, pad_x, pad_y, view_width, view_height,
            spec.name, score_threshold=0.10 if sliced else 0.14,
        )
        if sliced:
            current = [
                _remap_tile_detection(
                    value, view_width, view_height, offset_x, offset_y,
                    full_width, full_height,
                )
                for value in current
            ]
        detections.extend(current)
    return _nms(detections, 0.48), True


def _fallback_fusion(model_detections: list[list[Detection]]) -> list[Detection]:
    return _nms([item for group in model_detections for item in group], 0.50)


def _fuse_models(detections_by_model: list[list[Detection]], weights: list[float]) -> list[Detection]:
    nonempty = [(detections, weight) for detections, weight in zip(detections_by_model, weights) if detections]
    if not nonempty:
        return []
    if len(nonempty) == 1 or _weighted_boxes_fusion is None:
        return _fallback_fusion([value[0] for value in nonempty])

    keys = ["oculos", "capacete", "luvas"]
    key_to_id = {key: index for index, key in enumerate(keys)}
    id_to_key = {index: key for key, index in key_to_id.items()}
    boxes_list, scores_list, labels_list, active_weights = [], [], [], []
    for detections, weight in nonempty:
        boxes_list.append([list(value.box) for value in detections])
        scores_list.append([float(value.score) for value in detections])
        labels_list.append([key_to_id[value.key] for value in detections])
        active_weights.append(weight)

    boxes, scores, labels = _weighted_boxes_fusion(
        boxes_list, scores_list, labels_list, weights=active_weights,
        iou_thr=0.52, skip_box_thr=0.08, conf_type="avg",
    )
    return [
        Detection(
            id_to_key[int(round(label))], float(score),
            tuple(float(value) for value in box), "wbf",
        )
        for box, score, label in zip(boxes, scores, labels)
    ]


def _item_from_score(key: str, score: Optional[float]) -> dict[str, Any]:
    label = EPI_LABELS[key]
    if score is None:
        return {
            "label": label, "status": "nao_avaliavel", "confidence": 0.0,
            "note": "Modelo treinado não produziu evidência suficiente para este item.",
        }

    strong = {
        "oculos": 0.48, "capacete": 0.46, "luvas": 0.45,
    }[key]
    weak = strong * 0.58
    if score >= strong:
        return {
            "label": label, "status": "detectado", "confidence": min(0.98, score),
            "note": "EPI identificado por detectores treinados e fusão de probabilidades.",
        }
    if score >= weak:
        return {
            "label": label, "status": "incerto", "confidence": score,
            "note": "Há evidência de EPI, mas a confiança do conjunto de detectores é intermediária.",
        }
    return {
        "label": label, "status": "nao_avaliavel", "confidence": score,
        "note": "Os detectores treinados não produziram evidência forte o suficiente.",
    }


def analyze_ppe_ensemble(image_base64: str, pose_keypoints: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    image = _decode_image(image_base64)
    height, width = image.shape[:2]
    scale = min(1.0, 1400.0 / max(height, width))
    if scale < 1.0:
        image = cv2.resize(
            image, (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    detections_by_model, weights, active_models = [], [], []
    for spec in MODEL_SPECS:
        detections, loaded = _run_model(spec, image)
        if loaded:
            active_models.append(spec.name)
            detections_by_model.append(detections)
            weights.append(spec.weight)

    fused = _fuse_models(detections_by_model, weights)

    scores: dict[str, Optional[float]] = {
        "oculos": None, "capacete": None, "luvas": None,
    }
    for detection in fused:
        current = scores[detection.key]
        scores[detection.key] = detection.score if current is None else max(current, detection.score)

    return {
        "items": {key: _item_from_score(key, score) for key, score in scores.items()},
        "active_models": active_models,
        "model_status": model_runtime_status(),
        "sliced_inference": max(image.shape[:2]) >= 760,
        "wbf": _weighted_boxes_fusion is not None and len(active_models) >= 2,
        "detections": [
            {"key": value.key, "score": value.score, "box": list(value.box)}
            for value in fused
        ],
    }
