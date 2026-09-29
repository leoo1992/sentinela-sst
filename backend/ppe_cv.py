import base64
import binascii
from typing import Any, Optional

import cv2
import numpy as np

LABELS = {
    "oculos": "Óculos",
    "capacete": "Capacete",
    "luvas": "Luvas",
    "protetorAuricular": "Protetor auricular",
}


def _item(key: str, status: str, confidence: float, note: str) -> dict[str, Any]:
    return {
        "label": LABELS[key],
        "status": status,
        "confidence": float(max(0.0, min(1.0, confidence))),
        "note": note,
    }


def _decode(image_base64: str) -> np.ndarray:
    try:
        raw = base64.b64decode(image_base64, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("Imagem base64 inválida.") from error

    data = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError("Não foi possível decodificar a imagem.")

    height, width = image.shape[:2]
    scale = min(1.0, 1100.0 / max(height, width))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return image


def _clip(image: np.ndarray, x: float, y: float, width: float, height: float) -> Optional[np.ndarray]:
    image_height, image_width = image.shape[:2]
    x1 = max(0, min(image_width - 1, int(round(x))))
    y1 = max(0, min(image_height - 1, int(round(y))))
    x2 = max(x1 + 1, min(image_width, int(round(x + width))))
    y2 = max(y1 + 1, min(image_height, int(round(y + height))))
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


def _stats(region: Optional[np.ndarray]) -> dict[str, float]:
    if region is None or region.size == 0:
        return {
            "skin": 0.0,
            "dark": 0.0,
            "saturated": 0.0,
            "edge": 0.0,
            "variance": 0.0,
        }

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    _, saturation, value = cv2.split(hsv)
    ycrcb = cv2.cvtColor(region, cv2.COLOR_BGR2YCrCb)
    _, cr, cb = cv2.split(ycrcb)

    skin = (cr >= 133) & (cr <= 177) & (cb >= 77) & (cb <= 135)
    dark = value <= 76
    saturated = saturation >= 78

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 55, 145) > 0

    total = float(region.shape[0] * region.shape[1])
    return {
        "skin": float(np.count_nonzero(skin) / total),
        "dark": float(np.count_nonzero(dark) / total),
        "saturated": float(np.count_nonzero(saturated) / total),
        "edge": float(np.count_nonzero(edges) / total),
        "variance": float(np.var(gray)),
    }


def _largest_face(image: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if detector.empty():
        return None

    height, width = image.shape[:2]
    minimum = max(28, min(height, width) // 16)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=4,
        minSize=(minimum, minimum),
    )

    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda value: int(value[2]) * int(value[3]))
    return int(x), int(y), int(w), int(h)


def _pose_point(
    pose_keypoints: Optional[dict[str, Any]],
    name: str,
    width: int,
    height: int,
) -> Optional[tuple[float, float, float]]:
    if not pose_keypoints:
        return None

    raw = pose_keypoints.get(name)
    if not isinstance(raw, dict):
        return None

    try:
        score = float(raw.get("score", 0))
        x = float(raw.get("x", 0)) * width
        y = float(raw.get("y", 0)) * height
    except (TypeError, ValueError):
        return None

    if score < 0.20:
        return None

    return x, y, score


def _point_region(
    image: np.ndarray,
    point: Optional[tuple[float, float, float]],
    radius: float,
) -> Optional[np.ndarray]:
    if point is None:
        return None
    x, y, _ = point
    return _clip(image, x - radius, y - radius, radius * 2, radius * 2)


def _helmet_item(stats: dict[str, float]) -> dict[str, Any]:
    signal = (
        stats["edge"] * 5.0
        + stats["saturated"] * 0.9
        + stats["dark"] * 0.6
        + min(1.0, stats["variance"] / 1800.0) * 0.75
    )

    if stats["skin"] < 0.45 and signal >= 0.78:
        return _item(
            "capacete",
            "detectado",
            min(0.94, 0.58 + signal * 0.18),
            "Objeto consistente com capacete identificado na cabeça, sem depender de cor, modelo ou especificação.",
        )

    if signal >= 0.52:
        return _item(
            "capacete",
            "incerto",
            0.47,
            "Há um objeto na região da cabeça, mas a identificação de capacete é inconclusiva.",
        )

    return _item(
        "capacete",
        "nao_detectado",
        0.56,
        "Nenhum objeto consistente com capacete foi identificado na cabeça.",
    )


def _glasses_item(stats: dict[str, float]) -> dict[str, Any]:
    signal = (
        stats["edge"] * 5.6
        + stats["dark"] * 0.85
        + min(1.0, stats["variance"] / 1300.0) * 0.65
    )

    if signal >= 0.88:
        return _item(
            "oculos",
            "detectado",
            min(0.90, 0.57 + signal * 0.16),
            "Estrutura compatível com óculos identificada ao redor dos olhos, independentemente do tipo.",
        )

    if signal >= 0.62:
        return _item(
            "oculos",
            "incerto",
            0.46,
            "Há contornos compatíveis com óculos, mas a identificação é inconclusiva.",
        )

    return _item(
        "oculos",
        "nao_detectado",
        0.54,
        "Óculos não foram identificados na região dos olhos.",
    )


def _gloves_item(stats_list: list[dict[str, float]]) -> dict[str, Any]:
    if not stats_list:
        return _item("luvas", "nao_avaliavel", 0.0, "Mãos não localizadas.")

    skin = float(np.mean([value["skin"] for value in stats_list]))
    material = float(np.mean([
        value["saturated"] * 0.7
        + value["dark"] * 0.5
        + value["edge"] * 3.2
        + min(1.0, value["variance"] / 1600.0) * 0.45
        for value in stats_list
    ]))

    if skin < 0.40 and material >= 0.66:
        return _item(
            "luvas",
            "detectado",
            min(0.91, 0.58 + material * 0.18),
            "Material diferente de pele identificado nas mãos, independentemente do tipo ou material da luva.",
        )

    if skin >= 0.58:
        return _item("luvas", "nao_detectado", 0.62, "As mãos apresentam forte padrão de pele exposta.")

    return _item("luvas", "incerto", 0.44, "As mãos estão visíveis, mas não foi possível confirmar luvas.")


def _hearing_item(stats_list: list[dict[str, float]]) -> dict[str, Any]:
    if not stats_list:
        return _item("protetorAuricular", "nao_avaliavel", 0.0, "Orelhas não localizadas.")

    signals = [
        (1.0 - value["skin"]) * 0.42
        + value["saturated"] * 0.62
        + value["dark"] * 0.48
        + value["edge"] * 3.6
        + min(1.0, value["variance"] / 1700.0) * 0.42
        for value in stats_list
    ]

    strongest = max(signals)

    if strongest >= 0.86:
        return _item(
            "protetorAuricular",
            "detectado",
            min(0.89, 0.56 + strongest * 0.18),
            "Objeto/material compatível com protetor auricular plug ou abafador identificado na região da orelha.",
        )

    if strongest >= 0.64:
        return _item(
            "protetorAuricular",
            "incerto",
            0.46,
            "Há alteração visual na região da orelha, mas não foi possível confirmar plug ou abafador.",
        )

    return _item(
        "protetorAuricular",
        "nao_detectado",
        0.55,
        "Protetor auricular não foi identificado na região das orelhas.",
    )


def analyze_ppe_cv(
    image_base64: str,
    module: str,
    pose_keypoints: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    image = _decode(image_base64)
    height, width = image.shape[:2]
    face = _largest_face(image)

    nose = _pose_point(pose_keypoints, "nose", width, height)
    left_eye = _pose_point(pose_keypoints, "left_eye", width, height)
    right_eye = _pose_point(pose_keypoints, "right_eye", width, height)
    left_ear = _pose_point(pose_keypoints, "left_ear", width, height)
    right_ear = _pose_point(pose_keypoints, "right_ear", width, height)
    left_wrist = _pose_point(pose_keypoints, "left_wrist", width, height)
    right_wrist = _pose_point(pose_keypoints, "right_wrist", width, height)
    left_shoulder = _pose_point(pose_keypoints, "left_shoulder", width, height)
    right_shoulder = _pose_point(pose_keypoints, "right_shoulder", width, height)

    shoulder_width = None
    if left_shoulder and right_shoulder:
        shoulder_width = max(28.0, abs(right_shoulder[0] - left_shoulder[0]))

    if nose and shoulder_width:
        head_region = _clip(
            image,
            nose[0] - shoulder_width * 0.42,
            nose[1] - shoulder_width * 0.82,
            shoulder_width * 0.84,
            shoulder_width * 0.74,
        )
    elif face:
        x, y, fw, fh = face
        head_region = _clip(image, x - 0.28 * fw, y - 0.92 * fh, 1.56 * fw, 1.02 * fh)
    else:
        head_region = _clip(image, width * 0.20, 0, width * 0.60, height * 0.32)

    if left_eye and right_eye:
        eye_distance = max(18.0, abs(right_eye[0] - left_eye[0]))
        eye_region = _clip(
            image,
            min(left_eye[0], right_eye[0]) - eye_distance * 0.50,
            min(left_eye[1], right_eye[1]) - eye_distance * 0.40,
            eye_distance * 2.0,
            eye_distance * 0.88,
        )
    elif face:
        x, y, fw, fh = face
        eye_region = _clip(image, x + 0.03 * fw, y + 0.18 * fh, 0.94 * fw, 0.38 * fh)
    else:
        eye_region = None

    body_reference = shoulder_width or (face[2] if face else min(width, height) * 0.20)
    hand_radius = max(18.0, body_reference * 0.30)
    ear_large_radius = max(14.0, body_reference * 0.23)
    ear_small_radius = max(8.0, body_reference * 0.12)

    hand_stats = [
        _stats(_point_region(image, left_wrist, hand_radius)),
        _stats(_point_region(image, right_wrist, hand_radius)),
    ]
    hand_stats = [value for point, value in zip([left_wrist, right_wrist], hand_stats) if point is not None]

    ear_stats: list[dict[str, float]] = []
    for ear in [left_ear, right_ear]:
        if ear is None:
            continue
        ear_stats.append(_stats(_point_region(image, ear, ear_large_radius)))
        ear_stats.append(_stats(_point_region(image, ear, ear_small_radius)))

    if not ear_stats and face:
        x, y, fw, fh = face
        ear_stats = [
            _stats(_clip(image, x - fw * 0.18, y + fh * 0.22, fw * 0.30, fh * 0.50)),
            _stats(_clip(image, x + fw * 0.88, y + fh * 0.22, fw * 0.30, fh * 0.50)),
        ]

    items = {
        "oculos": _glasses_item(_stats(eye_region)) if eye_region is not None else _item("oculos", "nao_avaliavel", 0.0, "Região dos olhos não localizada."),
        "capacete": _helmet_item(_stats(head_region)),
        "luvas": _gloves_item(hand_stats),
        "protetorAuricular": _hearing_item(ear_stats),
    }

    return {
        "items": items,
        "person_detected": bool(face or nose or left_shoulder or right_shoulder or pose_keypoints),
        "face_detected": bool(face),
    }
