import base64
import binascii
from typing import Any, Optional

import cv2
import numpy as np

LABELS = {
    "capacete": "Capacete",
    "oculos": "Óculos de proteção",
    "colete": "Colete/vestimenta refletiva",
    "luvas": "Luvas",
    "calcado": "Calçado fechado",
    "cinturao": "Cinturão paraquedista",
    "talabarte": "Talabarte",
    "travaQuedas": "Trava-quedas",
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
    scale = min(1.0, 1024.0 / max(height, width))
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
            "highvis": 0.0,
            "reflective": 0.0,
            "dark": 0.0,
            "edges": 0.0,
            "skin": 0.0,
            "tan": 0.0,
            "saturated": 0.0,
        }

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)

    yellow = (hue >= 18) & (hue <= 40) & (saturation >= 90) & (value >= 90)
    orange = (hue >= 5) & (hue < 18) & (saturation >= 100) & (value >= 80)
    lime = (hue >= 40) & (hue <= 90) & (saturation >= 70) & (value >= 80)
    blue = (hue >= 90) & (hue <= 135) & (saturation >= 70) & (value >= 70)
    red = ((hue <= 5) | (hue >= 170)) & (saturation >= 90) & (value >= 70)
    bright_neutral = (saturation <= 70) & (value >= 165)
    dark = value <= 75
    tan = (hue >= 6) & (hue <= 30) & (saturation >= 45) & (value >= 55) & (value <= 235)

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 70, 160) > 0

    ycrcb = cv2.cvtColor(region, cv2.COLOR_BGR2YCrCb)
    _, cr, cb = cv2.split(ycrcb)
    skin = (cr >= 133) & (cr <= 177) & (cb >= 77) & (cb <= 135)

    total = float(region.shape[0] * region.shape[1])
    return {
        "highvis": float(np.count_nonzero(yellow | orange | lime | blue | red) / total),
        "reflective": float(np.count_nonzero(bright_neutral) / total),
        "dark": float(np.count_nonzero(dark) / total),
        "edges": float(np.count_nonzero(edges) / total),
        "skin": float(np.count_nonzero(skin) / total),
        "tan": float(np.count_nonzero(tan) / total),
        "saturated": float(np.count_nonzero(saturation >= 80) / total),
    }


def _largest_face(image: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if detector.empty():
        return None

    height, width = image.shape[:2]
    minimum = max(30, min(height, width) // 14)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=5,
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
    value = pose_keypoints.get(name)
    if not isinstance(value, dict):
        return None
    try:
        score = float(value.get("score", 0))
        x = float(value.get("x", 0)) * width
        y = float(value.get("y", 0)) * height
    except (TypeError, ValueError):
        return None
    if score < 0.28 or not (0 <= x <= width and 0 <= y <= height):
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


def analyze_ppe_cv(
    image_base64: str,
    module: str,
    pose_keypoints: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    image = _decode(image_base64)
    height, width = image.shape[:2]
    face = _largest_face(image)

    if face:
        x, y, face_w, face_h = face
        head_region = _clip(
            image,
            x - 0.35 * face_w,
            y - 0.92 * face_h,
            1.70 * face_w,
            1.10 * face_h,
        )
        eye_region = _clip(
            image,
            x + 0.03 * face_w,
            y + 0.18 * face_h,
            0.94 * face_w,
            0.38 * face_h,
        )
        torso_region = _clip(
            image,
            x - 0.90 * face_w,
            y + 0.65 * face_h,
            2.80 * face_w,
            2.85 * face_h,
        )
    else:
        head_region = _clip(image, 0.25 * width, 0.02 * height, 0.50 * width, 0.30 * height)
        eye_region = _clip(image, 0.28 * width, 0.12 * height, 0.44 * width, 0.17 * height)
        torso_region = _clip(image, 0.15 * width, 0.30 * height, 0.70 * width, 0.55 * height)

    head = _stats(head_region)
    eyes = _stats(eye_region)
    torso = _stats(torso_region)

    if head["highvis"] >= 0.075:
        capacete = _item(
            "capacete",
            "detectado",
            min(0.95, 0.72 + head["highvis"] * 0.9),
            "Cor e volume compatíveis com capacete foram identificados na região da cabeça.",
        )
    elif head["reflective"] >= 0.32 and head["edges"] >= 0.008:
        capacete = _item(
            "capacete",
            "incerto",
            0.58,
            "Há objeto claro sobre a cabeça; confirme visualmente se é capacete de segurança.",
        )
    elif face:
        capacete = _item(
            "capacete",
            "nao_detectado",
            0.58,
            "A cabeça está visível, mas o OpenCV não encontrou padrão forte de capacete.",
        )
    else:
        capacete = _item(
            "capacete",
            "nao_avaliavel",
            0.15,
            "A cabeça não pôde ser localizada com confiança.",
        )

    if torso["highvis"] >= 0.115 or (
        torso["highvis"] >= 0.055 and torso["reflective"] >= 0.07
    ):
        colete = _item(
            "colete",
            "detectado",
            min(0.94, 0.68 + torso["highvis"] * 0.75 + torso["reflective"] * 0.25),
            "Vestimenta de alta visibilidade e/ou faixas refletivas foram identificadas no tronco.",
        )
    elif torso["saturated"] >= 0.32:
        colete = _item(
            "colete",
            "incerto",
            0.48,
            "Há vestimenta colorida no tronco, mas a característica refletiva é inconclusiva.",
        )
    else:
        colete = _item(
            "colete",
            "nao_detectado",
            0.56,
            "Não foi identificado padrão forte de vestimenta refletiva no tronco.",
        )

    if face and eyes["edges"] >= 0.105 and eyes["dark"] >= 0.15:
        oculos = _item(
            "oculos",
            "incerto",
            0.62,
            "Há estrutura com contornos compatíveis com proteção ocular; a confirmação semântica fica a cargo da análise complementar.",
        )
    elif face:
        oculos = _item(
            "oculos",
            "nao_avaliavel",
            0.25,
            "Óculos transparentes não podem ser confirmados com segurança apenas por contraste.",
        )
    else:
        oculos = _item(
            "oculos",
            "nao_avaliavel",
            0.10,
            "A região dos olhos não foi localizada.",
        )

    reference_radius = max(14.0, min(width, height) * 0.065)
    wrist_regions = [
        _point_region(image, _pose_point(pose_keypoints, "left_wrist", width, height), reference_radius),
        _point_region(image, _pose_point(pose_keypoints, "right_wrist", width, height), reference_radius),
    ]
    wrist_stats = [_stats(region) for region in wrist_regions if region is not None]

    if wrist_stats:
        skin = float(np.mean([value["skin"] for value in wrist_stats]))
        tan = float(np.mean([value["tan"] for value in wrist_stats]))
        saturated = float(np.mean([value["saturated"] for value in wrist_stats]))
        if (skin < 0.28 and saturated > 0.25) or tan > 0.42:
            luvas = _item(
                "luvas",
                "detectado",
                min(0.86, 0.58 + max(tan, saturated) * 0.30),
                "Material não semelhante à pele foi identificado nas regiões das mãos.",
            )
        elif skin > 0.56:
            luvas = _item(
                "luvas",
                "nao_detectado",
                0.58,
                "As regiões das mãos apresentam forte padrão de pele exposta.",
            )
        else:
            luvas = _item(
                "luvas",
                "incerto",
                0.44,
                "As mãos estão visíveis, mas o material não pôde ser classificado com segurança.",
            )
    else:
        lower_center = _clip(image, 0.18 * width, 0.52 * height, 0.64 * width, 0.38 * height)
        lower_stats = _stats(lower_center)
        if lower_stats["tan"] > 0.24 and lower_stats["saturated"] > 0.35:
            luvas = _item(
                "luvas",
                "incerto",
                0.46,
                "Há material compatível com luvas na região central inferior, mas faltam pontos confiáveis das mãos.",
            )
        else:
            luvas = _item(
                "luvas",
                "nao_avaliavel",
                0.15,
                "As mãos não foram localizadas com confiança.",
            )

    ankle_regions = [
        _point_region(image, _pose_point(pose_keypoints, "left_ankle", width, height), reference_radius * 1.2),
        _point_region(image, _pose_point(pose_keypoints, "right_ankle", width, height), reference_radius * 1.2),
    ]
    ankle_stats = [_stats(region) for region in ankle_regions if region is not None]

    if ankle_stats:
        skin = float(np.mean([value["skin"] for value in ankle_stats]))
        dark = float(np.mean([value["dark"] for value in ankle_stats]))
        if skin < 0.20 and dark > 0.16:
            calcado = _item(
                "calcado",
                "detectado",
                0.58,
                "Material fechado foi identificado nas regiões dos pés.",
            )
        elif skin > 0.48:
            calcado = _item(
                "calcado",
                "nao_detectado",
                0.54,
                "Há forte presença de pele nas regiões dos pés.",
            )
        else:
            calcado = _item(
                "calcado",
                "incerto",
                0.40,
                "Os pés estão visíveis, mas o tipo de calçado é inconclusivo.",
            )
    else:
        calcado = _item(
            "calcado",
            "nao_avaliavel",
            0.10,
            "Os pés não estão visíveis ou não foram localizados.",
        )

    cinturao = _item(
        "cinturao",
        "nao_avaliavel",
        0.18,
        "O OpenCV não confirma cinturão paraquedista sem tiras e pontos de conexão claramente visíveis.",
    )
    talabarte = _item(
        "talabarte",
        "nao_avaliavel",
        0.15,
        "Talabarte exige cabo/fita e conectores claramente visíveis.",
    )
    trava_quedas = _item(
        "travaQuedas",
        "nao_avaliavel",
        0.15,
        "Trava-quedas exige dispositivo e linha de vida claramente visíveis.",
    )

    items = {
        "capacete": capacete,
        "oculos": oculos,
        "colete": colete,
        "luvas": luvas,
        "calcado": calcado,
        "cinturao": cinturao,
        "talabarte": talabarte,
        "travaQuedas": trava_quedas,
    }

    requested = (
        ["capacete", "cinturao", "talabarte", "travaQuedas"]
        if module == "altura"
        else ["capacete", "oculos", "colete", "luvas", "calcado"]
    )

    return {
        "items": {key: items[key] for key in requested},
        "person_detected": bool(face or pose_keypoints),
        "face_detected": bool(face),
    }
