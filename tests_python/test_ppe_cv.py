import base64

import cv2
import numpy as np

from api.index import PpeImageRequest, analyze_ppe_image


def _synthetic_worker_image() -> tuple[str, dict]:
    image = np.full((600, 600, 3), 180, dtype=np.uint8)

    # Cabeça/face.
    cv2.ellipse(image, (300, 205), (70, 90), 0, 0, 360, (120, 170, 210), -1)

    # Capacete azul escuro: prova que a detecção não depende de amarelo/laranja.
    cv2.ellipse(image, (300, 125), (105, 58), 0, 180, 360, (160, 60, 20), -1)
    cv2.rectangle(image, (190, 120), (410, 145), (160, 60, 20), -1)

    # Óculos pretos.
    cv2.rectangle(image, (238, 185), (292, 218), (20, 20, 20), 7)
    cv2.rectangle(image, (308, 185), (362, 218), (20, 20, 20), 7)
    cv2.line(image, (292, 201), (308, 201), (20, 20, 20), 5)

    # Luvas nas mãos.
    cv2.rectangle(image, (150, 420), (235, 505), (35, 35, 190), -1)
    cv2.rectangle(image, (365, 420), (450, 505), (35, 35, 190), -1)

    ok, encoded = cv2.imencode(".jpg", image)
    assert ok

    pose = {
        "nose": {"x": 0.50, "y": 0.34, "score": 0.99},
        "left_eye": {"x": 0.44, "y": 0.33, "score": 0.99},
        "right_eye": {"x": 0.56, "y": 0.33, "score": 0.99},
        "left_shoulder": {"x": 0.36, "y": 0.50, "score": 0.99},
        "right_shoulder": {"x": 0.64, "y": 0.50, "score": 0.99},
        "left_wrist": {"x": 0.32, "y": 0.77, "score": 0.99},
        "right_wrist": {"x": 0.68, "y": 0.77, "score": 0.99},
    }

    return base64.b64encode(encoded.tobytes()).decode("ascii"), pose


def test_opencv_detecta_oculos_capacete_e_luvas():
    image_base64, pose = _synthetic_worker_image()

    response = analyze_ppe_image(
        PpeImageRequest(
            module="epi",
            image_base64=image_base64,
            mime_type="image/jpeg",
            pose_keypoints=pose,
        )
    )

    assert response["engines"]["opencv"] is True
    assert response["items"]["oculos"]["status"] == "detectado"
    assert response["items"]["capacete"]["status"] == "detectado"
    assert response["items"]["luvas"]["status"] == "detectado"
