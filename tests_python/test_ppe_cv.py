import base64

import cv2
import numpy as np

from api.index import PpeImageRequest, analyze_ppe_image


def _synthetic_worker_image() -> str:
    image = np.zeros((512, 512, 3), dtype=np.uint8)

    # Capacete amarelo na região superior.
    cv2.rectangle(image, (150, 25), (360, 145), (0, 220, 255), -1)

    # Vestimenta laranja de alta visibilidade no tronco.
    cv2.rectangle(image, (90, 175), (420, 470), (0, 110, 255), -1)

    # Faixas refletivas claras.
    cv2.rectangle(image, (90, 245), (420, 275), (220, 220, 220), -1)
    cv2.rectangle(image, (90, 360), (420, 390), (220, 220, 220), -1)

    # Luvas marrons/tan na região central inferior.
    cv2.rectangle(image, (145, 300), (250, 405), (20, 120, 160), -1)
    cv2.rectangle(image, (265, 300), (370, 405), (20, 120, 160), -1)

    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def test_opencv_detecta_capacete_vestimenta_e_luvas():
    response = analyze_ppe_image(
        PpeImageRequest(
            module="epi",
            image_base64=_synthetic_worker_image(),
            mime_type="image/jpeg",
        )
    )

    assert response["engines"]["opencv"] is True
    assert response["items"]["capacete"]["status"] == "detectado"
    assert response["items"]["colete"]["status"] == "detectado"
    assert response["items"]["luvas"]["status"] == "detectado"
