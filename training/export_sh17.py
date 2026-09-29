from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from ultralytics import YOLO

SOURCES = {
    "sh17-yolo8n": "https://github.com/ahmadmughees/SH17dataset/releases/download/v1/yolo8n.pt",
    "sh17-yolo10n": "https://github.com/ahmadmughees/SH17dataset/releases/download/v1/yolo10n.pt",
}


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 1_000_000:
        return
    urllib.request.urlretrieve(url, destination)


def export_model(name: str, url: str, output_dir: Path) -> Path:
    weights_dir = output_dir / "source"
    weights_dir.mkdir(parents=True, exist_ok=True)
    pt_path = weights_dir / f"{name}.pt"
    download(url, pt_path)

    model = YOLO(str(pt_path))
    exported = Path(model.export(
        format="onnx", imgsz=640, opset=12, simplify=True,
        dynamic=False, half=False, nms=False,
    ))
    final_path = output_dir / f"{name}.onnx"
    exported.replace(final_path)
    return final_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="models")
    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported: list[str] = []
    errors: dict[str, str] = {}
    for name, url in SOURCES.items():
        try:
            exported.append(str(export_model(name, url, output_dir)))
        except Exception as error:
            errors[name] = f"{error.__class__.__name__}: {error}"
            print(f"Falha ao exportar {name}: {errors[name]}")

    if not exported:
        raise SystemExit("Nenhum modelo SH17 pôde ser exportado.")

    metadata = {
        "source_dataset": "SH17",
        "source_release": "https://github.com/ahmadmughees/SH17dataset/releases/tag/v1",
        "classes": {"8": "glasses", "9": "gloves", "10": "helmet"},
        "models": exported,
        "errors": errors,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
