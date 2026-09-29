from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import RTDETR, YOLO


def train_yolo26(data: str, output: Path, epochs: int, imgsz: int) -> Path:
    model = YOLO("yolo26n.pt")
    results = model.train(data=data, epochs=epochs, imgsz=imgsz, project=str(output), name="yolo26-ppe")
    best = Path(results.save_dir) / "weights" / "best.pt"
    return Path(YOLO(str(best)).export(format="onnx", imgsz=imgsz, opset=12, simplify=True))


def train_rtdetr(data: str, output: Path, epochs: int, imgsz: int) -> Path:
    model = RTDETR("rtdetr-l.pt")
    results = model.train(
        data=data, epochs=epochs, imgsz=imgsz, project=str(output),
        name="rtdetr-ppe", deterministic=False,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    return Path(RTDETR(str(best)).export(format="onnx", imgsz=imgsz, opset=16, simplify=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina o ensemble de próxima geração do Sentinela SST.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="training/runs")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=832)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    print("YOLO26:", train_yolo26(args.data, output, args.epochs, args.imgsz))
    print("RT-DETR:", train_rtdetr(args.data, output, args.epochs, args.imgsz))


if __name__ == "__main__":
    main()
