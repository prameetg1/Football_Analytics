#!/usr/bin/env python3
"""Train the football player/ball detector on the merged multi-source dataset.

Ingests `data/player_detection_merged/` (v11 1280^2 + minaehyeon 640^2, both
ball/goalkeeper/player/referee) via its data.yaml. Tuned for the local GTX 1060
(6GB): a small model at imgsz=1280, batch 2, AMP. For a yolov8x-quality model,
run `notebooks/train_player_colab.ipynb` instead (T4/A100).

Usage:
    PYTHONPATH=. .venv/bin/python scripts/train_player_detector.py \
        --model yolov8m.pt --imgsz 1280 --batch 2 --epochs 50 --project runs/detect --name merged
"""

import argparse
import os
from pathlib import Path

from src.config import PLAYER_DETECTION_DATA


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="yolov8m.pt")
    p.add_argument("--data", default=PLAYER_DETECTION_DATA)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--project", default="runs/detect")
    p.add_argument("--name", default="merged")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    from ultralytics import YOLO

    if args.resume:
        last = Path(args.project) / args.name / "weights" / "last.pt"
        if not last.is_file():
            raise SystemExit(f"no checkpoint to resume: {last}")
        YOLO(str(last)).train(resume=True)
        return

    model = YOLO(args.model)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        epochs=args.epochs,
        amp=True,
        plots=True,
        workers=min(8, os.cpu_count() or 1),
        project=str(Path(args.project).resolve()),  # absolute: avoids ultralytics
        name=args.name,                             # double-nesting runs/run
    )


if __name__ == "__main__":
    main()
