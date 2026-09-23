"""Compare a detection model against the held-out test split.

A single, consistent COCO-style evaluator so the bake-off is apples-to-apples:
the SAME code scores a local `.pt` model and a hosted Roboflow id on the SAME
91-image test split, producing per-class Precision, Recall (at the operating
confidence) and mAP50 / mAP50-95.

Usage:
    python scripts/eval_model_on_test.py --model runs/detect/merged/weights/best.pt
    python scripts/eval_model_on_test.py --model football-players-detection-3zvbc/11
    python scripts/eval_model_on_test.py --model <...> --conf 0.3 --iou0 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.config import PLAYER_DETECTION_DATA
from src.analytics.adapters.video.detection.detection import Detection
from src.analytics.adapters.video.detection.detector import Detector


def _load_data_and_labels(yaml_path: str):
    """Return (images, names, gt) where gt[image] = {class_id: [gt boxes]}.

    Boxes are stored in normalized (0..1) form and converted to raw-image
    pixel space per image (test images are not all the same resolution).
    """
    import yaml

    data = yaml.safe_load(Path(yaml_path).read_text())
    names = data["names"]
    root = Path(data["path"])
    split_dir = root / "test"
    images = sorted((split_dir / "images").glob("*.jpg"))
    labels = split_dir / "labels"

    gt: dict[str, dict[int, list[np.ndarray]]] = {}
    for img_path in images:
        lab = labels / (img_path.stem + ".txt")
        rows = []
        if lab.exists():
            for line in lab.read_text().splitlines():
                c, cx, cy, w, h = (float(v) for v in line.split())
                rows.append((int(c), np.array([cx, cy, w, h])))
        gt[str(img_path)] = rows
    return images, names, gt


def _gt_to_pixels(norm_box: np.ndarray, w: int, h: int) -> np.ndarray:
    """Normalized (cx, cy, w, h) -> pixel (x1, y1, x2, y2) in (w, h) space."""
    cx, cy, w_n, h_n = norm_box
    x1, y1 = (cx - w_n / 2) * w, (cy - h_n / 2) * h
    x2, y2 = (cx + w_n / 2) * w, (cy + h_n / 2) * h
    return np.array([x1, y1, x2, y2])


def _load_gt_count(gt: dict, class_ids: list[int]) -> dict[int, int]:
    counts = {c: 0 for c in class_ids}
    for rows in gt.values():
        for c, _box in rows:
            counts[c] += 1
    return counts


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    xi1 = max(a[0], b[0]); yi1 = max(a[1], b[1])
    xi2 = min(a[2], b[2]); yi2 = min(a[3], b[3])
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _ap_at_iou(scored_preds: list[tuple[float, float, int]], n_gt: int,
               iou_thr: float) -> float:
    """Area under the precision-recall curve at a fixed IoU threshold.

    scored_preds: (confidence, best_match_iou, best_match_gt_idx) sorted by
    confidence descending. Greedy matching: each ground-truth box counts at
    most once (COCO semantics). Precision-recall uses COCO-style 101-point
    recall interpolation -> trapezoidal area.
    """
    preds = sorted(scored_preds, key=lambda p: p[0], reverse=True)
    if n_gt == 0 or not preds:
        return 0.0

    matched: set[int] = set()
    tps = []
    for _conf, iou, gt_idx in preds:
        if iou >= iou_thr and gt_idx not in matched:
            matched.add(gt_idx)
            tps.append(1)
        else:
            tps.append(0)

    cum_tp = np.cumsum(tps)
    cum_fp = np.cumsum([1 - t for t in tps])
    recalls = cum_tp / n_gt
    precs = cum_tp / np.maximum(cum_tp + cum_fp, 1)

    # COCO: max precision over recalls >= r (right), sampled at 101 points.
    r_points = np.linspace(0.0, 1.0, 101)
    prec_interp = np.array([
        precs[recalls >= r].max() if np.any(recalls >= r) else 0.0
        for r in r_points
    ])
    return float(np.trapz(prec_interp, r_points))


def evaluate(model: str, conf: float, imgsz: int = 1280,
             tiled: bool = False, tile: int = 640) -> dict:
    det = Detector(weights=model, confidence=conf, imgsz=imgsz)
    images, names, gt = _load_data_and_labels(PLAYER_DETECTION_DATA)
    class_ids = list(range(len(names)))
    ball_id = next(c for c in class_ids if names[c] == "ball")

    # per-class: (confidence, best_match_iou, best_match_gt_idx) per prediction
    # for AP, plus operating-point TP/FP (greedy) for P & R at `conf`.
    pred_meta = {c: [] for c in class_ids}
    op_matches: dict[int, set[tuple[str, int]]] = {c: set() for c in class_ids}
    op_fp = {c: 0 for c in class_ids}
    gt_count = _load_gt_count(gt, class_ids)

    for img_path in images:
        frame = cv2.imread(str(img_path))
        h, w = frame.shape[:2]
        key = str(img_path)
        dets: list[Detection] = det.infer(frame)
        # Tiled ball recovery: the full-frame pass misses tiny balls; add
        # ball detections from overlapping upscaled tiles, then NMS so the same
        # ball found by both passes is counted once.
        if tiled:
            dets = det.nms(dets + det.infer_tiled(frame, tile=tile, imgsz=imgsz,
                                                  conf=conf, classes={ball_id}))
        for d in dets:
            c = d.class_id
            if c not in class_ids:
                continue
            box = np.asarray(d.xyxy, dtype=np.float64)
            best_iou, best_gt = 0.0, -1
            for gi, (gc, gbox) in enumerate(gt[key]):
                if gc != c:
                    continue
                iou = _iou(box, _gt_to_pixels(gbox, w, h))
                if iou > best_iou:
                    best_iou, best_gt = iou, gi
            pred_meta[c].append((d.confidence, best_iou, (key, best_gt)))
            if best_iou >= 0.5 and (key, best_gt) not in op_matches[c]:
                op_matches[c].add((key, best_gt))
            else:
                op_fp[c] += 1

    ious = [round(0.5 + 0.05 * i, 2) for i in range(10)]
    metrics = {}
    for c in class_ids:
        n_gt = gt_count[c]
        prec = len(op_matches[c]) / max(len(op_matches[c]) + op_fp[c], 1)
        rec = len(op_matches[c]) / max(n_gt, 1)
        ap50 = _ap_at_iou(pred_meta[c], n_gt, 0.5)
        ap5095 = np.mean([_ap_at_iou(pred_meta[c], n_gt, t) for t in ious])
        metrics[c] = {
            "name": names[c], "images": len(images), "instances": n_gt,
            "P@conf": prec, "R@conf": rec, "mAP50": ap50, "mAP50-95": ap5095,
        }

    all_m = metrics
    all_ap50 = np.mean([m["mAP50"] for m in all_m.values()])
    all_ap5095 = np.mean([m["mAP50-95"] for m in all_m.values()])
    metrics[-1] = {
        "name": "all", "images": len(images),
        "instances": sum(gt_count.values()),
        "P@conf": np.mean([m["P@conf"] for m in all_m.values()]),
        "R@conf": np.mean([m["R@conf"] for m in all_m.values()]),
        "mAP50": all_ap50, "mAP50-95": all_ap5095,
    }
    return metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="local .pt path or hosted Roboflow model id")
    p.add_argument("--conf", type=float, default=0.3,
                   help="operating confidence (also the sweep floor)")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--tiled", action="store_true",
                   help="add SAHI-style tiled ball recovery to the full-frame pass")
    p.add_argument("--tile", type=int, default=640,
                   help="tile size (px) for --tiled recovery")
    args = p.parse_args()

    print(f"Evaluating {args.model} on test split (conf floor {args.conf}"
          + (f", tiled ball recovery tile={args.tile}" if args.tiled else "")
          + ")")
    metrics = evaluate(args.model, args.conf, args.imgsz,
                       tiled=args.tiled, tile=args.tile)

    header = f"{'class':<12}{'imgs':>5}{'inst':>6}{'P@c':>8}{'R@c':>8}{'mAP50':>8}{'mAP50-95':>9}"
    print(header)
    print("-" * len(header))
    for c in sorted(metrics, key=lambda k: (-(k == -1), k)):
        m = metrics[c]
        print(f"{m['name']:<12}{m['images']:>5}{m['instances']:>6}"
              f"{m['P@conf']:>8.3f}{m['R@conf']:>8.3f}"
              f"{m['mAP50']:>8.3f}{m['mAP50-95']:>9.3f}")


if __name__ == "__main__":
    main()