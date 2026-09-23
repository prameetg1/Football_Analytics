"""Object detection — football-entity bounding boxes.

Wraps a fine-tuned YOLO (Ultralytics) model. When no custom weights are given
we load a pretrained fallback so the pipeline is runnable end-to-end; production
uses the fine-tuned model produced by `References/train_player_detector.py`.

Output objects are plain dataclasses (no hard dependency on a specific tracker /
annotation library) so detection results are easy to store, test and pass to the
tracking and team stages.
"""

from __future__ import annotations

import numpy as np

from src.config import (
    DETECTION_CONFIDENCE,
    DETECTION_FALLBACK_MODEL,
    DETECTION_IMAGE_SIZE,
    DETECTION_MODEL_WEIGHTS,
    DETECTION_ROI,
)
from src.analytics.adapters.video.detection.detection import Detection


def _as_tuple(xy) -> tuple[float, float, float, float]:
    arr = np.asarray(xy, dtype=np.float64).ravel()
    return (float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))


def _iou_xyxy(a: tuple[float, float, float, float],
              b: tuple[float, float, float, float]) -> float:
    xi1, yi1 = max(a[0], b[0]), max(a[1], b[1])
    xi2, yi2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    if inter <= 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


class Detector:
    """Object detector — local YOLO (.pt) OR hosted Roboflow model id.

    Args:
        weights: a local '.pt' path for Ultralytics inference, OR a Roboflow
            hosted model id (e.g. "org/project/version") that does not end in
            .pt (implies hosted inference with ROBOFLOW_API_KEY).
        confidence: detection confidence threshold.
        imgsz: inference input size (pixels); used only for the local path.
    """

    def __init__(self, weights: str | None = None, confidence: float | None = None,
                 imgsz: int | None = None, roi: tuple[float, float, float, float] | None = None):
        self.weights = weights or DETECTION_MODEL_WEIGHTS
        self.confidence = confidence if confidence is not None else DETECTION_CONFIDENCE
        self.imgsz = imgsz or DETECTION_IMAGE_SIZE
        # Stage 5: optional region of interest (normalized 0..1 fractions).
        self.roi = roi if roi is not None else DETECTION_ROI

        from src.config import ROBOFLOW_API_KEY
        self._api_key = ROBOFLOW_API_KEY
        self._hosted = bool(self.weights) and not self.weights.endswith(".pt")

        if self._hosted:
            self._build_hosted()
        else:
            self._build_local()

    def _roi_to_pixels(self, frame: np.ndarray) -> tuple[int, int, int, int] | None:
        if self.roi is None:
            return None
        h, w = frame.shape[:2]
        x1 = int(self.roi[0] * w)
        y1 = int(self.roi[1] * h)
        x2 = int(self.roi[2] * w)
        y2 = int(self.roi[3] * h)
        return max(0, x1), max(0, y1), max(x1, min(w, x2)), max(y1, min(h, y2))

    def _infer_cropped(self, frame: np.ndarray, roi_px: tuple[int, int, int, int]) -> list[Detection]:
        """Infer on the ROI crop; shift boxes back to full-frame coordinates."""
        x1, y1, x2, y2 = roi_px
        crop = frame[y1:y2, x1:x2]
        if self._hosted:
            dets = self._infer_hosted(crop)
        else:
            dets = self._infer_local(crop)
        for d in dets:
            d.xyxy = (d.x1 + x1, d.y1 + y1, d.x2 + x1, d.y2 + y1)
        return dets

    def infer(self, frame: np.ndarray) -> list[Detection]:
        roi_px = self._roi_to_pixels(frame)
        if roi_px is not None:
            return self._infer_cropped(frame, roi_px)
        if self._hosted:
            return self._infer_hosted(frame)
        return self._infer_local(frame)

    def _build_local(self):
        from ultralytics import YOLO
        model_path = self.weights or DETECTION_FALLBACK_MODEL
        self._model = YOLO(model_path)
        self.class_names = self._model.names

    def _build_hosted(self):
        if not self._api_key:
            raise RuntimeError(
                f"{self.weights!r} is a hosted id but ROBOFLOW_API_KEY is not set "
                "(add it to .env or the environment)."
            )
        from inference import get_model
        self._model = get_model(self.weights, api_key=self._api_key)
        self.class_names = getattr(self._model, "class_names", None)

    def infer(self, frame: np.ndarray) -> list[Detection]:
        roi_px = self._roi_to_pixels(frame)
        if roi_px is not None:
            return self._infer_cropped(frame, roi_px)
        if self._hosted:
            return self._infer_hosted(frame)
        return self._infer_local(frame)

    def infer_region(self, frame: np.ndarray, center_xy: tuple[float, float],
                     half: float, imgsz: int | None = None,
                     conf: float | None = None) -> list[Detection]:
        """Infer on a square window centred at `center_xy` in full-frame pixels.

        The window is cropped, optionally upscaled for inference (`imgsz`), and
        the detections are shifted back into full-frame coordinates. Used by the
        ball-recovery pass to give a tiny ball many more pixels than the
        full-frame run. `half` (px) bounds the crop on all sides.
        """
        h, w = frame.shape[:2]
        cx, cy = center_xy
        x1 = int(max(0.0, cx - half))
        y1 = int(max(0.0, cy - half))
        x2 = int(min(w, cx + half))
        y2 = int(min(h, cy + half))
        if x2 - x1 < 32 or y2 - y1 < 32:
            return []
        crop = frame[y1:y2, x1:x2]
        if self._hosted:
            dets = self._infer_hosted(crop)
        else:
            dets = self._infer_local(crop, imgsz=imgsz or self.imgsz,
                                     conf=conf if conf is not None else self.confidence)
        for d in dets:
            d.xyxy = (d.x1 + x1, d.y1 + y1, d.x2 + x1, d.y2 + y1)
        return dets

    def infer_tiled(self, frame: np.ndarray, tile: int = 640,
                    imgsz: int | None = None, conf: float | None = None,
                    overlap: float = 0.25,
                    classes: set[int] | None = None) -> list[Detection]:
        """SAHI-style tiled inference; boxes merged back to full-frame pixels.

        Runs the model on overlapping upscaled tiles so small objects (the
        ball) get more pixels than a single full-frame pass at `imgsz`.
        `classes` optionally restricts each tile's output (e.g. {ball_id}).
        Duplicate detections across overlapping tiles are collapsed with
        class-aware NMS so the same object is counted once.
        """
        h, w = frame.shape[:2]
        step = int(tile * (1.0 - overlap))
        ys = list(range(0, max(1, h - tile), step))
        if not ys or ys[-1] + tile < h:
            ys.append(max(0, h - tile))
        xs = list(range(0, max(1, w - tile), step))
        if not xs or xs[-1] + tile < w:
            xs.append(max(0, w - tile))
        out: list[Detection] = []
        for y in ys:
            for x in xs:
                dets = self.infer_region(frame, (x + tile / 2, y + tile / 2),
                                         tile / 2, imgsz=imgsz, conf=conf)
                for d in dets:
                    if classes is not None and d.class_id not in classes:
                        continue
                    out.append(d)
        return self.nms(out)

    @staticmethod
    def nms(dets: list[Detection], iou_thr: float = 0.5) -> list[Detection]:
        """Class-aware greedy NMS over full-frame detection boxes.

        Keeps the highest-confidence box per class and suppresses any box of the
        same class with IoU above `iou_thr`. Collapses duplicate detections from
        overlapping tiles while letting distinct objects of the same class pass.
        """
        if not dets:
            return []
        order = sorted(dets, key=lambda d: d.confidence, reverse=True)
        keep: list[Detection] = []
        for d in order:
            dup = False
            for k in keep:
                if k.class_id != d.class_id:
                    continue
                if _iou_xyxy(d.xyxy, k.xyxy) > iou_thr:
                    dup = True
                    break
            if not dup:
                keep.append(d)
        return keep

    def _infer_local(self, frame: np.ndarray, imgsz: int | None = None,
                     conf: float | None = None) -> list[Detection]:
        results = self._model(
            frame, conf=conf if conf is not None else self.confidence,
            imgsz=imgsz or self.imgsz,
            classes=None, verbose=False,
        )
        detections: list[Detection] = []
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return detections
        boxes = r.boxes.xyxy.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        names = self._model.names
        for xyxy, c, conf in zip(boxes, clss, confs):
            detections.append(Detection(
                class_id=int(c),
                class_name=names.get(int(c), str(c)),
                confidence=float(conf),
                xyxy=_as_tuple(xyxy),
            ))
        return detections

    def _infer_hosted(self, frame: np.ndarray) -> list[Detection]:
        import supervision as sv
        result = self._model.infer(frame, confidence=self.confidence)[0]
        detections = sv.Detections.from_inference(result)
        out = []
        for i in range(len(detections)):
            xyxy = detections.xyxy[i]
            cid = int(detections.class_id[i])
            conf = float(detections.confidence[i])
            names = detections.data.get("class_name")
            name = names[i] if names is not None and len(names) > 0 else str(cid)
            out.append(Detection(
                class_id=cid, class_name=str(name),
                confidence=conf, xyxy=_as_tuple(xyxy),
            ))
        return out