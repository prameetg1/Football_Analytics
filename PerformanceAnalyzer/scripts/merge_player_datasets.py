#!/usr/bin/env python3
"""Merge multiple Roboflow YOLO-format player/ball detection datasets.

Both `football-players-detection-3zvbc` (v11, 1280x1280) and
`minaehyeon/football-player-jrjtj` (v5, 640x640) share the same four classes
(ball, goalkeeper, player, referee). This script merges their train/valid/test
splits into a single self-contained dataset with a canonical class mapping:

    data/player_detection_merged/
        train/images, train/labels
        valid/images, valid/labels
        test/images,  test/labels
        data.yaml

Class ids are remapped to the canonical order, filenames are prefixed with the
source tag to avoid collisions, and a merged data.yaml is written so the
Ultralytics trainer can ingest it directly:

    yolo detect train data=data/player_detection_merged/data.yaml ...

The ball is the tiny, hard class (~11px median). Because the inference pipeline
pads ball boxes by BALL_BOX_PAD_PX before tracking/projecting them (mirroring
the Roboflow reference), we optionally mirror that on the training side too:
`--ball-pad-scale` expands the ball (class 0) GT boxes around their centre so
the model learns the padded target it will actually be asked to track.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/merge_player_datasets.py \
        --out data/player_detection_merged \
        --source v11:data/player_detection_v11 \
        --source minae:data/player_detection_minaehyeon \
        --ball-pad-scale 2.5
"""

import argparse
import shutil
from pathlib import Path

CANONICAL_CLASSES = ["ball", "goalkeeper", "player", "referee"]
BALL_CLASS_ID = CANONICAL_CLASSES.index("ball")
SPLITS = ["train", "valid", "test"]


def parse_source(spec: str) -> tuple[str, Path]:
    tag, path = spec.split(":", 1)
    return tag, Path(path)


def read_class_names(data_yaml: Path) -> list[str]:
    """Parse class names from a Roboflow data.yaml.

    Handles both the bullet format (`- ball`) and the inline list format
    (`names: ['ball', 'goalkeeper', ...]`) that newer Roboflow exports use.
    """
    names: list[str] = []
    for line in data_yaml.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("names:"):
            inner = line[len("names:"):].strip()
            if inner.startswith("[") and inner.endswith("]"):
                inner = inner[1:-1]
            names.extend(
                t.strip().strip("'\"") for t in inner.split(",") if t.strip())
        elif line.startswith("-"):
            names.append(line[1:].strip().strip("'\""))
    return names


def pad_ball_box(parts: list[str], scale: float) -> list[str] | None:
    """Expand a ball GT box by `scale` around its centre; clamp to [0, 1].

    Takes a YOLO-format label line (class + normalized cx cy w h) and returns
    the padded line, or None if the padded box is degenerate (e.g. pushed off
    the image edge).
    """
    if scale <= 1.0:
        return parts
    cx, cy, w, h = (float(parts[1]), float(parts[2]),
                    float(parts[3]), float(parts[4]))
    nw = w * scale
    nh = h * scale
    nx = max(0.0, min(1.0, cx - (nw - w) / 2.0))
    ny = max(0.0, min(1.0, cy - (nh - h) / 2.0))
    nw = max(0.0, min(1.0 - nx, nw))
    nh = max(0.0, min(1.0 - ny, nh))
    if nw <= 0.0 or nh <= 0.0:
        return None
    return [parts[0], f"{nx:.6f}", f"{ny:.6f}", f"{nw:.6f}", f"{nh:.6f}"]


def merge(args) -> None:
    sources = [parse_source(s) for s in args.source]
    if not sources:
        raise SystemExit("no --source given")

    out = Path(args.out)
    canonical = {name: i for i, name in enumerate(CANONICAL_CLASSES)}

    summary = {}
    for tag, src in sources:
        data_yaml = src / "data.yaml"
        names = read_class_names(data_yaml)
        if not names:
            raise SystemExit(f"{tag}: no class names parsed from {data_yaml}")
        src_to_canon = {name: canonical.get(name) for name in names}
        missing = [k for k, v in src_to_canon.items() if v is None]
        if missing:
            raise SystemExit(
                f"{tag}: classes not in canonical set {CANONICAL_CLASSES}: {missing}")

        for split in SPLITS:
            img_dir = src / split / "images"
            if not img_dir.is_dir():
                print(f"[{tag}] no '{split}' split, skipping")
                continue
            n_copied = 0
            n_skipped = 0
            for img in sorted(img_dir.glob("*")):
                if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                lab = img.with_suffix(".txt")
                if lab.parent.name != "labels":
                    lab = src / split / "labels" / (img.stem + ".txt")

                new_stem = f"{tag}__{img.stem}"
                dst_img = out / split / "images" / (new_stem + img.suffix)
                dst_lab = out / split / "labels" / (new_stem + ".txt")
                dst_img.parent.mkdir(parents=True, exist_ok=True)
                dst_lab.parent.mkdir(parents=True, exist_ok=True)

                shutil.copy2(img, dst_img)
                lines: list[str] = []
                if lab.is_file():
                    for raw in lab.read_text().splitlines():
                        parts = raw.split()
                        if not parts:
                            continue
                        try:
                            cls_name = names[int(parts[0])]
                        except (ValueError, IndexError):
                            continue
                        cid = src_to_canon.get(cls_name)
                        if cid is None:
                            continue
                        out_parts = [str(cid)] + parts[1:]
                        if cid == BALL_CLASS_ID:
                            out_parts = pad_ball_box(out_parts, args.ball_pad_scale)
                        if out_parts is None:
                            continue
                        lines.append(" ".join(out_parts))
                dst_lab.write_text("\n".join(lines))
                n_copied += 1

            key = f"{tag}/{split}"
            summary[key] = n_copied
            print(f"[{tag}] {split}: {n_copied} images")

    # merged data.yaml
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\n"
        f"train: train/images\n"
        f"val: valid/images\n"
        f"test: test/images\n\n"
        f"names: {CANONICAL_CLASSES}\n"
        f"nc: {len(CANONICAL_CLASSES)}\n"
    )
    print(f"\nmerged data.yaml written to {out / 'data.yaml'}")
    print(f"total: {sum(summary.values())} images across {len(summary)} split/source pairs")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--source", action="append", required=True,
                   help="tag:path pairs, e.g. --source v11:data/player_detection_v11")
    p.add_argument("--ball-pad-scale", type=float, default=1.0,
                   help="expand ball (class 0) GT boxes by this factor around "
                        "their centre (e.g. 2.5); 1.0 = unchanged")
    merge(p.parse_args())


if __name__ == "__main__":
    main()
