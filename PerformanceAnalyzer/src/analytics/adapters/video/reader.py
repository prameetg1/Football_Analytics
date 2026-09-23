"""Video frame I/O and metadata.

Thin OpenCV wrapper aligned with supervision's frame-generator pattern so we
can drop in `sv.get_video_frames_generator` later without changing callers.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2


@dataclass
class VideoInfo:
    """Static properties of a source video."""
    width: int
    height: int
    fps: float
    total_frames: int

    @classmethod
    def from_path(cls, path: str) -> "VideoInfo":
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {path}")
        info = cls(
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(cap.get(cv2.CAP_PROP_FPS)),
            total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
        cap.release()
        return info


def frame_generator(source_path: str, stride: int = 1, start_frame: int = 0):
    """Yield frames from a video.

    Args:
        source_path: path to a video file.
        stride: yield one frame every `stride` frames (e.g. 30 == ~1fps @ 30fps).
        start_frame: first frame index to begin at.
    """
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {source_path}")

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index >= start_frame and (frame_index - start_frame) % stride == 0:
            yield frame
        frame_index += 1
    cap.release()


def write_video(output_path: str, frames, fps: float,
                width: int | None = None, height: int | None = None) -> str:
    """Write an iterable of BGR frames to an mp4 using the H.264 codec."""
    import os

    first = next(frames, None)
    if first is None:
        raise ValueError("No frames to write")
    width = width or int(first.shape[1])
    height = height or int(first.shape[0])

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    writer.write(first)
    for frame in frames:
        writer.write(frame)
    writer.release()
    return output_path