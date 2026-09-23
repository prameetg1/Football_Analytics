"""Locate kick-off in the BBC first-half video: ball projected near centre spot."""
import cv2, numpy as np, json
from src.analytics.adapters.video.homography.keypoint_detector import PitchKeypointDetector
from src.analytics.adapters.video.homography.pitch_model import PitchConfig
from src.analytics.adapters.video.detection.detector import Detector
from src.config import PITCH_KEYPOINT_CONFIDENCE, CLASS_PLAYER, CLASS_BALL

verts = PitchConfig().vertices
cap = cv2.VideoCapture("data/raw/wc2022_final_bbc_1st_half.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
kdet = PitchKeypointDetector()
pdet = Detector(weights=None)

cands = []
i = 0
STRIDE = int(round(fps * 3))
while True:
    ok, f = cap.read()
    if not ok:
        break
    if i % STRIDE == 0:
        kp = kdet.detect(f).filtered(PITCH_KEYPOINT_CONFIDENCE)
        if len(kp.xy) >= 6:
            Hm, inl = cv2.findHomography(kp.xy.astype(np.float64), verts[kp.indices],
                                         cv2.RANSAC, 3.0)
            if Hm is not None and inl is not None and inl.sum() >= 8:
                dets = pdet.infer(f)
                balls = [d for d in dets if d.class_name == CLASS_BALL]
                players = [d for d in dets if d.class_name == CLASS_PLAYER
                           and d.confidence > 0.35]
                if balls and len(players) >= 16:
                    b = max(balls, key=lambda d: d.confidence)
                    px = np.array([[(b.xyxy[0]+b.xyxy[2])/2,
                                    (b.xyxy[1]+b.xyxy[3])/2]], dtype=np.float32)
                    P = cv2.perspectiveTransform(px.reshape(-1,1,2),
                                                 Hm.astype(np.float32))[0,0]
                    dist = float(np.linalg.norm(P - np.array([60,35])))
                    if dist < 7:
                        cands.append({"t": i/fps, "dist": dist,
                                      "players": len(players)})
                        print(f"t={i/fps:7.1f}s dist={dist:.1f}m players={len(players)}",
                              flush=True)
    i += 1
cap.release()
json.dump(cands, open("/tmp/kickoff_bbc.json", "w"))
print("DONE", len(cands), "candidates", flush=True)
