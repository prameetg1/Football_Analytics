"""Pipeline configuration for PerformanceAnalyzer.

Central home for weights, model ids, class maps, paths and thresholds so the
pipeline stays deterministic and reproducible (matching DefenseAnalytics' config
conventions).
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Secrets (read from environment; optionally load a local .env).
# ---------------------------------------------------------------------------
def _load_dotenv():
    candidates = [
        PROJECT_ROOT / ".env",                 # project-level override
        PROJECT_ROOT.parent / ".env",          # shared Football_Analytics/.env
    ]
    for dotenv_path in candidates:
        if not dotenv_path.exists():
            continue
        for line in dotenv_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip().strip(), val.strip())


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_load_dotenv()

ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "").strip()
# HF key appears under several names depending on how it was exported; accept
# the common aliases and prefer HF_TOKEN when set.
HF_TOKEN = (os.environ.get("HF_TOKEN")
            or os.environ.get("HuggingFace_token")
            or os.environ.get("HUGGINGFACE_HUB_TOKEN")
            or os.environ.get("HUGGINGFACE_TOKEN")
            or "").strip()

# ---------------------------------------------------------------------------
# Class map (matches the Roboflow football-players-detection dataset, order
# is the model's class_id: 0..3). Mapped by name for robustness across exports.
# ---------------------------------------------------------------------------
CLASS_BALL = "ball"
CLASS_GOALKEEPER = "goalkeeper"   # dataset label may be "goalkeeper" or "GK"
CLASS_PLAYER = "player"
CLASS_REFEREE = "referee"

# Minimal set of labels we semantically care about (everything else ignored).
INTERESTING_CLASSES = {CLASS_BALL, CLASS_GOALKEEPER, CLASS_PLAYER, CLASS_REFEREE}

# ---------------------------------------------------------------------------
# --- Object detection -------------------------------------------------------
# ---------------------------------------------------------------------------
# Path to a fine-tuned YOLO weights file (e.g. best.pt produced by
# References/train_player_detector.py). If None, we fall back to a
# downloaded pretrained Ultralytics model for smoke-testing only.
#
# Winner of the Stage 1 bake-off (local best.pt beat hosted v11 on ball
# mAP50/recall and every other class on the shared test split).
#
# To use the HOSTED fine-tuned football detector instead, set this to a
# Roboflow model id like "roboflow-jvuqo/football-players-detection-3zvbc/11"
# and provide ROBOFLOW_API_KEY (env or .env). The Detector auto-detects
# "hosted id" (no .pt) vs "local .pt path".
DETECTION_MODEL_WEIGHTS = "runs/detect/merged/weights/best.pt"
DETECTION_FALLBACK_MODEL = "yolov8s.pt"   # pretrained fallback (not football-tuned)

# Hosted-fine-tuned reference (enabled when DETECTION_MODEL_WEIGHTS is set to
# a Roboflow id OR --detector is passed one; see Detector).
ROBOFLOW_PLAYER_MODEL_ID = "football-players-detection-3zvbc/11"
PITCH_KEYPOINT_MODEL_ID = "football-field-detection-f07vi/14"

DETECTION_CONFIDENCE = 0.3
DETECTION_IOU = 0.5
DETECTION_IMAGE_SIZE = 1280

# Player/ball training data: merged multi-source dataset (see
# scripts/merge_player_datasets.py + scripts/train_player_detector.py).
PLAYER_DETECTION_DATA = "data/player_detection_merged/data.yaml"

# Ball is handled separately: pad the box and keep it out of the tracker.
BALL_BOX_PAD_PX = 10

# Stage 4b: ball-recovery second pass. The full-frame detector misses tiny
# balls (test recall 0.705) because they occupy only a few pixels at imgsz.
# When the frame's best ball detection falls below BALL_RECOVERY_MIN_CONF, we
# re-run inference on an upscaled window centred on the Kalman-predicted ball
# position (back-projected to pixels) so the small ball gets far more pixels.
BALL_RECOVERY_ENABLED = True
BALL_RECOVERY_MIN_CONF = 0.5       # full-frame best-ball conf below this -> recover
BALL_RECOVERY_WINDOW_PX = 640      # window size (square) around the prediction
BALL_RECOVERY_IMGSZ = 1280         # upscaled inference size for the window
BALL_RECOVERY_CONF = 0.15          # lower floor inside the recovery window

# Stage 4: ball-hypothesis Kalman tracker (pitch-plane, metres).
# Reserved tracker_id used for the ball's hypothesis row in exports.
BALL_TRACKER_ID = -1

# Video/CV provider adapter is DORMANT by default (parked during the pivot).
# Set this to True to let `VideoAdapter` run the CV pipeline against video and
# emit normalized events instead of raising `VideoAdapterDormant`.
VIDEO_ADAPTER_ENABLED = False
# Constant-velocity Kalman tuning. process_noise scales the white-acceleration
# process model (higher = trusts motion more, smoother predictions through
# occlusions); measurement_noise is the projection noise in m.
BALL_KALMAN_PROCESS_NOISE = 2.0
BALL_KALMAN_MEASUREMENT_NOISE = 1.0
# Mahalanobis association gate (m): a projected ball farther than this from the
# prediction is treated as a new ball event (pass/restart) rather than a track
# update — also stops homography glitches teleporting the trail.
BALL_ASSOCIATION_GATE_M = 8.0
# Frames the tracker keeps predicting through a detection miss before the
# hypothesis is declared lost (and the row disappears).
BALL_MAX_PREDICT_FRAMES = 5

# Max allowed per-frame jump (metres) in a tracked object's pitch position
# before it is treated as a projection outlier and dropped (the reference uses
# a 500 cm threshold for its ball path; we express it in metres).
PITCH_MAX_STEP_M = 5.0

# ---------------------------------------------------------------------------
# Tracking (ByteTrack-like)
# ---------------------------------------------------------------------------
TRACK_MIN_CONFIDENCE = 0.3
TRACK_CLASS_AGNOSTIC_NMS = True
TRACK_NMS_THRESHOLD = 0.5
TRACK_MAX_AGE = 30          # frames a track may be lost before expiry

# ---------------------------------------------------------------------------
# Team classification (SigLIP -> UMAP -> KMeans)
# ---------------------------------------------------------------------------
SIGLIP_MODEL = "google/siglip-base-patch16-224"
TEAM_CROP_STRIDE = 30       # sample one frame every N frames when bootstrapping
N_TEAMS = 2                 # two outfield teams; referee is an excluded cluster
UMAP_N_COMPONENTS = 3
EMBEDDING_BATCH = 32
TEAM_BOOTSTRAP_SECONDS = 30  # seconds of footage used to fit the classifier
TEAM_BOOTSTRAP_CROPS = 256   # player crops collected before fitting once
DEVICE = "cuda"

# Stage 5: adaptive team re-fit (drift detection + bounded refit).
# Rolling window of player crops used for periodic drift checks.
TEAM_REFIT_WINDOW_CROPS = 384
# Run a drift check every N frames after bootstrap.
TEAM_REFIT_CHECK_INTERVAL_FRAMES = 90
# Refit when the recent window's cluster tightness exceeds the bootstrap
# baseline by this ratio (kit/lighting drift makes appearance embeddings
# scatter, inflating within-cluster distances).
TEAM_REFIT_TIGHTNESS_RATIO = 1.6
# Half-precision SigLIP embeddings (halves VRAM/latency on CUDA).
SIGLIP_FP16 = True

# Stage 5: adaptive frame-rate + optional ROI.
# Detection/tracking run at the source's full FPS (track continuity), but the
# exported tracking table / metric layer are sampled down to this rate to keep
# metric compute (and table size) sane. Common football-analysis rates are
# 4-6 Hz. 0/None disables downsampling.
METRICS_EXPORT_HZ = 5.0
# Optional detector region of interest as normalized (0..1) fractions of the
# frame: (x1, y1, x2, y2). Detections are cropped to it before inference and
# boxes are shifted back to full-frame coordinates; anything fully outside is
# dropped. None == full frame (no ROI).
DETECTION_ROI = None

# ---------------------------------------------------------------------------
# Stage 7: StatsBomb ground-truth gate.
# ---------------------------------------------------------------------------
# Base URL of the StatsBomb open-data mirror (raw GitHub file paths are joined
# onto this). Used by src/statsbomb/loader.py to fetch match data.
STATSBOMB_DATA_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
STATSBOMB_DATA_DIR = PROJECT_ROOT / "data" / "statsbomb"
# Timestamp alignment: StatsBomb events carry mm:ss.ms since kick-off; we map
# them onto the tracking table via this nominal fps (frame_sec = clock_sec).
STATSBOMB_NOMINAL_FPS = 25.0
# Coordinate conversion: StatsBomb uses x in [0,120], y in [0,80]. We rescale
# the width to our canonical pitch [0,70] (see PitchConfig) so GT and CV live
# on the same metres. 120-x reframes attacker->defender direction.
STATSBOMB_PITCH_LENGTH = 120.0
STATSBOMB_PITCH_WIDTH = 80.0
# CV-vs-GT validation thresholds for the metrics gate. A metric table is only
# surfaced when the gate is enabled AND the match's validation report stays
# inside these bounds.
METRICS_GATE_ENABLED = False
GATE_MAX_POSITION_ERROR_M = 5.0        # mean per-player position error (m)
GATE_MIN_PASS_RECALL = 0.4             # fraction of GT passes CV recovered
GATE_MIN_PASS_PRECISION = 0.4          # fraction of CV passes that match a GT pass
# Generic per-event-class thresholds (shots, turnovers, clearances, ...) used
# by the Stage-9 gate. Reuse the pass bounds unless a class needs stricter ones.
GATE_MIN_EVENT_RECALL = 0.3
GATE_MIN_EVENT_PRECISION = 0.3

# ---------------------------------------------------------------------------
# Stage 9: event extraction from the tracking table.
# ---------------------------------------------------------------------------
# A "pass" is registered when the ball-owner changes to another player of the
# same team AND the ball has travelled at least this far (m) between the two
# ownership samples (kills per-frame ownership flicker being read as passes).
EVENT_MIN_PASS_MOVE_M = 2.0
# A same-team owner change is a pass; a cross-team owner change is a turnover.
# A turnover also implies a regain for the gaining team.
EVENT_MIN_TURNOVER_MOVE_M = 2.0
# Shot detection: the ball must enter the goal mouth (distance from the goal
# line <= this, m) at at least this speed (m/s) to count as a shot.
EVENT_SHOT_GOAL_LINE_M = 4.0
EVENT_SHOT_MIN_SPEED_MS = 6.0
# The last owner before a shot is attributed as the shooter's team.
# After a shot is registered, ignore further ball-in-mouth samples until the
# ball leaves the mouth again (avoid counting the same shot multiple times).

# Clearance: a ball leaving the defensive third (x <= this) that reaches the
# halfway line (x >= this) quickly (<= seconds).
EVENT_CLEARANCE_DEFENSIVE_THIRD_M = 40.0
EVENT_CLEARANCE_MIDFIELD_M = 60.0
EVENT_CLEARANCE_MAX_SEC = 3.0

# Pressure: an opposition player within this many metres of the ball while it
# is owned, for at least this many consecutive ball samples.
EVENT_PRESSURE_RADIUS_M = 3.0
EVENT_PRESSURE_MIN_SAMPLES = 2

# Carry / dribble: the same owner keeps the ball while it travels at least
# this many metres without an ownership change.
EVENT_CARRY_MIN_M = 3.0
EVENT_DRIBBLE_MIN_M = 3.0          # >= carry distance counts as a dribble run

# Duel: both teams have a player within this radius of a *loose* ball. A duel
# is reported once per contiguous run of at least `EVENT_DUEL_MIN_FRAMES`
# frames where the condition holds (per-frame noise and double-counting).
EVENT_DUEL_RADIUS_M = 5.0
EVENT_DUEL_MIN_FRAMES = 2

# Through-ball: a pass whose receiver is beyond the opposition's last man by
# at least this margin (m). The defensive line is the 3 deepest outfielders.
EVENT_THROUGH_BALL_LINE_M = 15.0
EVENT_THROUGH_BALL_DEFENDERS = 3

# ---------------------------------------------------------------------------
# Stage 9, Stage C: pitch control / EPV.
# ---------------------------------------------------------------------------
# Pitch control (Spearman-style) grid resolution (cells).
PITCH_CONTROL_NX = 24
PITCH_CONTROL_NY = 16
# C1 pitch control, physics model: each point on the pitch is a "race" players
# run with constant max acceleration `PITCH_CONTROL_A_MAX_MS2` (m/s^2), given
# their current velocity. Control goes to whichever team reaches a point first;
# `PITCH_CONTROL_TIME_SIGMA` (s) is the logistic sharpness over the arrival-time
# gap (uncertainty in the race outcome).
PITCH_CONTROL_A_MAX_MS2 = 7.0
PITCH_CONTROL_TIME_SIGMA = 0.3
# EPV value iteration: how far ahead the possession value horizon is, in
# transition steps, and the scoring-value discount applied each step.
EPV_HORIZON_STEPS = 15
EPV_DISCOUNT = 0.9
# Probability the ball stays on the same team after each transition step
# (opposition turnover at each step = 1 - this). Used by the EPV Markov chain.
EPV_KEEP_PROB = 0.85
# Distance (m) a player "carries" the value in a single transition step.
EPV_STEP_M = 5.0
# C2 scoring probability: fitted logistic shot model (real StatsBomb shots).
# The model file is produced by `scripts/fit_shot_model.py`; when absent, the
# hand-set exponential fallback is used.
EPV_SHOT_MODEL_PATH = STATSBOMB_DATA_DIR / "shot_model.json"
# C5 space creation: a run counts when the runner stays in cells where the
# opponent's pitch control is below this threshold, advancing at least
# `SPACE_CREATION_MIN_RUN_M` over consecutive frames (gaps up to
# `SPACE_CREATION_FRAME_GAP` tolerated).
SPACE_CREATION_OPP_CONTROL_MAX = 0.3
SPACE_CREATION_MIN_RUN_M = 5.0
SPACE_CREATION_FRAME_GAP = 2

# ---------------------------------------------------------------------------
# Stage 9, Stage D: team shape analytics.
# ---------------------------------------------------------------------------
# Zone thresholds for ball-zone-conditioned shape (m along pitch length).
SHAPE_DEFENSIVE_THIRD_M = 40.0
SHAPE_FINAL_THIRD_M = 80.0

# ---------------------------------------------------------------------------
# Pitch / homography
# ---------------------------------------------------------------------------
PITCH_LENGTH_M = 105.0      # canonical metric pitch used for projections
PITCH_WIDTH_M = 68.0
# Confidence gate applied at the HOMOGRAPHY stage (not in the detector). The
# reference uses 0.5, but low-confidence keypoints are mis-detected outliers
# that poison the homography; 0.7-0.9 measurably reduces back-projection error.
KEYPOINT_CONFIDENCE_MIN = 0.5
HOMOGRAPHY_MAXLEN = 5       # deque length for temporal smoothing of H

# Stage 6: stateful homography + shot-cut reset.
# Per-keypoint exponential smoothing before fitting H (noise reduction).
KEYPOINT_SMOOTH_ALPHA = 0.5
# Shot-cut threshold (pixels): mean displacement of the canonical pitch points
# between the freshly-fitted H and the smoothed H. A broadcast cut moves every
# point by far more than this; within-shot pan/zoom moves them by less.
SHOT_CUT_THRESHOLD_PX = 40.0

# Local pitch keypoint detector (fine-tuned pose model, 32 keypoints in
# SoccerPitchConfiguration order). Trained via References/
# train_pitch_keypoint_detector.py; weights land in runs/pose/<name>/weights/best.pt.
PITCH_KEYPOINT_MODEL_WEIGHTS = "runs/pose/runs/pose/field_v12/weights/best.pt"
PITCH_KEYPOINT_CONFIDENCE = 0.25    # detector's box/kpt confidence threshold
PITCH_KEYPOINT_IMGSZ = 640

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"
# Per-source raw acquisition cache for the retrieval layer. Each retriever
# writes its downloaded/scraped payloads under
#   RETRIEVAL_DIR/<source>/...
# so adapter layers can read normalized-converted data offline.
RETRIEVAL_DIR = DATA_DIR / "retrieval"
# Dense-tracking open dataset drops land under this subdir (IDSSE, SkillCorner,
# PFF, ...) so a single tracking adapter can find them without knowing source
# specifics at import time.
TRACKING_DATA_DIR = RETRIEVAL_DIR / "tracking"
# Wyscout (Pappalardo et al. 2019) open event dataset cache dir.
WYSCOUT_DATA_DIR = RETRIEVAL_DIR / "wyscout"

for _d in (DATA_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR, OUTPUT_DIR,
           RETRIEVAL_DIR, TRACKING_DATA_DIR, WYSCOUT_DATA_DIR):
    _d.mkdir(parents=True, exist_ok=True)