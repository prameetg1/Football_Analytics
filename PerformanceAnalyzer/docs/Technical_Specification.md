# PerformanceAnalyzer: Computer Vision Spec (Gate Check Reference)

## 1. Vision & The Analytics Opportunity

### 1.1 Why This Project Exists
DefenseAnalytics (the sibling project) extracts performance statistics from **structured StatsBomb event JSON**. PerformanceAnalyzer is the direct successor: it extracts the **same class of metrics directly from raw football video** by generating its own tracking data via Computer Vision. This is the open-source bleeding edge — it removes the dependency on commercial event-data vendors and unlocks metrics that only tracking data can produce (velocity, sprint counts, time-to-intercept, true defensive line shape).

### 1.2 The Central Reality Check
**Extracting data from raw video is exponentially harder than parsing event data.** The dominant challenges are not "identifying a player" but:

| Challenge | Why It is Hard |
|-----------|----------------|
| **Camera Motion** | Broadcast cameras constantly pan, tilt, and zoom. A pixel coordinate means nothing in isolation; the pitch is not static in the frame. |
| **Ball Tiny + Fast** | The ball is ~22cm wide, travels at up to 100km/h, blurs at broadcast frame rates (25/50fps), and is frequently < 16px across on nondescript crops. |
| **Occlusion** | Players block each other and the ball. Boxes merge; trackers swap IDs; the ball vanishes behind bodies for multiple frames. |
| **Fine-motion precision** | Getting defensive-line height ±0.5m from a broadcast image is a far stricter requirement than drawing a bounding box. |

### 1.3 Net Architecture
The pipeline chains four distinct ML stages. Each stage has a dedicated failure mode that the next stage must tolerate.

```
         ┌───────────────┐   ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
 VIDEO ─▶│ 1. Detection  │──▶│ 2. Tracking   │──▶│ 3. Team       │──▶│ 4. Homography │──▶ 2D pitch
         │  YOLO bboxes  │   │  ByteTrack ID │   │  Classification│   │  pitch mapping│     (x, y)
         └───────────────┘   └───────────────┘   └───────────────┘   └───────────────┘
              per frame          per frame             static logic         per frame, stack
```

---

## 2. Pipeline Architecture

### Step 1 — Object Detection (Bounding Boxes)

**Goal:** Identify every relevant entity in a single frame: players, goalkeepers, referees, and the ball.

**Tech Stack:** YOLOv8 / YOLOv10 (Ultralytics) with PyTorch.

**Key challenge:** Out-of-the-box YOLO finds people well but **fails on footballs in motion blur**. The ball is small, fast, and often partially occluded.

**Action:** Fine-tune a pre-trained YOLO on a football-specific dataset. The Roboflow open pipeline already provides a proven detection model and dataset to fine-tune from:
- **Reference model:** `football-players-detection-3zvbc/11` (Roboflow Universe), fine-tuned YOLOv8, classes: **ball=0, goalkeeper=1, player=2, referee=3**. Train source dataset: `football-players-detection-3zvbc`, `yolov8` export, `imgsz=1280`.
- **Signal the ball carefully:** Development shows the ball is intrinsically hard; the reference excludes the ball from tracking entirely and handles it as a separate stream (padded box, `BOTTOM_CENTER` anchor — see Steps 2 and 4). For blur robustness you may run a **dedicated small-ball detector** (YOLO-nano over a fine-tuned ball-crop) in parallel and label-fuse.

**Config (`src/detection/`):**
- Model weights path, confidence threshold (players ~0.3), NMS IoU, input size (640–1280 for broadcast), class-id map (ball=0, goalkeeper=1, player=2, referee=3).
- See `References/train_player_detector.py` for the exact fine-tuning recipe.

### Step 2 — Multi-Object Tracking (MOT)

**Goal:** Connect detections across frames so each player keeps one stable ID ("tracked-run").

**Tech Stack:** **ByteTrack** (primary, for its occlusion handling) with DeepSORT as the alternative. Both integrate with Ultralytics via `model.track()`.

**Key challenge — Occlusion:**
- Boxes merge when a defender steps in front of an attacker.
- IDs swap after separation.
- **ByteTrack is preferred** for sports because it recovers IDs better when players cross paths, using low-confidence detections to bridge occlusion. DeepSORT leans on appearance re-identification, which fails for same-kit players.

**Tech Stack:** **ByteTrack** (supervision's `sv.ByteTrack`, primary) with DeepSORT as the alternative. Reference uses `sv.ByteTrack`.

**Key challenge — Occlusion:**
- Boxes merge when a defender steps in front of an attacker.
- IDs swap after separation.
- **ByteTrack is preferred** for sports because it recovers IDs better when players cross paths, using low-confidence detections to bridge occlusion. DeepSORT leans on appearance re-identification, which fails for same-kit players.
- The reference applies **class-agnostic NMS** (`with_nms(threshold=0.5, class_agnostic=True)`) before tracking to de-overlap players/keepers/referees.

**Ball is handled separately:** the reference does **not** feed the ball into the tracker. It: (1) isolates ball detections, (2) **pads** the box (`sv.pad_boxes(detections, px=10)`) to counter the tiny-box problem, and (3) anchors the ball at `BOTTOM_CENTER`. Ball trajectory is reconstructed post-hoc (see Step 4), optionally cleaned by a distance-based outlier filter (positions further than a threshold from the last valid point are dropped; reference uses 500 pitch-units on DFL footage).

**Output:** per-frame detection list `(track_id, class, x1, y1, x2, y2, conf)` for players/keepers/referees; separate ball boxes.

### Step 3 — Team Classification (Kit Appearance + Clustering)

**Goal:** Assign each tracked player to *Team A / Team B*.

**Tech Stack:** embedding + dimension-reduction + clustering. **The Roboflow reference does NOT use HSV** — it uses a learned appearance model, which is far more robust to lighting, shadows, and similar-colored kits:

**Algorithm (SigLIP → UMAP → K-Means, wrapped as `TeamClassifier`):**
1. **Sample crops:** over the first N seconds (`stride` ~ 30 frames), crop every detected `player` box via `sv.crop_image`.
2. **Embed:** run `SigLIP` (`google/siglip-base-patch16-224`) vision-transformer on each crop, take the mean of `last_hidden_state` → 768-dim embedding per crop.
3. **Reduce:** `UMAP(n_components=3)` projects to 3D.
4. **Cluster:** `KMeans(n_clusters=2)` splits into the two teams.
5. `TeamClassifier.fit(crops)` learns the model once; then `TeamClassifier.predict(crops)` per frame assigns each player `class_id ∈ {0, 1}`.

**Goalkeeper resolution (Reference heuristic):** goalkeepers are *not* clustered. Compute the mean `BOTTOM_CENTER` centroid of each team's players; assign each goalkeeper to the team whose centroid is closer.

**Referees** are given their own class. No per-frame re-fitting — the palette is fit once from a bootstrap sample, which prevents mid-game label flips.

**Output:** `team_id ∈ {0, 1}` for players/keepers; referee stays separate.

### Step 4 — Homography & Pitch Mapping (The Final Boss)

**Goal:** Map `(pixel_x, pixel_y)` → normalized pitch coordinate `(x: 0–105, y: 0–68)`.

**Tech Stack:** OpenCV (`cv2.findHomography`, `cv2.warpPerspective`).

**The math:** a homography `H` (3×3 matrix) maps pixel points to pitch points with `p' = H p`. `findHomography` needs **at least 4 point correspondences** per frame.

**Key challenge — camera motion:**
- Broadcast pans/tilts/zooms → `H` differs almost every frame.
- **You must re-derive H per frame** — a stable (`src/homography/` module).

**Solution — automated keypoint association (reference approach):**
1. **Pitch keypoint detection:** use a fine-tuned **YOLO pose/keypoint** model on the `football-field-detection-f07vi` dataset (reference model `football-field-detection-f07vi/14`). It predicts N=28 pitch keypoints (line intersections, corners, penalty spot), each with a confidence score, 1:1 corresponding to a canonical `SoccerPitchConfiguration` (in the `roboflow/sports` package — real 105×68 geometry).
2. **Confidence-gate:** keep only keypoints with `confidence > 0.5` (skip low-confidence garbage points).
3. **Correspondence:** the kept frame keypoints pair index-by-index with `CONFIG.vertices`.
4. **Fit:** `ViewTransformer(source=frame_pts, target=pitch_pts)` builds the homography `H` (a `cv2.findHomography`-style reprojection) between the camera and pitch planes.
5. **Temporal stabilization (crucial):** accumulate `H` over a fixed-size deque (`MAXLEN=5`) and apply the **mean** matrix (`transformer.m = np.mean(stack, axis=0)`) each frame. This averages pan/zoom noise and bridges frames where keypoints briefly fail — satisfying "recompute `H` per frame" in a numerically stable way.
6. **Map:** project each player's + ball's `BOTTOM_CENTER` anchor to pitch `(x, y)` via `transform_points`.

**Key precision:** defensive metrics need the *defensive line height* to ±0.5–1 m. `H` comes from the pitch keypoints, not player positions, so measurement error is not reintroduced into the line.

---

## 3. Module Layout

```
PerformanceAnalyzer/
├── data/
│   ├── raw/              # input videos + frames
│   ├── processed/         # detections, tracks, homographies (parquet/csv)
│   ├── models/            # downloaded/finetuned model weights
│   └── baseline/          # textual/ground-truth labels for validation (optional)
├── docs/
│   ├── Technical_Specification.md   # THIS document
│   ├── Innovation_Plan.md           # what we adopt vs. what we build beyond the reference
│   ├── reference/                   # Roboflow reference implementations
│   │   ├── football_ai.py               # canonical end-to-end Football AI notebook
│   │   ├── train_player_detector.py    # fine-tune YOLOv8 player detector recipe
│   │   └── train_pitch_keypoint_detector.py  # fine-tune pitch keypoint model recipe
│   ├── Architecture_Plan.md
│   └── Sprint_Roadmap.md
├── src/
│   ├── __init__.py
│   ├── config.yaml               # pipeline constants (weights, paths, thresholds)
│   ├── video/
│   │   ├── __init__.py
│   │   ├── reader.py             # frame iterator, frame-rescale, fps/collection
│   │   └── augment.py            # motion-blur / scale-variant ball detection
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── detector.py           # YOLO wrapper (ball + player path)
│   │   └── finetune.py           # dataset glue + training loop (Ultralytics)
│   ├── tracking/
│   │   ├── __init__.py
│   │   ├── tracker.py            # ByteTrack wrapper, stable-ID guarantees
│   │   └── reid.py               # (optional) appearance embedding bridge
│   ├── homography/
│   │   ├── __init__.py
│   │   ├── pitch_model.py        # canonical SoccerPitchConfiguration (105×68 vertices/edges)
│   │   ├── keypoints.py          # YOLO-pose pitch keypoint detector + confidence gate
│   │   └── homography.py         # per-frame ViewTransformer + deque temporal smoothing
│   ├── teams/
│   │   ├── __init__.py
│   │   └── classifier.py         # SigLIP embedding → UMAP → K-Means team_id
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── base.py               # MetricBase interface (mirrors DefenseAnalytics)
│   │   ├── position.py           # per-frame (x, y, speed, distance)
│   │   ├── ppda.py               # pressing intensity (event-free, tracking-based)
│   │   ├── defensive_line.py     # defensive line height from team positions
│   │   ├── high_turnover.py      # win-in-opp-third → shot within 10s (temporal)
│   │   └── sprints.py            # distance/velocity-derived (Phase 2+)
│   └── visualization/
│       ├── __init__.py
│       ├── annotated.py          # bounding box + team overlay on movie
│       └── tactical.py           # dot-plot on SSD pitch
├── main.py                        # CLI entry point (mirrors DefenseAnalytics main.py)
├── tests/
│   ├── test_detection.py
│   ├── test_tracking.py
│   ├── test_teams.py
│   ├── test_homography.py
│   └── test_metrics.py
├── requirements.txt
└── README.md
```

---

## 4. Functional Requirements (FRs)

| ID | Component | Description |
|----|-----------|-------------|
| **FR-01** | Video Capture | The system must read a video stream frame-by-frame at a configurable target FPS/resolution and emit annotated results. |
| **FR-02** | Detection | The system must detect players, goalkeepers, referees, and the ball in each frame with YOLO, tuned for ball in motion blur. |
| **FR-03** | Tracking | The system must associate detections to stable per-player IDs across frames, tolerating occlusion without ID swap. |
| **FR-04** | Team Assignment | The system must classify each tracked player to Team A / Team B (referee excluded) via learned appearance embeddings (SigLIP → UMAP → K-Means), goalkeepers by centroid heuristic. |
| **FR-05** | Pitch Mapping | The system must compute a per-frame homography and map every tracked player + ball to normalized pitch coordinates. |
| **FR-06** | Metric Extraction | The system must compute defensively-relevant metrics (PPDA-, defensive-line-height-, high-turnover-conversion-style) from the generated tracking data. |
| **FR-07** | Output & Export | The system must render annotated output video/pitch plots and export per-frame tracking to a structured format. |

---

## 5. Non-Functional Requirements (NFRs)

| Requirement | How We Satisfy It |
|-------------|-------------------|
| **Modularity** | Every stage is an independent module with a defined input/output contract; any model can be swapped (YOLOv8 ↔ YOLOv10, ByteTrack ↔ DeepSORT) without touching others. `MetricBase` mirrors the sibling project. |
| **Performance / Latency** | Target **>10 fps** on a modern NVIDIA GPU for the full chain; pre-load models; run ball-detection in parallel to player-detection; cache intermediate homographies. Groundtruth-limited "post-match debrief" (not live) is the Phase 1 target. |
| **Extensibility** | New metrics = new module in `src/metrics` implementing `compute(df, team_id)`; the pipeline is unaware. |
| **Robustness** | Sanitization gates at Homography (RANSAC + residual gates) and Tracking (bridge gaps); never let one bad frame corrupt the match. |
| **Determinism / Reproducibility** | Pinned model weights, fixed thresholds in `config.py`, seeded RNG in K-Means and any stochastic step. |

---

## 6. Validation Strategy

Because we generate our own ground-truth, we must validate against **reference ground-truth**. Key enabler: the Roboflow `football_ai.py` (in `References/`) is a complete, working end-to-end baseline on the DFL-Bundesliga clip `121364_0.mp4` — use it as both a **cross-check and as Sprint 0's executable slice**:
- **Detection:** mAP@0.5:0.95 and precision/recall per class vs. a held-out annotated frame set.
- **Tracking:** MOTA, HOTA, IDF1 on a short labeled clip; specifically assert identity-stability during crossing events.
- **Team:** accuracy of `TeamClassifier` on a random sample of manually labeled team boxes.
- **Homography / Mapping:** mean-projection error (reproject landmarks through candidate `H`) plus distance error (m) vs. a match whose positions we know (a tactical-camera feed that does not pan, hence an almost-stable `H` to calibrate against).

> **Strategic note — from reference to product:** see **[`docs/Innovation_Plan.md`](Innovation_Plan.md)** for how we adopt the reference (A1–A9) vs. innovate beyond it (ball-hypothesis+ballistic tracking, stateful/cut-shot homography, adaptive appearance, efficiency, and hybrid StatsBomb ground-truth), with a weaknesses→sprints mapping (W1–W8 → S0–S6) that keeps the pipeline runnable at every step.

---

## 7. Starting Small — Tactical Camera First

Broadcast footage must be deferred. Phase 1 targets a **static, wide-angle tactical camera** (high-up, no pan/zoom). This isolates the homography problem (a near-static `H`, or only slow drift) so you can prove metric extraction end-to-end, then generalize to professional broadcast footage.

**Recommended bootstrapping sprint:**
1. Find a tactical-camera clip (ideally, same match we already have in StatsBomb — e.g., a WC 2018 match with 360° / tactical feed).
2. Wire YOLOv8 to draw bounding boxes on a handful of frames and inspect them visually.
3. THEN tackle homography (Step 4) because geometry, not detection, is where most projects stall.

---

## 8. Agile Sprints

Adopt **vertical slicing** — deliver a runnable, end-to-end command per sprint.

| Sprint | Scope | Exit Criteria |
|--------|-------|--------------|
| **S0 — Scaffold** | Repo skeleton, `config.py`, video reader/writer, test harness | `main.py` reads a clip and prints frame count; FR-01. |
| **S1 — Detection** | YOLOv8 fine-tuning setup + per-frame inference | Bounded player/ball overlay image produced; FR-02. |
| **S2 — Tracking** | ByteTrack wrapper + stable-ID plumbing | ID-stable annotated video; FR-03. |
| **S3 — Teams** | SigLIP embeddings → UMAP → K-Means + goalkeeper heuristic | Team-colored overlay on tactical feed; FR-04. |
| **S4 — Homography** | Pitch keypoint model (YOLOv8-pose) + confidence gate + per-frame `ViewTransformer` with deque smoothing | Players mapped onto a normalized 2D pitch; ball-projection error runs valid; FR-05. |
| **S5 — Metrics** | `MetricBase` + top 3 metrics | Daily per-match metric report from raw video; FR-06, FR-07. |
| **S6 (stretch)** | Broadcast robustness, ball-tracking refinements, more metrics | Pipeline on a broadcast feed; latency budget met. |

---

## 9. Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|-----------|------------|
| Ball miss / blur in broadcast | High | High | Downscale-augmented fusion detector + dedicated tiny/ball model + track then run. |
| ID-swap during crossings | Medium | High | ByteTrack, appearance re-ID bridge, occlusion-gap timeout <1s. |
| Homography per-frame drift / pan-zoom | High | High | RANSAC + residual thresholds + temporal interpolation of `H`; running-outlier rejection; start on static camera. |
| Team-color flips (same-color kits) | Medium | Medium | Persistent reference palette, high-confidence bootstrap, not per-frame centroids |
| Ground truth unavailable | Medium | High | Use tactical feed as self-ground-truth; when same match has events, compare to StatsBomb |
| Latency below target | Medium | Medium | Vectorized OPENCV ops, reduce model size (tiny/nano), batch frames, GPU async, decouple |

---

## 10. References & Further Reading

1. Roboflow — **Football AI** end-to-end notebook (this plan's canonical reference): `References/football_ai.py` → [football-ai.ipynb](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/football-ai.ipynb)
2. Roboflow — train soccer player detector: `References/train_player_detector.py`
3. Roboflow — train pitch keypoint detector: `References/train_pitch_keypoint_detector.py`
4. Roboflow `sports` repo (ByteTrack, `TeamClassifier`, `ViewTransformer`, `SoccerPitchConfiguration`, pitch annotators) — github.com/roboflow/sports
5. Roboflow Universe — `football-players-detection-3zvbc`, `football-field-detection-f07vi`
6. DFL Bundesliga Data (Kaggle) — broadcast clips for training/validation
7. `supervision` library — `sv.ByteTrack`, `sv.Detections`, `sv.KeyPoints`, annotators
8. Ultralytics YOLOv8 — object detection + pose/keypoint fine-tuning
9. ByteTrack: Multi-Object Tracking with Tracklets-matched Detection (arXiv:2110.18156)
10. SigLIP / UMAP / scikit-learn KMeans — the improved team-clustering stack (used by `TeamClassifier`)
11. mplsoccer — tactical pitch plotting for visualization (reuse from DefenseAnalytics)

_Current date note: the sibling `DefenseAnalytics` docs cite 2025/2026 sources because that project's roadmap lives a future cycle; this spec anchors code to stable 2024-era tooling (YOLOv8 / YOLOv10, ByteTrack) to keep a reproducible baseline._