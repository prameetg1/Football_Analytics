# PerformanceAnalyzer: Innovation Plan

> Purpose: separate **what we adopt** from the Roboflow reference (proven, low-risk) from **what we build beyond it** (our differentiated engineering and product value), and sequence the two so the pipeline stays runnable at every step.

The starting point (`References/football_ai.py`) is an excellent **demo**: it proves the 4-stage CV chain works on real broadcast footage and gives us a working baseline. But it is (a) a notebook, not a pipeline, (b) stops at Voronoi/possession — it computes **no performance metrics**, and (c) solves several hard problems only "well enough to look good." This plan turns that demo into a production, defensible analytics product.

---

## 1. What We Adopt From the Reference (Baseline, unchanged-first)

These are proven and we only refactor them, never redesign them in Sprint 1-2.

| # | Inherited capability | Reference source | Why we keep it |
|---|----------------------|------------------|----------------|
| A1 | YOLO object detection (ball/keeper/player/referee) | `football_ai.py` + `train_player_detector.py` | battle-tested class separation |
| A2 | Class-agnostic NMS before tracking | `detections.with_nms(class_agnostic=True)` | de-dupes overlapping boxes |
| A3 | **ByteTrack** via `sv.ByteTrack` | supervision | best-in-class occlusion recovery |
| A4 | **Ball handled as a separate stream** (padded, `BOTTOM_CENTER`) | ball div. logic | sound separation of responsibilities |
| A5 | **SigLIP→UMAP→K-Means team classifier** (`TeamClassifier`) | `TeamClassifier.fit/predict` | far more robust than HSV; refute naive color clustering |
| A6 | **Goalkeeper centroid** heuristic | `resolve_goalkeepers_team_id` | simple, effective |
| A7 | **Pitch keypoint model** + confidence gate (`>0.5`) | `football-field-detection-f07vi` | learned pitch geometry, not Hough |
| A8 | **Canonical `SoccerPitchConfiguration`** (105×68 vertices/edges) | `sports/configs/soccer.py` | standard pitch frame for all projections |
| A9 | `ViewTransformer` (homography) | `sports/common/view.py` | compacts `findHomography` code |

**Adaptation principle:** these become modules with stable interfaces, wrapped in our `src/` layout, kept **behaviorally identical** in Sprint 1–2 (parity check vs. reference output on the same clip).

---

## 2. Where the Reference Leaves the Door Open (Weaknesses)

These are our **opportunities**. Each is a concrete engineering gap the demo does not solve:

| # | Shadow (weakness) | Reference behavior | Why it limits us |
|---|-------------------|--------------------|------------------|
| W1 | **No performance metrics** | Only possession, speed, Voronoi | Our core product is metrics (PPDA, line height, turnover conversion) |
| W2 | **Ball "track" = naive distance-outlier filter** | `replace_outliers_based_on_distance` | Ball vanishes behind bodies; false/teleport; no physics |
| W3 | **Deque-mean of homography** | rolling `np.mean` of last 5 matrices | Averaging matrices is not geometrically "pure"; can distort zoom & flicks |
| W4 | **Shot-change / replay cuts not detected** | none | a replay or cut box poisons `H` silently |
| W5 | **Team classifier fit-once** | fit on first min towards, then never updated | subs, kit swaps, differing light collapse |
| W6 | **Fps 25/50 full-res per-frame inference** | `sv.get_video_generator` per frame | expensive; metrics need lower rate |
| W7 | **Self-labeled only** | no external ground truth | cannot prove we are correct |
| W8 | **No metrics of quality gate** | demo only | no precision assurance |

---

## 3. Our Adaptations & Innovations (the moves)

For each weakness, the fix. The first three are our **principal engineering innovations** (hard), the rest are hardening.

### S1 — Metric Layer (closing gap W1) — *product core*
- Rebuild the DefenseAnalytics `MetricBase` interface over the tracking output.
- Compute **PPDA (tracking-based), defensive-line height, high-turnover conversion** from per-player-per-frame `(x,y)` + ball track, using the framework in `src/metrics/`.
- Explainability-first output: tactics in words + adjustable weight sliders (inherit symmetric philosophy).

### S2 — Ball-Hypothesis + Ballistic Tracking (closing W2)
- Instead of dropping outliers, build a **ball-holder hypothesis**: when the ball is unobserved, predict it with a **Kalman / physics-informed (drag + bounce) predictor**; bridge occlusion gaps until held by a player/keeper again.
- Use `last toucher's team` + direction to break ambiguous occlusions.
- Track at the re-sampled rate and output a **continuous ball path** (plotted via `draw_paths_on_pitch` but continuous).

### S3 — Stateful Homography (closing W3 & W4)
- Replace the deque-mean with per-frame **RANSAC homography with a prior** from the previous valid frame's `H`.
- **Keypoint smoothing:** apply a temporal filter (e.g., exponential / Kalman per keypoint) *before* fitting.
- **Shot-change detector:** score continuity of `H`/keypoints per frame; on a discontinuity (replay cut / camera cut) **reset** the stateful prior rather than smoothing across the tail.
- Few-keypoints fallback: interpolation across short gaps, never corrupting a the whole match.

### S3.5 — Adaptive Appearance (closing W5)
- Fit signature once, but **confidence-gated, semi-supervised re-fit** over the match, and keep **per-`track_id` embeddings** so a new-entering player (sub) is assigned rather than breaking.
- Keep the referee as a rejected out-of-2 cluster.

### S4 — Efficiency (closing W6)
- **Adaptive frame-rate**: detect at full FPS, but compute/export metrics at 4–6 Hz resample; ROI for video; async producer/consumer; optional model quantization (half-precision / int on GPU).
- Because metrics are mean/derivative of trajectories, a subsampled stable rate is both fast and resistance-comparative.

### S5 — Hybrid StatsBomb Ground-Truth (closing W7 & W8) — the big accuracy lever
- We already cache StatsBomb event data for the same WC/DFL (sibling). For matches we have both:
  - **Validate**: compare CV-derived player positions / passes / recoveries against StatsBomb events (position error m, recall/precision).
  - **Correct**: where the event data is authoritative (ball-possession state, shot xg), blend it in; tracking fills positions/vor and adds what events lack (velocity, defensive shape, sprint distance).
- Produce a *metrics contract* (threshold tests) that gates each sprint's acceptance.

---

### Assert all other (A1–A9) are retained

---

## 4. Innovation Touch — Product, not just CV

Two differentiators that elevate the product beyond the reference demo:

1. **Video + Event hybrid** — the portfolio never had both CV tracking *and* StatsBomb events unified; neither is sufficient alone. This is the novel combination.
2. **Explainability-first, metrics-first output** — the reference is a radar / Voronoi with no takeaway. We deliver a **tactical narrative + adjustable weight sliders** (sibling east/advances).

---

## 5. Sprint Mapping against the Weaknesses

| Sprint | Scope | Closes | Reference algorithm change? |
|--------|-------|--------|------------------------------|
| S0 | Run reference as baseline on `121364_0.mp4` | (none) | pure validate — record output to compare |
| S1 | Port layers into `src/` modules; parity with reference | W8, W1 (start) | No (behavioral parity) |
| S2 | Metric layer (PPDA, line-height, turnover conversion) | W1 | Adds scalar layer |
| S3 | Ball-hypothesis + Kalman ballistic tracking | W2 | **Change** |
| S4 | Stateful camera + shot-cut reset + keypoint filtering | W3, W4 | **Change** |
| S5 | Adaptive appearance re-fit + efficiency (FPS/resample) | W5, W6 | **Change** |
| S6 | Hybrid StatsBomb validation + metrics gate | W7, W8 | **Change** |

Each sprint **keeps the pipeline runnable** — no rewrite brick, incremental evolution.