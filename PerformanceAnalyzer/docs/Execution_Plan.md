# PerformanceAnalyzer: Stage-by-Stage Execution Plan

> Companion to `docs/Innovation_Plan.md` (the *why* / sprint mapping) and
> `docs/Technical_Specification.md` (the *what*). This file is the *when / in
> what order*, with per-stage scope, entry points, and acceptance criteria.
> Every stage keeps the pipeline runnable (no rewrite bricks).

## Stage Map (from `References/football_ai.py`)

Reference stages (all adopted, behaviorally, into `src/`):
Detection -> Tracking -> Teams -> Pitch keypoints -> Homography -> Projection ->
Ball tracking -> Viz/radar.

Our **planned additions beyond the reference**: metrics (PPDA/line-height/turnover),
ball-hypothesis Kalman tracking, stateful homography with shot-cut reset, adaptive
team re-fit, efficiency, and a StatsBomb ground-truth gate.

## Execution Order & Dependency Graph

```
            ┌─────────────────────────────────────────────────────────┐
            │ Stage 0  Dataset + training pipeline  █ done (wiring)    │
            └───────────────┬─────────────────────────────────────────┘
                            ▼
            ┌─────────────────────────────────────────────────────────┐
            │ Stage 1  Detector bake-off (merged @1280 vs hosted v11)  │
            └───────┬───────────────────┬───────────────┬─────────────┘
                    ▼                   ▼               ▼
            ┌──────────────┐    ┌───────────────┐  ┌──────────────────┐
            │ Stage 2      │    │ Stage 4       │  │ Stage 6          │
            │ Tracking     │    │ Ball-hypoth.  │  │ Stateful homog.  │
            │ export       │    │ + Kalman      │  │ + shot-cut reset │
            └──────┬───────┘    └───────────────┘  └──────────────────┘
                   ▼
            ┌──────────────┐    ┌───────────────┐
            │ Stage 3      │    │ Stage 5       │
            │ Metric layer │    │ Adaptive team │
            │ (product)    │    │ re-fit + eff. │
            └──────┬───────┘    └───────────────┘
                   ▼
            ┌──────────────┐    ┌───────────────┐
            │ Stage 7      │◀───│ Stage 8 Viz   │
            │ StatsBomb GT │    │ radar/Voronoi │
            │ gate         │    │               │
            └──────────────┘    └───────────────┘
```

**Critical path (first):** 0 -> 1 -> 2 -> 3 -> 7. **Parallelizable after Stage 1:**
4, 5, 6 (independent reliability upgrades). 8 can start as soon as Stage 3 lands.

---

## Stage 0 — Dataset & training pipeline  *(status: done)*

**Objective:** A reproducible, merged training corpus and a one-command train entry.

**Scope / deliverables**
- `data/player_detection_v11/` + `data/player_detection_minaehyeon/` (raw sources).
- `data/player_detection_merged/` (763 train / 182 val / 91 test; ball 658 train
  instances; canonical classes ball/gk/player/referee) + `data.yaml`.
- `scripts/merge_player_datasets.py` (idempotent merge, class remap, dedup names)
  with `--ball-pad-scale 2.5`: expands the ball (class 0) GT boxes ~2.5x around
  their centre, so the training target matches the +10px-padded box the inference
  pipeline feeds to tracking/projection (see Stage 4 ball stream).
- `scripts/train_player_detector.py` (local entry: yolov8m @1280, batch 2).
- `notebooks/train_player_colab.ipynb` (yolov8x @1280, batch 6, 50 ep — the
  recipe that reached mAP 92.91 as hosted v11).
- `src/config.py`: `PLAYER_DETECTION_DATA`.

**Remaining work**
- Run the training to completion on padded-ball data (in progress).
- Optional: dedup overlap between minaehyeon and v11 for a strict eval split.

**Acceptance:** `check_det_dataset(merged/data.yaml)` passes; train script launches.

---

## Stage 1 — Detector bake-off  *(status: done)*

**Objective:** Pick the detection model for production by measuring ball accuracy,
not vibes.

**Scope / deliverables**
- Per-class validation (esp. **ball mAP50 + ball recall**) on the held-out test
  split for: hosted v11, local `best.pt` (merged), and any Colab yolov8x result.
- Wire the winner into `DETECTION_MODEL_WEIGHTS` (local `.pt`) or
  `ROBOFLOW_PLAYER_MODEL_ID` (hosted).
- Optional SAHI/tiled inference note: ball at 1280 input is ~7px after letterboxing
  from 1080p; tiled inference on native res is the fallback if recall is short.

**Acceptance:** ball recall target defined vs hosted v11 (baseline 89.6% recall,
92.9% mAP); model with the best ball mAP chosen and wired.

---

## Stage 2 — Tracking export  *(status: done)*

**Objective:** Emit a per-frame tracking table the metric layer can consume.

**Scope / deliverables**
- `Pipeline.to_dataframe()` -> rows `[frame_idx, timestamp_sec, track_id, team_id,
  pitch_x, pitch_y, class_name]` conforming to `src/metrics/base.py`
  `TrackingFrame.REQUIRED_COLUMNS`.
- Export to CSV/parquet for a clip; sanity check column coverage & continuity.

**Acceptance:** a 60-frame clip produces a non-empty DataFrame with all required
columns; unit test on the export.

---

## Stage 3 — Metric layer (product core)  *(status: done)*

**Objective:** Turn tracking data into match analytics — the thing the reference
does not compute.

**Scope / deliverables** (in order of value)
1. **Velocity / total & sprint distance** (per player, per team) — needs tracking
   export + timestamps; simplest, highest value.
2. **Ball possession** (which team, % time) — needs ball track + team assignment.
3. **PPDA** (passes allowed per defensive action in final 60%) — needs ball events.
4. **Line height / defensive shape** (mean & std of back-line y) — needs team rows.
5. **Turnover conversion** — needs possession-state changes.

**Goalkeeper policy (mirrors the reference):** the keeper is a tracked entity
(class `goalkeeper`, ByteTrack id, bottom-center projection) but is **excluded
from team clustering** — team id comes from the nearest-player-centroid heuristic
(`resolve_goalkeeper_team_ids`). In the metrics:
- keeper is **included** in velocity / distance / sprint / possession
  (`PLAYER_CLASSES = (player, goalkeeper)`), since he covers distance and
  receives back-passes;
- keeper is **excluded** from line height by default (`LineHeightMetric`
  uses outfield `player` only, `include_goalkeeper=True` opts in) — he always
  sits deepest and would otherwise collapse the defensive line to the goal.

**Deliverables:** concrete metric classes under `src/metrics/`, a `MetricsRunner`
(multi-metric, per-team), and a report writer (CSV/JSON). Update the
`TrackingFrame` schema contract only if a metric forces it.

**Acceptance:** every metric returns a scalar/table per team on the tracked clip;
a `pytest` covers each with synthetic tracking data.

---

## Stage 4 — Ball-hypothesis + Kalman ballistic tracking  *(status: done)*

**Objective:** Fix the reference's weakest stage — the ball is tiny, fast, and
occluded; detections alone produce broken trails.

**Scope / deliverables**
- Ball hypothesis stream (padded box kept out of ByteTrack).
- Kalman filter with ballistic (gravity/drag) motion model; predict when the ball
  is undetected; associate detections to the hypothesis.
- Interpolate/smooth the projected ball path; feed possession (Stage 3) and
  outlier rejection (existing `is_position_outlier`).

**Status:** built (`src/tracking/ball_tracker.py` + pipeline integration +
`scripts/ball_continuity_report.py`). Constant-velocity pitch-plane Kalman
(constant-velocity is the honest 2D choice: the ball's 3D flight is not
recoverable from its bottom-center projection), Mahalanobis-gated association
(glitch-safe, re-anchors on new ball events).

**Acceptance:** ball trail continuity on `0bfacc_0.mp4` (e.g., % frames with a
valid hypothesis >> % with a raw detection); possession delta measurable vs raw.
Measured on the full clip (750 frames): hypothesis 80.9% vs raw detection 64.4%
(+16.5 pts); 46.4% of miss frames recovered; possession shifted measurably
(team0 Δ−0.045, team1 Δ+0.015). **Met.**

---

## Stage 5 — Adaptive team re-fit + efficiency  *(status: done)*

**Objective:** Stay robust when kit/lighting drifts and keep runtime sane.

**Scope / deliverables**
- Team classifier drift detector -> bounded re-fit (extend `TeamAssigner`).
  Goalkeeper handling stays as-is: re-fitting updates the player clusters, then
  each keeper is re-assigned via the same nearest-player-centroid heuristic
  against the refreshed team ids (no separate keeper clustering).
- Adaptive frame-rate (detect full FPS, export metrics at 4-6 Hz), optional ROI,
  half-precision inference.

**Status:** drift detector + bounded refit built. `TeamClassifier.tightness()`/`refit()`
added; `refit` re-clusters KMeans on a fresh window but pins new cluster centres back to
their pre-refit identity via `match_cluster_centers` (Hungarian) so labels never flip.
`TeamAssigner` now keeps a rolling window of player crops (`TEAM_REFIT_WINDOW_CROPS`),
checks drift every `TEAM_REFIT_CHECK_INTERVAL_FRAMES`, and re-fits when window tightness
exceeds `TEAM_REFIT_TIGHTNESS_RATIO` x baseline. SigLIP runs in FP16 on CUDA
(`SIGLIP_FP16`) — halves embed VRAM/latency. GK handling unchanged.

Efficiency: detection/tracking stay at full source FPS (track continuity) while
`Pipeline.to_dataframe(analysis_hz=...)` samples the exported table down to
`METRICS_EXPORT_HZ` (5 Hz default) — distance/velocity recompute dt from exported
timestamps, preserving physics. Optional normalized detector ROI (`DETECTION_ROI`)
crops the frame before inference and shifts boxes back to full-frame coordinates.

**Acceptance:** no label flips on a longer clip (unit-tested: refit preserves identity;
drift triggers re-fit, no-drift doesn't); pipeline FPS documented and stable.

---

## Stage 6 — Stateful homography + shot-cut reset  *(status: done)*

**Objective:** Stabilize the pitch map across pans/zooms and recover instantly on
camera cuts.

**Scope / deliverables**
- Temporal keypoint smoothing (exp/Kalman) **before** fitting H.
- Shot-change detector: score keypoint/H continuity; on discontinuity reset the
  stateful prior (never smooth across a replay cut).
- Reuse `StablePitchMapper`; extend `validate_pitch.py` with a cut-injection test.

**Status:** built into `StablePitchMapper` (per-index exp keypoint smoothing via
`smooth_keypoints`; shot-change detector in `update()` via `_to_pixels`
displacement against the smoothed state, resetting the H deque + keypoint prior
on a cut). Wired into `Pipeline._project`; `validate_pitch.py --self-test` is the
synthetic cut-injection harness.

**Acceptance:** homography RMSE stays <= ~17px within a shot and recovers within a
few frames after an injected cut (val harness). Self-test: within-shot RMSE
0.08px; recovered on frame 1 after the injected cut. Real clip: 40/40 valid
frames, avg reproj RMSE 17.5px, no false cuts. **Met.**

---

## Stage 7 — Hybrid StatsBomb ground-truth gate  *(status: done — infra, gate, synthetic GT)*

**Objective:** Prove the CV output, not just demo it.

**Scope / deliverables**
- Acquire StatsBomb open data for a public match (id matched to footage if
  possible; else a closest-footage protocol).
- Alignment: frame/timestamp mapping, team/player linking (jersey to appearance).
- Validation: CV-vs-event position error (m), pass/recovery recall & precision.
- Blending: where events are authoritative (ball possession state, shots xg) use
  them; tracking supplies what events lack (velocity, defensive shape, sprints).
- Metrics gate: refuse to emit a metric unless the gate passes.

**Status:** fetched + parsed StatsBomb open data end-to-end. `src/statsbomb/`
loader (`StatsBombLoader`: events/360/lineups, cached under `data/statsbomb/`),
alignment (`convert_pitch_xy` 120x80->120x70, `build_gt_frame` from 360
freeze-frames, `extract_pass_events`), validation (`position_error` via optimal
assignment, `pass_match` recall/precision, `ValidationReport`). `MetricsGate`
(+`MetricsRunner.gate`) raises `MetricGateBlocked` when enabled and the report
fails or is absent; off by default (`METRICS_GATE_ENABLED`).
`scripts/validate_against_statsbomb.py` is the closest-footage CLI. Verified
against real match 3788741: synthetic CV (=GT+1m noise) -> pos_err 1.41m,
pass recall/precision 1.0, gate PASS. Tests use synthetic StatsBomb-shaped GT.

**Acceptance:** a validation report (position error m, recall/precision) for one
match; metrics gated on it. **Built.** (Exact real-footage alignment deferred:
our excerpt clip is an unidentifiable clip — closest-footage protocol is used.)

---

## Stage 8 — Visualization / product HUD  *(status: done)*

**Objective:** Presentable output — the reference's best asset, extended to metrics.

**Scope / deliverables**
- Video-game radar + Voronoi overlay (reference parity).
- Metric HUD (possession %, line height, sprints) on the render.
- Reuse `scripts/render_pitch_video.py` (fix at `--scale 10`).

**Status:** built. `src/render/hud.py` adds `finite_voronoi_polygons` (clipped
Voronoi ownership cells), `draw_voronoi_radar` (team-colored dots + translucent
ownership cells + ball), `draw_hud` (translucent metric panel) and
`radar_with_hud`. `scripts/render_pitch_video.py` defaults to the local bake-off
winner (`DETECTION_MODEL_WEIGHTS`), accumulates a live tracking table, recomputes
the metric HUD every `--hud-interval` frames (possession %, defensive-line
height, sprint distance), and writes a broadcast-frame | radar split video.

**Acceptance:** rendered clip shows team-colored radar + at least one live metric.
**Met** — `output/pitch_radar_hud.mp4` (150 frames) verified programmatically:
team-colored Voronoi cells present, HUD panel drawn.

---

## Stage 9 — Event extraction + team shape + pitch control  *(status: in progress — A1-A9 + B1-B6 + C1-C5 + D1-D13 built)*

**Objective:** Turn the tracking table into a real performance-event and
positional-analytics layer — the differentiator beyond event data (StatsBomb is
free; positional/spatial inference is not). Gate every metric against real data
wherever a gate exists; synthetic GT (real 360 positions + injected noise) stands
in until real footage of a StatsBomb-covered match is added.

**Gate-first principle (hard requirement):**
Every new metric ships with an accuracy gate against real data before it is
reported as product output:
- Event classes -> StatsBomb event match (recall/precision) on match 3788741
  (WC 2022 final): Pass 1059, Carry 862, Pressure 382, Ball Recovery 92,
  Duel 51, Interception 47, Clearance 39, Shot 27, Dribble 32.
- Shape/spatial metrics -> 360 freeze-frame player positions (real positions,
  no event matching needed) + position-error sensitivity (1.4 m measured).
- Until real footage arrives: synthetic GT (real 360 positions + Gaussian noise)
  validates extractor logic and gate plumbing end-to-end.

**Scope / deliverables**

*Stage A — event extraction (src/events/):*
- A1 Passes — ball-owner transition A->B + spatial distance + completion.
- A2 Shots / on-target — goal-box entry at velocity; outcome vs GT.
- A3 Turnovers / regains — ownership flips; won vs conceded.
- A4 Clearances — long ball exiting defensive third under pressure.
- A5 Set-pieces — throw-in / goal-kick / corner / free-kick (play_pattern).
- A6 Pressures — opposition players within N m of ball-carrier.
- A7 Carries / dribbles — player-in-possession ball movement.
- A8 Through-balls — passes into space behind the defensive line.
- A9 Aerial duels — contested high balls.

*Stage B — event aggregates (src/metrics/):*
- B1 Pass completion %, passes/min, forward-pass %.
- B2 Possession sequences — passes, duration, touches before turnover.
- B3 PPDA — passes per defensive action.
- B4 Shots per possession, chances created.
- B5 Turnover rate — forced/unforced, final-third regains.
- B6 Pressing success, pressures/min.

*Stage C — pitch control / spatial (src/spatial/):*
- C1 Pitch control surface (Spearman-style) per location.
- C2 Expected Possession Value (EPV) — Markov grid value (the cutting-edge core).
- C3 Expected Threat / danger maps — value movement per second.
- C4 Field tilt / territory — time in final vs own third.
- C5 Space creation — runs into uncovered space, threatening runs.

*Stage D — team shape analytics (src/shape/):*
- D1 Spread — distance of players from team centroid.
- D2 Length / Width — bounding extent along/against the attacking axis.
- D3 Stretch index — length x width.
- D4 Compactness — mean pairwise / centroid distance (out of possession).
- D5 Spatial entropy — positional dispersion / unpredictability.
- D6 Defensive line height + lateral shift (height already in Stage 3).
- D7 Block position — midfield block depth, pressing line height.
- D8 Team centroid + velocity — shape momentum.
- D9 Moment of inertia — formation rotation around centroid.
- D10 Shape in vs out of possession (attack width vs defensive compactness).
- D11 Shape during transitions — stretch spike after turnovers.
- D12 Formation recovery time — frames to compact shape after ball loss.
- D13 Shape-vs-ball-zone — compactness with ball in own vs final third.

**Validation:**
- A/B: gate recall/precision per event class vs StatsBomb (extend the Stage 7
  gate skeleton to shots/turnovers/clearances).
- C: indirect gate — position-error sensitivity + possession-outcome correlation.
- D: shape metrics on real 360 positions as reference; synthetic known-shape
  teams for exact error bounds.

**Build order (critical path):** A1 passes -> A2 shots -> A3 turnovers ->
C1/C2 pitch control + EPV -> D1-D5 core shape -> D6-D13 structure/transition.
D (shape) is independent of events and can be built in parallel with A.

**Acceptance:** each shipped metric reports gate PASS/BLOCK; suite green.

---

## Sequencing rationale

1. **0 -> 1 first:** all downstream stages consume detections; a validated,
   better detector raises everything else (ball recall feeds possession, Kalman,
   GT gate).
2. **2 -> 3 before 4/5/6:** metrics are the product core and only need the export,
   so they land on the critical path early and de-risk the whole project.
3. **4, 5, 6 parallel:** independent reliability upgrades; each improves a
   different metric's fidelity (4 -> possession, 6 -> positions, 5 -> teams).
4. **7 after 3:** the gate validates metrics; it also motivates 4/6 because it
   will surface their failure modes quantitatively.
5. **8 opportunistic:** can start once Stage 3 gives something to display.
6. **9 after 7/8:** event extraction reuses the Stage 7 gate and the Stage 8
   render; shape analytics needs only positions, so D can start immediately.
   Real-footage gates depend on acquiring a clip of a StatsBomb-covered match.

## Status log

- Stage 0 wiring: done (merge tool, train script, merged data, config).
- Stage 1: done — local `best.pt` beat hosted v11 on the shared test split
  (ball mAP50 0.704 vs 0.644, ball recall 0.705 vs 0.641); wired into
  `DETECTION_MODEL_WEIGHTS`.
- Stage 4: done — ball Kalman hypothesis (80.9% hypothesis vs 64.4% raw det
  frames on `0bfacc_0.mp4`); acceptance met.
- Stage 6: done — keypoint smoothing + shot-cut reset; self-test recovers on
  frame 1 after an injected cut; real clip 17.5px reproj RMSE, no false cuts.
- Stages 2-3, 5, 7-8: done (section headers updated to match).
- Stage 9: planned — full event + shape + pitch-control spec above; gate-first
  (real StatsBomb events/360 positions, synthetic GT until real footage lands).
- Stage 9 progress: A1/A2/A3 built (`src/events/` — Pass, Ball Recovery,
  Shot extractors), gate generalized (`event_match`, per-kind
  `ValidationReport`, `validate_events_against_gt`), D1-D5 built
  (`SpreadMetric`, `StretchIndexMetric`, `CompactnessMetric`,
  `SpatialEntropyMetric`; D2/D6 from `LineHeightMetric`), shape metrics
  added to runner defaults. 121 tests passing. Real-clip smoke test:
  31 passes / 6 recoveries in 150 s of `0bfacc_0.mp4`.
- Stage 9 progress (latest): all Stage-A detectors built in `src/events/`
  (A4 Clearance, A5 Throw-in/Goal Kick/Corner, A6 Pressure, A7 Carry,
  A8 Through-ball, A9 Duel) plus B1-B6 aggregates
  (`src/metrics/event_aggregates.py`), C1-C4 pitch control / EPV / xT /
  field tilt (`src/spatial/pitch_control.py`), and D6/D8/D10/D11-D13
  shape-structure + transition metrics (`shape.py`, `transition.py`). Duel
  events temporally clustered (2-frame minimum run) to avoid double-counting.
  Tests for all of the above added (`test_events` 32, `test_event_aggregates`
  15, `test_spatial` 10, `test_transition` 6); full suite 167 passing. Smoke
  test on 150 s of `0bfacc_0.mp4`: 31 Pass / 21 Carry / 43 Pressure /
  7 Duel / 6 Ball Recovery. Remaining: C5 space creation, D7 block
  position / pressing line, D9 moment of inertia.
- Stage 9 progress (complete): C5 `space_creation_runs` in
  `src/spatial/pitch_control.py` (runs into cells where opponent control is
  below a ceiling, with a `threatening` final-third flag), D7
  `BlockPositionMetric` (midfield block depth between the back and front
  lines), D9 `MomentOfInertiaMetric` (radius of gyration) and
  `FormationAngleMetric` (principal-axis orientation, 0-180 deg). All three
  shape metrics added to runner defaults. Tests added for C5/D7/D9
  (`test_spatial` 13, `test_shape` 18); full suite 176 passing. Real-clip
  smoke: block position 58.3/47.7 m, moment of inertia 17.8/15.5 m,
  formation angle 95/88 deg, 5 team-0 space runs (mean 10.6 m, 0
  threatening). Stage 9 metric surface now complete; remaining work is
  real-footage StatsBomb gating (blocked on user-provided footage of a
  StatsBomb-covered match).

## Post-Stage-8 accuracy work

- **Possession metric fixed:** the 2 m ownership radius mislabeled ~82% of ball
  frames "loose" (real tracking distances median 3.5 m). Replaced with
  nearest-player + stateful hysteresis (12-frame switch commit, >8 m loose
  safety net) in `src/metrics/possession.py`; shares now sum to ~1.0
  (0.845 / 0.155 on the full clip).
- **Ball-recall pass:** full-frame detector missed tiny balls (test recall
  0.705; 21/23 misses had zero prediction). Added prior-guided recovery in
  `src/pipeline.py` (`_recover_ball`) — upscaled window around the
  Kalman-predicted ball position — plus `Detector.infer_region` /
  `infer_tiled` / `nms` and a `--tiled` eval mode. Video ball coverage
  607 -> 746 / 750 frames; possession 0.845/0.155 -> 0.901/0.099; test-split
  ball recall 0.705 -> 0.744, mAP50 0.704 -> 0.720. 91 tests passing.

## Stage-9 accuracy upgrade: C1 physics + C2 fitted shot model

- **C1 — Spearman-style pitch control** (`src/spatial/pitch_control.py`):
  replaced the heuristic reach/speed model with the industry-standard
  acceleration-constrained race. Each pitch point is won by the team whose
  player gets there first under constant acceleration
  (`PITCH_CONTROL_A_MAX_MS2=7.0`) from the player's position **and velocity**
  (momentum matters). Time-to-reach: sprinting toward a point
  `(-v+sqrt(v^2+2ad))/a`, moving away `|v|/a + sqrt(2d/a+(v/a)^2)`; arrival
  gap -> control via a logistic (`PITCH_CONTROL_TIME_SIGMA=0.3` s).
  `_track_velocities` (central differences), `_velocity_map` (built once per
  clip, reused per frame -> 143 s -> 16 s) and `_race_surface` (fast path).
  NaNs are skipped (967 occlusion rows no longer poison the surface).
- **C2 — EPV from a fitted shot model** (`src/statsbomb/shot_model.py`,
  `scripts/fit_shot_model.py`): P(goal) is a logistic over distance and
  angle-to-goal, `logit = 0.8747 - 0.1039*d - 0.1856*|y-35|`, **fit on 428 real
  StatsBomb shots (77 goals) from all 16 WC-2022 knockout matches** (the d^2
  feature was dropped: it produced a spurious near-goal peak; empirical
  conversion is monotone 0.375 at 3-6 m -> 0.028 at 20-25 m). Model cached at
  `data/statsbomb/shot_model.json` (two-feature refit); exponential
  fallback retained. EPV update is absorbing: `v = score + discount*keep*
  (1-score)*v[next]`; near-goal EPV 0.645, halfway 0.006.
- **C5 — space creation reworked** to measure opponent threat at the runner's
  **exact position** (nearest opponent's time-to-reach, momentum included,
  logistic-mapped) instead of the nearest grid center (which let an opponent
  "beat" a runner to a point up to 3.5 m away they don't occupy). Runs now:
  11 (team 0) / 14 (team 1) on the real clip, 6-16 m, max threat 0.14-0.30;
  `min_opp_control` renamed `max_opp_control`.
- Tests: momentum test proves a 10 m/s sprinter wins a cell a static player
  loses (`test_spatial` 15); new `tests/test_shot_model.py` (extract, monotone
  fit, grid shape, fallback, cache round-trip). **Full suite 187 passing.**
  Remaining: real-footage StatsBomb gating of C1/C2 (blocked on user footage).
- **Render upgrade** (`src/render/hud.py`, `scripts/render_pitch_video.py`):
  the radar now visualizes the upgraded surfaces instead of raw Voronoi.
  `draw_control_surface` paints the live C1 race surface (blue team 0 / red
  team 1, refreshed every `--hud-interval` frames; `--radar voronoi` falls
  back to ownership cells) and `draw_epv_heatmap` renders the C2 value grid
  as a green/viridis strip beneath the radar (`--no-epv` disables). Radar
  height is fixed by always stacking the strip (the earlier frame-15 writer
  drop was a silent size mismatch). New tests for both helpers; full render
  `output/pitch_radar_hud_v4.mp4` (750 frames, 47 MB). **Suite 190 passing.**
## Pivot: provider-agnostic analytics layer

Direction change after the broadcast/tactical-camera debugging arc: **statistical
modelling is now the foundation, fed by normalized third-party event data
(StatsBomb first), with Video/CV demoted to optional enrichment** (that path is
parked). Research workstream (Soccermatics + The Numbers Game digests) is saved
in `research/` and informs the model choices.

### New package: `src/analytics/`
- **`schema.py`** — provider-agnostic normalized types (`Match`, `Team`,
  `Player`, `Event`, `PassEvent`, `ShotEvent`, `FreezeFrame`) on the canonical
  120x70 pitch frame (defending goal at x=0 for both sides). Models never see
  provider JSON.
- **`adapters/base.py`** — `ProviderAdapter` Protocol (match/events/passes/
  shots/freeze_frames).
- **`adapters/statsbomb.py`** — `StatsBombAdapter` reusing `StatsBombLoader`
  (fetch+cache), `convert_pitch_xy`/`seconds_from_timestamp`/`team_map`.
  Handles Pass (end_location, outcome, receiver), Shot (result, `statsbomb_xg`),
  and 360 `FreezeFrame` snapshots (3683 for the WC final). Any other provider
  (Opta/Wyscout) plugs in behind the same Protocol.
- **`models/core.py`** — passing network (Soccermatics §7, ≥3-link threshold),
  possession/passing-rate clock (gap attribution), slippage funnel
  (Numbers Game §1: possessions→shots→goals), Poisson expected points
  (Numbers Game §3) with win/draw probability, and goal-line valuation
  (prevention ≈ scoring).
- **`models/xg.py`** — xG aggregation using provider `statsbomb_xg` or the
  fitted shot model (`shot_model.py`) fallback; per-side + per-player.
- **`analyze.py`** — `analyze_match(adapter, id)` bundles everything into a
  `MatchReport` (per-side possession, passing rate, shots/xG/goals, funnel).

### Verified on the locally-cached WC-2022 Final (3869685, no network)
- Possession share France 0.657 / Argentina 0.343; passing rates sane.
- Shots/xG/goals: France 14 / 5.41 / 5, Argentina 24 / 5.89 / 7 (includes the
  penalty shootout — a faithful reflection of StatsBomb's event data).
- Passing network edges (3090→20572 ×18), top degree player 165 connections.
- Poisson expected points: balanced ~1.37, strong side (2.2/1.0) ~2.13.
- **Whole suite 203 passing** (190 baseline + 13 new `test_analytics.py`).

### Fixes found along the way
- `_possession_chains`: turnover event now *opens* the next chain (was appended
  to the old one, cross-contaminating team possession).
- Poisson expected-points loop: win probability had used P(opponent≥k) instead
  of P(opponent<k) — inverted (strong side appeared to earn less than weak).
- `FreezeFrame.players` tuple was passed as 5 positional args (fixed).

### Next
- Document the adapter contract in `docs/`; add a passing-network/possession
  report script; optionally wire the xG model into the radar EPV surface and
  expose DefenseAnalytics-facing summaries.

## Reorganization: Video/CV becomes a dormant adapter

Folder structure unified so the model layer is provider-agnostic and every input
source is an adapter. The entire CV/video pipeline moved under the adapter tree
and is **dormant** (parked during the pivot) — it stays a first-class source,
but refuses to run unless explicitly enabled.

### New layout under `src/analytics/`
```
analytics/
  schema.py               normalized types (single interchange format)
  common.py               shared contracts (TrackingFrame, PLAYER_CLASSES) — NEW
  analyze.py              adapter -> MatchReport
  models/                 statistical models
    core.py               passing network / possession / funnel / expected points
    xg.py                 xG aggregation
    pitch_control.py      (moved from src/spatial) control/EPV/threat/territory
  adapters/
    base.py               ProviderAdapter protocol
    statsbomb/            ACTIVE structured data source (moved from src/statsbomb)
      adapter.py          (was analytics/adapters/statsbomb.py)
      loader.py alignment.py validate.py shot_model.py
    video/                DORMANT video/CV source (moved from src/*)
      adapter.py          VideoAdapter -> raises VideoAdapterDormant unless enabled
      pipeline.py reader.py
      detection/ homography/ tracking/ teams/ events/ metrics/ render/
```

### What moved
- CV stack `detection/homography/tracking/teams/events/metrics/render` +
  `pipeline.py` + `video/reader.py` -> `src/analytics/adapters/video/`.
- `src/statsbomb/*` -> `src/analytics/adapters/statsbomb/` (loader/alignment/
  validate/shot_model + adapter).
- `src/spatial/pitch_control.py` -> `src/analytics/models/pitch_control.py`.
- All absolute import prefixes rewritten across 53 files (src/tests/scripts/main).

### Decoupling
- `analytics/common.py` owns `TrackingFrame` + `PLAYER_CLASSES` (single source);
  the statsbomb source adapter and the video metrics both consume it, so the
  data-source side never imports from the video pipeline.
- `pitch_control` (a model) imports `PLAYER_CLASSES` from `common`, not video.
- Video heavy imports (torch/ultralytics) are **lazy** — importing the module
  never pulls the CV stack.

### Dormant gate
- `config.VIDEO_ADAPTER_ENABLED = False`; `VideoAdapter()` raises
  `VideoAdapterDormant` on every provider method unless `enabled=True`.
- `_runner.run_video_to_events` is the parked seam where re-enabling Video/CV
  resumes (CV tracking+event vocabulary -> normalized schema).
- Adapter surface now: `StatsBombAdapter` (active), `VideoAdapter` (dormant).

### Tests
- 3 new dormant-adapter tests; **full suite 206 passing** (was 203). StatsBomb
  end-to-end report still works (France vs Argentina WC final).
- Safety backup of pre-reorganization src/tests kept in /tmp.

---

## Enriched Models Layer (all four directions)

### Direction 1 — 360 + event-time pitch control
- `models/snapshot.py`: `to_tracking_df` (freeze-frame -> (t,x,y,team) frames),
  `snapshot_summary` (per-snapshot possession + territory share from 360),
  `territory_series` (event-time pitch control / territory over a match).
- Reuses canonical pitch frame (x∈[0,120], y∈[0,70]) and `convert_pitch_xy`.

### Direction 2 — Flow fields + average-position formations
- `models/flow.py` (team-level): `team_formation` (centroid/width/length/n),
  `flow_field` (bin-paired consecutive-snapshot velocities, nearest-position
  greedy matching, nan-safe).
- **Key data finding**: StatsBomb 360 freeze-frames carry NO player identity
  (only {teammate, actor, keeper, location}). So per-player formations/flow are
  impossible from 360; models are **team-level aggregates** only. Per-player
  analysis requires real tracking (dormant VideoAdapter / premium provider).
- Cleared dead `actor_count` logic (players unlabelled -> every entry an
  independent position sample).

### Direction 3 — Pressure / pressing map
- `models/pressure.py` (Soccermatics Ch 9): `press_map` (press counts by
  third and x-band, counter-press count, pressed-against), `press_windows`
  (on-ball ≤2.3 s vs deep ≥5.5 s press win windows from event durations).
- Verified on WC-Final: France 376 press events, Argentina 393; both ~1.0 s avg
  press duration (mostly on-ball presses).

### Direction 4 — Richer passing/possession
- `models/attacking.py`: `shot_map` (shots/goals/xG bucketed into an nx×ny
  pitch heatmap), `progressive_summary` (progressive/backward/through-ball
  pass counts per team; progressive = ≥10 m advancement).
- Verified: France 17% progressive share, Argentina 15%; 3 through-balls (ARG).
- (Shots total includes penalty shootout — xG/shot maps on open play minus
  penalties if strict.)

### Wiring & verification
- `analyze.py` `MatchReport` extended: `shot_map`, `passing_progress`,
  per-side `press`/`press_windows`, and opt-in `spatial` (`SpatialReport`:
  formation + flow + sampled territory) via `analyze_match(..., with_spatial=True)`.
- `models/__init__.py` exports all new models.
- **10 new tests** in `tests/test_analytics.py` (snapshot, flow, pressure,
  attacking, enriched events/freeze-frames). **Full suite: 216 passing**
  (was 206).

---

## Data Acquisition — StatsBomb 360 (bulk)

Added `scripts/fetch_statsbomb_360.py` — enumerates matches with
`match_status_360 == "available"` from the open-data match index and downloads
`events` + `lineups` + `three-sixty` for each into `data/statsbomb/` (same
`match_{id}_{kind}.json` naming, re-runnable / skip-if-cached / per-match
failure-tolerant).

### Acquired: 312 matches w/ 360 + events + lineups (938 JSON, 2.6 GB)
| Competition | Season | # 360 |
|---|---|---|
| FIFA World Cup | 2022 | 64 |
| UEFA Euro | 2024 | 51 |
| Women's World Cup | 2023 | 64 |
| Bundesliga | 2023/24 | 34 |
| La Liga | 2020/21 | 35 |
| Ligue 1 | 2021/22 | 26 |
| MLS | 2023 | 6 |
| Women's Euro | 2025 | 31 |

Also backfilled lineups for the 15 WC-knockout matches that previously lacked them.

### Verified
- All 312 event files parse cleanly; adapter loads matches across 7 comps
  (WC/Women's-WC/Euro24/Bundesliga/LaLiga/Ligue1/Women's-Euro25).
- Metadata `shot_fidelity_version`, `match_status_360` from live match index.
- Full test suite still **216 passing**.

### Data caveat — MLS 360 is not event-pairable
- Pairing rate of 360 `event_uuid` -> events `id`:
  **7 of 8 comps = 100%**; **MLS = 0%** (all 6 matches). The MLS 360 freezeframe
  event_uuids appear *nowhere* in the MLS events export (verified against every
  uuid-like field), so the adapter returns 0 frames for MLS. MLS raw 360 exists
  (3.1–6.2 MB each) but cannot be joined to events by UUID with the current
  schema. **MLS matches removed** (per user decision). Active usable set:
  **306 matches** (events + lineups + 360 each) out of the 312 originally
  acquired.

---

## Multi-Source Architecture — Retrievers, Adapters, Resolver

Extended the provider layer so dense-tracking and scraped sources feed the same
statistical models while keeping the model layer provider-agnostic.

### Capability mixins (`src/analytics/adapters/base.py`)
`ProviderAdapter` is now capability mixins a source may implement together:
`EventsProvider`, `SparseTracking` (event-keyed snapshots, e.g. StatsBomb 360),
`DenseTracking` (continuous per-player series with identity), `LineupsProvider`.
StatsBomb = Events + Sparse + Lineups; SkillCorner = Dense + Events + Lineups;
Wyscout = Events + Lineups. Detection prefers the explicit `CAPABILITIES` set
class-attr so a `[]`-returning stub is never mistaken for a real capability.

### Retriever layer (`src/analytics/retrieval/`)
Acquisition half: `Retriever` protocol + `retrieval.registry` (12 built-in
sources). Retrievers cache payloads under `data/retrieval/<source>/`; adapters
read them offline. Season-scoped sources cache under
`data/retrieval/<source>/<season>/` so re-running for a new season never
clobbers an older one.

| Retriever        | Source                                              | Capability added |
|------------------|-----------------------------------------------------|------------------|
| `StatsBombRetriever` | cached `data/statsbomb/` (306 matches)          | events / sparse / lineups |
| `SkillCornerRetriever` | SkillCorner Open Data (GitHub, MIT) A-League 24/25 | **dense** (10 fps) |
| `WyscoutRetriever`    | koenvo mirror of Pappalardo et al. 2019 (CC-BY 4) | events / lineups |
| `HTTPFileRetriever`   | generic cached HTTPS downloader w/ **Git LFS** resolution | files |
| `IDSSERetriever`      | IDSSE/Sportec DFL Bundesliga (pysport HF mirror, 25 Hz) | dense (7 matches cached) |
| `MetricaRetriever`    | Metrica sample-data (GitHub, MIT) 25 Hz tracking | dense / events / lineups |
| `OpenFootballRetriever` | OpenFootball CC0 `football.json` (`en.1` EPL) | events / lineups |
| `soccerdata` factory  | FBref / Understat / WhoScored / SofaScore / ESPN | events (scraped) |

soccerdata scaffold corrected to the 1.9.x API surface: class is `Sofascore`
(lowercase s) not `SofaScore`; ESPN league key is `ENG-Premier League` (not
`ENG.1`); `football-data` removed (not a soccerdata class — separate site/API).
`available_resources()` reports each source's real capabilities (FBref =
schedule/player_season/team_season/events; Understat = schedule/player_season/
shots; WhoScored = schedule/events; ESPN & Sofascore = schedule only).

HTTP layer added Git LFS pointer resolution (SkillCorner stores the 89 MB
tracking JSONL behind an LFS pointer on raw.githubusercontent).

### Adapters
* **`SkillCornerAdapter`** (Dense + Events + Lineups): parses
  `{id}_tracking_extrapolated.jsonl` into 59k `Frame`s per match with per-player
  identity + canonical [0,120]x[0,70] coords (meter origin-centre rescaled);
  `dynamic_events.csv` possession-actions -> our event vocabulary via `end_type`
  (pass/shot/possession); lineups from `{id}_match.json`.
* **`WyscoutAdapter`** (Events + Lineups): [0,100] coords (home = left->right,
  top-left origin) -> canonical frame; `tags` (1801 = accurate) -> pass outcome;
  keyed team/player dicts flattened. Loads via the kloppy-compatible mirror.
* **`Resolver` / `build_resolver()`**: maps a model's primitive needs onto the
  best single adapter per capability; cross-source merge (e.g. StatsBomb events
  + SkillCorner dense tracking) when no single source suffices. `ResolvedProvider`
  exposes `events/passes/shots/freeze_frames/tracking/match`.
* **`IDSSEAdapter`** (Dense 25 Hz + Events + Lineups): streams the 418 MB
  `positions_raw_observed` XML with `ElementTree.iterparse` into `Frame`s,
  bucketing persons by (half, N) — N resets per half (10000../100000..) — into
  per-timestamp canonical frames; events from `events_raw` XML (Pass/Cross
  inside `<Play>`, `ShotAtGoal` + result element, incl. nested penalty shots);
  lineups from `matchinformation` XML (DFL `TW` = keeper). Verified across all
  7 cached Bundesliga matches (Köln 1-2 Bayern, 3 goals; etc.).
* **`alignment` layer** (`src/analytics/alignment.py`): timestamp-level glue
  (`nearest_frame_index` / `frame_at` / `align_events_to_frames` /
  `possession_at`) pairing a discrete event with the tracking `Frame` describing
  the pitch at that instant, on the shared `timestamp_sec` axis — the state
  models need at event time when events + tracking come from different sources.

### Retrieved data (cache now populated)
* `data/retrieval/skillcorner/1886347/` — Auckland FC 2-0 Newcastle Utd
  (A-League 24/25): 59,061 tracking frames, 5,079 events (902 passes, 23 shots),
  36 players. Retrieved live, incl. LFS-resolved 89 MB tracking.
* `data/retrieval/wyscout/matches/2499841.json` — Huddersfield 0-5 Man City
  (17/18): 1,593 events (855 passes, 15 shots), 36 players, teams metadata.
  (Plus 1694390/1694391/2576338; several sampled ids 404 on the mirror.)
* `data/retrieval/idsse/{J03WMX,J03WN1,J03WOH,J03WOY,J03WPY,J03WQQ,J03WR9}/`
  — full 7-match 25 Hz DFL set (~2.4 GB): matchinfo + events_raw +
  positions_raw_observed per match, retrieved live from the pysport HF mirror.
* `data/retrieval/{understat,espn,sofascore}/2021/schedule.json` + 
  `data/retrieval/fbref/2021/{schedule,player_season,team_season}.json` — live
  soccerdata scrapes of the EPL 20-21 season (380-game schedules; fbref 532-row
  player season). **Caveats**: whoScored is captcha/proxy-blocked headless
  (`NoneType` on schedule/events); scraped tables are personal-use only (no
  redistribution licence).
* **TODO (blocked, do later) — 2025-26 full-season FBref.** Added `--season`
  to `scripts/retrieve_sources.py` and made the soccerdata retriever
  season-scoped (`data/retrieval/<source>/<season>/`, e.g. `fbref/2026/`) so a
  second season never clobbers 20-21. Fetch 25-26 with
  `python scripts/retrieve_sources.py --sources fbref --season 2026`.
  Currently BLOCKED: fbref.com returns HTTP 403 (IP/anti-bot block) from this
  host, and the sandbox reaps long-lived background processes, so a live
  scrape can't complete from here. Retry later when FBref's block window
  expires; the per-resource cache resumes partial progress.
* FIFA World Cup 2022 (PFF) — manual drop only (no auto-fetch; licence-gated).

### Cross-source end-to-end
```
build_resolver(SkillCornerAdapter(...))
  .resolve(match_id, needs={"events","dense_tracking","lineups"})
```
resolves events/lineups -> StatsBombAdapter and dense_tracking ->
SkillCornerAdapter; models consume `frames` (formation/flow) and `events`
(possession/funnel) without knowing the source. **20 tests** in
`tests/test_retrieval.py` (offline, cached data) + **5 tests** in
`tests/test_alignment.py` (timestamp glue). **Full suite: 232 passing**
(was 228; 13 video/CV tests deselected).

---

## Season-aware providers + dense-tracking/OpenFootball wiring

Seven-item closeout: Understat 2025-26, season-scoped caching, two new data
providers (Metrica, OpenFootball), Resolver end-to-end, and the full-suite gate.

### 1. Understat 2025-26 shots (fetched)
- `python scripts/retrieve_sources.py --sources understat --season 2025`
  cached `schedule.json`, `player_season.json`, and `shots.json` under
  `data/retrieval/understat/2025/` (shots.json 3.19 MB, 9,524 shots) — the full
  completed 2025-26 Premier League season.
- **Season keying**: soccerdata input `N` -> season `N(N+1)`; `2025` = 2025-26
  (finished), `2026` = 2026-27 (ongoing). Cache dirs follow the input value.

### 2. SoccerDataAdapter season-aware
- `SoccerDataAdapter` now takes `season` and passes it to the soccerdata source;
  verified through the adapter for 2021-22 and 2025-26
  (`tests/test_soccerdata_adapter.py`, 8 passing incl. `test_season_awareness`).
- New tests link `SoccerDataRetriever` to schedule/player_season/shots resources.

### 3. Metrica Sports dense-tracking + events (new provider)
- `MetricaRetriever` (`src/analytics/retrieval/tracking.py`) fetches the two MIT
  sample games from the metrica-sports GitHub: `tracking_home.csv` /
  `tracking_away.csv` (25 Hz, ~32.8 MB each, home/away row-aligned) +
  `events.csv`. Both SG1/SG2 cached under `data/retrieval/metrica/`.
- `MetricaAdapter` (`src/analytics/adapters/metrica/`) implements Events + Dense
  + Lineups. Coordinates: `x_canon = x_norm*120`, `y_canon = (1-y_norm)*70`
  (fixed broadcast frame, y-flip from top-left origin; kickoff (0.5,0.5) ->
  (60,35)). `NaN`/out-of-camera rows are skipped (`to_finite`), never surfaced
  to downstream models. `sample_every=5` default downsamples 25 Hz -> 5 Hz.
  Events vocab maps to schema types (SHOT subtypes -> goal/saved/off_target/
  blocked/post). Registered in `register_tracking_builtins`.
- **6 tests** (`TestMetricaAdapter`) + `test_registered_sources` asserts
  `"metrica"`.

### 4. OpenFootball CC0 provider (new provider)
- Dataset survey: `openfootball/football.json` (CC0) is score/meta only in JSON;
  the England repo `league.txt` carries **scorer annotations for 2025-26**
  (2024-25 and earlier are inline-score only). `Yggdrasil-Org/free-soccer-data`
  is gone (404), so OpenFootball wraps the EPL `en.1` data.
- `OpenFootballRetriever` (`src/analytics/retrieval/openfootball.py`) fetches
  `league.json` + `league.txt` per season into
  `data/retrieval/openfootball/<season>/en.1/` (2024-25 and 2025-26 cached).
- `OpenFootballAdapter` (`src/analytics/adapters/openfootball/`) implements
  Lineups + Goal-only Events: parses `(Scorer MIN'[, MIN']*; ...)` annotations
  into named goal events (penalty/own-goal tagging), with a scoreline-fallback
  for seasons/badges without annotations. Goal events default to
  ball-at-kickoff (60,35) with qualifiers `scorer` / `event_type` / `source`.
- Goal-annotation fix: block annotation lines are **after the
  team/score line** (the continuation line of a two-line annotation does not
  start with `(`); parsing now joins every line after the score line. 2025-26
  row 0 -> 6 real goals; **924/952** goals scorer-annotated season-wide.
- **6 tests** (`TestOpenFootballAdapter`) + registry asserts `"openfootball"`.
- Retriever registry now enumerates **12 sources** (statsbomb, skillcorner,
  wyscout, http_file, idsse, metrica, openfootball, fbref, understat,
  whoscored, sofascore, espn).

### 5. Resolver end-to-end (cross-source integration)
- `TestResolver` extended: StatsBomb events/lineups + Metrica dense tracking
  resolved as one `ResolvedProvider` (`test_cross_source_metrica_integration`);
  models consume the merged feed source-agnostically — `team_formation` /
  `flow_field` over `rp.tracking()`, `progressive_summary` / `shot_map` over
  `rp.passes()` / `rp.shots()`.
- SkillCorner path covered (`test_cross_source_skillcorner_metrics`) ->
  `team_formation` over resolved SkillCorner frames.
- **5 resolver tests passing.** Resolver tests are offline (cached StatsBomb
  match 3869685 + SkillCorner dir + Metrica sample game).

### 6. Full test suite green
- `pytest tests/ -q` — **265 passed, 0 failed** (the whole suite, including the
  CV/video files, which need no exclusion now; earlier convention deselected 13
  video/CV tests). New Math-domain count: 165 analytics/adapter/season tests +
  104 CV/video tests.
- Soccerdata season-id warning (`"2025" is ambiguous`) is expected: soccerdata
  interprets the input value as `NN-NN+1`; the adapter documents this mapping.

### 7. Docs
- This section. Retriever table, registry count, and status log updated to
  reflect the season-scoped cache layout and the four live trackers
  (StatsBomb / SkillCorner / Metrica / OpenFootball) behind the Resolver.

### Remaining (blocked / parked)
- FBref 2025-26: HTTP 403 (IP/anti-bot). Retry `--season 2026` later.
- WhoScored: captcha-blocked headless.
- FIFA World Cup 2022 (PFF): manual drop only (licence-gated).

---

## Model layer: centralization, score-curve, ratings, VAEP

Four new provider-agnostic models landed in `src/analytics/models/` (all consume
`schema`/plain numbers only, none touch provider JSON; 16 new tests).

### 1. Passing-network centrality (`core.py`)
- `PassingNetwork.betweenness(weights)` — Brandes on the pass graph (fewest
  intervening players, volume tiebreak); normalised to [0,1].
- `PassingNetwork.decentralization(weights)` — Freeman centralization: 1.0 = one
  hub dominates shortest paths, 0.0 = fully distributed. Lower = the
  decentralized build-up Soccermatics §3 links to ~8% higher scoring.
- `summarize_network()` → `PassNetworkSummary` (degree-top, betweenness,
  decentralization, most-central player).

### 2. Win-probability score curve (`winprob.py`)
- `win_probability(lam_for, lam_against, minute, score_for, score_against)` —
  P(win/draw/loss) from two truncated-Poisson goal processes raced over the
  remaining minutes.
- `score_curve()` — the full Soccermatics ch.3 score-curve from kickoff.
- `goal_value_by_minute()` — marginal E[pts] of the next goal per minute; the
  Numbers Game non-linearity (equaliser at 90' >> early icing goal) falls out
  of the model: late goal values rise toward full time.

### 3. Dixon-Coles double-Poisson ratings (`rating.py`)
- `MatchResult` (provider/dataset-agnostic goal pairs, e.g. from OpenFootball or
  Understat fixtures) → `fit_dixon_coles()`: ML attack/defence strengths per
  team + home-advantage factor + low-score ρ (BFGS, attack anchored at mean 1).
- `predict_match()` — outcome probabilities + expected goals for a fixture
  (normalised, τ-corrected); `table_from_results()` — actual vs expected points
  and per-team luck. Verified on synthetic leagues: strength order recovered,
  home advantage > 1, ρ < 0 as expected for modern football.

### 4. VAEP on-ball valuation (`vaep.py`)
- `hazard_surface(attacking_goal_x=0.0)` — cell scoring-probability grid
  decaying from the attacking goal; the default matches the alignment-mirrored
  canonical events (both sides attack x=0), pass `120.0` for attacking-frame
  coordinates.
- `value_action()` — value = Δ(scoring prob) − Δ(conceding prob): shots use
  provider xG (scoring-before = 0); passes/carries move between cells; lost
  balls hand the opponent a 40% share of the destination hazard.
- `vaep()` → per-action list + per-player/per-team summary. Real-data smoke on
  the WC-Final: 1,301 actions valued; top player #3009 = +2.96 across the match.

**Full suite: 281 passing** (was 265; +16 model tests). Constants pithy and
importable; each model exported from `src/analytics.models`.

## Analytics agents layer

A team of **deterministic model+knowledge** agents (no LLM required) under
`src/analytics/agents/`, consumed via a knowledge base, that turn adapter +
model outputs into a structured, human-readable match analysis. 13 new tests;
**full suite now 294 passing** (was 281).

### Files
- `knowledge_store/football.md` — 15-section curated knowledge base of the key
  methods (xG, VAEP, pitch control/EPV, pressing, networks, win-prob,
  Dixon-Coles, …) with formulas and when-to-apply guidance.
- `base.py` — `AgentContext` (match_id, match, events, passes, shots, frames,
  extras), `Agent` protocol, `AgentRegistry`.
- `knowledge_store.py` — `KnowledgeStore.load()` parses the markdown into
  section entries; `search(query)` does keyword scoring to pull the relevant
  passages; `section(id)` fetches by number.
- `tactician.py` — `TacticianAgent` → `TacticalReport` (formation centroid /
  width / length from frames, passing-network decentralization + most-central
  player, press maps: total events + final-third share + deep-press count,
  flow magnitude) plus readable insights.
- `stat_scientist.py` — `StatScientistAgent` → `StatisticalReport` (xG, goals,
  over/under-performance, shots, VAEP value+actions, possession share, passing
  rate, funnel). Also computes expected-points, goal-line valuation, the
  win-probability score curve, and per-minute goal value.
- `visualizer.py` — `VisualizerAgent` → `VisualizationReport` with matplotlib
  figures: shot map (size = xG), xG timeline, passing networks per side, press
  heatmaps per side, win-probability curve. `save_all()` writes PNGs to
  `output/`.
- `domain_expert.py` — `DomainExpertAgent` → `NarrativeReport`: headline, TL;DR,
  tactical + statistical summaries, key moments (goals + high-xG misses), player
  highlights (top VAEP), coaching insights from the knowledge base, and a full
  markdown narrative.
- `orchestrator.py` — `orchestrate(adapter, match_id)` runs the full chain
  Tactician → StatScientist → Visualizer → DomainExpert and assembles a
  `MatchInsightReport`. Each agent runs in a try/except so one failure is
  recorded in `report.errors` without aborting the pipeline.

### Entry point
```python
from src.analytics.agents import orchestrate
report = orchestrate(adapter, match_id, save_figures=True)
report.narrative.headline      # "Liverpool 2-0 Arsenal"
report.narrative.full_narrative
report.tactical.home.decentralization
report.statistical.home.xg
report.visual.saved_paths      # PNGs in output/
```

## HTML reports per match — timestamped analysis folders

Every orchestrated match now produces a **shared, timestamped analysis folder**:

    output/analyses/<YYYY-mm-dd_HH-MM-SS>_<match_id>/
        figures/fig_*.png                 12 football-analysis charts
        tactician.html                    tactical report page
        stat_scientist.html               statistical report page
        visualizer.html                   all charts, inline
        domain_expert.html                narrative report page
        index.html                        links all four + key visuals

### `html.py`
- `MatchOutputDir` — one folder per match owned jointly by every agent.
  Handles folder creation, `write_text()`, `save_figure(fig, name)` (PNG →
  `figures/`), and `figure_data_uri(name)` (base64 data URI for inline embeds).
- Shared page scaffolding: dark theme, `html_page()` wrapper, `esc()` escaping,
  `stat_card()`/`figure_img()`/`index_link()`/`match_header()` builders.

### `renderers.py`
- `render_tactician` / `render_stat_scientist` / `render_visualizer` /
  `render_domain_expert` — one styled page per agent, figures inlined as
  base64 so each page is fully self-contained (renders from anywhere).
- `render_index` — front page: match header, xG/VAEP snapshot, key visuals,
  and cards linking every agent report.

### `visualizer.py` — richer football-analysis chart set
- **Shot map** on a broadcast-frame pitch with goal mouths; marker size = xG,
  star = goal, 0.50+ xG labelled, off-target greyed.
- **xG flow** — cumulative xG per side with goals marked plus the white dashed
  net-xG (home − away) differential line.
- **Pass maps** per side (complete solid / incomplete dashed arrows, average
  player positions as nodes) and **passing networks** (directed, arrow width
  ∝ volume).
- **Press heatmaps** per side, **win-prob curve** with goal markers, **team
  comparison radar** (xG, shots on target, VAEP, possession, pass rate,
  funnel), **top-VAEP contributor bars** per side, and (when tracking frames
  exist) **average-position formation maps**.
- All figures save into the match folder then embed into the + index pages.

### `orchestrator.py`
- Creates the `MatchOutputDir` up front and seeds `ctx.extras["output_dir"]`
  so **every agent writes into the same folder** for the same match.
- After each agent runs, its HTML is written into the folder; the index is
  written last (always, even if an agent failed — errors are collected in
  `report.errors` and pages degrade gracefully).

**Full suite: 301 passing** (was 294; +7 HTML/reporting tests).
