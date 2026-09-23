<div align="center">

# PerformanceAnalyzer

**End-to-end football analytics: from raw match video and event/tracking feeds to
xG, VAEP, pitch control and agentic match reports.**

</div>

---

## Table of contents
- [Problem space](#-problem-space)
- [Solution space](#-solution-space)
- [Architecture](#architecture)
- [Results — see it working](#results)
- [Getting started](#getting-started)
- [Testing](#testing)
- [The thinking behind the work](#the-thinking-behind-the-work)
- [Data & licensing](#data--licensing)
- [Roadmap](#roadmap)
- [License](#license)

---

## Problem space

Football analytics has a *capture* problem: every honest performance question —
*"Did we actually create the better chances?"*, *"Which player changed the game
when he came on?"*, *"How much press did we really generate?"* — needs clean,
structured, position-aware data. But that data is fragmented and hostile to work
with:

1. **Every source speaks a different dialect.** StatsBomb encodes events with
   360° freeze-frames; Metrica provides dense broadcast tracking; SkillCorner
   gives optical tracking; OpenFootball (FIFA/GIF public feeds) uses its own
   terminology; Wyscout, Understat, FBref and WhoScored each have their own
   schemas and, in many cases, anti-bot walls. Building one analysis on one
   source means betting your entire project on that vendor.

2. **The analytics methods rarely ship with the data.** The interesting metrics —
   expected goals (xG), expected possession value (VAEP), pitch control /
   expected possession value (EPV), passing networks and centralisation — sit
   scattered across academic papers (`Soccermatics`, *The Numbers Game*, plus a
   mature Statistics & Machine Learning literature). Reproducing them from
   scratch is the real work, and most people never get past a single notebook.

3. **Broadcast video is the messiest source of all.** The team, the ball and
   the pitch are moving simultaneously, the camera cuts without notice, the
   pitch keypoints vanish between frames, and any tracking error compounds
   through every downstream metric.

4. **Results are not reproducible.** A one-off notebook that renders one match
   cannot be audited, re-run across 300 matches, or compared against a held-out
   ground truth.

**The objective of this project is a single, source-agnostic analytics stack
that turns *any* supported feed — broadcast video, event data, or tracking data —
into defensible, explainable match reports, validated against ground truth, and
ready to run on hundreds of matches.**

---

## Solution space

PerformanceAnalyzer is two layers, both fully built and tested:

### 1. Computer-vision pipeline (broadcast video → events)

| Stage | Approach | Outcome |
|-------|----------|---------|
| Object detection | Fine-tuned YOLO (`best.pt`, trained locally) vs hosted v11 | **Ball mAP50 0.72 / recall 0.74** local beat hosted inference (0.644/mAP); released-stage model pushed to 0.720 |
| Multi-object tracking | ByteTrack (players) with a **ball-hypothesis stream** kept out of the tracker | Ball coverage **80.9% (hypothesis) vs 64.4% (raw detection)** over 750 frames |
| Team classification | SigLIP embeddings → UMAP → KMeans, with drift-and-refit | Stable per-team assignment, FP16 class-ready |
| Homography | Keypoint homography with **deque-mean temporal smoothing + shot-cut reset** | Within-shot reprojection **RMSE 17.5 px**; recovers on frame 1 after a cut; no false cuts |
| Metric layer | Tracking-derived possession, velocity, distance, sprints, line | Per-team metric report via CLI (`--metrics`) |

### 2. Analytics layer (events/tracking → models → reports)

- **A retrieval + resolver layer** covering 12 providers (StatsBomb + 360,
  SkillCorner, Metrica, OpenFootball, Wyscout, Understat, FBref, WhoScored,
  SofaScore, ESPN, …) with caching, capability resolution, and **cross-source
  merging** — e.g. StatsBomb events fused with Metrica or SkillCorner tracking
  for one match.
- **A model library (~50 exports)** implemented from the literature:
  possession, passing networks + centrality, ball-recovery funnel, Poisson
  expected points, **xG** (fit on WC2022 shots), **win-probability score
  curves**, Dixon-Coles ratings, **VAEP on-ball valuation** (via `xg`-conditioned
  sampling), **pitch control / EPV / space creation**, press maps, and flow
  formation sequences.
- **An agent layer** that turns the models into a person-readable result:
  - `TacticianAgent` → formation, decentralisation, most-central player, press maps
  - `StatScientistAgent` → xG, over/under-performance, VAEP, expected points, goal line
  - `VisualizerAgent` → shot maps, xG flow with net-xG differential, pass maps and
    networks, press heatmaps, win-prob curve, team radar, top-VAEP bars, formations
  - `DomainExpertAgent` → headline, TL;DR, narrative keyed to a curated football
    knowledge base
  - `orchestrator.orchestrate(adapter, match_id)` runs the chain and writes a
    **timestamped HTML report folder** — one self-contained page per agent with
    base64-inlined charts, shared by all agents for one match.

**301 tests** cover the detection through analytics stack and CI-style habits
(fixtures, caching, no network in tests).

---

## Architecture

```
src/
├── config.py                  # paths, model dirs, token env (never hardcoded)
├── analytics/
│   ├── retrieval/             # 12 source providers + cache + resolver
│   ├── adapters/              # normalized events/tracking/tracking+lineups API
│   │   ├── statsbomb/ metrica/ skillcorner/ idsse/ openfootball/
│   │   ├── wyscout/ soccerdata/ video/        ← video is dormant-by-design
│   │   └── resolver.py        # capability-based cross-source merge
│   ├── schema.py              # Event/Pass/Shot/FreezeFrame/Team/Player/Match
│   ├── models/                # ~50 methods: xg, vaep, winprob, rating,
│   │                          #   pressure, flow, attacking, snapshot,
│   │                          #   pitch_control, core
│   └── agents/                # tactician, stat_scientist, visualizer,
│                              #   domain_expert, orchestrator, html, renderers,
│                              #   knowledge_store/football.md
├── events/                    # Stage-9 event detectors (pass, recovery, ...)
├── video/  detection/  tracking/  teams/  homography/  metrics/  overlays/
main.py                        # CV CLI (--detect --metrics --dry-run)
scripts/                       # retrieval, eval, render, validate, fit_shot_model
examples/agents_demo.py        # one-match orchestration demo
results/                       # committed sample reports (see below)
tests/                         # 301 tests
```

### Orchestration in one call

```python
from src.analytics.agents import orchestrate
from src.analytics.adapters.metrica.adapter import MetricaAdapter

adapter = MetricaAdapter()
report = orchestrate(adapter, 1, save_figures=True)

report.narrative.headline       # "Liverpool 2-0 Arsenal" (any match)
report.statistical.home.xg      # e.g. 2.10
report.tactical.away.decentralization
report.visual.saved_paths       # PNGs in the match folder
report.html_written              # index + per-agent pages
```

All adapters implement the same minimal API (`match`/`events`/`passes`/`shots`/
`freeze_frames`), so `orchestrate` works identically on StatsBomb, Metrica, or
any supported source — this is the *source-agnostic* bet made concrete.

---

## Results

Real reports generated end-to-end and committed so you can see the output
without owning the data:

### World Cup 2022 Final — France 5-7 Argentina  *(StatsBomb, event + 360)*
- **xG:** France 5.41 (5 goals, −0.41), Argentina 5.89 (7 goals)
- **Verdict:** Argentina created the better chances despite the scoreline
- **Pressing:** France 376 defensive events vs Argentina 393 · 14 charts

### Metrica — Sample Game 1  *(dense broadcast tracking + events)*
- **xG:** home 2.10 (3 goals), away 0.17 (0 goals)
- **12 charts**: shot map, xG flow + net-xG, pass maps/networks, press
  heatmaps, win-prob curve, team radar, VAEP bars

Explore them:

```
results/
├── wc2022-final/
│   ├── index.html          # front page (open in a browser)
│   ├── tactician.html
│   ├── stat_scientist.html
│   ├── domain_expert.html
│   └── figures/            # 14 PNGs
└── metrica-sample-game-1/
    └── (same layout, 12 figures)
```

> The committed reports are **static exports**. Data volumes (StatsBomb's open
> corpus, Metrica/SkillCorner tracking) are licensed and large — they are not
> committed (see [Data & licensing](#data--licensing)). Re-run any report with
> `examples/agents_demo.py` after fetching your cached copy of the feeds.

---

## Getting started

```bash
# 1. Clone
git clone <repo-url>
cd PerformanceAnalyzer

# 2. Environment (Python 3.10+)
python -m venv .venv && source .venv/bin/activate
pip install -e ".[ml, teams, scrapers, dev]"   # or: pip install -r requirements.txt

# 3. Environment template (optional — only for hosted Roboflow/HF inference)
cp .env.example .env   # fill in keys you actually use; .env never committed

# 4. Sanity check
pytest -q
```

### Run a full match report (offline, from local cache)

```bash
PYTHONPATH=. python examples/agents_demo.py
```

prints the resolver mapping, each agent's typed outputs, the HTML report
folder, and the full narrative — then open the printed `file://…/index.html`.

### CV pipeline (needs a video + optionally model weights)

```bash
python main.py --video path/to/clip.mp4 --dry-run     # smoke the whole stack
python main.py --video path/to/clip.mp4 --detect
python main.py --video path/to/clip.mp4 --metrics --metrics-out table.csv
```

---

## Testing

```bash
PYTHONPATH=. pytest -q          # 301 passing
```

Tests follow real engineering habits:
- **No network.** Every test uses locally cached fixtures (StatsBomb `3869685`
  final, Metrica sample game, synthetic tracking).
- **Deterministic synthetic fixtures** for shape/pass/VAEP so numbers are
  asserted exactly.
- **CI-style structure**: `tests/test_retrieval.py`, `test_analytics.py`,
  `test_agents.py`, `test_detection.py`, `test_tracking.py`, `test_teams.py`,
  `test_homography.py`, `test_metrics.py`, `test_events.py`, … — the CV half
  validates against injected cuts and held-out frames.

---

## The thinking behind the work

A few decisions that shaped the codebase, and the reasons:

> **"One canonical event model, not one per vendor."**
> StatsBomb, Metrica, Wyscout and OpenFootball are all mapped into the same
> `Event`/`Pass`/`Shot`/`Player` schema (`src/analytics/schema.py`) with a
> single canonical pitch (both teams attack toward `x→0` — the StatsBomb
> convention — with Metrica's fixed broadcast frame transformed to match). The
> resolver then serves the *best available* source per capability
> (`events`/`dense_tracking`/`sparse_tracking`/`lineups`) and merges across
> adapters. The alternative — "one adapter per metric" — would have multiplied
> the modelling work by twelve.

> **"Track the ball as a hypothesis stream, not another bounding box."**
> The ball is the smallest, fastest object in the frame and ByteTrack
> fundamentally assumes pedestrians. The pipeline keeps a padded ball box
> *out* of the tracker and maintains a ballistic Kalman hypothesis, only
> associating detections when they agree. That single decision took ball
> coverage from 64.4% → 80.9%.

> **"Homography must survive camera cuts."**
> Broadcuts are routine, so the homography fuses keypoints through a
> deque-mean temporal filter and **resets explicitly on shot-cut detection** —
> validated to recover on frame 1 after an injected cut (17.5 px within-shot
> RMSE, no false cuts) rather than averaged across a cut boundary.

> **"A metric with no ground truth is a claim, not a result."**
> xG and VAEP are derived models; the `scripts/validate_against_statsbomb.py`
> harness and the released-stage recall gains (0.705 → 0.744) exist to keep the
> *measurement* honest, and match reports always pair xG against actuals with an
> explicit over/under-performance figure.

> **"The video adapter is dormant on purpose."**
> The CV pipeline is real, but as machine learning engineers we gate claims
> about it on real-footage ground truth (the StatsBomb held-out gate) before
> wiring broadcast-derived events into analytics. Better an explicit
> `VideoAdapterDormant` seam than a silently-validated claim.

> **"No secrets in the repo, not even 'just for now'.**"
> Every token flows through environment variables read in `src/config.py`; the
> committed `.env.example` holds empty placeholders, `data/` (licensed feeds)
> is gitignored, and NO `*.pt` weights are committed.

---

## Data & licensing

This repository ships **code and static result exports only**.

- **StatsBomb open data** — free for research (attribution required). Fetch the
  corpus independently (`scripts/fetch_statsbomb_360.py`); locally cached under
  `data/statsbomb/` (gitignored).
- **Metrica sports data** — free research dataset; `data/retrieval/metrica/`
  (gitignored).
- **SkillCorner / IDSSE / Wyscout / FBref / Understat / WhoScored** — each has
  its own licence and, for scraped sources, usage policies. Cache them locally
  under `data/retrieval/` (gitignored) via `scripts/retrieve_sources.py`.
- **Broadcast footage / trained weights** are not distributed.

`data/`, `output/`, `runs/`, `*.pt`, `.env.*` (except the template) are
gitignored. See `scripts/` for the fetch/eval tooling.

---

## Roadmap

- [x] Detection / tracking / homography pipeline (validated vs injected cuts)
- [x] Cross-source retrieval + resolver (12 providers)
- [x] Model library (xG, VAEP, pitch control/EPV, win-prob, Dixon-Coles, presses, networks)
- [x] Agent layer + HTML match reports (per-match, timestamped, render-anywhere)
- [ ] Human-readable *narrative* enrichment over the domain-expert output
- [ ] Wire CV-extracted events through the StatsBomb ground-truth gate into the
      same adapter API (unblocks the dormant video adapter)
- [ ] Refresh blocked feeds (FBref 403, WhoScored captcha) when policies allow

---

## License

[MIT](LICENSE) © 2026 Prameet Ghosh.