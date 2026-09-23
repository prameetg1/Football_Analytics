# Soccermatics — Analyst's Digest

**David Sumpter, *Soccermatics: Mathematical Adventures in the Beautiful Game* (2016, Pro-Edition = 15 ch / 4 parts)**

Compiled for the PerformanceAnalyzer CV pipeline: what metrics/visualizations an analyst can extract and their feasibility from broadcast-video tracking.

> **Feasibility tags:** **[CV]** = derivable directly from player/ball tracking · **[EVENT]** = needs event/outcome data (StatsBomb or engineering detectors) · **[CV+EVENT]** = both.

---

## 1. Passing networks (Ch 2, Ch 7)

- **Passing network** = the "tactical map" (Ch 7): nodes = players; links = partnerships with **≥13 successful passes**; **line thickness ∝ pass count**. Canonical figure: Italy–England Euro 2012 (Italy 68% possession, 36 shots; Pirlo 115 passes; England direct Hart→Carroll). One image that explains why a team wins/loses.
- **Two core metrics (Grund):**
  - **Passing rate** = successful passes per minute of possession → **+2 vs ~3 passes/min ≈ ~20% more goals**.
  - **Network centrality** → decentralized teams ≈ **8% higher scoring rate**.
- Passing rate **beats raw possession** as an outcome predictor.
- Ch 2: **formation minimum spanning trees** (shortest + second-shortest links between players' average positions; Barcelona–Panathinaikos) and **Delaunay triangles** among Xavi/Iniesta/Messi/Pedro.

**Feasibility:** **[EVENT/CV+EVENT]** — needs pass ownership (ball-owner attribution per frame) to build true networks; from pure coordinates you can dump trajectory sequences between players and threshold them into provisional links.

---

## 2. Space / pitch control / xG

- The book has **no "pitch control as a race"** model — that's Sumpter's *readthedocs course Lesson 6* (our separate implementation). The 2016 book uses **Voronoi diagrams** (nearest-player partitioning; Voronoi⇄Delaunay duals, Ch 2) and notes Voronoi "started to find its place" with analysts (Ch 15, Opta Pro Forum 2016).
- **xG for the book (Ch 12):** three shooting zones — **3.4%** outside box, **12.4%** in box, **32.2%** six-yard box.
- Ch 15 design principle: shot-position contours are the tactical target ("shoot from within the contours"), counter = **packing the box**.

**Feasibility:** **Voronoi/Delaunay/half-space occupation are directly computable** from tracking **[CV]**; xG needs shot events (detectable from video: x/y + header/volley/cross flags → fit zonal model) **[CV+EVENT]**.

---

## 3. Win probability / "score curve"

- **"Score curve" / "momentum" are NOT in this book** (verified absent). Nearest ideas:
  - Ch 1: goals are **memoryless** — probability **0.031 per minute** (2.79/match ÷ 90) → memoryless-Poisson baseline against which any "momentum" claim must be tested.
  - Ch 11–13: win-probability-in-time via **in-play betting odds** (e.g., Atlético 2-0 down at 31', odds ~7.00); quantification of luck and xG-vs-scoreline.
- The named "score curve" chart is Sumpter's **blog/course** material, not a chapter.

**Feasibility:** **[EVENT]** — needs a calibrated Poisson goal-rate model (not tracking).

---

## 4. Possession & domination

- **Raw possession is weak**; passing rate and pitch-position dominate (Ch 7 Italy example). Ch 12 ranking strategy = passing rate + xG-for.
- **Pressing = measurable domination (Ch 9, Prozone, 260M points):**
  - Counter-press: **first presser within 2.3 s** of a lost ball, **second presser within 5.5 s** (Guardiola's "six-second rule").
  - Deep press: **exactly one** defender steps, others hold channels.
- Ch 6: three-points-for-a-win game theory (incentives).

**Feasibility:** possession %, territory, and **2.3/5.5 s press windows** all derived directly from tracking **[CV]**.

---

## 5. Player movement & space (Ch 3, Ch 9, Ch 15)

- Ch 3: GPS ~5 Hz position data → **flow fields** (spatial movement vectors per zone) and **heat maps** (Pirlo vs Schweinsteiger).
- Ch 9: **learn team formation by clustering per-half average player positions** (Bialkowski 2014; 22 tracked at 200/s, ball at 2000/s) — origin of "formations from tracking data".
- Ch 15: **half-spaces** (5-space grid: LW/LH/C/RH/RW vs Van Gaal's 6×3 grid), the "blind area" a player faces, Guardiola vs Mourinho 4-4-2/5-4-1 spacing (7 m vs 5 m), false fullbacks.

**Feasibility:** all direct from tracking **[CV]** — average positions, clustering, flow fields, zone occupancy. Highest-value, lowest-dependency material in the book.

---

## 6. Key models / formulas

- **Poisson (Ch 1):** goal prob/min = **0.031**; 4 params/team (home/away scored & conceded) for match simulation & top-4 prediction.
- **Ch 12 Poisson regression** (predicts scoring rate): rate = **0.13 × passing rate + 0.76 × log(xG-for)**.
- **xG zones:** 3.4 / 12.4 / 32.2 %.
- **Markov value-of-possession (Ch 14, Sarah Rudd 2011):** states M/W/B/G/L; per-state goal probabilities **Box 25%, Midfield 15%, Wing 12%**; later expanded (11 locations × 2 pressure levels → hundreds/thousands of states). This *is* the "possession potential" — ancestor of EPV/xT and of our **C2 EPV pipeline**.
- **Other building blocks:** defensive convex hulls (Lichtsteiner example, Fig 7.14), "packing" (players taken out by a pass), possession-chain classification (Kwiatkowski/Brentford), player radar charts (Ted Knutson), Kelly criterion wager sizing, Euro Club Index / SPI / Elo baselines.

---

## 7. "90-minute attack" / momentum

- **No such chapter; "momentum" absent from the text.** Only adjacent themes: memoryless-Poisson timing (Ch 1), six-second ball recovery (Ch 9), crowd **contagion** (Ch 10, S-curves), betting markets (Ch 11). The "90-minute attack" framing is from Sumpter's blog/podcast.

---

## 8. Visualization techniques (highest leverage)

- **Poisson fit histogram** (goals/match vs fitted curve) — any-ratio sanity chart (Fig 1.2).
- **Voronoi / Delaunay** maps; **minimum spanning trees** for formation (Ch 2, Fig 15.1).
- **Passing network** — thresholded links (≥13), thickness ∝ volume (Fig 7.1).
- **"Tactical map"** — a *single image* for a coach.
- **Convex hulls** for defensive shape/territory (Fig 7.14–7.16).
- **Flow fields + heat maps** for movement (Ch 3).
- **Average-position formation markers** (Fig 9.2).
- **xG maps** — chance-quality squares/contours; Michael Caley's Arsenal maps (Ch 13).
- Ch 15 design rule: every viz ends with **"so what?"** — build around a decision a coach acts on.

---

## 9. CV-feasibility summary

- **Direct from tracking:** Voronoi/Delaunay, flow fields, heat maps, average-position formations, convex hulls, half-space occupancy, press windows (2.3/5.5 s), possession/territory. **[CV]**
- **Needs event detection/modeling:** passing networks (ownership), xG (shot detection + zonal model), Markov state values, win-probability curves **[EVENT/CV+EVENT]**.
- **Not in the book — get from Sumpter's course/blog:** pitch-control-as-a-race (readthedocs Lesson 6), "score curve" win-probability, 90-minute attack.
