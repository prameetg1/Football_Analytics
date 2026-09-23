# The Numbers Game — Analyst's Digest

**Chris Anderson & David Sally, *The Numbers Game: Why Everything You Know About Football Is Wrong* (2013)**

Compiled for the PerformanceAnalyzer CV pipeline. A pre-xG book whose empirical findings are the intellectual foundation of modern xG pipelines. Their "new formula" for football: **2.66, 50/50, 53.4, <58<73<79, and 0 > 1**.

> **Feasibility tags:** **[CV]** = derivable from player/ball tracking · **[EVENT]** = needs event/outcome data (StatsBomb or engineer detectors) · **[CV+EVENT]** = both.

---

## 1. What predicts wins (the xG precursor)

- **Headline:** outcomes are predicted far better by **goals scored/conceded (goal difference)** than possession or passing. The skill component lives in **creating and preventing good chances**, not keeping the ball.
- **Model decision:** use GF/GA (or GD) — **not** possession or pass count — as the dependent variable. **[CV]** — goals and shots are CV-detectable; a "goal" is a ball-in-net event you already tag.
- **Shots, especially shots on target, are the next-closest proxy** to goals (closer than corners or possession). League avg ≈ **14 shots, 4.7 on target**/team/game; ≈ 1 goal per 9–11 shots. **[CV]** — on-target label needs care (flag ball crossing plane within goal frame).
- They're arguing for **xG before xG existed**: chance *location/quality* matters more than raw shot count. **[CV+EVENT]**
- **Debunked myths (use as priors, re-check on your data):**
  - *Possession:* >50% side won only **39.4%** of games vs 31.6% for the under-50 side.
  - *Pass completion:* barely predictive; over-valued.
  - *"More shots = more wins":* the team that shoots more wins **less than half** the time (shot counts are score-state contaminated — you shoot more when behind). **To use shots predictively, adjust for score state.**
  - *"Vulnerable right after scoring":* false — teams are *least* likely to concede immediately after scoring. Don't build "momentum dip" features.
  - *Corners:* a corner → shot only **20.5%** of the time, goal 1-in-9 of those ≈ **0.022 goals** (~1 corner-goal per 10 games/team); GD value ≈ zero after counter-attack risk. Don't weight corner volume. **[CV]** corner/restart detection.
- **The "slippage funnel":** possessions → passes → shots → shots-on-target → goals. Only **6 in 100 possessions** end in a shot, **0.74 in 100** in a goal. **[CV]** — compute per team; big on-target surplus/deficit vs goals = luck-adjusted over/under-performance.

---

## 2. Central findings: randomness, Poisson, beta-binomial

- **Football is ~50/50 luck/skill.** "Half the goals you see, half the results, are down not to skill but to luck." Individual matches are near coin-flips; aggregate results are reliable.
- **Favorites win only ~50% of the time** — worst predictability of the major team sports.
- **Scarcity:** >30% of matches end with ≤1 goal; most wins are by one goal. Each goal has outsized, mostly-random impact.
- **Goals are Poisson** (~**2.66 goals/match**, top-5 European leagues 1993–2011): a single Poisson predicts a season's scoreline frequencies with no tactical input (PL: ~30 goalless, ~70 one-goal wins, ~95 two-goal, ~80 three-goal, ~55 four-goal, ~50 five-plus).
- **Beta-binomial framing:** each team has a stable latent "talent" rate; observed outcomes draw from it via a counting process. Short samples = binomial noise dominates → a season still leaves ~half the variance unexplained.
- **Analyst applications:**
  - **Expected-points tables from goals/xG via Poisson** (formula below); luck = actual − expected. **[CV]** goals/xG.
  - **Shrinkage/hierarchical models** for team & player rates — never trust a single season/game ("hot hand" largely myth).
  - Use **Poisson win-probability**, not linear models, for in-game decisioning.

---

## 3. Expected points from a Poisson goal model

For scoring rate λ_F and conceded rate λ_A:

```
P(X=k) = e^-λ · λ^k / k!          (Poisson PMF)
P(win)   = Σ_k P_F(k) · (1 − Σ_{j≤k} P_A(j))
P(draw)  = Σ_k P_F(k) · P_A(k)
E[pts]   = 3·P(win) + 1·P(draw)
```

- Correct engine for win-probability, expected-points standings, and **goal → points valuation.**
- **Goal values are non-linear:** the **2nd goal is most valuable (~+0.99 EP)**; by the **5th** victory is effectively guaranteed. → model goal-arrival dynamics, not static aggregates. **[CV]** — with live score you get live EPV curves.
- **The "55/45 lesson":** a 55/45 better team gains only **~14.75 points/season**; **±1% ≈ ~3 points** — the entire relegation safety margin. Edges look trivially small but are season-defining → **justifies a ~1% CV-model improvement.**

---

## 4. Rebalancing attack vs defense ("0 > 1")

- **Prevention ≈ scoring, and cheaper:** 2001–11, +10 goals ≈ **+2.30 wins**; −10 conceded ≈ **+2.16 wins**. On wins, attack ≈ defense.
- **Defense wins *avoiding defeat*:** 10 fewer conceded = **−2.35 defeats** vs −1.76 for +10 scored → **each goal NOT conceded ≈ 33% more valuable than each scored** for survival.
- **Clean sheet ≈ 2.5 points/match** vs **a goal scored ≈ 1 point/match** → "not conceding is more than twice as valuable."
- **Modeling decisions:**
  - Value actions **symmetrically on GD**, but weight *prevention* more for not-losing contexts (survival, away).
  - **Judge players equally:** a defender removing N high-value chances has the same GD impact as a striker creating N. **[CV]** defensive actions (blocks, clearances, interceptions, pressures) extractable from tracking; classify by danger zone.

---

## 5. Team vs player performance; "weakest-link" game

- **Teams:** skill shows over many games; a season is ~half luck. **Players:** a season of a striker's goals is mostly noise — shrink toward career/league baselines.
- **Football is a "weakest-link" game, not "strongest-link":** the *worst* player (errors: needless fouls, lost duels) decides more outcomes than the superstar. **[CV]** track error-prone events (dispossession under pressure, failed recoveries, mispresses); **[EVENT]** actual fouls/cards.
- **Touches ≠ influence:** avg touches/90 — defenders 63, midfielders 73, forwards 51; the average player has the ball only **53.4 seconds/match** (running 191 m with it). The meta-game is **who makes the fewest costly errors in those seconds**.
- **Concrete valuation:** value a player by **Δ(team chances created) + Δ(team chances conceded) attributable to them**, weighted by chance quality — an on/off or rate-difference in CV-extracted xG-areas, not goals/assists headlines. **[CV]**

---

## 6. Moneyball for football / market inefficiencies

Market **over-prices** what's visible and glamorous (brand, dribbles, "galactico" strikers); **under-prices:**
1. **Defenders / goal-prevention** (pay more for defenders, less for goalscorers),
2. **Direct, set-piece, transition football** (anti-tiki-taka; Reep: 2 of 9 goals come from moves with >3 passes — Route One under-rated),
3. **Proven, boring scorers** (the "Darren Bent" thesis),
4. **Weak-link fixing** — upgrading the 10th/11th starter or reducing error rates beats a superstar. **[EVENT]** — needs transfer prices joined to CV/event skill estimates.

- **Trade-off (game theory):** goals are cheap to concede and expensive to score → **defensive solidity and turnover-minimization are the cheapest wins**; attack/defense is a balance, not "attack always beats defense."
- **Modeling:** **down-weight** possession volume, pass completion, dribbles, corner volume; **up-weight** danger-zone turnovers prevented/forced, chance-creation location, set-piece/transition output.

---

## 7. Named findings: home advantage, red cards, subs, managers

- **Home advantage** ≈ **0.3–0.5 goals / ~53–57% of points** (~1.5 pts/ml vs 1.2 away). Largely referee behavior (fouls/cards/pens) — **not CV-detectable**; treat as a constant offset, not a tracked feature. **[EVENT/CV+EVENT]**
- **Red cards:** it is **easier for the 11-man side**; a home red card drops the shorthanded team ~0.86–0.97 GD (~−19pp win prob); early red ~ −1.0 GD, effect collapses after ~min 60–80. **[EVENT]** — not CV-detectable; must ingest, then use as a conditioning variable (shot data is card-state dependent).
- **Substitution rule "<58<73<79":** if losing, sub before the 58th/73rd/79th min; waiting (anchoring) forfeits recoverable points. **[CV]** — predict the window from physical decay (sprint drop, space leaking) rather than hard-code.
- **Managers ≈ 15–20% of results:** wages explain ~59% of table variation, luck ~50%; but **sacking yields no better results than keeping** (regression to the mean as a false "bounce"). Don't build manager-change features. **[EVENT]**
- **Cross-league universality:** goals, shots (~14), SoT (~4.7), corners (~5), pens (0.14/game) are statistically indistinguishable across EPL/La Liga/BuLi/Serie A — style narratives are cosmetic. Don't add league-style effects.

---

## 8. Visualization concepts → pipeline

- **Poisson overlay histograms** (goals/game vs fitted curve).
- **"Which league?" scoreline test** — normalized benchmarking across competitions.
- **Slippage-funnel / conversion waterfall** (possession→shots→SoT→goals with stage loss rates; ancestor of xG-vs-actual and xG-flow). Excellent coaching + luck-auditing viz.
- **Shot maps** — chance location matters more than count (today's xG shot-distribution maps).
- **Pass/regain maps** — regains high up the pitch are the best chance sources (~30% of regained possessions in the opp. box → shots; ~half of all goals trace to regained possessions) → **"dangerous regains / turnovers in final 40 m" heatmap**.
- **Win-probability / EPV-vs-scoreline curves** from the Poisson model (drives "2nd goal most valuable").
- **"Luck vs skill" decomposition** — season actual vs Poisson-expected points (bubble = residue).

---

## 9. Bottom line for the pipeline

1. **Target variable:** goals scored/conceded → Poisson/xG-based expected-points model; never possession or pass completion as targets/headlines.
2. **Best CV features:** shots + on-target + shot location/type (→ xG), dangerous regains/turnover map, pressure/space metrics, defensive danger-zone actions, score/card-state conditioning, attacker identity via re-ID.
3. **Event data to enrich:** scoreline provenance, points, bookings/subs, penalties & corner classifications, fouls, market value/wages, identities (StatsBomb or engineered detectors).
4. **Analytic discipline:** assume ~50% noise on single-match judgment; shrink every estimate; report expected/luck-adjusted numbers (actual − expected).

> **Caveat:** several headline stats come from 1–3 season windows (corners: 134 PL games 2010–11; possession: 1,140 matches). Treat as priors to re-validate — which is exactly the job of the CV pipeline.

---

## Cross-reference with PerformanceAnalyzer

| Book idea | Pipeline component |
|---|---|
| Markov value-of-possession (Rudd) | `src/spatial/pitch_control.py` C2 EPV |
| Passing networks / tactical map | downstream on `tracking_full.csv` |
| Voronoi occupancy | `render.py --radar voronoi`, C1 control surface |
| Pressing 2.3/5.5 s windows | needs tracking timestamps + ball-loss events |
| xG zones 3.4/12.4/32.2% | `src/statsbomb/shot_model.py` zonal model |
| Poisson expected-points | downstream model on events |
| "0 > 1" prevention value | downstream weighting |
