# Football Analytics Knowledge Base

## 1. Expected Goals (xG)

**What:** Probability a shot results in a goal, estimated from pre-shot state (location, body part, game state, pressure).

**Core model (Dixon-Coles 1997):**
```
logit(P(goal)) = α + β₁·dist_to_goal + β₂·angle_to_goal + ρ·τ(h,a)
```
τ(h,a) is the low-score correction (0-0, 1-0, 0-1, 1-1 cells adjusted for correlation between scoring/conceding).

**Modern extensions:**
- Shot location + body part + header/foot + through ball + counter-attack + big chance
- Freeze-frame: number/positions of defenders and goalkeeper at shot time (StatsBomb 360, Wyscout event+360)
- Game state: score differential at shot time
- Pre-shot sequence: is this the first shot of the possession? Has there been a pass into the box?

**When to use:** Player/team finishing quality (goals − xG = over/under-performance), match report shot charts, player recruitment.

**Caution:** xG ≠ shot quality alone — context matters. A 0.7 xG open-play header differs from a 0.7 xG penalty (identical xG, different context). Always label the context.

---

## 2. VAEP (Valuing Actions by Estimating Probabilities)

**What:** Value every on-ball action by how much it changes the probability of scoring/conceding next.

```
value(action) = ΔP(scoring) − ΔP(conceding)
              = [P(goal|after) − P(goal|before)] − [P(conceded|after) − P(conceded|before)]
```

**Inputs:** start/end locations of each action, xG on shots, ball-lost flag for incomplete actions, opponent hazard surface.

**Reference:** Tom Decroos et al. (2019) "Actions Speak Louder than Goals: Valuing Player Actions in Soccer" — KDD 2019.

**Variant — xT (Expected Threat):** Karun Singh's grid-based expected threat model. States: ball location; transitions: pass/carry to adjacent grid cell; P(goal|state) is the long-run scoring probability from that cell. Value of an action = xT(destination) − xT(origin). Simpler than VAEP but less granular (ignores game state, pressure, body part).

**When to use:** Player ratings (compare midfielders by value-creating actions, not just goals/assists), action selection analysis, build-up quality assessment.

**Caution:** Needs a large event corpus to fit the hazard surface. Small samples → noisy cell estimates. Dampening/smoothing helps.

---

## 3. Pitch Control / Expected Possession Value (EPV)

**What:** At any moment, each location on the pitch has a "possession value" — who is most likely to get there first, and what is the long-run probability of scoring from that possession?

**Pitch control surface (Spearman 2018):**
- For each cell, compute which team's nearest player arrives first (under constant acceleration model, max a=7 m/s²).
- Arrival time: sprinting toward point (-v + √(v² + 2ad))/a; moving away |v|/a + √(2d/a + (v/a)²)
- Smooth arrival-time difference through a logistic (σ=0.3s) → control probability per cell.

**EPV (Fernández, Bornn 2019):**
```
EPV = control(cell) × P(goal | possession at cell, game state)
```
The EPV grid uses fitted goal-probability surface (Dixon-Coles or logistic xG).

**Field tilt:** fraction of final-third time (or events) in each team's favour. Proxy for territorial dominance.

**When to use:** Tactical assessment (where is a team dominant?), player off-ball movement (does a run create space in high-EPV zones?), match momentum.

---

## 4. Pressing Analysis (PPDA & Press Windows)

**What:** Quantify how aggressively and where a team presses out of possession.

**PPDA (Passes Per Defensive Action):**
```
PPDA = opponent's completed passes / team's defensive actions in final 60%
```
Lower PPDA = more aggressive pressing. Opponent's PPDA = mirror. High PPDA opponent → team sits deep.

**Press windows (Soccermatics Ch 9):**
- On-ball press: defensive event within 2.3s of opponent's possession action.
- High press win: sustained pressing lasting ≥5.5s.
- Press map: defensive action count by pitch zone (third × lane).

**When to use:** Team style classification (high press vs low block), match tactical changes (PPDA shift at halftime), pressing efficiency.

---

## 5. Passing Network Analysis

**What:** Graph-theoretic view of a team's passing patterns.

**Adjacency matrix:** weight(i,j) = number of completed passes from player i to player j.

**Metrics:**
- Degree centrality: total pass volume per player (in+out). High-degree = involved in build-up.
- Betweenness centrality: fraction of shortest paths (fewest intermediary passers) through a player. High betweenness = the hub. Low betweenness + high degree = distributed passing (Soccermatics §3: decentralized ≈8% higher scoring rate).
- Clustering coefficient: how interconnected are a player's passing partners.
- Pass-flow: weighted directed edges, visualised as a network graph with edge thickness = pass count.

**When to use:** Build-up style (centralized hub vs distributed), key player identification, tactical changes at halftime.

---

## 6. Formation & Shape Analysis

**What:** Measure team compactness, width, length, and moment of inertia from positional data.

**Metrics (Football++ / K-master / Soccermatics):**
- Spread: mean distance of players from team centroid.
- Length: bounding extent along attacking axis (longitudinal spread).
- Width: bounding extent perpendicular to attacking axis.
- Compactness: mean pairwise distance among outfield players (out-of-possession).
- Spatial entropy: Shannon entropy of binned player positions — high = unpredictable, low = organized.
- Moment of inertia: Σ mᵢ·rᵢ² about centroid — measures formation rotation/dispersion.
- Formation angle: principal-axis orientation (from PCA) — 0°=flat 4-4-2, 90°=diamond.

**Shape dynamics:**
- Shape vs ball zone: compactness when ball is in own third vs final third.
- Shape during transition: stretch spike after turnovers (moment of inertia spike = team expanding into attack).
- Formation recovery time: frames to return to baseline compactness after ball loss.

**When to use:** Tactical profiling (compact defensive block vs stretched attack), coach debriefs, transition analysis.

---

## 7. Win Probability & Goal Value (Score Curve)

**What:** Real-time win probability from scoreline + time remaining + team scoring rates.

**Model (Soccermatics Ch 3):**
```
P(win at minute m | score s, λ_for, λ_against)
  = Σ_h Σ_a P(final = s+H, s+A) · 1(H > A)
```
Where H, A ~ Poisson(λ_for·(90−m)/90), Poisson(λ_against·(90−m)/90).

**Goal value (Numbers Game Ch 3):** marginal E[points] of the next goal at each minute. The equaliser at 90' is worth ~2× what it's worth at 0' (more game states become decisive). The 2nd goal is the most valuable overall.

**When to use:** Live match commentary (score-curve overlays), goal significance rating, substitution impact analysis.

---

## 8. Dixon-Coles Team Ratings

**What:** Infer team attack/defence strengths from observed results using a Poisson regression.

**Model:**
```
P(home=h, away=a) = τ(h,a) · Poisson(λ_h, h) · Poisson(λ_a, a)
λ_h = μ · att_home · def_away · home_adv
λ_a = μ · att_away · def_home
```
Where μ = league-average goals per team, τ = low-score adjustment.

**Estimation:** Maximum likelihood via BFGS. Attack strengths anchored at mean 1; home advantage > 1.

**When to use:** League prediction, fixture difficulty rating, team strength comparison across seasons, expected points from goals.

---

## 9. Progressive Passing & Threat Creation

**What:** Classify passes by their contribution to advancing the ball toward goal.

**Progressive pass (Soccermatics Ch 7):** a completed pass that advances the ball ≥10m closer to the opponent's goal.

**Through ball:** a pass that puts a teammate in behind the defensive line (destination beyond the second-deepest outfield player).

**Progressive carry:** a carry that moves the ball ≥5m toward the opponent's goal.

**When to use:** Player comparison (playmakers vs ball-retainers), team build-up style, assist quality vs shot quality.

---

## 10. Shot Quality vs Finishing Skill

**What:** Distinguish between getting into good positions (xG) and putting the ball in the net (finishing).

**xG over-performance (goals − xG):**
- Over a large sample, elite strikers sustain +0.05 to +0.15 xG over-performance.
- One season of over-performance is mostly noise; requires 300+ shots to be confident.
- Rates shrink to career/league baselines (regression to the mean).

**Finishing skill decomposition (FBref method):**
- Post-shot xG (PSxG): xG calculated from the shot's final trajectory (on target + placement), not just location.
- Goals minus PSxG: pure finishing quality (placement + power), independent of location.

**When to use:** Player recruitment (is this striker a good finisher or just getting good chances?), contract negotiation.

---

## 11. Network + Possession Chain Analysis

**What:** Sequence of passes/carry events forming a possession, valued by how far they advance and whether they lead to a shot.

**Chain metrics:**
- Chain length: number of passes + carries before turnover.
- Chain duration: time from first to last event.
- Chain xT gain: xT(end) − xT(start) of the chain.
- Chain outcome: shot created / goal / turnover / ball out.

**Slippage funnel (Numbers Game Ch 1):**
```
Possessions → Shots → On target → Goals
```
Conversion rates at each stage characterize team efficiency.

**When to use:** Build-up quality (which possessions lead to shots?), turnover cost, team-level efficiency profiling.

---

## 12. Player Comparison Framework

**What:** Structured approach to comparing players in the same role.

**Method:**
1. Filter by position + minutes played threshold (e.g., ≥900 min).
2. Compute per-90-minute rates: xG, xA (expected assists from shot assists), progressive passes, progressive carries, pressures in final third, aerials won%.
3. Percentile rank within the league/competition for each metric.
4. Spider/radar chart for visual comparison.

**Caution:** Percentile ranks are league-dependent. A 90th-percentile xG in a low-scoring league may be 50th in the Premier League. Always label the population.

---

## 13. Key Data Sources & Licences

| Source | Data | Licence | Notes |
|--------|------|---------|-------|
| StatsBomb | Events + 360 + lineups | Free (open-data GitHub) | 306 matches available; WC/Euro/Bundesliga/LaLiga |
| Metrica | Dense tracking (25Hz) + events | MIT (sample data) | 2 sample games; fixed broadcast camera |
| SkillCorner | Dense tracking (10fps) | MIT (open-data GitHub) | A-League 24/25; LFS-resolved |
| IDSSE/DFL | Dense tracking (25Hz) + events | Academic use (pysport mirror) | 7 Bundesliga matches |
| OpenFootball | Match results + goal scorers | CC0 | EPL 2024-26; season-scoped |
| Wyscout | Events + lineups | Pappalardo et al. 2019 (CC-BY 4) | Mirror via koenvo |
| FBref / Understat | Shots, player stats, schedules | Personal use only | Scraped via soccerdata; blocked periodically |
| WhoScored | Events, schedules | Blocked (captcha) | Not currently accessible |

---

## 14. Canonical Pitch Convention

All models operate on a shared pitch frame:
- **x ∈ [0, 120]** (length), **y ∈ [0, 70]** (width)
- Goal mouth at y = 35
- Attacking direction: both teams normalized so their attacking goal is at x = 0 (adapter-level mirroring).
- The shot model / EPV grid uses x = 120 as the attacking goal (raw StatsBomb convention); hazard_surface() in vaep.py defaults to x = 0 matching the adapter events. Pass `attacking_goal_x=120.0` for attacking-frame coordinates.

---

## 15. Measurement & Validation Principles

**Gate-first:** every metric ships with a validation gate against real data before being treated as product output.

**Synthetic GT:** when real footage is unavailable, use real 360 positions + injected noise as a stand-in. This validates the model plumbing but not real-world accuracy.

**Regression to the mean:** player-level metrics (goals, xG over-performance) shrink toward career baselines over large samples. Never over-interpret one season of extreme performance.

**Independence of observations:** events within a match are not independent (a goal changes subsequent behavior). Treat them as correlated; avoid naive p-values.
