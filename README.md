# Football Analytics Monorepo

Analysing football with **computer vision**, **event/tracking data**, and a
**provider-agnostic analytics layer** — one umbrella for the projects that
live side by side on disk.

| Project | Description | Status |
|---------|-------------|--------|
| [`PerformanceAnalyzer/`](./PerformanceAnalyzer) | End-to-end match analytics: broadcast-video CV pipeline (detection, tracking, homography) + provider-agnostic analytics layer (xG, VAEP, pitch control, passing networks) with agentic HTML match reports. | Active, documented in its own `README.md` |
| `DefenseAnalytics/` | Event-based defensive analytics (sibling project). | Coming soon |
| `DataStore/` | Shared local data (licensed feeds, weights, caches). Not committed. | Local only |

## Conventions

- Each project is fully self-contained: source, tests, docs and its own
  `README.md`. Open a project folder to work on it.
- Licensed datasets and trained weights live in `DataStore/` and are never
  committed here.
- `PerformanceAnalyzer/` carries its own git history (this monorepo was built
  by re-rooting that project's commits into its directory).