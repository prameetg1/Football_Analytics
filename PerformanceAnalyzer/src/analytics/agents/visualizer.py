"""Visualization analyst agent: football-first matplotlib charts.

Builds the figures that back the HTML report pages. Every figure is saved into
the match's `MatchOutputDir.figures/` and later inlined into the pages as a
base64 data URI, so the HTML is fully self-contained.

Chart library (per football analysis conventions):
  - shot map on a pitch with goal mouths (marker size = xG, star = goal)
  - cumulative xG timeline with goal markers + net-xG line
  - per-side pass maps (complete/incomplete arrows, node = avg position)
  - per-side passing networks (node degree by volume, weighted arrows)
  - per-side pressing heatmaps
  - win-probability area chart with goals annotated
  - team comparison radar (xG, shots on target, VAEP, possession, press, pass)
  - per-side top-VAEP player bars
  - per-side average-position formation map (when frames are available)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

from src.analytics.agents.base import AgentContext, KNOWLEDGE_DIR
from src.analytics.agents.html import MatchOutputDir

PITCH_LENGTH = 120.0
PITCH_WIDTH = 70.0
OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "output"

# canonical colour set
HOME = "#38bdf8"
AWAY = "#f87171"
GOLD = "#ffb648"
PITCH_GREEN = "#2d5a27"


def _draw_pitch(ax: plt.Axes) -> None:
    """A broadcast-frame pitch: home attacks right (x=120)."""
    ax.set_xlim(-6, PITCH_LENGTH + 6)
    ax.set_ylim(-4, PITCH_WIDTH + 4)
    ax.set_aspect("equal")
    ax.set_facecolor(PITCH_GREEN)
    ax.plot([0, PITCH_LENGTH, PITCH_LENGTH, 0, 0],
            [0, 0, PITCH_WIDTH, PITCH_WIDTH, 0], color="white", lw=1.2)
    ax.axvline(PITCH_LENGTH / 2, color="white", lw=0.8, ls="--", alpha=0.8)
    circle = plt.Circle((PITCH_LENGTH / 2, PITCH_WIDTH / 2), 9.15,
                         fill=False, color="white", lw=0.8)
    ax.add_patch(circle)
    for x0, y0 in ((0, (PITCH_WIDTH - 40.3) / 2),
                   (PITCH_LENGTH - 16.5, (PITCH_WIDTH - 40.3) / 2)):
        box = mpatches.FancyBboxPatch((x0, y0), 16.5, 40.3,
                                      boxstyle="round,pad=0", fill=False,
                                      color="white", lw=0.8)
        ax.add_patch(box)
    _draw_goal(ax, PITCH_LENGTH, "right")
    _draw_goal(ax, 0, "left")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_goal(ax: plt.Axes, x: float, side: str) -> None:
    """Goal mouth at the given goal line (posts + crossbar)."""
    y_bot, y_top = PITCH_WIDTH / 2 - 3.66, PITCH_WIDTH / 2 + 3.66
    dirn = 1 if side == "left" else -1
    gx = x + dirn * 1.5
    ax.plot([gx, gx], [y_bot, y_top], color="white", lw=2)
    ax.plot([gx, x + dirn * 0.4], [y_top, y_top], color="white", lw=1.2)
    ax.plot([gx, x + dirn * 0.4], [y_bot, y_bot], color="white", lw=1.2)


def _shot_color(s) -> str:
    if s.result == "goal":
        return GOLD
    if s.result == "off_target" or s.result == "post":
        return "#9aa7b4"
    return HOME if s.team_side == 0 else AWAY


def _draw_shot_map(ctx: AgentContext) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 7))
    _draw_pitch(ax)
    if not ctx.shots:
        ax.set_title("Shot Map — no shots in data", fontsize=11,
                     color="white", fontweight="bold", pad=10)
        return fig
    for s in ctx.shots:
        size = max((s.xg or 0.05) * 450, 25)
        color = _shot_color(s)
        marker = "*" if s.result == "goal" else "o"
        ax.scatter(s.x, s.y, s=size, c=color, marker=marker,
                   alpha=0.85, zorder=5, edgecolors="black", linewidths=0.6)
        if (s.xg or 0) >= 0.5:
            ax.annotate(f"{s.xg:.2f}", (s.x, s.y - 3), color="white",
                        fontsize=7, ha="center", va="top", zorder=7)
    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=HOME,
               markersize=9, label="Home"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=AWAY,
               markersize=9, label="Away"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#9aa7b4",
               markersize=9, label="Off target"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor=GOLD,
               markersize=13, label="Goal"),
    ]
    ax.legend(handles=legend, loc="upper left", fontsize=8, framealpha=0.85)
    ax.set_title("Shot Map — size = xG, star = goal", fontsize=12,
                 color="white", fontweight="bold", pad=10)
    fig.tight_layout()
    return fig


def _draw_xg_timeline(ctx: AgentContext) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    if not ctx.shots:
        ax.set_title("xG Timeline — no shots in data", fontsize=11,
                     fontweight="bold")
        return fig
    for side, color, label in ((0, HOME, "Home"), (1, AWAY, "Away")):
        side_shots = sorted([s for s in ctx.shots if s.team_side == side],
                            key=lambda s: s.timestamp_sec)
        cum = np.cumsum([s.xg or 0.0 for s in side_shots])
        mins = [s.timestamp_sec / 60.0 for s in side_shots]
        ax.step(mins, cum, where="post", color=color, linewidth=2.2,
                label=label)
        for m, c, s in zip(mins, cum, side_shots):
            if s.result == "goal":
                ax.plot(m, c, "*", color=GOLD, markersize=13,
                        zorder=6, markeredgecolor="black")
            elif (s.xg or 0) >= 0.4:
                ax.plot(m, c, "o", color=color, markersize=5, alpha=0.6)
    # net-xG differential line (home - away), the classic xG-flow view
    all_shots = sorted(ctx.shots, key=lambda s: s.timestamp_sec)
    if len(all_shots) > 1:
        t = [s.timestamp_sec / 60.0 for s in all_shots]
        net = np.cumsum([(s.xg or 0.0) * (1 if s.team_side == 0 else -1)
                         for s in all_shots])
        ax.plot(t, net, color="white", linewidth=1.2, ls="--", alpha=0.9,
                label="Net xG (home − away)")
        ax.axhline(0, color="white", lw=0.6, alpha=0.4)
    ax.set_xlabel("Minute", color="white")
    ax.set_ylabel("Cumulative xG", color="white")
    ax.set_title("xG Flow — cumulative xG with goals (*) and net-xG", fontsize=12,
                 fontweight="bold", color="white")
    ax.tick_params(colors="white")
    ax.legend(framealpha=0.8, loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)
    fig.patch.set_facecolor("#0f1419")
    ax.set_facecolor("#0f1419")
    fig.tight_layout()
    return fig


def _pass_vectors(ctx: AgentContext, side: int):
    complete, incomplete = [], []
    for p in ctx.passes:
        if p.team_side != side or p.end_x is None or p.end_y is None:
            continue
        vec = (p.x, p.y, p.end_x, p.end_y)
        if p.outcome == "complete":
            complete.append(vec)
        else:
            incomplete.append(vec)
    return complete, incomplete


def _draw_pass_map(ctx: AgentContext, side: int, team_name: str) -> plt.Figure:
    from matplotlib.collections import LineCollection
    fig, ax = plt.subplots(figsize=(11, 7))
    _draw_pitch(ax)
    complete, incomplete = _pass_vectors(ctx, side)
    color = HOME if side == 0 else AWAY
    lines = [( (x1, y1), (x2, y2) ) for x1, y1, x2, y2 in complete]
    widths = [1.0 + 4.0 / max(1 + ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5, 1.0)
              for x1, y1, x2, y2 in complete]
    if lines:
        lc = LineCollection(lines, colors=color, linewidths=widths,
                            alpha=0.45, zorder=2)
        ax.add_collection(lc)
    if incomplete:
        inc = [((x1, y1), (x2, y2)) for x1, y1, x2, y2 in incomplete]
        ilc = LineCollection(inc, colors="#e04f4f", linewidths=1.2,
                             alpha=0.6, ls="--", zorder=2)
        ax.add_collection(ilc)
    # node = average position of each receiver/ passer
    for pid, px, py in _avg_positions(ctx, side):
        ax.scatter(px, py, s=240, c="#ffd98a", edgecolors="black",
                   linewidths=0.8, zorder=6)
        ax.annotate(str(pid), (px, py), color="#1a1a1a", fontsize=7,
                    ha="center", va="center", zorder=7, fontweight="bold")
    legend = [
        Line2D([0], [0], color=color, lw=2.5, label="Complete pass"),
        Line2D([0], [0], color="#e04f4f", lw=1.5, ls="--", label="Incomplete"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#ffd98a",
               markersize=8, label="Player (avg position)"),
    ]
    ax.legend(handles=legend, loc="upper left", fontsize=8, framealpha=0.85)
    ax.set_title(f"{team_name} — Pass Map", fontsize=12, color="white",
                 fontweight="bold", pad=10)
    fig.tight_layout()
    return fig


def _avg_positions(ctx: AgentContext, side: int) -> list[tuple[int, float, float]]:
    pos: dict[int, list[float]] = {}
    for p in ctx.passes:
        if p.team_side != side:
            continue
        for pid, x, y in ((p.player_id, p.x, p.y),
                          (p.receiver_id, p.end_x, p.end_y)):
            if pid is None or x is None or y is None:
                continue
            pos.setdefault(pid, []).extend([x, y])
    out = []
    for pid, xy in pos.items():
        if len(xy) >= 2:
            out.append((pid, float(np.mean(xy[0::2])),
                        float(np.mean(xy[1::2]))))
    return out


def _draw_passing_network(ctx: AgentContext, side: int, team_name: str) -> plt.Figure:
    import networkx as nx
    from src.analytics.models.core import PassingNetwork
    fig, ax = plt.subplots(figsize=(10, 6.5))
    _draw_pitch(ax)
    net = PassingNetwork()
    for p in ctx.passes:
        if (p.outcome == "complete" and p.team_side == side
                and p.player_id is not None and p.receiver_id is not None):
            net.add(p.player_id, p.receiver_id)
    if not net.edges:
        ax.set_title(f"{team_name} Passing Network — no data", fontsize=10,
                     color="white")
        return fig
    G = nx.DiGraph()
    for (a, b), w in net.edges.items():
        G.add_edge(a, b, weight=w)
    pos = {n: (np.mean([p.x for p in ctx.passes
                       if p.player_id == n or p.receiver_id == n
                       and p.x is not None]),
               np.mean([p.y for p in ctx.passes
                        if p.player_id == n or p.receiver_id == n
                        and p.y is not None]))
           for n in G.nodes}
    pos = {n: (px if np.isfinite(px) else PITCH_LENGTH / 2,
               py if np.isfinite(py) else PITCH_WIDTH / 2)
           for n, (px, py) in pos.items()}
    volumes = dict(net.degree_top(k=len(G.nodes)))
    node_sizes = [max(volumes.get(n, 0) * 14, 90) for n in G.nodes]
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(weights) if weights else 1
    widths = [1.5 + 4.0 * w / max_w for w in weights]
    color = HOME if side == 0 else AWAY
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                           node_color="#ffd98a", alpha=0.95,
                           edgecolors="black", linewidths=0.7)
    nx.draw_networkx_edges(G, pos, ax=ax, width=widths, edge_color=color,
                           alpha=0.65, arrows=True, arrowsize=10,
                           arrowstyle="->", connectionstyle="arc3,rad=0.12",
                           min_source_margin=8, min_target_margin=8)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7, font_color="#1a1a1a",
                            font_weight="bold")
    ax.set_title(f"{team_name} — Passing Network (edge weight = volume)",
                 fontsize=12, color="white", fontweight="bold", pad=10)
    fig.tight_layout()
    return fig


def _draw_press_map(ctx: AgentContext, side: int, team_name: str) -> plt.Figure:
    from matplotlib.colors import LinearSegmentedColormap
    fig, ax = plt.subplots(figsize=(11, 7))
    _draw_pitch(ax)
    events = [e for e in ctx.events if e.team_side != side and e.under_pressure]
    if not events:
        ax.set_title(f"{team_name} press targets — no under-pressure data",
                     fontsize=10, color="white")
        return fig
    xs = [e.x for e in events]
    ys = [e.y for e in events]
    cmap = LinearSegmentedColormap.from_list("press",
                                             ["#2ecc71", "#f39c12", "#e74c3c"])
    h = ax.hist2d(xs, ys, bins=[16, 12],
                  range=[[0, PITCH_LENGTH], [0, PITCH_WIDTH]],
                  cmap=cmap, alpha=0.75)
    cb = plt.colorbar(h[3], ax=ax, label="Press events", shrink=0.7)
    cb.ax.tick_params(colors="white")
    ax.set_title(f"{team_name} — Where They Press (defensive-action density)",
                 fontsize=12, color="white", fontweight="bold", pad=10)
    fig.tight_layout()
    return fig


def _draw_win_prob_curve(ctx: AgentContext, wp_curve) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    fig.patch.set_facecolor("#0f1419")
    ax.set_facecolor("#0f1419")
    if not wp_curve:
        ax.set_title("Win Probability — no data", fontsize=11, color="white")
        return fig
    minutes = [c.minute for c in wp_curve]
    ax.stackplot(minutes,
                 [c.p_win for c in wp_curve],
                 [c.p_draw for c in wp_curve],
                 [c.p_loss for c in wp_curve],
                 labels=["Home win", "Draw", "Away win"],
                 colors=[HOME, "#9aa7b4", AWAY], alpha=0.85)
    for s in ctx.shots:
        if s.result == "goal":
            ax.axvline(s.timestamp_sec / 60.0, color="white", lw=0.9,
                       ls=":", alpha=0.5)
    ax.set_xlabel("Minute", color="white")
    ax.set_ylabel("Probability", color="white")
    ax.set_title("Win Probability Curve (dotted = goal)",
                 fontsize=12, fontweight="bold", color="white")
    ax.tick_params(colors="white")
    ax.set_ylim(0, 1)
    ax.legend(loc="center right", fontsize=8, framealpha=0.8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def _draw_team_radar(ctx: AgentContext, stat) -> plt.Figure:
    labels = ["xG", "Shots on target", "VAEP value", "Possession",
              "Pass rate", "Funnel"]
    if stat is None:
        return _empty_figure("Team comparison radar")
    def _val(sa, metric):
        if metric == "xG":
            return sa.xg
        if metric == "Shots on target":
            return float(sa.shots_on_target)
        if metric == "VAEP value":
            return max(sa.vaep_value, 0.0)
        if metric == "Possession":
            return sa.possession_share * 100.0
        if metric == "Pass rate":
            return min(sa.passing_rate, 50.0)
        return sa.funnel_possessions_to_goal * 100.0
    home_vals = [_val(stat.home, m) for m in labels]
    away_vals = [_val(stat.away, m) for m in labels]
    for i in range(len(labels)):
        mx = max(home_vals[i], away_vals[i], 0.01)
        home_vals[i] = home_vals[i] / mx
        away_vals[i] = away_vals[i] / mx
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    home_vals += home_vals[:1]
    away_vals += away_vals[:1]
    fig = plt.figure(figsize=(6.5, 6.5))
    fig.patch.set_facecolor("#0f1419")
    ax = plt.subplot(111, polar=True)
    ax.set_facecolor("#0f1419")
    ax.plot(angles, home_vals, color=HOME, lw=2.5, label="Home")
    ax.fill(angles, home_vals, color=HOME, alpha=0.2)
    ax.plot(angles, away_vals, color=AWAY, lw=2.5, label="Away")
    ax.fill(angles, away_vals, color=AWAY, alpha=0.2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color="white", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.tick_params(colors="white")
    ax.set_title("Team Comparison Radar (per-90, normalised)",
                 pad=22, color="white", fontweight="bold", fontsize=12)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1),
              fontsize=9, framealpha=0.8)
    fig.tight_layout()
    return fig


def _draw_player_vaep(ctx: AgentContext, side: int, team_name: str,
                      vaep_summary) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0f1419")
    ax.set_facecolor("#0f1419")
    if vaep_summary is None:
        ax.set_title("Player VAEP — no data", fontsize=11, color="white")
        return fig
    vals = [(pid, v["value"]) for pid, v in vaep_summary.by_player.items()
            if _player_side(pid, ctx) == side]
    vals.sort(key=lambda t: t[1])
    top = vals[-10:]
    if not top:
        ax.set_title(f"{team_name} VAEP — no valued actions",
                     fontsize=11, color="white")
        return fig
    pids = [f"#{pid}" for pid, _ in top]
    values = [v for _, v in top]
    color = HOME if side == 0 else AWAY
    ax.barh(pids, values, color=color, alpha=0.85)
    ax.axvline(0, color="white", lw=0.6)
    ax.set_xlabel("VAEP value", color="white")
    ax.set_title(f"{team_name} — Top Contributors by VAEP",
                 fontsize=12, fontweight="bold", color="white")
    ax.tick_params(colors="white")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    return fig


def _draw_formation_map(ctx: AgentContext, side: int, team_name: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 7))
    _draw_pitch(ax)
    frames = [f for f in ctx.frames if hasattr(f, "players")]
    if not frames:
        ax.set_title(f"{team_name} — Formation (no tracking frames)",
                     fontsize=10, color="white")
        return fig
    player_pos: dict[int, list] = {}
    for f in frames:
        for pid, pside, x, y, _gk in getattr(f, "players", []):
            if pside == side and pid is not None:
                player_pos.setdefault(pid, []).append((x, y))
    color = HOME if side == 0 else AWAY
    for pid, pts in player_pos.items():
        if len(pts) < 5:
            continue
        mx = np.mean([p[0] for p in pts])
        my = np.mean([p[1] for p in pts])
        ax.scatter(mx, my, s=260, c=color, edgecolors="black",
                   linewidths=0.8, zorder=6, alpha=0.9)
        ax.annotate(str(pid), (mx, my), color="white", fontsize=7,
                    ha="center", va="center", zorder=7, fontweight="bold")
    ax.set_title(f"{team_name} — Average Player Position",
                 fontsize=12, color="white", fontweight="bold", pad=10)
    fig.tight_layout()
    return fig


def _empty_figure(title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.set_title(title, fontsize=11, color="white")
    ax.set_facecolor("#0f1419")
    fig.patch.set_facecolor("#0f1419")
    return fig


@dataclass
class VisualizationReport:
    """Collected figures from the visualizer agent."""
    figures: dict[str, plt.Figure] = field(default_factory=dict)
    saved_paths: list[str] = field(default_factory=list)
    output_dir: MatchOutputDir | None = None

    def save_all(self, directory: str | Path | None = None, dpi: int = 150) -> list[str]:
        """Persist every figure; closes the open figures as it goes.

        With an `output_dir` present, the figures/layout uses the match folder;
        otherwise the legacy `directory` (or project `output/`) is used.
        """
        if self.output_dir is not None:
            paths = []
            for name, fig in self.figures.items():
                p = self.output_dir.save_figure(fig, name, dpi=dpi)
                paths.append(str(p))
            self.output_dir.figures_dir.mkdir(parents=True, exist_ok=True)
            self.saved_paths = [str(self.output_dir.figures_dir / f"fig_{n}.png")
                                for n in self.figures]
            return paths
        directory = Path(directory or OUTPUT_DIR)
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        for name, fig in self.figures.items():
            p = directory / f"{name}.png"
            fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
            paths.append(str(p))
            plt.close(fig)
        self.saved_paths = paths
        return paths


class VisualizerAgent:
    """Agent that produces the football-first chart set for a match."""

    name = "visualizer"

    def process(self, ctx: AgentContext) -> VisualizationReport:
        vr = VisualizationReport()
        out = ctx.extras.get("output_dir")
        vr.output_dir = out

        vr.figures["shot_map"] = _draw_shot_map(ctx)
        vr.figures["xg_timeline"] = _draw_xg_timeline(ctx)
        vr.figures["win_prob_curve"] = _draw_win_prob_curve(
            ctx, ctx.extras.get("wp_curve"))
        if ctx.passes:
            vr.figures["pass_map_home"] = _draw_pass_map(
                ctx, 0, _team_name(ctx, 0))
            vr.figures["pass_map_away"] = _draw_pass_map(
                ctx, 1, _team_name(ctx, 1))
            vr.figures["passing_network_home"] = _draw_passing_network(
                ctx, 0, _team_name(ctx, 0))
            vr.figures["passing_network_away"] = _draw_passing_network(
                ctx, 1, _team_name(ctx, 1))
        vr.figures["press_map_home"] = _draw_press_map(ctx, 0, _team_name(ctx, 0))
        vr.figures["press_map_away"] = _draw_press_map(ctx, 1, _team_name(ctx, 1))
        vr.figures["team_radar"] = _draw_team_radar(
            ctx, ctx.extras.get("statistical_report"))
        vaep_summary = ctx.extras.get("vaep_summary")
        if vaep_summary is not None:
            vr.figures["player_vaep_home"] = _draw_player_vaep(
                ctx, 0, _team_name(ctx, 0), vaep_summary)
            vr.figures["player_vaep_away"] = _draw_player_vaep(
                ctx, 1, _team_name(ctx, 1), vaep_summary)
        if ctx.frames:
            vr.figures["formation_home"] = _draw_formation_map(
                ctx, 0, _team_name(ctx, 0))
            vr.figures["formation_away"] = _draw_formation_map(
                ctx, 1, _team_name(ctx, 1))

        if out is not None:
            vr.save_all()
        return vr


def _player_side(player_id: int, ctx: AgentContext) -> int:
    for e in ctx.events:
        if e.player_id == player_id and e.team_side is not None:
            return e.team_side
    return 0


def _team_name(ctx: AgentContext, side: int) -> str:
    if side < len(ctx.match.teams):
        return ctx.match.teams[side].name
    return f"Team {side}"