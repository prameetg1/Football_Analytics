"""HTML page renderers: one styled, self-contained page per agent.

Each renderer takes the agent's typed report (plus the shared `MatchOutputDir`)
and returns the full HTML document (figures inlined as base64 data URIs). The
orchestrator writes each page into the same timestamped match folder and builds
an index linking them all.
"""

from __future__ import annotations

from src.analytics.agents.html import (
    MatchOutputDir,
    esc,
    figure_img,
    html_page,
    index_link,
    match_header,
    stat_card,
)


def _two_team_cards(title: str, rows: list[tuple[str, str, str]]) -> str:
    """A labelled header + a two-column stat grid (home vs away)."""
    cards = []
    cards.append(f"<h2>{esc(title)}</h2>")
    cards.append('<div class="grid">')
    cards.append(stat_card("Home", "Home", value_span="home"))
    cards.append(stat_card("Away", "Away", value_span="away"))
    cards.append("</div>")
    for label, home_v, away_v in rows:
        cards.append(
            f'<div class="grid"><div class="stat"><div class="k">{esc(label)} '
            f'<span class="home">●</span></div><div class="v home">{home_v}</div></div>'
            f'<div class="stat"><div class="k">{esc(label)} '
            f'<span class="away">●</span></div><div class="v away">{away_v}</div></div>'
            f'</div>')
    return "\n".join(cards)


# --------------------------------------------------------------------------- #
# Tactician
# --------------------------------------------------------------------------- #

def render_tactician(tac, out: MatchOutputDir, agent: str) -> str:
    home, away = tac.home, tac.away
    body = [match_header(f"{home.team_name} vs {away.team_name} — Tactical",
                         out, agent)]

    body.append(_two_team_cards("Team shape", [
        ("Decentralization", f"{home.decentralization:.3f}",
         f"{away.decentralization:.3f}"),
        ("Formation width", f"{home.formation_width:.1f}m",
         f"{away.formation_width:.1f}m"),
        ("Formation length", f"{home.formation_length:.1f}m",
         f"{away.formation_length:.1f}m"),
        ("Max flow magnitude",
         f"{home.flow_magnitude_max:.2f}", f"{away.flow_magnitude_max:.2f}"),
    ]))

    body.append("<h2>Pressing</h2>")
    body.append('<div class="grid">')
    body.append(stat_card("Home press events", str(home.press_events_total)))
    body.append(stat_card("Away press events", str(away.press_events_total)))
    body.append(stat_card("Home final-third press",
                          f"{home.press_final_third_share:.0%}"))
    body.append(stat_card("Away final-third press",
                          f"{away.press_final_third_share:.0%}"))
    body.append("</div>")

    if home.knowledge_refs or away.knowledge_refs:
        body.append("<h2>Method references</h2><div>")
        for r in sorted(set(home.knowledge_refs) | set(away.knowledge_refs)):
            body.append(f'<span class="tag">{esc(r)}</span>')
        body.append("</div>")

    if tac.insights:
        body.append("<h2>Insights</h2><ul>")
        for i in tac.insights:
            body.append(f"<li>{esc(i)}</li>")
        body.append("</ul>")

    return html_page(f"Tactical — {home.team_name} vs {away.team_name}",
                     "".join(body), agent=agent)


# --------------------------------------------------------------------------- #
# Statistical scientist
# --------------------------------------------------------------------------- #

def render_stat_scientist(stat, out: MatchOutputDir, agent: str) -> str:
    home, away = stat.home, stat.away
    body = [match_header(f"{home.team_name} vs {away.team_name} — Statistics",
                         out, agent)]

    body.append(f"<h1>{home.team_name} {home.goals}–{away.goals} {away.team_name}</h1>")
    body.append(_two_team_cards("Chance creation", [
        ("xG", f"{home.xg:.2f}", f"{away.xg:.2f}"),
        ("xG over-performance",
         f"{home.xg_overperformance:+.2f}", f"{away.xg_overperformance:+.2f}"),
        ("Shots", str(home.shots), str(away.shots)),
        ("Shots on target", str(home.shots_on_target),
         str(away.shots_on_target)),
    ]))

    body.append("<h2>Value & control</h2>")
    body.append('<div class="grid">')
    body.append(stat_card("Home VAEP value", f"{home.vaep_value:.2f}"))
    body.append(stat_card("Away VAEP value", f"{away.vaep_value:.2f}"))
    body.append(stat_card("Home actions valued", str(home.vaep_actions)))
    body.append(stat_card("Away actions valued", str(away.vaep_actions)))
    body.append(stat_card("Home possession", f"{home.possession_share:.1%}"))
    body.append(stat_card("Away possession", f"{away.possession_share:.1%}"))
    body.append(stat_card("Home passing rate",
                          f"{home.passing_rate:.1f}/min"))
    body.append(stat_card("Away passing rate",
                          f"{away.passing_rate:.1f}/min"))
    body.append("</div>")

    if stat.expected_points is not None:
        ep = stat.expected_points.expected_points
        body.append("<h2>Model expected points (from shot-based rates)</h2>")
        body.append('<div class="grid">')
        body.append(stat_card("Home xP", f"{ep.get(0, 0.0):.2f}"))
        body.append(stat_card("Away xP", f"{ep.get(1, 0.0):.2f}"))
        body.append("</div>")

    if stat.insights:
        body.append("<h2>Insights</h2><ul>")
        for i in stat.insights:
            body.append(f"<li>{esc(i)}</li>")
        body.append("</ul>")

    return html_page(f"Statistics — {home.team_name} vs {away.team_name}",
                     "".join(body), agent=agent)


# --------------------------------------------------------------------------- #
# Visualizer
# --------------------------------------------------------------------------- #

def render_visualizer(vis, out: MatchOutputDir, agent: str) -> str:
    home = "Home"
    away = "Away"
    if vis.output_dir is not None:
        out = vis.output_dir
    body = [match_header(f"Match visualizations — match {out.match_id}",
                         out, agent)]

    ordered = ["shot_map", "xg_timeline", "win_prob_curve",
               "team_radar", "player_vaep_home", "player_vaep_away",
               "pass_map_home", "pass_map_away",
               "passing_network_home", "passing_network_away",
               "press_map_home", "press_map_away",
               "formation_home", "formation_away"]
    captions = {
        "shot_map": "All shots: marker size scales with xG; stars are goals; "
                    "only 0.50+ xG are labelled.",
        "xg_timeline": "Cumulative xG per side with goals marked; the white "
                       "dashed line is the net-xG (home minus away) flow.",
        "win_prob_curve": "Win/draw/loss probabilities over the match from "
                          "two Poisson goal processes; dotted lines mark goals.",
        "team_radar": "Both teams normalised on key per-90 metrics.",
        "player_vaep_home": "Top home contributors by on-ball value (VAEP).",
        "player_vaep_away": "Top away contributors by on-ball value (VAEP).",
        "pass_map_home": "Complete (solid) and incomplete (dashed) passes with "
                         "average positions as nodes.",
        "pass_map_away": "",
        "passing_network_home": "Directed network; arrow width scales with "
                                "pass volume.",
        "passing_network_away": "",
        "press_map_home": "Where the home team pressed (defensive-action "
                          "density).",
        "press_map_away": "",
        "formation_home": "Average player position from tracking frames.",
        "formation_away": "",
    }
    for name in ordered:
        if name in vis.figures or (vis.output_dir is not None and
                                   (out.figures_dir / f"fig_{name}.png").exists()):
            body.append(figure_img(name, out, captions.get(name, "")))

    if not vis.saved_paths and not (out.figures_dir / "fig_shot_map.png").exists():
        body.append("<p class='meta'>No figures generated (no data).</p>")

    return html_page(f"Visuals — match {out.match_id}", "".join(body),
                     agent=agent)


# --------------------------------------------------------------------------- #
# Domain expert
# --------------------------------------------------------------------------- #

def render_domain_expert(nr, out: MatchOutputDir, agent: str,
                         home_name: str = "Home",
                         away_name: str = "Away") -> str:
    body = [match_header(nr.headline, out, agent)]

    body.append(f"<p><b>TL;DR:</b> {esc(nr.tldr)}</p>")

    if nr.tactical_summary:
        body.append("<h2>Tactical</h2>")
        body.append(f"<div class='card'>{esc(nr.tactical_summary)}</div>")
    if nr.statistical_summary:
        body.append("<h2>Statistical</h2>")
        body.append(f"<div class='card'>{esc(nr.statistical_summary)}</div>")

    if nr.key_moments:
        body.append("<h2>Key moments</h2><table>")
        body.append("<tr><th>Minute</th><th>Category</th><th>Moment</th></tr>")
        for m in nr.key_moments:
            body.append(
                f"<tr><td>{esc(m.minute)}&rsquo;</td>"
                f"<td><span class='tag'>{esc(m.category)}</span></td>"
                f"<td>{esc(m.description)}</td></tr>")
        body.append("</table>")

    if nr.player_highlights:
        body.append("<h2>Player highlights</h2><ul>")
        for ph in nr.player_highlights:
            side = "Home" if ph.team_side == 0 else "Away"
            body.append(f"<li>{esc(side)} #{ph.player_id} — {esc(ph.summary)}"
                        f" <span class='tag'>{esc(ph.key_stat)}</span></li>")
        body.append("</ul>")

    if nr.coaching_insights:
        body.append("<h2>Coaching insights</h2><ul>")
        for ci in nr.coaching_insights:
            body.append(f"<li>{esc(ci)}</li>")
        body.append("</ul>")

    return html_page(f"{nr.headline} — Full analysis", "".join(body),
                     agent=agent)


# --------------------------------------------------------------------------- #
# Index page
# --------------------------------------------------------------------------- #

def render_index(out: MatchOutputDir, *, tactical, statistical, visual,
                 domain) -> str:
    """The match front page linking all four agent reports."""
    body = [
        f"<h1>Match {out.match_id} — Analysis</h1>",
        f"<div class='meta'>Generated {out.created_at.strftime('%Y-%m-%d %H:%M:%S')}"
        " &middot; timestamped analysis folder "
        f"<code>{esc(out.folder.name)}</code></div>",
    ]

    if domain and domain.headline:
        body.append(f"<div class='card'><b>{esc(domain.headline)}</b><br>"
                    f"{esc(domain.tldr)}</div>")

    if statistical:
        body.append("<h2>Snapshot</h2><div class='grid'>")
        h, a = statistical.home, statistical.away
        body.append(stat_card("Home xG", f"{h.xg:.2f}", value_span="home"))
        body.append(stat_card("Away xG", f"{a.xg:.2f}", value_span="away"))
        body.append(stat_card("Home VAEP", f"{h.vaep_value:.2f}",
                              value_span="home"))
        body.append(stat_card("Away VAEP", f"{a.vaep_value:.2f}",
                              value_span="away"))
        body.append("</div>")

    if visual and (visual.figures or (out.figures_dir / "fig_shot_map.png").exists()):
        body.append("<h2>Key visuals</h2>")
        for name in ("shot_map", "xg_timeline", "win_prob_curve"):
            if (out.figures_dir / f"fig_{name}.png").exists():
                body.append(figure_img(name, out))

    body.append("<h2>Agent reports</h2>")
    body.append('<div class="grid">')
    body.append(index_link("Tactician",
                           "Formation, pressing, network & shape",
                           "tactician.html", "tactical"))
    body.append(index_link("Statistical scientist",
                           "xG, VAEP, win-prob, ratings & possession",
                           "stat_scientist.html", "statistical"))
    body.append(index_link("Visualizer",
                           "Shot maps, pass maps, networks & radars",
                           "visualizer.html", "charts"))
    body.append(index_link("Domain expert",
                           "Synthesised narrative & coaching insights",
                           "domain_expert.html", "narrative"))
    body.append("</div>")

    return html_page(f"Match {out.match_id} — Analysis", "".join(body),
                     extra_styles="")