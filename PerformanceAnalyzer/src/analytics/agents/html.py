"""Match output directory + shared HTML rendering helpers.

Each analysed match gets its own timestamped folder under an analyses root:

    <analyses_root>/<YYYY-mm-dd_HH-MM-SS>_<match_id>/
        fig_shot_map.png
        fig_xg_timeline.png
        ...
        tactician.html
        stat_scientist.html
        visualizer.html
        domain_expert.html
        index.html

Figures are saved next to the HTML and inlined into the pages as base64 data
URIs so every page is fully self-contained (renders from anywhere, no
relative-path fragility). All agents for the same match write into the *same*
`MatchOutputDir` instance.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

# root that holds one timestamped folder per analysed match
ANALYSES_ROOT = (Path(__file__).parent.parent.parent.parent / "output" / "analyses")


@dataclass
class MatchOutputDir:
    """The timestamped analysis folder shared by all agents for one match.

    Dependencies (figures) are registered with `add_figure` so the HTML writers
    can embed them. The folder is created lazily on first write.
    """

    match_id: int
    analyses_root: Path | None = None
    created_at: datetime = field(default_factory=datetime.now)
    extra_subdirs: tuple[str, ...] = ("figures",)

    def __post_init__(self) -> None:
        self.analyses_root = Path(self.analyses_root) if self.analyses_root \
            else ANALYSES_ROOT
        stamp = self.created_at.strftime("%Y-%m-%d_%H-%M-%S")
        self.folder = self.analyses_root / f"{stamp}_{self.match_id}"
        self.figures_dir = self.folder / "figures"

    def ensure(self) -> Path:
        """Create the folder structure and return `self.folder`."""
        self.folder.mkdir(parents=True, exist_ok=True)
        for sub in self.extra_subdirs:
            (self.folder / sub).mkdir(parents=True, exist_ok=True)
        return self.folder

    def write_text(self, rel_path: str | Path, content: str) -> Path:
        """Write text content into the match folder, returning its path."""
        self.ensure()
        p = self.folder / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return p

    def save_figure(self, fig, name: str, dpi: int = 150) -> Path:
        """Save a matplotlib figure into this folder's figures/ subdir.

        Returns the path to the saved PNG.
        """
        self.ensure()
        p = self.figures_dir / f"fig_{name}.png"
        fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        import matplotlib.pyplot as plt
        plt.close(fig)
        return p

    def figure_data_uri(self, name: str) -> str:
        """Return a base64 data URI for a previously-saved figure."""
        p = self.figures_dir / f"fig_{name}.png"
        if not p.exists():
            return ""
        b64 = base64.b64encode(p.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{b64}"


# --------------------------------------------------------------------------- #
# Shared HTML scaffolding
# --------------------------------------------------------------------------- #

_AGENT_STYLES = """
:root {
  --bg: #0f1419; --panel: #1a212b; --panel2: #222c38;
  --text: #e8edf2; --muted: #9aa7b4; --accent: #ffb648;
  --home: #38bdf8; --away: #f87171; --good: #34d399; --bad: #f87171;
  --line: #2a3541;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--text);
  font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  line-height: 1.55;
}
.wrap { max-width: 1100px; margin: 0 auto; padding: 24px 20px 64px; }
h1 { font-size: 1.7rem; margin: 0 0 4px; font-weight: 700; }
h2 { font-size: 1.25rem; margin: 28px 0 10px; color: var(--accent);
      border-bottom: 1px solid var(--line); padding-bottom: 6px; }
h3 { font-size: 1.05rem; margin: 18px 0 6px; }
.meta { color: var(--muted); font-size: 0.9rem; margin-bottom: 18px; }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
        padding: 16px 18px; margin: 12px 0; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px,1fr));
        gap: 14px; }
.stat { background: var(--panel2); border: 1px solid var(--line); border-radius: 8px;
        padding: 10px 12px; }
.stat .k { color: var(--muted); font-size: 0.75rem; text-transform: uppercase;
           letter-spacing: .04em; }
.stat .v { font-size: 1.3rem; font-weight: 700; }
.home { color: var(--home); } .away { color: var(--away); }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 0.92rem; }
th,td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--line); }
th { color: var(--muted); font-weight: 600; text-transform: uppercase;
     font-size: 0.75rem; letter-spacing: .04em; }
img.chart { width: 100%; height: auto; border-radius: 8px; border: 1px solid var(--line);
            margin: 6px 0; display: block; }
.tag { display:inline-block; background:#25303c; color: var(--accent);
       border-radius: 12px; padding: 2px 10px; font-size: .78rem; margin-right:6px; }
ul { margin: 6px 0 6px 20px; }
code { background:#11161d; padding:2px 6px; border-radius:4px; font-size:.85em; }
.bar { height: 8px; border-radius: 4px; background:var(--panel2); overflow:hidden; margin:4px 0; }
.bar > span { display:block; height:100%; }
"""

_LINK_STYLES = """
.grid a.card { display:block; text-decoration:none; color:var(--text); }
.grid a.card:hover { border-color: var(--accent); }
.grid a.card .v { color: var(--accent); }
"""


def html_page(title: str, body: str, *, agent: str = "",
              extra_styles: str = "") -> str:
    """Wrap an HTML body fragment in a full, styled document."""
    nav: list[str] = []
    if agent:
        nav.append(
            f'<a href="index.html" style="color:var(--muted);text-decoration:none;'
            f'margin-right:12px;font-size:.85rem">&#8592; All reports</a>')
    return (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
        "<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"<title>{title}</title>\n"
        f"<style>{_AGENT_STYLES}{_LINK_STYLES}{extra_styles}</style>\n"
        "</head>\n<body>\n<div class=\"wrap\">\n"
        + "".join(nav)
        + body
        + "</div>\n</body>\n</html>\n"
    )


def esc(text: object) -> str:
    """HTML-escape a value (render as text, not markup)."""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(
        ">", "&gt;").replace('"', "&quot;")


def stat_card(key: str, value: str,
              value_span: str = "") -> str:
    cls = f' class="{value_span}"' if value_span else ""
    return f'<div class="stat"><div class="k">{esc(key)}</div><div class="v"{cls}>{value}</div></div>'


def figure_img(name: str, out: MatchOutputDir, caption: str = "") -> str:
    """A width-filling, self-contained <img> (base64 data URI) for a figure."""
    uri = out.figure_data_uri(name)
    if not uri:
        return f'<p class="meta">Figure {esc(name)} unavailable.</p>'
    cap = f'<div class="meta">{esc(caption)}</div>' if caption else ""
    return (f'<div class="card"><img class="chart" '
            f'src="{uri}" alt="{esc(name)}">{cap}</div>')


def index_link(title: str, subtitle: str, href: str, badge: str = "") -> str:
    badge_html = f' <span class="tag">{esc(badge)}</span>' if badge else ""
    return (f'<a class="card" href="{esc(href)}">'
            f'<div class="v" style="font-size:1.1rem">{esc(title)}{badge_html}</div>'
            f'<div class="meta" style="margin:6px 0 0">{esc(subtitle)}</div>'
            f'</a>')


def match_header(title: str, out: MatchOutputDir, agent: str) -> str:
    """Standard page header/hero for an agent report page."""
    return (
        f"<h1>{esc(title)}</h1>\n"
        f"<div class='meta'>Agent: {esc(agent)} &middot; match {esc(out.match_id)} "
        f"&middot; generated {out.created_at.strftime('%Y-%m-%d %H:%M:%S')}</div>\n"
    )
