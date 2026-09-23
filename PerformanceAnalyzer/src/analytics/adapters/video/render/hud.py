"""Stage 8: video-game radar + Voronoi ownership + live metric HUD.

Composable render helpers used by `scripts/render_pitch_video.py`:

  * `finite_voronoi_polygons` — clipped Voronoi cells (scipy's classic recipe)
    so every player owns exactly one convex region inside the pitch.
  * `draw_voronoi_radar` — team-colored player dots + Voronoi ownership cells
    painted onto a top-down pitch image.
  * `draw_hud` — a translucent metric panel (possession %, line height,
    sprint distance, ...) overlaid on a frame.

All draw onto a supplied image and return a new BGR uint8 image, so the render
loop stays pure (easy to unit test, no global state).
"""

from __future__ import annotations

import numpy as np

from src.analytics.adapters.video.homography.pitch_model import render_pitch


def finite_voronoi_polygons(points: np.ndarray) -> list[np.ndarray]:
    """Clipped 2-D Voronoi cells for `points` (N, 2).

    The canonical `voronoi_finite_polygons_2d` recipe from the scipy docs:
    unbounded regions are closed by extending their infinite edges to a far
    ring around the data, so every input point owns exactly one bounded convex
    polygon. Returns a list of (K_i, 2) arrays aligned with the input rows.
    """
    from scipy.spatial import Voronoi

    if len(points) < 2:
        return [np.empty((0, 2)) for _ in points]

    vor = Voronoi(points)
    center = vor.points.mean(axis=0)
    if np.max(np.linalg.norm(points - center, axis=1)) == 0:
        return [np.empty((0, 2)) for _ in points]
    radius = float(np.ptp(points, axis=0).max()) * 2.0

    # ridge_vertices (v1, v2) -> the point index on each side
    ridge_points_map: dict[tuple[int, int], int] = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        ridge_points_map[(v1, v2)] = p1
        ridge_points_map[(v2, v1)] = p2

    polygons: list[np.ndarray] = []
    for point_idx, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if not region or -1 not in region:
            # bounded: close the loop
            polygons.append(vor.vertices[region + [region[0]]])
            continue

        # unbounded: assemble the region's edges, order the finite vertices,
        # and extend the two dangling rays out to the far ring.
        edges = []
        for r1, r2 in zip(region, region[1:] + region[:1]):
            if (r1, r2) in ridge_points_map:
                edges.append(ridge_points_map[(r1, r2)])
            elif (r2, r1) in ridge_points_map:
                edges.append(ridge_points_map[(r2, r1)])
            else:
                edges.append(point_idx)

        # finite vertices of this region, ordered around its centroid
        finite = [vor.vertices[i] for i in region if i != -1]
        if not finite:
            polygons.append(np.empty((0, 2)))
            continue
        local = np.mean(finite, axis=0)
        angles = np.arctan2([p[1] - local[1] for p in finite],
                            [p[0] - local[0] for p in finite])
        ordered = np.asarray(finite)[np.argsort(angles)]

        # rays off the two extreme vertices of the closed finite polygon
        ray1 = ordered[0] + np.sign(ordered[0] - center) * radius
        ray2 = ordered[-1] + np.sign(ordered[-1] - center) * radius
        polygons.append(np.vstack([ordered, ray1, ray2]))
    return polygons


def _fill_cell(img: np.ndarray, poly: np.ndarray, color,
               alpha: float = 0.35) -> None:
    """Fill a polygon with a translucent team colour."""
    if len(poly) < 3:
        return
    pts = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
    overlay = img.copy()
    cv2 = __import__("cv2")
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)


def draw_control_surface(pitch: np.ndarray, surface: np.ndarray,
                         scale: float = 10.0, alpha: float = 0.45) -> np.ndarray:
    """Overlay a C1 race control surface on a rendered pitch.

    `surface` is an (ny, nx) probability grid (team 0 control share). Cells
    paint blue (team 0) / red (team 1) proportionally to the share, so a
    contested cell goes white. Returns a new BGR image.
    """
    import cv2

    from src.analytics.models.pitch_control import PITCH_LENGTH, PITCH_WIDTH

    img = pitch.copy()
    ny, nx = surface.shape
    pad = 50  # matches PitchConfig.padding used by render_pitch
    cw = PITCH_LENGTH / nx * scale
    ch = PITCH_WIDTH / ny * scale
    overlay = img.copy()
    for iy in range(ny):
        for ix in range(nx):
            s = float(surface[iy, ix])
            if s != s:
                continue
            b = int(np.clip(255 * s, 0, 255))
            r = int(np.clip(255 * (1 - s), 0, 255))
            x0 = int(ix * cw + pad)
            y0 = int(iy * ch + pad)
            cv2.rectangle(overlay, (x0, y0),
                          (int(x0 + cw), int(y0 + ch)), (b, 0, r), -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)
    return img


def draw_epv_heatmap(pitch: np.ndarray, grid: np.ndarray,
                     scale: float = 10.0, alpha: float = 0.5,
                     label: str = "EPV") -> np.ndarray:
    """Overlay a C2 EPV value grid as a green heatmap on a rendered pitch."""
    import cv2

    from src.analytics.models.pitch_control import PITCH_LENGTH, PITCH_WIDTH

    img = pitch.copy()
    ny, nx = grid.shape
    g = np.nan_to_num(grid, nan=0.0)
    vmax = float(g.max())
    if vmax <= 1e-9:
        return img
    pad = 50  # matches PitchConfig.padding used by render_pitch
    w, h = int(PITCH_LENGTH * scale), int(PITCH_WIDTH * scale)
    g8 = np.clip(g / vmax, 0, 1)
    up = cv2.resize((g8 * 255).astype(np.uint8), (w, h),
                    interpolation=cv2.INTER_LINEAR)
    heat = cv2.applyColorMap(up, cv2.COLORMAP_VIRIDIS)   # dark -> yellow-hot
    overlay = img.copy()
    overlay[pad:pad + h, pad:pad + w] = heat
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)
    cv2.putText(img, f"{label} {vmax:.2f} max",
                (pad, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return img


def draw_voronoi_radar(pitch: np.ndarray,
                       positions: np.ndarray,          # (N, 2) metres
                       team_ids: np.ndarray,           # (N,) 0/1/other
                       team_colors: dict[int, tuple],
                       scale: float = 10.0,
                       ball_pos: np.ndarray | None = None,
                       ball_color: tuple = (0, 255, 255),
                       show_cells: bool = True) -> np.ndarray:
    """Team-colored radar + Voronoi ownership on a top-down pitch image.

    `positions` are pitch metres (0..length, 0..width); `team_colors` maps team
    id -> BGR tuple; unassigned (ref, None) players render grey with no cell.
    With `show_cells=False` only the player dots and ball are drawn.
    """
    import cv2

    img = pitch.copy()
    pad = 50  # matches PitchConfig.padding used by render_pitch

    px = positions * scale + pad
    cell_ids = team_ids.copy()
    # only colour cells for assigned teams
    owned = cell_ids >= 0
    if show_cells and owned.any() and len(px) >= 2:
        cells = finite_voronoi_polygons(px[owned])
        for poly, tid in zip(cells, cell_ids[owned]):
            color = team_colors.get(int(tid), (200, 200, 200))
            _fill_cell(img, poly, color, alpha=0.3)

    for (x, y), tid in zip(px, team_ids):
        if tid is None or tid < 0:
            color = (255, 255, 255)
        else:
            color = team_colors.get(int(tid), (200, 200, 200))
        cv2.circle(img, (int(x), int(y)), 5, color, -1)
        cv2.circle(img, (int(x), int(y)), 5, (0, 0, 0), 1)

    if ball_pos is not None:
        bx, by = ball_pos * scale + pad
        cv2.circle(img, (int(bx), int(by)), 4, ball_color, -1)

    return img


def draw_hud(frame: np.ndarray, lines: list[str],
             anchor: tuple[int, int] = (12, 24),
             font_scale: float = 0.55, bg_alpha: float = 0.55) -> np.ndarray:
    """Draw a translucent metric panel with `lines` of text."""
    import cv2

    img = frame.copy()
    fh = 18
    box_w = max(220, max((int(cv2.getTextSize(
        ln, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0][0]) + 16
        for ln in lines), default=220))
    box_h = fh * len(lines) + 14
    x0, y0 = anchor
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, bg_alpha, img, 1 - bg_alpha, 0, dst=img)
    for i, ln in enumerate(lines):
        cv2.putText(img, ln, (x0 + 8, y0 + fh * i + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
    return img


def radar_with_hud(pitch_config, positions: np.ndarray, team_ids: np.ndarray,
                   team_colors: dict[int, tuple], lines: list[str],
                   scale: float = 10.0,
                   ball_pos: np.ndarray | None = None) -> np.ndarray:
    """Top-down pitch radar + Voronoi + HUD panel, composed in one call.

    `pitch_config` is the `PitchConfig` used to render the empty pitch.
    """
    pitch = render_pitch(pitch_config, scale=scale)
    radar = draw_voronoi_radar(pitch, positions, team_ids, team_colors,
                               scale=scale, ball_pos=ball_pos)
    return draw_hud(radar, lines)
