import io
import json
import heapq
from datetime import datetime

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Terrain Viewer",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

SOILS = {
    "Карьерный песок средний": 1.20,
    "Карьерный песок мелкий": 1.25,
    "Песок намывной": 1.15,
    "ПГС": 1.18,
    "Щебень": 1.10,
}

DEMO_SCENARIOS = [
    "Ровный участок с уклоном (30 x 22 м)",
    "Дом на плато и низина (30 x 22 м)",
    "Участок с локальной ямой (30 x 22 м)",
    "Большой участок с L-образным домом (60 x 40 м)",
    "Большой участок с двумя строениями (80 x 50 м)",
    "Большой участок без дома: овраг и холм (100 x 60 м)",
    "Большой участок без дома: ровный склон (90 x 70 м)",
]

VIS_GROUPS = {
    "Рельеф": [
        "Карта высот",
        "Карта высот с сеткой",
        "Горизонтали",
        "Карта высот + горизонтали",
        "Карта уклонов",
        "Карта направления уклона",
    ],
    "Подсыпка": [
        "Карта толщины подсыпки",
        "Карта выемка / насыпь",
    ],
    "Вода": [
        "Сток воды (D8)",
        "Подтопление по уровню",
        "Время стекания",
        "Потенциальные водотоки",
        "Анимация подтопления",
    ],
    "Профили": [
        "Продольный профиль",
        "Поперечный профиль",
        "Разрез по двум точкам",
    ],
    "Контроль данных": [
        "Карта качества данных",
        "Карта неопределенности интерполяции",
    ],
    "Планирование": [
        "Карта допустимых зон",
        "Маршрут техники",
    ],
}


def init_state():
    defaults = {
        "project_name": "Новый проект",
        "df_records": None,
        "data_source": "Демо-участок",
        "demo_name": DEMO_SCENARIOS[0],
        "mode": "2D модель",
        "vis_group": "Рельеф",
        "surface_2d_type": "Карта высот + горизонтали",
        "surface_type": "Поверхность + каркас",
        "target_h": -0.20,
        "soil": "Карьерный песок средний",
        "exclude_zero": True,
        "x_min": None,
        "x_max": None,
        "y_min": None,
        "y_max": None,
        "water_level": None,
        "flow_rain_mm": 20,
        "flow_runoff_fraction": 0.50,
        "flow_vector_step": 2,
        "flow_show_vectors": True,
        "fill_slope_x": 0.0,
        "fill_slope_y": 0.0,
        "profile_x1": None,
        "profile_y1": None,
        "profile_x2": None,
        "profile_y2": None,
        "profile_samples": 150,
        "stream_threshold": 30.0,
        "outlier_threshold": 0.35,
        "suitability_max_slope": 5.0,
        "suitability_min_h": -99.0,
        "suitability_max_h": 99.0,
        "route_x_start": None,
        "route_y_start": None,
        "route_x_end": None,
        "route_y_end": None,
        "route_max_slope": 12.0,
        "route_avoid_foundation": True,
        "route_slope_penalty": 5.0,
        "animation_max_level": None,
        "animation_frames": 18,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def build_grid(df):
    grid = df.pivot_table(index="Y", columns="X", values="H", aggfunc="mean")
    return grid.sort_index().sort_index(axis=1)


def load_grid(file):
    df = pd.read_excel(file)
    required = {"X", "Y", "H"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"В XLSX нет колонок: {', '.join(sorted(missing))}")

    df = df[["X", "Y", "H"]].dropna().copy()
    for col in ("X", "Y", "H"):
        df[col] = df[col].astype(float)

    return df, build_grid(df)


def get_grid_step(grid):
    xs = np.sort(grid.columns.to_numpy(dtype=float))
    ys = np.sort(grid.index.to_numpy(dtype=float))
    x_step = float(np.min(np.diff(xs))) if len(xs) > 1 else 1.0
    y_step = float(np.min(np.diff(ys))) if len(ys) > 1 else 1.0
    return x_step, y_step, x_step * y_step


def make_demo_terrain(name):
    if name == "Ровный участок с уклоном (30 x 22 м)":
        width, height, house_kind = 30, 22, "rect"
    elif name == "Дом на плато и низина (30 x 22 м)":
        width, height, house_kind = 30, 22, "rect"
    elif name == "Участок с локальной ямой (30 x 22 м)":
        width, height, house_kind = 30, 22, "rect"
    elif name == "Большой участок с L-образным домом (60 x 40 м)":
        width, height, house_kind = 60, 40, "l_shape"
    elif name == "Большой участок с двумя строениями (80 x 50 м)":
        width, height, house_kind = 80, 50, "two_buildings"
    elif name == "Большой участок без дома: овраг и холм (100 x 60 м)":
        width, height, house_kind = 100, 60, "none"
    else:
        width, height, house_kind = 90, 70, "none"

    rows = []
    for y in range(height + 1):
        for x in range(width + 1):
            xn, yn = x / width, y / height

            if name == "Ровный участок с уклоном (30 x 22 м)":
                h = -0.18 - 0.018 * (height - y) - 0.006 * x
            elif name == "Дом на плато и низина (30 x 22 м)":
                h = -0.22 - 0.010 * x - 0.025 * (height - y)
                h -= 0.55 * np.exp(-((x - 23) ** 2 / 65 + (y - 2) ** 2 / 8))
            elif name == "Участок с локальной ямой (30 x 22 м)":
                h = -0.24 - 0.012 * x - 0.022 * (height - y)
                h -= 0.45 * np.exp(-((x - 8) ** 2 / 24 + (y - 5) ** 2 / 10))
                h += 0.18 * np.exp(-((x - 24) ** 2 / 30 + (y - 17) ** 2 / 18))
            elif name == "Большой участок с L-образным домом (60 x 40 м)":
                h = -0.10 - 0.55 * (1 - yn) - 0.22 * xn
                h -= 0.42 * np.exp(-(((x - 47) / 12) ** 2 + ((y - 5) / 6) ** 2))
                h += 0.18 * np.exp(-(((x - 12) / 8) ** 2 + ((y - 31) / 7) ** 2))
            elif name == "Большой участок с двумя строениями (80 x 50 м)":
                h = -0.14 - 0.48 * (1 - yn) - 0.25 * xn
                h -= 0.30 * np.exp(-(((x - 67) / 11) ** 2 + ((y - 8) / 8) ** 2))
                h += 0.14 * np.exp(-(((x - 37) / 13) ** 2 + ((y - 32) / 10) ** 2))
            elif name == "Большой участок без дома: овраг и холм (100 x 60 м)":
                h = -0.22 - 0.35 * (1 - yn) - 0.10 * xn
                ravine_center = 12 + 0.38 * x
                h -= 0.70 * np.exp(-((y - ravine_center) ** 2) / 22)
                h += 0.48 * np.exp(-(((x - 20) / 15) ** 2 + ((y - 46) / 12) ** 2))
            else:
                h = -0.05 - 0.85 * (1 - yn) - 0.22 * xn
                h += 0.035 * np.sin(x / 8) * np.cos(y / 11)

            foundation = False
            if house_kind == "rect":
                foundation = 4 <= x <= 14 and 7 <= y <= 15
            elif house_kind == "l_shape":
                foundation = ((17 <= x <= 37 and 20 <= y <= 27) or
                              (17 <= x <= 24 and 12 <= y <= 33))
            elif house_kind == "two_buildings":
                foundation = ((14 <= x <= 32 and 25 <= y <= 40) or
                              (51 <= x <= 60 and 8 <= y <= 17))

            if foundation:
                h = 0.0

            rows.append({"X": float(x), "Y": float(y), "H": round(float(h), 3)})
    return pd.DataFrame(rows)


def get_gradients(grid):
    z = grid.to_numpy(dtype=float)
    x_step, y_step, _ = get_grid_step(grid)
    dz_dy, dz_dx = np.gradient(z, y_step, x_step)
    slope_ratio = np.sqrt(dz_dx ** 2 + dz_dy ** 2)
    slope_pct = slope_ratio * 100
    slope_deg = np.degrees(np.arctan(slope_ratio))
    # Aspect: 0=N, 90=E, 180=S, 270=W, downhill direction.
    aspect_deg = (np.degrees(np.arctan2(-dz_dx, -dz_dy)) + 360) % 360
    return dz_dx, dz_dy, slope_pct, slope_deg, aspect_deg


def project_surface(X, Y, target_h, slope_x_pct, slope_y_pct):
    Xg, Yg = np.meshgrid(X, Y)
    return target_h + (slope_x_pct / 100) * (Xg - X.min()) + (slope_y_pct / 100) * (Yg - Y.min())


def calc_fill_volume(grid, x_min, x_max, y_min, y_max, target_surface, exclude_zero=True):
    z = grid.to_numpy(dtype=float)
    xs = grid.columns.to_numpy(dtype=float)
    ys = grid.index.to_numpy(dtype=float)
    _, _, cell_area = get_grid_step(grid)

    volume_fill = 0.0
    volume_cut = 0.0
    area = 0.0
    cells = 0

    for r in range(len(ys) - 1):
        for c in range(len(xs) - 1):
            if not (x_min <= xs[c] and xs[c + 1] <= x_max and y_min <= ys[r] and ys[r + 1] <= y_max):
                continue

            h = z[r:r + 2, c:c + 2]
            p = target_surface[r:r + 2, c:c + 2]

            if np.isnan(h).any() or np.isnan(p).any():
                continue
            if exclude_zero and np.allclose(h, 0):
                continue

            delta = float(np.mean(p - h))
            if delta > 0:
                volume_fill += delta * cell_area
            else:
                volume_cut += abs(delta) * cell_area

            area += cell_area
            cells += 1

    return volume_fill, volume_cut, cells, area


def calculate_d8_flow(grid):
    z = grid.to_numpy(dtype=float)
    rows, cols = z.shape
    valid = np.isfinite(z)
    x_step, y_step, cell_area = get_grid_step(grid)

    to_row = np.full((rows, cols), -1, dtype=int)
    to_col = np.full((rows, cols), -1, dtype=int)
    distance = np.full((rows, cols), np.nan, dtype=float)
    local_slope = np.zeros((rows, cols), dtype=float)

    neighbours = [
        (-1, -1, np.hypot(x_step, y_step)), (-1, 0, y_step), (-1, 1, np.hypot(x_step, y_step)),
        (0, -1, x_step),                                         (0, 1, x_step),
        (1, -1, np.hypot(x_step, y_step)),  (1, 0, y_step),  (1, 1, np.hypot(x_step, y_step)),
    ]

    for r in range(rows):
        for c in range(cols):
            if not valid[r, c]:
                continue
            best_slope = 0.0
            for dr, dc, d in neighbours:
                rr, cc = r + dr, c + dc
                if 0 <= rr < rows and 0 <= cc < cols and valid[rr, cc]:
                    slope = (z[r, c] - z[rr, cc]) / d
                    if slope > best_slope + 1e-12:
                        best_slope = slope
                        to_row[r, c] = rr
                        to_col[r, c] = cc
                        distance[r, c] = d
            local_slope[r, c] = best_slope

    accumulation = np.where(valid, cell_area, 0.0)
    valid_indices = np.flatnonzero(valid.ravel())
    high_to_low = valid_indices[np.argsort(z.ravel()[valid_indices])[::-1]]

    for index in high_to_low:
        r, c = np.unravel_index(index, z.shape)
        rr, cc = to_row[r, c], to_col[r, c]
        if rr >= 0:
            accumulation[rr, cc] += accumulation[r, c]

    interior = np.ones((rows, cols), dtype=bool)
    interior[[0, -1], :] = False
    interior[:, [0, -1]] = False
    # H=0 часто используется как условный фундамент. Плоская отметка фундамента
    # иначе порождает много ложных "понижений" в D8, поэтому ее исключаем.
    sinks = valid & interior & (to_row < 0) & ~np.isclose(z, 0.0)

    # Время до края/понижения. Модель очень упрощенная: скорость зависит от уклона.
    time_seconds = np.full((rows, cols), np.nan)
    low_to_high = valid_indices[np.argsort(z.ravel()[valid_indices])]
    for index in low_to_high:
        r, c = np.unravel_index(index, z.shape)
        rr, cc = to_row[r, c], to_col[r, c]
        at_edge = r in (0, rows - 1) or c in (0, cols - 1)

        if rr < 0:
            time_seconds[r, c] = 0.0 if at_edge else np.nan
        else:
            downstream = time_seconds[rr, cc]
            velocity = max(0.02, 0.25 * np.sqrt(max(local_slope[r, c], 1e-6)))
            if np.isfinite(downstream):
                time_seconds[r, c] = downstream + distance[r, c] / velocity

    return {
        "to_row": to_row,
        "to_col": to_col,
        "distance": distance,
        "local_slope": local_slope,
        "accumulation": accumulation,
        "sinks": sinks,
        "valid": valid,
        "time_seconds": time_seconds,
    }


def build_flow_line_segments(X, Y, flow, step=2, min_accumulation=0.0):
    line_x, line_y = [], []
    rows, cols = flow["accumulation"].shape
    for r in range(0, rows, max(1, int(step))):
        for c in range(0, cols, max(1, int(step))):
            rr, cc = flow["to_row"][r, c], flow["to_col"][r, c]
            if rr >= 0 and flow["accumulation"][r, c] >= min_accumulation:
                line_x.extend([X[c], X[cc], None])
                line_y.extend([Y[r], Y[rr], None])
    return line_x, line_y


def analyse_flooding(grid, water_level):
    z = grid.to_numpy(dtype=float)
    valid = np.isfinite(z)
    flooded_all = valid & (z <= water_level)
    rows, cols = flooded_all.shape

    flooded_open = np.zeros_like(flooded_all, dtype=bool)
    stack = []

    for c in range(cols):
        for r in (0, rows - 1):
            if flooded_all[r, c] and not flooded_open[r, c]:
                flooded_open[r, c] = True
                stack.append((r, c))
    for r in range(rows):
        for c in (0, cols - 1):
            if flooded_all[r, c] and not flooded_open[r, c]:
                flooded_open[r, c] = True
                stack.append((r, c))

    neighbours = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    while stack:
        r, c = stack.pop()
        for dr, dc in neighbours:
            rr, cc = r + dr, c + dc
            if (0 <= rr < rows and 0 <= cc < cols and flooded_all[rr, cc] and not flooded_open[rr, cc]):
                flooded_open[rr, cc] = True
                stack.append((rr, cc))

    flooded_closed = flooded_all & ~flooded_open
    water_depth = np.where(flooded_all, water_level - z, 0.0)
    closed_depth = np.where(flooded_closed, water_level - z, 0.0)

    return {
        "valid": valid,
        "flooded_all": flooded_all,
        "flooded_open": flooded_open,
        "flooded_closed": flooded_closed,
        "water_depth": water_depth,
        "closed_depth": closed_depth,
    }


def nearest_indices(values, points):
    values = np.asarray(values)
    p = np.asarray(points)
    idx = np.searchsorted(values, p)
    idx = np.clip(idx, 1, len(values) - 1)
    left = values[idx - 1]
    right = values[idx]
    idx -= (p - left < right - p)
    return idx


def sample_profile(grid, x1, y1, x2, y2, samples):
    xs = grid.columns.to_numpy(dtype=float)
    ys = grid.index.to_numpy(dtype=float)
    z = grid.to_numpy(dtype=float)

    t = np.linspace(0, 1, int(samples))
    x = x1 + (x2 - x1) * t
    y = y1 + (y2 - y1) * t
    c = nearest_indices(xs, x)
    r = nearest_indices(ys, y)
    h = z[r, c]
    dist = np.hypot(x - x1, y - y1)
    return dist, x, y, h


def route_dijkstra(grid, start_x, start_y, end_x, end_y, max_slope_pct, avoid_foundation, slope_penalty):
    z = grid.to_numpy(dtype=float)
    xs = grid.columns.to_numpy(dtype=float)
    ys = grid.index.to_numpy(dtype=float)
    rows, cols = z.shape
    x_step, y_step, _ = get_grid_step(grid)
    valid = np.isfinite(z)
    dz_dx, dz_dy, slope_pct, _, _ = get_gradients(grid)

    sr = int(nearest_indices(ys, [start_y])[0])
    sc = int(nearest_indices(xs, [start_x])[0])
    er = int(nearest_indices(ys, [end_y])[0])
    ec = int(nearest_indices(xs, [end_x])[0])

    if avoid_foundation:
        blocked = np.isclose(z, 0.0)
        blocked[sr, sc] = False
        blocked[er, ec] = False
    else:
        blocked = np.zeros_like(valid, dtype=bool)

    dist = np.full((rows, cols), np.inf)
    prev_r = np.full((rows, cols), -1, dtype=int)
    prev_c = np.full((rows, cols), -1, dtype=int)
    dist[sr, sc] = 0.0
    queue = [(0.0, sr, sc)]

    neighbours = [
        (-1, -1, np.hypot(x_step, y_step)), (-1, 0, y_step), (-1, 1, np.hypot(x_step, y_step)),
        (0, -1, x_step),                                         (0, 1, x_step),
        (1, -1, np.hypot(x_step, y_step)),  (1, 0, y_step),  (1, 1, np.hypot(x_step, y_step)),
    ]

    while queue:
        current, r, c = heapq.heappop(queue)
        if current != dist[r, c]:
            continue
        if (r, c) == (er, ec):
            break

        for dr, dc, step_distance in neighbours:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < rows and 0 <= cc < cols and valid[rr, cc] and not blocked[rr, cc]):
                continue

            edge_slope = abs(z[rr, cc] - z[r, c]) / step_distance * 100
            if edge_slope > max_slope_pct:
                continue

            cost = step_distance * (1 + slope_penalty * edge_slope / 100)
            candidate = current + cost
            if candidate < dist[rr, cc]:
                dist[rr, cc] = candidate
                prev_r[rr, cc] = r
                prev_c[rr, cc] = c
                heapq.heappush(queue, (candidate, rr, cc))

    if not np.isfinite(dist[er, ec]):
        return None

    path = []
    r, c = er, ec
    while r >= 0 and c >= 0:
        path.append((r, c))
        if (r, c) == (sr, sc):
            break
        r, c = prev_r[r, c], prev_c[r, c]

    path.reverse()
    return {
        "path_rows": [p[0] for p in path],
        "path_cols": [p[1] for p in path],
        "cost": float(dist[er, ec]),
        "length": float(sum(np.hypot(xs[path[i][1]] - xs[path[i - 1][1]], ys[path[i][0]] - ys[path[i - 1][0]]) for i in range(1, len(path)))),
        "start": (sr, sc),
        "end": (er, ec),
    }


def data_quality(df, grid, outlier_threshold):
    xs = np.sort(grid.columns.to_numpy(dtype=float))
    ys = np.sort(grid.index.to_numpy(dtype=float))
    x_step, y_step, _ = get_grid_step(grid)

    duplicate_counts = df.groupby(["X", "Y"]).size().reset_index(name="count")
    duplicates = duplicate_counts[duplicate_counts["count"] > 1][["X", "Y", "count"]]

    expected = {(round(float(x), 10), round(float(y), 10)) for x in np.arange(xs.min(), xs.max() + x_step * 0.5, x_step)
                for y in np.arange(ys.min(), ys.max() + y_step * 0.5, y_step)}
    actual = {(round(float(x), 10), round(float(y), 10)) for x, y in df[["X", "Y"]].to_numpy()}
    missing = np.array(sorted(expected - actual)) if expected - actual else np.empty((0, 2))

    z = grid.to_numpy(dtype=float)
    outliers = []
    for r in range(1, z.shape[0] - 1):
        for c in range(1, z.shape[1] - 1):
            center = z[r, c]
            neigh = z[r - 1:r + 2, c - 1:c + 2].copy()
            neigh[1, 1] = np.nan
            median = np.nanmedian(neigh)
            if np.isfinite(center) and np.isfinite(median) and abs(center - median) >= outlier_threshold:
                outliers.append((xs[c], ys[r], center, center - median))

    return duplicates, missing, pd.DataFrame(outliers, columns=["X", "Y", "H", "Отклонение"])


def interpolation_uncertainty(df, grid):
    xs = np.sort(grid.columns.to_numpy(dtype=float))
    ys = np.sort(grid.index.to_numpy(dtype=float))
    Xg, Yg = np.meshgrid(xs, ys)
    points = df[["X", "Y"]].drop_duplicates().to_numpy(dtype=float)
    uncertainty = np.zeros_like(Xg, dtype=float)

    # Честная визуальная метрика: расстояние до ближайшей исходной точки.
    for r in range(Xg.shape[0]):
        dx = Xg[r, :, None] - points[None, :, 0]
        dy = Yg[r, :, None] - points[None, :, 1]
        uncertainty[r, :] = np.sqrt(np.min(dx * dx + dy * dy, axis=1))
    return uncertainty


def add_selected_rect(fig, x1, x2, y1, y2):
    fig.add_shape(
        type="rect",
        x0=x1, x1=x2, y0=y1, y1=y2,
        line=dict(color="red", width=3),
        fillcolor="rgba(255,0,0,0.08)",
    )


def project_to_json(df, settings):
    payload = {
        "app": "terrain-viewer",
        "version": 3,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "settings": settings,
        "points": df[["X", "Y", "H"]].to_dict(orient="records"),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def load_project_json(file):
    payload = json.load(file)
    if "points" not in payload:
        raise ValueError("В JSON нет блока points.")
    df = pd.DataFrame(payload["points"])[["X", "Y", "H"]].dropna()
    for col in ("X", "Y", "H"):
        df[col] = df[col].astype(float)
    return df, payload.get("settings", {})


def apply_project_settings(settings):
    allowed = {
        "project_name", "target_h", "soil", "exclude_zero", "mode", "vis_group",
        "surface_2d_type", "surface_type", "x_min", "x_max", "y_min", "y_max",
        "fill_slope_x", "fill_slope_y", "profile_x1", "profile_y1", "profile_x2",
        "profile_y2", "profile_samples", "stream_threshold", "outlier_threshold",
        "suitability_max_slope", "suitability_min_h", "suitability_max_h",
        "route_x_start", "route_y_start", "route_x_end", "route_y_end",
        "route_max_slope", "route_avoid_foundation", "route_slope_penalty",
    }
    for key, value in settings.items():
        if key in allowed:
            st.session_state[key] = value


def heatmap_figure(X, Y, values, title, colorscale, colorbar_title, x_min=None, x_max=None, y_min=None, y_max=None, zmid=None):
    kwargs = {"colorscale": colorscale, "colorbar": dict(title=colorbar_title)}
    if zmid is not None:
        kwargs["zmid"] = zmid
    fig = go.Figure(data=go.Heatmap(x=X, y=Y, z=values, **kwargs))
    if None not in (x_min, x_max, y_min, y_max):
        add_selected_rect(fig, x_min, x_max, y_min, y_max)
    fig.update_layout(title=title, xaxis_title="X, м", yaxis_title="Y, м", height=820)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def render_heatmap(X, Y, Z, x_min, x_max, y_min, y_max, show_grid=False):
    fig = heatmap_figure(X, Y, Z, "Карта высот" + (" с сеткой" if show_grid else ""), "earth", "H, м", x_min, x_max, y_min, y_max)
    if show_grid:
        fig.update_xaxes(showgrid=True, gridcolor="rgba(40,40,40,0.35)", dtick=float(np.min(np.diff(X))) if len(X) > 1 else 1)
        fig.update_yaxes(showgrid=True, gridcolor="rgba(40,40,40,0.35)", dtick=float(np.min(np.diff(Y))) if len(Y) > 1 else 1)
    return fig


def render_contours(X, Y, Z, x_min, x_max, y_min, y_max, filled=False):
    fig = go.Figure(data=go.Contour(
        x=X, y=Y, z=Z, colorscale="earth",
        contours=dict(
            start=float(np.nanmin(Z)),
            end=float(np.nanmax(Z)),
            size=0.05 if filled else 0.10,
            coloring="heatmap" if filled else "lines",
            showlabels=True,
        ),
        line=dict(width=1),
        colorbar=dict(title="H, м"),
    ))
    add_selected_rect(fig, x_min, x_max, y_min, y_max)
    fig.update_layout(title="Карта высот + горизонтали" if filled else "Горизонтали", xaxis_title="X, м", yaxis_title="Y, м", height=820)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def render_plotly_3d(Xg, Yg, Z):
    fig = go.Figure(data=[go.Surface(x=Xg, y=Yg, z=Z, colorscale="earth", colorbar=dict(title="H, м"))])
    fig.update_layout(
        title="Интерактивная 3D модель",
        height=850,
        scene=dict(xaxis_title="X, м", yaxis_title="Y, м", zaxis_title="H, м", aspectratio=dict(x=1.4, y=1.0, z=0.15)),
    )
    return fig


def render_static_3d(X, Y, Xg, Yg, Z, surface_type):
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("terrain")
    norm = mcolors.Normalize(vmin=float(np.nanmin(Z)), vmax=float(np.nanmax(Z)))
    surface = None

    if surface_type == "Поверхность с сеткой":
        surface = ax.plot_surface(Xg, Yg, Z, cmap=cmap, edgecolor="black", linewidth=0.35, antialiased=True, rstride=1, cstride=1)
    elif surface_type == "Гладкая поверхность":
        surface = ax.plot_surface(Xg, Yg, Z, cmap=cmap, edgecolor="none", linewidth=0, antialiased=True, rstride=1, cstride=1)
    elif surface_type == "Только каркас":
        ax.plot_wireframe(Xg, Yg, Z, color="black", linewidth=0.6, rstride=1, cstride=1)
    else:
        surface = ax.plot_surface(Xg, Yg, Z, cmap=cmap, edgecolor="none", linewidth=0, antialiased=True, rstride=1, cstride=1, alpha=0.92)
        ax.plot_wireframe(Xg, Yg, Z, color="black", linewidth=0.3, rstride=1, cstride=1)

    ax.set_box_aspect((max(float(X.max() - X.min()), 1), max(float(Y.max() - Y.min()), 1), 2.2))
    ax.set_title("3D модель высот участка", pad=20)
    ax.set_xlabel("X, м", labelpad=10)
    ax.set_ylabel("Y, м", labelpad=10)
    ax.set_zlabel("H, м", labelpad=12)
    ax.view_init(elev=27, azim=-135)
    ax.zaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.grid(True)

    source = surface or plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    if surface is None:
        source.set_array(Z)
    fig.colorbar(source, ax=ax, shrink=0.65, pad=0.08).set_label("Высота, м")
    plt.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    buffer.seek(0)
    plt.close(fig)
    return buffer


def render_flow(X, Y, grid, rain_mm, runoff_fraction, show_vectors, vector_step, threshold=None):
    """
    Два разных режима:
    - threshold=None: полная карта накопления стока D8.
    - threshold=<м²>: только потенциальные русла, где водосбор достиг порога.
    """
    flow = calculate_d8_flow(grid)
    acc = flow["accumulation"]
    valid = flow["valid"]
    z = grid.to_numpy(dtype=float)

    if threshold is None:
        log_acc = np.where(valid, np.log10(np.maximum(acc, 1e-9)), np.nan)
        fig = heatmap_figure(
            X, Y, log_acc,
            "Распределение поверхностного стока (D8)",
            "blues",
            "log10 водосбора, м²",
        )

        if show_vectors:
            line_x, line_y = build_flow_line_segments(
                X, Y, flow, vector_step, min_accumulation=0.0
            )
            fig.add_trace(go.Scatter(
                x=line_x,
                y=line_y,
                mode="lines",
                line=dict(color="rgba(20,20,20,0.38)", width=1),
                hoverinfo="skip",
                name="Направление стока",
            ))

        sink_mask = flow["sinks"]
        sink_rows, sink_cols = np.where(sink_mask)
        if len(sink_rows):
            fig.add_trace(go.Scatter(
                x=X[sink_cols],
                y=Y[sink_rows],
                mode="markers",
                marker=dict(symbol="x", size=9, color="red"),
                name="Локальные понижения",
                hovertemplate="Локальное понижение<br>X: %{x} м<br>Y: %{y} м<extra></extra>",
            ))

    else:
        # Водотоки - только клетки, где к точке приходит водосбор не меньше порога.
        # Это принципиально отличается от общей D8-карты, где цветится весь участок.
        stream_mask = valid & (acc >= float(threshold))
        stream_acc = np.where(stream_mask, acc, np.nan)

        fig = go.Figure()

        # Нейтральный фон рельефа.
        fig.add_trace(go.Contour(
            x=X,
            y=Y,
            z=z,
            colorscale="Greys",
            opacity=0.36,
            contours=dict(
                start=float(np.nanmin(z)),
                end=float(np.nanmax(z)),
                size=0.10,
                coloring="heatmap",
                showlabels=False,
            ),
            line=dict(color="rgba(80,80,80,0.25)", width=0.5),
            colorbar=None,
            showscale=False,
            hoverinfo="skip",
            name="Рельеф",
        ))

        # Ярко показываем только отобранные русла.
        fig.add_trace(go.Heatmap(
            x=X,
            y=Y,
            z=stream_acc,
            colorscale="Blues",
            zmin=float(threshold),
            zmax=float(np.nanmax(acc)),
            colorbar=dict(title="Водосбор, м²"),
            hovertemplate=(
                "X: %{x} м<br>"
                "Y: %{y} м<br>"
                "Площадь водосбора: %{z:.1f} м²"
                "<extra></extra>"
            ),
            name="Потенциальный водоток",
        ))

        # Контур/линии направлений только внутри потенциальных русел.
        if show_vectors:
            line_x, line_y = build_flow_line_segments(
                X, Y, flow, vector_step, min_accumulation=float(threshold)
            )
            fig.add_trace(go.Scatter(
                x=line_x,
                y=line_y,
                mode="lines",
                line=dict(color="rgba(0,45,100,0.85)", width=2),
                hoverinfo="skip",
                name="Направление водотока",
            ))

        stream_rows, stream_cols = np.where(stream_mask)
        fig.add_trace(go.Scatter(
            x=X[stream_cols],
            y=Y[stream_rows],
            mode="markers",
            marker=dict(size=3, color="rgba(0,70,140,0.9)"),
            hoverinfo="skip",
            name="Клетки русла",
        ))

        fig.update_layout(
            title=f"Потенциальные водотоки: водосбор от {float(threshold):.0f} м²",
            xaxis_title="X, м",
            yaxis_title="Y, м",
            height=820,
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

    _, _, cell_area = get_grid_step(grid)
    valid_area = float(np.count_nonzero(valid) * cell_area)
    rain_volume = valid_area * rain_mm / 1000 * runoff_fraction
    max_point = float(np.nanmax(acc) * rain_mm / 1000 * runoff_fraction)
    return fig, flow, valid_area, rain_volume, max_point

def render_flood(X, Y, Z, grid, water_level, x_min, x_max, y_min, y_max):
    flood = analyse_flooding(grid, water_level)
    _, _, cell_area = get_grid_step(grid)
    classes = np.full(Z.shape, np.nan)
    classes[flood["valid"]] = 0
    classes[flood["flooded_open"]] = 1
    classes[flood["flooded_closed"]] = 2

    fig = go.Figure(data=go.Heatmap(
        x=X, y=Y, z=classes, zmin=0, zmax=2,
        colorscale=[[0.0, "#e8e1cf"], [0.32, "#e8e1cf"], [0.33, "#8cc7e8"], [0.66, "#8cc7e8"], [0.67, "#1769aa"], [1.0, "#1769aa"]],
        customdata=np.dstack([Z, flood["water_depth"]]),
        colorbar=dict(tickvals=[0, 1, 2], ticktext=["Суша", "Связано с границей", "Замкнутая низина"], title="Статус"),
        hovertemplate="X: %{x} м<br>Y: %{y} м<br>Рельеф: %{customdata[0]:.2f} м<br>Глубина: %{customdata[1]:.2f} м<extra></extra>",
    ))
    fig.add_trace(go.Contour(x=X, y=Y, z=Z, contours=dict(start=float(np.nanmin(Z)), end=float(np.nanmax(Z)), size=0.10, coloring="none"), line=dict(color="rgba(40,40,40,0.45)", width=1), showscale=False, hoverinfo="skip"))
    add_selected_rect(fig, x_min, x_max, y_min, y_max)
    fig.update_layout(title=f"Подтопление при уровне {water_level:.2f} м", xaxis_title="X, м", yaxis_title="Y, м", height=820)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    return fig, {
        "flooded_area": float(np.count_nonzero(flood["flooded_all"]) * cell_area),
        "closed_area": float(np.count_nonzero(flood["flooded_closed"]) * cell_area),
        "volume": float(np.nansum(flood["water_depth"] * cell_area)),
        "closed_volume": float(np.nansum(flood["closed_depth"] * cell_area)),
    }


def render_flood_animation(X, Y, grid, max_level, frames_count):
    z = grid.to_numpy(dtype=float)
    zmin = float(np.nanmin(z))
    levels = np.linspace(zmin, max_level, max(2, int(frames_count)))

    def classes_for(level):
        flood = analyse_flooding(grid, level)
        classes = np.full(z.shape, np.nan)
        classes[flood["valid"]] = 0
        classes[flood["flooded_open"]] = 1
        classes[flood["flooded_closed"]] = 2
        return classes

    colorscale = [[0.0, "#e8e1cf"], [0.32, "#e8e1cf"], [0.33, "#8cc7e8"], [0.66, "#8cc7e8"], [0.67, "#1769aa"], [1.0, "#1769aa"]]
    first = classes_for(levels[0])

    fig = go.Figure(
        data=[go.Heatmap(x=X, y=Y, z=first, zmin=0, zmax=2, colorscale=colorscale,
                         colorbar=dict(tickvals=[0,1,2], ticktext=["Суша","Открытая зона","Замкнутая низина"]))],
        frames=[
            go.Frame(data=[go.Heatmap(z=classes_for(level), zmin=0, zmax=2, colorscale=colorscale)], name=f"{level:.2f}")
            for level in levels
        ],
    )

    fig.update_layout(
        title=dict(
            text="Анимация повышения уровня воды",
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
        ),
        xaxis_title="X, м",
        yaxis_title="Y, м",
        height=860,
        margin=dict(t=115, b=125, l=70, r=40),
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "direction": "left",
            "x": 1.0,
            "xanchor": "right",
            "y": 1.10,
            "yanchor": "bottom",
            "pad": {"r": 6, "t": 0},
            "buttons": [
                {"label": "▶ Воспроизвести", "method": "animate", "args": [None, {"frame": {"duration": 300, "redraw": True}, "fromcurrent": True}]},
                {"label": "■ Стоп", "method": "animate", "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}]},
            ],
        }],
        sliders=[{
            "active": 0,
            "x": 0.10,
            "len": 0.80,
            "y": -0.12,
            "xanchor": "left",
            "yanchor": "top",
            "pad": {"t": 25, "b": 0},
            "steps": [{"label": f"{level:.2f} м", "method": "animate", "args": [[f"{level:.2f}"], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}]} for level in levels],
        }],
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def render_profile(dist, h, title, target_profile=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dist, y=h, mode="lines", name="Фактический рельеф"))
    if target_profile is not None:
        fig.add_trace(go.Scatter(x=dist, y=target_profile, mode="lines", name="Проектная поверхность"))
    fig.update_layout(title=title, xaxis_title="Расстояние по линии, м", yaxis_title="H, м", height=500)
    return fig


def make_aspect_colorscale():
    # cyclic colors, end repeats start
    return [
        [0.00, "#e31a1c"], [0.125, "#ff7f00"], [0.25, "#ffff33"],
        [0.375, "#33a02c"], [0.50, "#00bcd4"], [0.625, "#1f78b4"],
        [0.75, "#6a3d9a"], [0.875, "#e7298a"], [1.00, "#e31a1c"],
    ]


def route_figure(X, Y, Z, route, x_min, x_max, y_min, y_max):
    fig = heatmap_figure(X, Y, Z, "Маршрут техники", "earth", "H, м", x_min, x_max, y_min, y_max)
    if route:
        path_x = X[np.array(route["path_cols"], dtype=int)]
        path_y = Y[np.array(route["path_rows"], dtype=int)]
        fig.add_trace(go.Scatter(x=path_x, y=path_y, mode="lines+markers", line=dict(color="red", width=4), marker=dict(size=5), name="Маршрут"))
    return fig


# ---------- Streamlit UI ----------

init_state()
st.title("Terrain Viewer")
st.caption("Анализ рельефа по точкам X, Y, H: подсыпка, вода, профили, контроль измерений и планирование.")

with st.sidebar:
    st.header("Проект")
    st.text_input("Название проекта", key="project_name")

    st.header("Источник данных")
    source_options = ["Демо-участок", "Загрузить XLSX", "Проект JSON"]
    st.radio("Данные для расчета", source_options,
             index=source_options.index(st.session_state.data_source) if st.session_state.data_source in source_options else 0,
             key="data_source")
    df = None

    if st.session_state.data_source == "Демо-участок":
        st.selectbox("Сценарий демо-участка", DEMO_SCENARIOS,
                     index=DEMO_SCENARIOS.index(st.session_state.demo_name) if st.session_state.demo_name in DEMO_SCENARIOS else 0,
                     key="demo_name")
        df = make_demo_terrain(st.session_state.demo_name)
        st.session_state.df_records = df.to_dict(orient="records")
        st.caption("Демо-данные можно использовать без XLSX.")

    elif st.session_state.data_source == "Загрузить XLSX":
        uploaded = st.file_uploader("Загрузить XLSX с колонками X, Y, H", type=["xlsx"], key="xlsx_upload")
        if uploaded:
            try:
                df, _ = load_grid(uploaded)
                st.session_state.df_records = df.to_dict(orient="records")
                st.success("XLSX загружен.")
            except Exception as e:
                st.error(f"Ошибка XLSX: {e}")
        else:
            st.info("Загрузи XLSX-файл с колонками X, Y, H.")

    else:
        project_file = st.file_uploader("Открыть проект JSON", type=["json"], key="project_upload")
        if project_file is not None:
            try:
                df, settings = load_project_json(project_file)
                st.session_state.df_records = df.to_dict(orient="records")
                apply_project_settings(settings)
                st.success("Проект загружен.")
            except Exception as e:
                st.error(f"Не удалось открыть проект: {e}")
        elif st.session_state.df_records:
            df = pd.DataFrame(st.session_state.df_records)
            st.caption("Используются данные текущей сессии.")
        else:
            st.info("Загрузи JSON проекта.")

if df is None and st.session_state.df_records:
    df = pd.DataFrame(st.session_state.df_records)
if df is None:
    st.info("Выбери демо-участок, загрузи XLSX или открой JSON проекта.")
    st.stop()

grid = build_grid(df)
X = grid.columns.to_numpy(dtype=float)
Y = grid.index.to_numpy(dtype=float)
Z = grid.to_numpy(dtype=float)
Xg, Yg = np.meshgrid(X, Y)
x_step, y_step, cell_area = get_grid_step(grid)

# Default and safe settings bound to current grid
for key, value in {
    "x_min": float(X.min()), "x_max": float(X.max()),
    "y_min": float(Y.min()), "y_max": float(Y.max()),
    "profile_x1": float(X.min()), "profile_y1": float(np.median(Y)),
    "profile_x2": float(X.max()), "profile_y2": float(np.median(Y)),
    "route_x_start": float(X.min()), "route_y_start": float(Y.min()),
    "route_x_end": float(X.max()), "route_y_end": float(Y.max()),
    "animation_max_level": float(np.round(np.nanmedian(Z), 2)),
}.items():
    if st.session_state.get(key) is None:
        st.session_state[key] = value

for key, low, high in [
    ("x_min", X.min(), X.max()), ("x_max", X.min(), X.max()),
    ("y_min", Y.min(), Y.max()), ("y_max", Y.min(), Y.max()),
    ("profile_x1", X.min(), X.max()), ("profile_x2", X.min(), X.max()),
    ("profile_y1", Y.min(), Y.max()), ("profile_y2", Y.min(), Y.max()),
    ("route_x_start", X.min(), X.max()), ("route_x_end", X.min(), X.max()),
    ("route_y_start", Y.min(), Y.max()), ("route_y_end", Y.min(), Y.max()),
]:
    st.session_state[key] = max(float(low), min(float(st.session_state[key]), float(high)))

with st.sidebar:
    st.header("Визуализация")
    modes = ["2D модель", "3D модель", "3D статическая модель (Matplotlib)"]
    st.selectbox("Режим", modes, index=modes.index(st.session_state.mode) if st.session_state.mode in modes else 0, key="mode")
    is_2d = st.session_state.mode == "2D модель"

    if is_2d:
        # select current group based on persisted visualization
        current_group = next((group for group, items in VIS_GROUPS.items() if st.session_state.surface_2d_type in items), "Рельеф")
        if st.session_state.vis_group not in VIS_GROUPS:
            st.session_state.vis_group = current_group
        st.selectbox("Раздел 2D", list(VIS_GROUPS), index=list(VIS_GROUPS).index(st.session_state.vis_group), key="vis_group")
        current_options = VIS_GROUPS[st.session_state.vis_group]
        if st.session_state.surface_2d_type not in current_options:
            st.session_state.surface_2d_type = current_options[0]
        st.selectbox("Тип 2D поверхности", current_options, key="surface_2d_type")

        st.header("Зона и подсыпка")
        st.number_input("X min", min_value=float(X.min()), max_value=float(X.max()), step=x_step, key="x_min")
        st.number_input("X max", min_value=float(X.min()), max_value=float(X.max()), step=x_step, key="x_max")
        st.number_input("Y min", min_value=float(Y.min()), max_value=float(Y.max()), step=y_step, key="y_min")
        st.number_input("Y max", min_value=float(Y.min()), max_value=float(Y.max()), step=y_step, key="y_max")
        st.number_input("Целевая отметка, м", step=0.01, format="%.2f", key="target_h")
        st.selectbox("Тип грунта", list(SOILS), index=list(SOILS).index(st.session_state.soil), key="soil")
        st.checkbox("Исключать фундамент / H=0", key="exclude_zero")

        vis = st.session_state.surface_2d_type
        if vis in ("Карта толщины подсыпки", "Карта выемка / насыпь"):
            st.subheader("Проектная плоскость")
            st.number_input("Уклон проектной поверхности по X, %", step=0.1, key="fill_slope_x")
            st.number_input("Уклон проектной поверхности по Y, %", step=0.1, key="fill_slope_y")

        if vis in ("Сток воды (D8)", "Потенциальные водотоки", "Время стекания"):
            st.subheader("Параметры воды")
            st.slider("Условный дождь, мм", min_value=1, max_value=100, key="flow_rain_mm")
            st.slider("Доля поверхностного стока", min_value=0.0, max_value=1.0, step=0.05, key="flow_runoff_fraction")
            st.checkbox("Показывать направления стока", key="flow_show_vectors")
            st.slider("Шаг стрелок, м", min_value=1, max_value=10, key="flow_vector_step")
            if vis == "Потенциальные водотоки":
                st.number_input("Порог водосбора, м²", min_value=0.0, step=5.0, key="stream_threshold")

        if vis == "Подтопление по уровню":
            water_min = float(np.floor(np.nanmin(Z) * 100) / 100)
            water_max = float(np.ceil(np.nanmax(Z) * 100) / 100)

            # Streamlit не может создать slider с None в session_state.
            if st.session_state.water_level is None:
                st.session_state.water_level = float(np.round(np.nanmedian(Z), 2))
            st.session_state.water_level = max(
                water_min,
                min(float(st.session_state.water_level), water_max)
            )

            st.subheader("Уровень подтопления")
            st.slider(
                "Отметка уровня воды, м",
                min_value=water_min,
                max_value=water_max,
                step=0.01,
                key="water_level"
            )

        if vis == "Анимация подтопления":
            water_min = float(np.floor(np.nanmin(Z) * 100) / 100)
            water_max = float(np.ceil(np.nanmax(Z) * 100) / 100)

            if st.session_state.animation_max_level is None:
                st.session_state.animation_max_level = float(np.round(np.nanmedian(Z), 2))
            st.session_state.animation_max_level = max(
                water_min,
                min(float(st.session_state.animation_max_level), water_max)
            )

            st.subheader("Анимация подтопления")
            st.slider(
                "Конечный уровень воды, м",
                min_value=water_min,
                max_value=water_max,
                step=0.01,
                key="animation_max_level"
            )
            st.slider("Количество кадров", min_value=5, max_value=40, key="animation_frames")

        if vis in ("Продольный профиль", "Поперечный профиль", "Разрез по двум точкам"):
            st.subheader("Линия разреза")
            if vis == "Продольный профиль":
                st.session_state.profile_x1 = float(X.min())
                st.session_state.profile_x2 = float(X.max())
            elif vis == "Поперечный профиль":
                st.session_state.profile_y1 = float(Y.min())
                st.session_state.profile_y2 = float(Y.max())
            st.number_input("X1", min_value=float(X.min()), max_value=float(X.max()), step=x_step, key="profile_x1")
            st.number_input("Y1", min_value=float(Y.min()), max_value=float(Y.max()), step=y_step, key="profile_y1")
            st.number_input("X2", min_value=float(X.min()), max_value=float(X.max()), step=x_step, key="profile_x2")
            st.number_input("Y2", min_value=float(Y.min()), max_value=float(Y.max()), step=y_step, key="profile_y2")
            st.slider("Точек профиля", min_value=20, max_value=500, key="profile_samples")
            st.caption("Сейчас линия задается координатами. Выбор кликами по карте потребует отдельного компонента Streamlit.")

        if vis == "Карта качества данных":
            st.number_input("Порог выброса по высоте, м", min_value=0.01, max_value=5.0, step=0.01, key="outlier_threshold")

        if vis == "Карта допустимых зон":
            st.subheader("Критерии пригодности")
            st.number_input("Макс. уклон, %", min_value=0.1, max_value=100.0, step=0.5, key="suitability_max_slope")
            st.number_input("Мин. отметка, м", step=0.05, key="suitability_min_h")
            st.number_input("Макс. отметка, м", step=0.05, key="suitability_max_h")

        if vis == "Маршрут техники":
            st.subheader("Маршрут техники")
            st.number_input("Старт X", min_value=float(X.min()), max_value=float(X.max()), step=x_step, key="route_x_start")
            st.number_input("Старт Y", min_value=float(Y.min()), max_value=float(Y.max()), step=y_step, key="route_y_start")
            st.number_input("Финиш X", min_value=float(X.min()), max_value=float(X.max()), step=x_step, key="route_x_end")
            st.number_input("Финиш Y", min_value=float(Y.min()), max_value=float(Y.max()), step=y_step, key="route_y_end")
            st.number_input("Макс. уклон маршрута, %", min_value=1.0, max_value=100.0, step=1.0, key="route_max_slope")
            st.slider("Штраф за уклон", min_value=0.0, max_value=20.0, step=0.5, key="route_slope_penalty")
            st.checkbox("Обходить фундамент H=0", key="route_avoid_foundation")

    elif st.session_state.mode == "3D статическая модель (Matplotlib)":
        st.selectbox("Тип 3D поверхности", ["Поверхность с сеткой", "Гладкая поверхность", "Только каркас", "Поверхность + каркас"], key="surface_type")

    if not is_2d:
        st.info("Выбор зоны и расчет подсыпки доступны только в режиме «2D модель».")

x_min, x_max = sorted([st.session_state.x_min, st.session_state.x_max])
y_min, y_max = sorted([st.session_state.y_min, st.session_state.y_max])
target = project_surface(X, Y, st.session_state.target_h, st.session_state.fill_slope_x, st.session_state.fill_slope_y)
fill_vol, cut_vol, cells, selected_area = calc_fill_volume(grid, x_min, x_max, y_min, y_max, target, st.session_state.exclude_zero)
compaction = SOILS[st.session_state.soil]

if is_2d:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Шаг сетки", f"{x_step:g} x {y_step:g} м")
    c2.metric("Площадь зоны", f"{selected_area:.1f} м²")
    c3.metric("Подсыпка", f"{fill_vol:.2f} м³")
    c4.metric("Выемка", f"{cut_vol:.2f} м³")
    c5.metric("Заказать с уплотнением", f"{fill_vol * compaction:.2f} м³")
else:
    c1, c2, c3 = st.columns(3)
    c1.metric("Шаг сетки", f"{x_step:g} x {y_step:g} м")
    c2.metric("Точек высот", f"{len(df):,}".replace(",", " "))
    c3.metric("Диапазон H", f"{np.nanmin(Z):.2f} ... {np.nanmax(Z):.2f} м")

tabs = st.tabs(["Карта", "Расчет", "Данные", "Сохранение"])

with tabs[0]:
    if st.session_state.mode == "3D модель":
        st.plotly_chart(render_plotly_3d(Xg, Yg, Z), use_container_width=True)
        st.info("Для выбора зоны и расчета подсыпки переключитесь на режим «2D модель».")
    elif st.session_state.mode == "3D статическая модель (Matplotlib)":
        st.image(render_static_3d(X, Y, Xg, Yg, Z, st.session_state.surface_type), caption="Статическая 3D-модель", use_container_width=True)
        st.info("Для выбора зоны и расчета подсыпки переключитесь на режим «2D модель».")
    else:
        vis = st.session_state.surface_2d_type

        if vis == "Карта высот":
            st.plotly_chart(render_heatmap(X, Y, Z, x_min, x_max, y_min, y_max, False), use_container_width=True)

        elif vis == "Карта высот с сеткой":
            st.plotly_chart(render_heatmap(X, Y, Z, x_min, x_max, y_min, y_max, True), use_container_width=True)

        elif vis == "Горизонтали":
            st.plotly_chart(render_contours(X, Y, Z, x_min, x_max, y_min, y_max, False), use_container_width=True)

        elif vis == "Карта высот + горизонтали":
            st.plotly_chart(render_contours(X, Y, Z, x_min, x_max, y_min, y_max, True), use_container_width=True)

        elif vis == "Карта уклонов":
            _, _, slope_pct, slope_deg, _ = get_gradients(grid)
            fig = heatmap_figure(X, Y, slope_pct, "Карта уклонов", "YlOrRd", "Уклон, %", x_min, x_max, y_min, y_max)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Максимальный уклон: {np.nanmax(slope_pct):.1f}% ({np.nanmax(slope_deg):.1f}°).")

        elif vis == "Карта направления уклона":
            _, _, _, _, aspect = get_gradients(grid)
            fig = heatmap_figure(X, Y, aspect, "Карта направления уклона", make_aspect_colorscale(), "Азимут стока, °", x_min, x_max, y_min, y_max)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("0° - север, 90° - восток, 180° - юг, 270° - запад. Цвет показывает направление наибольшего понижения.")

        elif vis == "Карта толщины подсыпки":
            depth = np.maximum(target - Z, 0)
            fig = heatmap_figure(X, Y, depth, "Карта толщины подсыпки", "blues", "Толщина, м", x_min, x_max, y_min, y_max)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Показывает только подсыпку. Там, где проектная поверхность ниже фактической, на карте 0.")

        elif vis == "Карта выемка / насыпь":
            delta = target - Z
            fig = heatmap_figure(X, Y, delta, "Карта выемка / насыпь", "RdBu", "Проект - факт, м", x_min, x_max, y_min, y_max, zmid=0)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Синие зоны - нужна подсыпка, красные - выемка. Проектная поверхность задается целевой отметкой и уклонами X/Y.")

        elif vis in ("Сток воды (D8)", "Потенциальные водотоки"):
            threshold = st.session_state.stream_threshold if vis == "Потенциальные водотоки" else None
            fig, flow, valid_area, rain_volume, max_point = render_flow(
                X, Y, grid, st.session_state.flow_rain_mm, st.session_state.flow_runoff_fraction,
                st.session_state.flow_show_vectors, st.session_state.flow_vector_step, threshold
            )
            st.plotly_chart(fig, use_container_width=True)
            a, b, c, d = st.columns(4)
            a.metric("Площадь расчета", f"{valid_area:.0f} м²")
            b.metric("Сток от дождя", f"{rain_volume:.2f} м³")
            c.metric("Макс. приток к точке", f"{max_point:.2f} м³")
            d.metric("Внутренние понижения", f"{int(np.count_nonzero(flow['sinks']))}")
            st.caption("D8 - упрощенная модель. Не учитывает трубы, канавы, дренаж, растительность, размыв и фактическую инфильтрацию.")

        elif vis == "Подтопление по уровню":
            fig, metrics = render_flood(
                X, Y, Z, grid,
                float(st.session_state.water_level if st.session_state.water_level is not None else np.nanmedian(Z)),
                x_min, x_max, y_min, y_max
            )
            st.plotly_chart(fig, use_container_width=True)
            a, b, c, d = st.columns(4)
            a.metric("Площадь ниже уровня", f"{metrics['flooded_area']:.1f} м²")
            b.metric("Замкнутые низины", f"{metrics['closed_area']:.1f} м²")
            c.metric("Объем до уровня", f"{metrics['volume']:.1f} м³")
            d.metric("В низинах", f"{metrics['closed_volume']:.1f} м³")

        elif vis == "Время стекания":
            flow = calculate_d8_flow(grid)
            minutes = flow["time_seconds"] / 60
            fig = heatmap_figure(X, Y, minutes, "Оценка времени стекания до края", "viridis", "Минуты", x_min, x_max, y_min, y_max)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Условная оценка по D8 и скорости, зависящей от уклона. Внутренние понижения могут отображаться как пустые области.")

        elif vis == "Анимация подтопления":
            fig = render_flood_animation(X, Y, grid, st.session_state.animation_max_level, st.session_state.animation_frames)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Используй кнопку воспроизведения или ползунок под картой. Это геометрическая анимация уровня воды, а не гидродинамическая симуляция.")

        elif vis in ("Продольный профиль", "Поперечный профиль", "Разрез по двум точкам"):
            x1, y1 = st.session_state.profile_x1, st.session_state.profile_y1
            x2, y2 = st.session_state.profile_x2, st.session_state.profile_y2
            dist, px, py, h = sample_profile(grid, x1, y1, x2, y2, st.session_state.profile_samples)
            target_profile = st.session_state.target_h + (st.session_state.fill_slope_x / 100) * (px - X.min()) + (st.session_state.fill_slope_y / 100) * (py - Y.min())
            st.plotly_chart(render_profile(dist, h, vis, target_profile), use_container_width=True)
            line_map = render_heatmap(X, Y, Z, x_min, x_max, y_min, y_max, False)
            line_map.add_trace(go.Scatter(x=[x1, x2], y=[y1, y2], mode="lines+markers", line=dict(color="red", width=3), marker=dict(size=8), name="Линия профиля"))
            st.plotly_chart(line_map, use_container_width=True)

        elif vis == "Карта качества данных":
            duplicates, missing, outliers = data_quality(df, grid, st.session_state.outlier_threshold)
            fig = render_heatmap(X, Y, Z, x_min, x_max, y_min, y_max, True)
            if len(duplicates):
                fig.add_trace(go.Scatter(x=duplicates["X"], y=duplicates["Y"], mode="markers", marker=dict(symbol="circle-open", size=14, color="red", line=dict(width=2)), name="Дубли"))
            if len(missing):
                fig.add_trace(go.Scatter(x=missing[:, 0], y=missing[:, 1], mode="markers", marker=dict(symbol="x", size=10, color="orange"), name="Пропуски"))
            if len(outliers):
                fig.add_trace(go.Scatter(x=outliers["X"], y=outliers["Y"], mode="markers", marker=dict(symbol="diamond", size=10, color="yellow", line=dict(width=1, color="black")), name="Подозрительные точки"))
            st.plotly_chart(fig, use_container_width=True)
            a, b, c = st.columns(3)
            a.metric("Дубли координат", len(duplicates))
            b.metric("Пропуски сетки", len(missing))
            c.metric("Подозрительные точки", len(outliers))
            st.caption("Подозрительная точка отличается от медианы 8 соседей больше заданного порога. Это сигнал для проверки, а не автоматический диагноз ошибки.")

        elif vis == "Карта неопределенности интерполяции":
            unc = interpolation_uncertainty(df, grid)
            fig = heatmap_figure(X, Y, unc, "Расстояние до ближайшего исходного измерения", "magma", "м", x_min, x_max, y_min, y_max)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Это не статистическая погрешность, а прозрачная индикаторная метрика: чем дальше от исходной точки, тем менее надежна интерпретация поверхности.")

        elif vis == "Карта допустимых зон":
            _, _, slope_pct, _, _ = get_gradients(grid)
            suitable = (
                (slope_pct <= st.session_state.suitability_max_slope) &
                (Z >= st.session_state.suitability_min_h) &
                (Z <= st.session_state.suitability_max_h)
            )
            if st.session_state.exclude_zero:
                suitable &= ~np.isclose(Z, 0.0)
            values = suitable.astype(int)
            fig = go.Figure(data=go.Heatmap(
                x=X, y=Y, z=values,
                colorscale=[[0, "#c9c9c9"], [0.49, "#c9c9c9"], [0.5, "#43a047"], [1, "#43a047"]],
                zmin=0, zmax=1, colorbar=dict(tickvals=[0,1], ticktext=["Не подходит", "Подходит"]),
            ))
            add_selected_rect(fig, x_min, x_max, y_min, y_max)
            fig.update_layout(title="Карта допустимых зон", xaxis_title="X, м", yaxis_title="Y, м", height=820)
            fig.update_yaxes(scaleanchor="x", scaleratio=1)
            st.plotly_chart(fig, use_container_width=True)
            st.metric("Подходящая площадь", f"{np.count_nonzero(suitable) * cell_area:.1f} м²")

        elif vis == "Маршрут техники":
            route = route_dijkstra(
                grid,
                st.session_state.route_x_start, st.session_state.route_y_start,
                st.session_state.route_x_end, st.session_state.route_y_end,
                st.session_state.route_max_slope, st.session_state.route_avoid_foundation,
                st.session_state.route_slope_penalty,
            )
            st.plotly_chart(route_figure(X, Y, Z, route, x_min, x_max, y_min, y_max), use_container_width=True)
            if route:
                st.success(f"Маршрут найден: длина {route['length']:.1f} м. Условная стоимость пути: {route['cost']:.1f}.")
            else:
                st.error("Маршрут не найден. Увеличь допустимый уклон, измени точки или отключи обход фундамента.")

with tabs[1]:
    st.subheader("Расчет подсыпки")
    if st.session_state.mode != "2D модель":
        st.warning("Расчет зоны доступен только в режиме «2D модель».")
    else:
        summary = pd.DataFrame([
            {"Показатель": "Границы зоны X", "Значение": f"{x_min:g} - {x_max:g} м"},
            {"Показатель": "Границы зоны Y", "Значение": f"{y_min:g} - {y_max:g} м"},
            {"Показатель": "Базовая проектная отметка", "Значение": f"{st.session_state.target_h:.2f} м"},
            {"Показатель": "Проектный уклон X / Y", "Значение": f"{st.session_state.fill_slope_x:.1f}% / {st.session_state.fill_slope_y:.1f}%"},
            {"Показатель": "Тип грунта", "Значение": st.session_state.soil},
            {"Показатель": "Коэффициент уплотнения", "Значение": f"x{compaction:.2f}"},
            {"Показатель": "Расчетная площадь", "Значение": f"{selected_area:.2f} м²"},
            {"Показатель": "Подсыпка после уплотнения", "Значение": f"{fill_vol:.2f} м³"},
            {"Показатель": "Выемка", "Значение": f"{cut_vol:.2f} м³"},
            {"Показатель": "Заказать рыхлого грунта", "Значение": f"{fill_vol * compaction:.2f} м³"},
        ])
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.warning("Расчеты по сетке являются ориентировочными. Для строительства учитывай фактическую геологию, коэффициент уплотнения, откосы и инженерные сети.")

with tabs[2]:
    st.subheader("Точки высот")
    st.dataframe(df.sort_values(["Y", "X"], ascending=[False, True]), use_container_width=True, hide_index=True)
    st.download_button("Скачать точки CSV", df.to_csv(index=False).encode("utf-8"), "terrain_points.csv", "text/csv")

with tabs[3]:
    st.subheader("Сохранение проекта")
    settings = {
        key: st.session_state.get(key)
        for key in [
            "project_name", "target_h", "soil", "exclude_zero", "mode", "vis_group",
            "surface_2d_type", "surface_type", "x_min", "x_max", "y_min", "y_max",
            "fill_slope_x", "fill_slope_y", "profile_x1", "profile_y1", "profile_x2",
            "profile_y2", "profile_samples", "stream_threshold", "outlier_threshold",
            "suitability_max_slope", "suitability_min_h", "suitability_max_h",
            "route_x_start", "route_y_start", "route_x_end", "route_y_end",
            "route_max_slope", "route_avoid_foundation", "route_slope_penalty",
        ]
    }
    project_bytes = project_to_json(df, settings)
    safe_name = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in st.session_state.project_name.strip()) or "terrain_project"
    st.download_button("Скачать проект JSON", project_bytes, f"{safe_name}.json", "application/json")
    st.info("JSON проекта содержит все точки X,Y,H и основные настройки. Его можно открыть через «Проект JSON» в боковой панели.")
