import io
import json
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


def init_state():
    defaults = {
        "project_name": "Новый проект",
        "df_records": None,
        "data_source": "Демо-участок",
        "demo_name": DEMO_SCENARIOS[0],
        "target_h": -0.20,
        "soil": "Карьерный песок средний",
        "exclude_zero": True,
        "mode": "Карта высот + горизонтали",
        "surface_type": "Поверхность + каркас",
        "water_level": None,
        "flow_rain_mm": 20,
        "flow_runoff_fraction": 0.50,
        "flow_vector_step": 2,
        "flow_show_vectors": True,
        "x_min": None,
        "x_max": None,
        "y_min": None,
        "y_max": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def load_grid(file):
    df = pd.read_excel(file)
    required = {"X", "Y", "H"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"В XLSX нет колонок: {', '.join(sorted(missing))}")

    df = df[["X", "Y", "H"]].dropna()
    df["X"] = df["X"].astype(float)
    df["Y"] = df["Y"].astype(float)
    df["H"] = df["H"].astype(float)
    return df, build_grid(df)


def build_grid(df):
    grid = df.pivot_table(index="Y", columns="X", values="H")
    grid = grid.sort_index()
    grid = grid.sort_index(axis=1)
    return grid


def get_grid_step(grid):
    xs = np.sort(grid.columns.to_numpy(dtype=float))
    ys = np.sort(grid.index.to_numpy(dtype=float))

    x_step = float(np.min(np.diff(xs))) if len(xs) > 1 else 1.0
    y_step = float(np.min(np.diff(ys))) if len(ys) > 1 else 1.0
    cell_area = x_step * y_step

    return x_step, y_step, cell_area


def make_demo_terrain(name):
    if name == "Ровный участок с уклоном (30 x 22 м)":
        width, height = 30, 22
        house_kind = "rect"
    elif name == "Дом на плато и низина (30 x 22 м)":
        width, height = 30, 22
        house_kind = "rect"
    elif name == "Участок с локальной ямой (30 x 22 м)":
        width, height = 30, 22
        house_kind = "rect"
    elif name == "Большой участок с L-образным домом (60 x 40 м)":
        width, height = 60, 40
        house_kind = "l_shape"
    elif name == "Большой участок с двумя строениями (80 x 50 м)":
        width, height = 80, 50
        house_kind = "two_buildings"
    elif name == "Большой участок без дома: овраг и холм (100 x 60 м)":
        width, height = 100, 60
        house_kind = "none"
    elif name == "Большой участок без дома: ровный склон (90 x 70 м)":
        width, height = 90, 70
        house_kind = "none"
    else:
        width, height = 30, 22
        house_kind = "none"

    rows = []

    for y in range(height + 1):
        for x in range(width + 1):
            xn = x / width
            yn = y / height

            if name == "Ровный участок с уклоном (30 x 22 м)":
                h = -0.18 - 0.018 * (height - y) - 0.006 * x

            elif name == "Дом на плато и низина (30 x 22 м)":
                h = -0.22 - 0.010 * x - 0.025 * (height - y)
                lowland = 0.55 * np.exp(-((x - 23) ** 2 / 65 + (y - 2) ** 2 / 8))
                h -= lowland

            elif name == "Участок с локальной ямой (30 x 22 м)":
                h = -0.24 - 0.012 * x - 0.022 * (height - y)
                pit = 0.45 * np.exp(-((x - 8) ** 2 / 24 + (y - 5) ** 2 / 10))
                hill = 0.18 * np.exp(-((x - 24) ** 2 / 30 + (y - 17) ** 2 / 18))
                h = h - pit + hill

            elif name == "Большой участок с L-образным домом (60 x 40 м)":
                h = -0.10 - 0.55 * (1 - yn) - 0.22 * xn
                swale = 0.42 * np.exp(-(((x - 47) / 12) ** 2 + ((y - 5) / 6) ** 2))
                mound = 0.18 * np.exp(-(((x - 12) / 8) ** 2 + ((y - 31) / 7) ** 2))
                h = h - swale + mound

            elif name == "Большой участок с двумя строениями (80 x 50 м)":
                h = -0.14 - 0.48 * (1 - yn) - 0.25 * xn
                basin = 0.30 * np.exp(-(((x - 67) / 11) ** 2 + ((y - 8) / 8) ** 2))
                ridge = 0.14 * np.exp(-(((x - 37) / 13) ** 2 + ((y - 32) / 10) ** 2))
                h = h - basin + ridge

            elif name == "Большой участок без дома: овраг и холм (100 x 60 м)":
                h = -0.22 - 0.35 * (1 - yn) - 0.10 * xn
                ravine_center = 12 + 0.38 * x
                ravine = 0.70 * np.exp(-((y - ravine_center) ** 2) / 22)
                hill = 0.48 * np.exp(-(((x - 20) / 15) ** 2 + ((y - 46) / 12) ** 2))
                h = h - ravine + hill

            elif name == "Большой участок без дома: ровный склон (90 x 70 м)":
                h = -0.05 - 0.85 * (1 - yn) - 0.22 * xn
                h += 0.035 * np.sin(x / 8) * np.cos(y / 11)

            else:
                h = -0.20

            is_foundation = False

            if house_kind == "rect":
                is_foundation = 4 <= x <= 14 and 7 <= y <= 15
            elif house_kind == "l_shape":
                is_foundation = (
                    (17 <= x <= 37 and 20 <= y <= 27) or
                    (17 <= x <= 24 and 12 <= y <= 33)
                )
            elif house_kind == "two_buildings":
                main_house = 14 <= x <= 32 and 25 <= y <= 40
                outbuilding = 51 <= x <= 60 and 8 <= y <= 17
                is_foundation = main_house or outbuilding

            if is_foundation:
                h = 0.0

            rows.append({"X": float(x), "Y": float(y), "H": round(float(h), 3)})

    return pd.DataFrame(rows)


def calc_fill_volume(grid, x_min, x_max, y_min, y_max, target_h, exclude_zero=True):
    xs = np.sort(grid.columns.to_numpy(dtype=float))
    ys = np.sort(grid.index.to_numpy(dtype=float))
    x_step, y_step, cell_area = get_grid_step(grid)

    volume = 0.0
    cells = 0

    for x in xs[:-1]:
        x_next = x + x_step
        if not (x_min <= x and x_next <= x_max):
            continue

        for y in ys[:-1]:
            y_next = y + y_step
            if not (y_min <= y and y_next <= y_max):
                continue

            try:
                h = np.array([
                    grid.loc[y, x],
                    grid.loc[y, x_next],
                    grid.loc[y_next, x],
                    grid.loc[y_next, x_next],
                ], dtype=float)
            except KeyError:
                continue

            if exclude_zero and np.allclose(h, 0):
                continue

            h_avg = np.nanmean(h)
            fill_depth = max(0, target_h - h_avg)
            volume += fill_depth * cell_area
            cells += 1

    return volume, cells, cells * cell_area


def calculate_d8_flow(grid):
    z = grid.to_numpy(dtype=float)
    rows, cols = z.shape
    valid = np.isfinite(z)
    _, _, cell_area = get_grid_step(grid)

    to_row = np.full((rows, cols), -1, dtype=int)
    to_col = np.full((rows, cols), -1, dtype=int)

    x_step, y_step, _ = get_grid_step(grid)
    neighbours = [
        (-1, -1, np.sqrt(x_step**2 + y_step**2)), (-1, 0, y_step), (-1, 1, np.sqrt(x_step**2 + y_step**2)),
        (0, -1, x_step),                                                   (0, 1, x_step),
        (1, -1, np.sqrt(x_step**2 + y_step**2)),  (1, 0, y_step),  (1, 1, np.sqrt(x_step**2 + y_step**2)),
    ]

    for r in range(rows):
        for c in range(cols):
            if not valid[r, c]:
                continue

            best_slope = 0.0

            for dr, dc, distance in neighbours:
                rr, cc = r + dr, c + dc

                if 0 <= rr < rows and 0 <= cc < cols and valid[rr, cc]:
                    slope = (z[r, c] - z[rr, cc]) / distance

                    if slope > best_slope + 1e-12:
                        best_slope = slope
                        to_row[r, c] = rr
                        to_col[r, c] = cc

    accumulation = np.where(valid, cell_area, 0.0)

    valid_indices = np.flatnonzero(valid.ravel())
    order = valid_indices[np.argsort(z.ravel()[valid_indices])[::-1]]

    for index in order:
        r, c = np.unravel_index(index, z.shape)
        rr, cc = to_row[r, c], to_col[r, c]

        if rr >= 0:
            accumulation[rr, cc] += accumulation[r, c]

    interior = np.ones((rows, cols), dtype=bool)
    interior[[0, -1], :] = False
    interior[:, [0, -1]] = False
    sinks = valid & interior & (to_row < 0)

    return {
        "to_row": to_row,
        "to_col": to_col,
        "accumulation": accumulation,
        "sinks": sinks,
        "valid": valid,
    }


def build_flow_line_segments(X, Y, flow, step=2, min_accumulation=0.0):
    to_row = flow["to_row"]
    to_col = flow["to_col"]
    accumulation = flow["accumulation"]

    line_x, line_y = [], []
    rows, cols = accumulation.shape

    for r in range(0, rows, step):
        for c in range(0, cols, step):
            rr, cc = to_row[r, c], to_col[r, c]

            if rr >= 0 and accumulation[r, c] >= min_accumulation:
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

    neighbours = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    while stack:
        r, c = stack.pop()

        for dr, dc in neighbours:
            rr, cc = r + dr, c + dc

            if (
                0 <= rr < rows
                and 0 <= cc < cols
                and flooded_all[rr, cc]
                and not flooded_open[rr, cc]
            ):
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


def add_selected_rect(fig, x1, x2, y1, y2):
    fig.add_shape(
        type="rect",
        x0=x1,
        x1=x2,
        y0=y1,
        y1=y2,
        line=dict(color="red", width=3),
        fillcolor="rgba(255,0,0,0.08)",
    )


def project_to_json(df, settings):
    payload = {
        "app": "terrain-viewer",
        "version": 2,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "settings": settings,
        "points": df[["X", "Y", "H"]].to_dict(orient="records"),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def load_project_json(file):
    payload = json.load(file)
    if "points" not in payload:
        raise ValueError("В JSON нет блока points.")

    df = pd.DataFrame(payload["points"])
    df = df[["X", "Y", "H"]].dropna()
    df["X"] = df["X"].astype(float)
    df["Y"] = df["Y"].astype(float)
    df["H"] = df["H"].astype(float)
    settings = payload.get("settings", {})
    return df, settings


def apply_project_settings(settings):
    mapping = {
        "project_name": "project_name",
        "target_h": "target_h",
        "soil": "soil",
        "exclude_zero": "exclude_zero",
        "mode": "mode",
        "surface_type": "surface_type",
        "x_min": "x_min",
        "x_max": "x_max",
        "y_min": "y_min",
        "y_max": "y_max",
    }
    for src, dst in mapping.items():
        if src in settings:
            st.session_state[dst] = settings[src]


def render_heatmap(X, Y, Z, x_min, x_max, y_min, y_max):
    fig = go.Figure(data=go.Heatmap(
        x=X,
        y=Y,
        z=Z,
        colorscale="earth",
        colorbar=dict(title="H, м"),
    ))
    add_selected_rect(fig, x_min, x_max, y_min, y_max)
    fig.update_layout(title="Карта высот участка", xaxis_title="X, м", yaxis_title="Y, м", height=800)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def render_contours(X, Y, Z, x_min, x_max, y_min, y_max, filled=False):
    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=X,
        y=Y,
        z=Z,
        colorscale="earth",
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
    fig.update_layout(
        title="Карта высот + горизонтали" if filled else "Горизонтали участка",
        xaxis_title="X, м",
        yaxis_title="Y, м",
        height=800,
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def render_plotly_3d(Xg, Yg, Z):
    fig = go.Figure(data=[go.Surface(
        x=Xg,
        y=Yg,
        z=Z,
        colorscale="earth",
        colorbar=dict(title="H, м"),
    )])
    fig.update_layout(
        title="3D модель участка",
        height=850,
        scene=dict(
            xaxis_title="X, м",
            yaxis_title="Y, м",
            zaxis_title="H, м",
            aspectratio=dict(x=1.4, y=1.0, z=0.15),
        ),
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
    elif surface_type == "Поверхность + каркас":
        surface = ax.plot_surface(Xg, Yg, Z, cmap=cmap, edgecolor="none", linewidth=0, antialiased=True, rstride=1, cstride=1, alpha=0.92)
        ax.plot_wireframe(Xg, Yg, Z, color="black", linewidth=0.3, rstride=1, cstride=1)

    ax.set_box_aspect((float(X.max() - X.min()), float(Y.max() - Y.min()), 2.2))
    ax.set_title("3D модель высот участка", pad=20)
    ax.set_xlabel("X, м", labelpad=10)
    ax.set_ylabel("Y, м", labelpad=10)
    ax.set_zlabel("H, м", labelpad=12)
    ax.view_init(elev=27, azim=-135)
    ax.set_zlim(float(np.nanmin(Z)) - 0.05, float(np.nanmax(Z)) + 0.05)
    ax.zaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.tick_params(axis="z", labelsize=8, pad=4)

    colorbar_source = surface
    if colorbar_source is None:
        colorbar_source = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        colorbar_source.set_array(Z)

    colorbar = fig.colorbar(colorbar_source, ax=ax, shrink=0.65, pad=0.08)
    colorbar.set_label("Высота, м")
    ax.grid(True)
    plt.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    buffer.seek(0)
    plt.close(fig)
    return buffer


def render_flow(X, Y, grid, rain_mm, runoff_fraction, show_vectors, vector_step):
    flow = calculate_d8_flow(grid)
    accumulation = flow["accumulation"]
    valid = flow["valid"]
    log_accumulation = np.where(valid, np.log10(np.maximum(accumulation, 1e-9)), np.nan)

    fig = go.Figure(data=go.Heatmap(
        x=X,
        y=Y,
        z=log_accumulation,
        colorscale="blues",
        customdata=accumulation,
        colorbar=dict(title="log10 площади<br>водосбора, м²"),
        hovertemplate="X: %{x} м<br>Y: %{y} м<br>Площадь водосбора: %{customdata:.1f} м²<extra></extra>",
    ))

    if show_vectors:
        line_x, line_y = build_flow_line_segments(X, Y, flow, step=int(vector_step), min_accumulation=0.0)
        fig.add_trace(go.Scatter(
            x=line_x,
            y=line_y,
            mode="lines",
            line=dict(color="rgba(20, 20, 20, 0.45)", width=1),
            hoverinfo="skip",
            name="Направление стока",
        ))

    sink_rows, sink_cols = np.where(flow["sinks"])
    if len(sink_rows) > 0:
        fig.add_trace(go.Scatter(
            x=X[sink_cols],
            y=Y[sink_rows],
            mode="markers",
            marker=dict(symbol="x", size=10, color="red", line=dict(width=1, color="white")),
            name="Внутренние понижения",
            hovertemplate="Потенциальное локальное понижение<br>X: %{x} м<br>Y: %{y} м<extra></extra>",
        ))

    fig.update_layout(title="Распределение поверхностного стока по рельефу", xaxis_title="X, м", yaxis_title="Y, м", height=850)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    valid_area = float(np.nansum(np.where(valid, get_grid_step(grid)[2], 0.0)))
    rainfall_volume = valid_area * rain_mm / 1000 * runoff_fraction
    max_catchment = float(np.nanmax(accumulation))
    max_point_runoff = max_catchment * rain_mm / 1000 * runoff_fraction

    return fig, valid_area, rainfall_volume, max_point_runoff, int(np.count_nonzero(flow["sinks"]))


def render_flood(X, Y, Z, grid, water_level, x_min, x_max, y_min, y_max):
    flood = analyse_flooding(grid, water_level)
    _, _, cell_area = get_grid_step(grid)

    flood_classes = np.full(Z.shape, np.nan)
    flood_classes[flood["valid"]] = 0
    flood_classes[flood["flooded_open"]] = 1
    flood_classes[flood["flooded_closed"]] = 2

    fig = go.Figure(data=go.Heatmap(
        x=X,
        y=Y,
        z=flood_classes,
        zmin=0,
        zmax=2,
        colorscale=[
            [0.00, "#e8e1cf"], [0.32, "#e8e1cf"],
            [0.33, "#8cc7e8"], [0.66, "#8cc7e8"],
            [0.67, "#1769aa"], [1.00, "#1769aa"],
        ],
        customdata=np.dstack([Z, flood["water_depth"]]),
        colorbar=dict(
            tickvals=[0, 1, 2],
            ticktext=["Суша", "Низкая зона с выходом", "Замкнутая низина"],
            title="Статус",
        ),
        hovertemplate="X: %{x} м<br>Y: %{y} м<br>Рельеф: %{customdata[0]:.2f} м<br>Глубина до уровня: %{customdata[1]:.2f} м<extra></extra>",
    ))

    fig.add_trace(go.Contour(
        x=X,
        y=Y,
        z=Z,
        contours=dict(start=float(np.nanmin(Z)), end=float(np.nanmax(Z)), size=0.10, coloring="none", showlabels=False),
        line=dict(color="rgba(40,40,40,0.45)", width=1),
        showscale=False,
        hoverinfo="skip",
        name="Горизонтали",
    ))

    add_selected_rect(fig, x_min, x_max, y_min, y_max)
    fig.update_layout(title=f"Подтопление при уровне воды {water_level:.2f} м", xaxis_title="X, м", yaxis_title="Y, м", height=850)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    flooded_area = float(np.count_nonzero(flood["flooded_all"]) * cell_area)
    closed_area = float(np.count_nonzero(flood["flooded_closed"]) * cell_area)
    estimated_volume = float(np.nansum(flood["water_depth"] * cell_area))
    closed_volume = float(np.nansum(flood["closed_depth"] * cell_area))

    return fig, flooded_area, closed_area, estimated_volume, closed_volume


init_state()

st.title("Terrain Viewer")
st.caption("Анализ участка по точкам X, Y, H: высоты, объем подсыпки, сток воды и подтопление.")

with st.sidebar:
    st.header("Проект")
    st.text_input("Название проекта", key="project_name")

    st.header("Источник данных")
    data_source = st.radio(
        "Данные для расчета",
        ["Демо-участок", "Загрузить XLSX", "Проект JSON"],
        index=["Демо-участок", "Загрузить XLSX", "Проект JSON"].index(st.session_state.data_source)
        if st.session_state.data_source in ["Демо-участок", "Загрузить XLSX", "Проект JSON"] else 0,
        key="data_source",
    )

    df = None

    if data_source == "Демо-участок":
        demo_name = st.selectbox(
            "Сценарий демо-участка",
            DEMO_SCENARIOS,
            index=DEMO_SCENARIOS.index(st.session_state.demo_name) if st.session_state.demo_name in DEMO_SCENARIOS else 0,
            key="demo_name",
        )
        df = make_demo_terrain(demo_name)
        st.session_state.df_records = df.to_dict(orient="records")
        st.caption("Демо-данные подходят для проверки визуализаций и расчетов без собственного файла.")

    elif data_source == "Загрузить XLSX":
        uploaded = st.file_uploader(
            "Загрузить XLSX с колонками X, Y, H",
            type=["xlsx"],
            key="xlsx_upload",
        )
        if uploaded:
            try:
                df, _ = load_grid(uploaded)
                st.session_state.df_records = df.to_dict(orient="records")
                st.success("XLSX загружен.")
            except Exception as e:
                st.error(f"Ошибка XLSX: {e}")
        else:
            st.info("Загрузи XLSX-файл с колонками X, Y, H.")

    elif data_source == "Проект JSON":
        project_file = st.file_uploader(
            "Открыть проект JSON",
            type=["json"],
            key="project_upload",
        )
        if project_file is not None:
            try:
                df_project, project_settings = load_project_json(project_file)
                st.session_state.df_records = df_project.to_dict(orient="records")
                apply_project_settings(project_settings)
                df = df_project
                st.success("Проект загружен.")
            except Exception as e:
                st.error(f"Не удалось открыть проект: {e}")
        elif st.session_state.df_records:
            df = pd.DataFrame(st.session_state.df_records)
            st.caption("Используется ранее открытый проект из текущей сессии.")
        else:
            st.info("Загрузи сохраненный JSON проекта.")

if df is None and st.session_state.df_records:
    df = pd.DataFrame(st.session_state.df_records)

if df is None:
    st.info("Выбери демо-участок, загрузи XLSX или открой сохраненный JSON проекта.")
    st.stop()

grid = build_grid(df)
X = grid.columns.to_numpy(dtype=float)
Y = grid.index.to_numpy(dtype=float)
Z = grid.to_numpy(dtype=float)
Xg, Yg = np.meshgrid(X, Y)
x_step, y_step, cell_area = get_grid_step(grid)

x_default_min = float(X.min())
x_default_max = float(X.max())
y_default_min = float(Y.min())
y_default_max = float(Y.max())

for key, value in {
    "x_min": x_default_min,
    "x_max": x_default_max,
    "y_min": y_default_min,
    "y_max": y_default_max,
}.items():
    if st.session_state[key] is None:
        st.session_state[key] = value

# ограничим границы зоны реальными размерами данных
st.session_state.x_min = max(x_default_min, min(float(st.session_state.x_min), x_default_max))
st.session_state.x_max = max(x_default_min, min(float(st.session_state.x_max), x_default_max))
st.session_state.y_min = max(y_default_min, min(float(st.session_state.y_min), y_default_max))
st.session_state.y_max = max(y_default_min, min(float(st.session_state.y_max), y_default_max))

with st.sidebar:
    st.header("Зона и подсыпка")

    st.number_input("X min", min_value=x_default_min, max_value=x_default_max, step=x_step, key="x_min")
    st.number_input("X max", min_value=x_default_min, max_value=x_default_max, step=x_step, key="x_max")
    st.number_input("Y min", min_value=y_default_min, max_value=y_default_max, step=y_step, key="y_min")
    st.number_input("Y max", min_value=y_default_min, max_value=y_default_max, step=y_step, key="y_max")

    st.number_input("Целевая отметка, м", step=0.01, format="%.2f", key="target_h")
    st.selectbox("Тип грунта", list(SOILS.keys()), index=list(SOILS.keys()).index(st.session_state.soil), key="soil")
    st.checkbox("Исключать фундамент / H=0", key="exclude_zero")

    st.header("Визуализация")
    modes = [
        "Карта высот",
        "Горизонтали",
        "Карта высот + горизонтали",
        "Сток воды (D8)",
        "Подтопление по уровню",
        "3D модель",
        "3D статическая модель (Matplotlib)",
    ]
    st.selectbox("Режим", modes, index=modes.index(st.session_state.mode) if st.session_state.mode in modes else 2, key="mode")

    if st.session_state.mode == "3D статическая модель (Matplotlib)":
        surface_options = ["Поверхность с сеткой", "Гладкая поверхность", "Только каркас", "Поверхность + каркас"]
        st.selectbox(
            "Тип 3D поверхности",
            surface_options,
            index=surface_options.index(st.session_state.surface_type) if st.session_state.surface_type in surface_options else 3,
            key="surface_type",
        )

    if st.session_state.mode == "Сток воды (D8)":
        st.subheader("Параметры стока")
        st.slider("Условный дождь, мм", min_value=1, max_value=100, key="flow_rain_mm")
        st.slider("Доля поверхностного стока", min_value=0.0, max_value=1.0, step=0.05, key="flow_runoff_fraction")
        st.checkbox("Показывать направления стока", key="flow_show_vectors")
        st.slider("Шаг стрелок, м", min_value=1, max_value=10, key="flow_vector_step")

    if st.session_state.mode == "Подтопление по уровню":
        if st.session_state.water_level is None:
            st.session_state.water_level = float(np.round(np.nanmedian(Z), 2))
        st.subheader("Уровень подтопления")
        st.slider(
            "Отметка уровня воды, м",
            min_value=float(np.floor(np.nanmin(Z) * 100) / 100),
            max_value=float(np.ceil(np.nanmax(Z) * 100) / 100),
            step=0.01,
            key="water_level",
        )

x_min = min(st.session_state.x_min, st.session_state.x_max)
x_max = max(st.session_state.x_min, st.session_state.x_max)
y_min = min(st.session_state.y_min, st.session_state.y_max)
y_max = max(st.session_state.y_min, st.session_state.y_max)
target_h = st.session_state.target_h
soil = st.session_state.soil
compaction = SOILS[soil]

volume, cells, selected_area = calc_fill_volume(
    grid,
    x_min,
    x_max,
    y_min,
    y_max,
    target_h,
    exclude_zero=st.session_state.exclude_zero,
)
volume_compacted = volume * compaction

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Шаг сетки", f"{x_step:g} x {y_step:g} м")
kpi2.metric("Площадь зоны", f"{selected_area:.1f} м²")
kpi3.metric("Объем без уплотнения", f"{volume:.2f} м³")
kpi4.metric("Заказать с уплотнением", f"{volume_compacted:.2f} м³")

tabs = st.tabs(["Карта", "Расчет", "Данные", "Сохранение"])

with tabs[0]:
    mode = st.session_state.mode

    if mode == "Карта высот":
        st.plotly_chart(render_heatmap(X, Y, Z, x_min, x_max, y_min, y_max), use_container_width=True)

    elif mode == "Горизонтали":
        st.plotly_chart(render_contours(X, Y, Z, x_min, x_max, y_min, y_max, filled=False), use_container_width=True)

    elif mode == "Карта высот + горизонтали":
        st.plotly_chart(render_contours(X, Y, Z, x_min, x_max, y_min, y_max, filled=True), use_container_width=True)

    elif mode == "3D модель":
        st.plotly_chart(render_plotly_3d(Xg, Yg, Z), use_container_width=True)

    elif mode == "3D статическая модель (Matplotlib)":
        buffer = render_static_3d(X, Y, Xg, Yg, Z, st.session_state.surface_type)
        st.image(buffer, caption="Изометрическая 3D-модель высот участка", use_container_width=True)

    elif mode == "Сток воды (D8)":
        fig, valid_area, rainfall_volume, max_point_runoff, sink_count = render_flow(
            X,
            Y,
            grid,
            st.session_state.flow_rain_mm,
            st.session_state.flow_runoff_fraction,
            st.session_state.flow_show_vectors,
            st.session_state.flow_vector_step,
        )
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Площадь расчета", f"{valid_area:.0f} м²")
        c2.metric("Сток от дождя", f"{rainfall_volume:.2f} м³")
        c3.metric("Макс. приток к точке", f"{max_point_runoff:.2f} м³")
        c4.metric("Внутренние понижения", f"{sink_count}")

        st.caption("D8 - упрощенная модель по рельефу. Она не учитывает трубы, канавы, дренаж, растительность и реальную инфильтрацию.")

    elif mode == "Подтопление по уровню":
        fig, flooded_area, closed_area, estimated_volume, closed_volume = render_flood(
            X,
            Y,
            Z,
            grid,
            float(st.session_state.water_level),
            x_min,
            x_max,
            y_min,
            y_max,
        )
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Площадь ниже уровня", f"{flooded_area:.1f} м²")
        c2.metric("Замкнутые низины", f"{closed_area:.1f} м²")
        c3.metric("Объем до уровня", f"{estimated_volume:.1f} м³")
        c4.metric("Объем в низинах", f"{closed_volume:.1f} м³")

        st.caption("Подтопление оценочное по сетке. Для инженерного расчета нужны дренаж, водоприемники, фильтрация и рельеф между точками.")

with tabs[1]:
    st.subheader("Расчет подсыпки")

    summary = pd.DataFrame([
        {"Показатель": "Границы зоны X", "Значение": f"{x_min:g} - {x_max:g} м"},
        {"Показатель": "Границы зоны Y", "Значение": f"{y_min:g} - {y_max:g} м"},
        {"Показатель": "Целевая отметка", "Значение": f"{target_h:.2f} м"},
        {"Показатель": "Тип грунта", "Значение": soil},
        {"Показатель": "Коэффициент уплотнения", "Значение": f"x{compaction:.2f}"},
        {"Показатель": "Площадь расчетных ячеек", "Значение": f"{selected_area:.2f} м²"},
        {"Показатель": "Проектный объем", "Значение": f"{volume:.2f} м³"},
        {"Показатель": "Объем к заказу", "Значение": f"{volume_compacted:.2f} м³"},
    ])
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.warning("Коэффициент уплотнения - практическая оценка. Для закупки лучше добавлять отдельный запас на потери, разравнивание и фактическую влажность.")

with tabs[2]:
    st.subheader("Точки высот")
    st.dataframe(df.sort_values(["Y", "X"], ascending=[False, True]), use_container_width=True, hide_index=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Скачать точки CSV", data=csv_bytes, file_name="terrain_points.csv", mime="text/csv")

with tabs[3]:
    st.subheader("Сохранение проекта")

    settings = {
        "project_name": st.session_state.project_name,
        "target_h": st.session_state.target_h,
        "soil": st.session_state.soil,
        "exclude_zero": st.session_state.exclude_zero,
        "mode": st.session_state.mode,
        "surface_type": st.session_state.surface_type,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
    }

    project_bytes = project_to_json(df, settings)
    safe_name = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in st.session_state.project_name.strip()) or "terrain_project"

    st.download_button(
        "Скачать проект JSON",
        data=project_bytes,
        file_name=f"{safe_name}.json",
        mime="application/json",
    )

    st.info("JSON проекта содержит точки X,Y,H и настройки расчета. После сохранения его можно загрузить через 'Открыть проект JSON' в боковой панели.")
