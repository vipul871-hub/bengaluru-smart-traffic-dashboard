
import os
import math
from io import BytesIO

import numpy as np
import pandas as pd
import networkx as nx
import streamlit as st
import folium
from streamlit_folium import st_folium

st.set_page_config(
    page_title="Bengaluru Traffic Route Dashboard",
    page_icon="🚦",
    layout="wide",
)

DEFAULT_FILE = "Bengaluru_Traffic_Analysis_Converted.xlsx"


@st.cache_data(show_spinner=False)
def load_workbook(source):
    if isinstance(source, str):
        excel = pd.ExcelFile(source)
        return {
            sheet: pd.read_excel(source, sheet_name=sheet, header=2)
            for sheet in excel.sheet_names
        }

    raw = source.getvalue()
    excel = pd.ExcelFile(BytesIO(raw))
    return {
        sheet: pd.read_excel(
            BytesIO(raw), sheet_name=sheet, header=2
        )
        for sheet in excel.sheet_names
    }


def clean_num(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def traffic_label(ratio):
    if pd.isna(ratio):
        return "Unknown"
    if ratio <= 1.15:
        return "Low"
    if ratio <= 1.35:
        return "Moderate"
    if ratio <= 1.75:
        return "High"
    return "Severe"


def build_graph(df):
    graph = nx.DiGraph()

    for _, row in df.iterrows():
        required = [
            "Start Latitude",
            "Start Longitude",
            "End Latitude",
            "End Longitude",
        ]
        if any(pd.isna(row.get(col)) for col in required):
            continue

        start = (
            round(float(row["Start Latitude"]), 7),
            round(float(row["Start Longitude"]), 7),
        )
        end = (
            round(float(row["End Latitude"]), 7),
            round(float(row["End Longitude"]), 7),
        )

        if start == end:
            continue

        distance = pd.to_numeric(
            row.get("Distance (m)", 0), errors="coerce"
        )
        travel_seconds = pd.to_numeric(
            row.get("Average Travel Time (s)", 0), errors="coerce"
        )
        ratio = pd.to_numeric(
            row.get("Travel Time Ratio", 1), errors="coerce"
        )

        distance = 0 if pd.isna(distance) else float(distance)
        travel_seconds = (
            0 if pd.isna(travel_seconds) else float(travel_seconds)
        )
        ratio = 1 if pd.isna(ratio) else float(ratio)

        data = row.to_dict()
        data.update({
            "distance_km": max(distance / 1000, 0.001),
            "time_min": max(travel_seconds / 60, 0.01),
            "ratio": ratio,
        })

        graph.add_edge(start, end, **data)
        reverse_data = data.copy()
        reverse_data["_reversed"] = True
        graph.add_edge(end, start, **reverse_data)

    return graph


def edge_cost(data, time_weight, distance_weight, congestion_weight):
    return (
        time_weight * data["time_min"]
        + distance_weight * data["distance_km"]
        + congestion_weight * max(0, data["ratio"] - 1)
    )


def route_edges(graph, path):
    edges = []
    for start, end in zip(path[:-1], path[1:]):
        data = graph.get_edge_data(start, end)
        if data is None:
            return []
        edges.append(data)
    return edges


def route_metrics(path, edges, fuel_kmpl):
    distance = sum(edge["distance_km"] for edge in edges)
    time_min = sum(edge["time_min"] for edge in edges)

    ratio_numerator = sum(
        edge["ratio"] * edge["time_min"] for edge in edges
    )
    ratio = ratio_numerator / time_min if time_min else np.nan
    speed = distance / (time_min / 60) if time_min else np.nan

    labels = [traffic_label(edge["ratio"]) for edge in edges]
    streets = list(dict.fromkeys(
        str(edge.get("Street Name", "Unnamed Road"))
        for edge in edges
    ))

    fuel = distance / fuel_kmpl if fuel_kmpl > 0 else np.nan

    return {
        "distance_km": distance,
        "time_min": time_min,
        "ratio": ratio,
        "speed": speed,
        "traffic": traffic_label(ratio),
        "severe": labels.count("Severe"),
        "high": labels.count("High"),
        "moderate": labels.count("Moderate"),
        "streets": streets,
        "fuel_l": fuel,
    }


def nearest_nodes(graph, points, n=10):
    if not points:
        return []

    ranked = []
    for node in graph.nodes:
        distance = min(
            (node[0] - point[0]) ** 2
            + (node[1] - point[1]) ** 2
            for point in points
        )
        ranked.append((distance, node))

    ranked.sort(key=lambda item: item[0])
    return [node for _, node in ranked[:n]]


def get_shape_lookup(shape_df):
    required = ["Latitude", "Longitude", "Point Sequence"]
    if "New Segment ID" not in shape_df.columns:
        return {}

    shape_df = clean_num(shape_df.copy(), required)
    lookup = {}

    for segment_id, group in shape_df.groupby(
        "New Segment ID", dropna=False
    ):
        points = (
            group.sort_values("Point Sequence")[
                ["Latitude", "Longitude"]
            ]
            .dropna()
            .values
            .tolist()
        )
        if len(points) >= 2:
            lookup[str(segment_id)] = points

    return lookup


def route_map(route_paths, route_info, shape_lookup, center,
              traffic_df=None):
    map_view = folium.Map(
        location=center,
        zoom_start=13,
        control_scale=True,
        tiles=None,
    )

    folium.TileLayer(
        "OpenStreetMap", name="Street map", control=True
    ).add_to(map_view)
    folium.TileLayer(
        folium.TileLayer(
    tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    attr="© OpenStreetMap contributors",
    name="Street map",
    control=True,
).add_to(map_view)

folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
          "World_Street_Map/MapServer/tile/{z}/{y}/{x}",
    attr="Tiles © Esri",
    name="Detailed road map",
    control=True,
).add_to(map_view)

    if traffic_df is not None:
        traffic_layer = folium.FeatureGroup(
            name="Traffic congestion layer", show=True
        )

        colors = {
            "Low": "#16a34a",
            "Moderate": "#eab308",
            "High": "#f97316",
            "Severe": "#dc2626",
            "Unknown": "#64748b",
        }

        for _, row in traffic_df.iterrows():
            try:
                start = [
                    float(row["Start Latitude"]),
                    float(row["Start Longitude"]),
                ]
                end = [
                    float(row["End Latitude"]),
                    float(row["End Longitude"]),
                ]

                ratio = pd.to_numeric(
                    row.get("Travel Time Ratio", 1),
                    errors="coerce",
                )
                if pd.isna(ratio):
                    ratio = 1.0

                label = traffic_label(float(ratio))
                segment_id = str(row.get("New Segment ID", ""))
                points = shape_lookup.get(segment_id)

                coords = (
                    points if points and len(points) >= 2
                    else [start, end]
                )

                street = str(row.get("Street Name", "Unknown road"))
                speed = pd.to_numeric(
                    row.get("Average Speed (km/h)", np.nan),
                    errors="coerce",
                )

                tooltip = (
                    f"{street} | {label} | ratio {ratio:.2f}"
                )
                if pd.notna(speed):
                    tooltip += f" | speed {speed:.1f} km/h"

                folium.PolyLine(
                    coords,
                    color=colors[label],
                    weight=4,
                    opacity=0.72,
                    tooltip=tooltip,
                ).add_to(traffic_layer)

            except (TypeError, ValueError, KeyError):
                continue

        traffic_layer.add_to(map_view)

    route_colors = [
        "#059669", "#2563eb", "#9333ea",
        "#f97316", "#db2777", "#0891b2",
    ]

    for index, (path, info) in enumerate(
        zip(route_paths, route_info)
    ):
        layer = folium.FeatureGroup(
            name=f"Route {index + 1} — {info['traffic']}",
            show=True,
        )
        coords = []

        for edge in route_edges(info["_graph"], path):
            segment_id = str(edge.get("New Segment ID", ""))
            points = shape_lookup.get(segment_id)

            if points:
                points = list(points)
                if edge.get("_reversed"):
                    points.reverse()
            else:
                points = [
                    [
                        float(edge["Start Latitude"]),
                        float(edge["Start Longitude"]),
                    ],
                    [
                        float(edge["End Latitude"]),
                        float(edge["End Longitude"]),
                    ],
                ]
                if edge.get("_reversed"):
                    points.reverse()

            coords.extend(points if not coords else points[1:])

        if coords:
            folium.PolyLine(
                coords,
                color=route_colors[index % len(route_colors)],
                weight=8 if index == 0 else 5,
                opacity=0.95 if index == 0 else 0.78,
                tooltip=(
                    f"Route {index + 1}: "
                    f"{info['distance_km']:.2f} km | "
                    f"{info['time_min']:.1f} min | "
                    f"{info['traffic']}"
                ),
            ).add_to(layer)

        layer.add_to(map_view)

    if route_paths:
        folium.Marker(
            route_paths[0][0],
            tooltip="Selected start",
            icon=folium.Icon(color="green", icon="play"),
        ).add_to(map_view)

        folium.Marker(
            route_paths[0][-1],
            tooltip="Destination",
            icon=folium.Icon(color="red", icon="flag"),
        ).add_to(map_view)

    folium.LayerControl(collapsed=False).add_to(map_view)
    return map_view


st.title("🚦 Bengaluru Traffic & Optimal Route Dashboard")
st.caption(
    "Route estimates are calculated from the road segments "
    "in the uploaded Excel workbook. They are not live traffic feeds."
)

with st.sidebar:
    st.header("1. Dataset")
    uploaded = st.file_uploader(
        "Upload Excel file", type=["xlsx"]
    )

    source = (
        uploaded
        if uploaded is not None
        else (
            DEFAULT_FILE
            if os.path.exists(DEFAULT_FILE)
            else None
        )
    )

    if source is None:
        st.error(
            "Upload Bengaluru_Traffic_Analysis_Converted.xlsx "
            "to continue."
        )
        st.stop()

    st.success("Dataset selected")

try:
    sheets = load_workbook(source)
except Exception as error:
    st.error(f"Could not read the Excel workbook: {error}")
    st.stop()

required_sheets = [
    "Segment Time Analysis",
    "Segments",
    "Shape Points",
    "Route Summary",
]
missing_sheets = [
    name for name in required_sheets if name not in sheets
]
if missing_sheets:
    st.error(f"Missing workbook sheets: {missing_sheets}")
    st.stop()

sta = sheets["Segment Time Analysis"].copy()
shape = sheets["Shape Points"].copy()

sta = clean_num(sta, [
    "Start Latitude",
    "Start Longitude",
    "End Latitude",
    "End Longitude",
    "Distance (m)",
    "Average Speed (km/h)",
    "Average Travel Time (s)",
    "Travel Time Ratio",
    "Speed Reduction (%)",
])

windows = sorted(
    sta["Time Window"].dropna().astype(str).unique()
)
if not windows:
    st.error("No traffic time windows were found in the workbook.")
    st.stop()

with st.sidebar:
    st.header("2. Route settings")
    window = st.selectbox("Traffic time window", windows)
    departure_time = st.time_input(
        "Planned departure time",
        value=pd.to_datetime("09:00").time(),
    )
    n_routes = st.slider(
        "Number of route alternatives", 2, 6, 3
    )
    fuel_kmpl = st.number_input(
        "Vehicle fuel economy (km/L)",
        min_value=1.0,
        max_value=100.0,
        value=15.0,
        step=0.5,
    )
    st.caption(
        "Fuel use is estimated from distance and your entered "
        "fuel economy; it is not measured by the workbook."
    )

    st.subheader("Route optimization weights")
    wt = st.slider("Travel time weight", 0.0, 5.0, 3.0, 0.1)
    wd = st.slider("Distance weight", 0.0, 5.0, 1.0, 0.1)
    wc = st.slider("Congestion weight", 0.0, 5.0, 2.0, 0.1)

work = sta[sta["Time Window"].astype(str) == window].copy()
if work.empty:
    st.error("No records for the selected time window.")
    st.stop()

graph = build_graph(work)
if graph.number_of_edges() == 0:
    st.error("No usable road segments were found.")
    st.stop()

node_options = list(graph.nodes)
node_labels = {
    node: f"{node[0]:.6f}, {node[1]:.6f}"
    for node in node_options
}

if "Street Name" not in work.columns:
    st.error("The workbook is missing the Street Name column.")
    st.stop()

streets = sorted(work["Street Name"].dropna().astype(str).unique())
if not streets:
    st.error("No street names were found.")
    st.stop()

with st.sidebar:
    st.header("3. Start / destination")
    start_street = st.selectbox("Start street", streets, index=0)
    end_street = st.selectbox(
        "Destination street",
        streets,
        index=min(1, len(streets) - 1),
    )

start_rows = work[
    work["Street Name"].astype(str) == start_street
][[
    "Start Latitude", "Start Longitude",
    "End Latitude", "End Longitude",
]].dropna().values.tolist()

end_rows = work[
    work["Street Name"].astype(str) == end_street
][[
    "Start Latitude", "Start Longitude",
    "End Latitude", "End Longitude",
]].dropna().values.tolist()

start_points = [
    (float(point[0]), float(point[1]))
    for row in start_rows for point in [row[:2], row[2:]]
]
end_points = [
    (float(point[0]), float(point[1]))
    for row in end_rows for point in [row[:2], row[2:]]
]

start_candidates = nearest_nodes(graph, start_points, 40)
end_candidates = nearest_nodes(graph, end_points, 40)

if not start_candidates or not end_candidates:
    st.error("Could not find selectable start/destination points.")
    st.stop()

with st.sidebar:
    start_label = st.selectbox(
        "Start point",
        [node_labels[node] for node in start_candidates],
    )
    end_label = st.selectbox(
        "Destination point",
        [node_labels[node] for node in end_candidates],
        index=min(1, len(end_candidates) - 1),
    )

start = next(
    node for node in start_candidates
    if node_labels[node] == start_label
)
end = next(
    node for node in end_candidates
    if node_labels[node] == end_label
)

for _, _, data in graph.edges(data=True):
    data["route_weight"] = edge_cost(data, wt, wd, wc)

try:
    path_generator = nx.shortest_simple_paths(
        graph, start, end, weight="route_weight"
    )
    paths = []
    for path in path_generator:
        paths.append(path)
        if len(paths) >= n_routes:
            break
except (nx.NetworkXNoPath, nx.NodeNotFound, nx.NetworkXError):
    paths = []

if not paths:
    st.error(
        "No route was found between these points in this time window. "
        "Try different streets or points."
    )
    st.stop()

infos = []
for path in paths:
    edges = route_edges(graph, path)
    metrics = route_metrics(path, edges, fuel_kmpl)
    metrics["_graph"] = graph
    metrics["score"] = sum(
        edge_cost(edge, wt, wd, wc) for edge in edges
    )
    infos.append(metrics)

best_idx = int(np.argmin([item["score"] for item in infos]))
best = infos[best_idx]

st.subheader("Optimal route")
columns = st.columns(6)
kpis = [
    ("Travel time", f"{best['time_min']:.1f} min"),
    ("Distance", f"{best['distance_km']:.2f} km"),
    ("Average speed", f"{best['speed']:.1f} km/h"),
    ("Traffic", best["traffic"]),
    ("Travel-time ratio", f"{best['ratio']:.2f}"),
    ("Estimated fuel", f"{best['fuel_l']:.2f} L"),
]
for column, (label, value) in zip(columns, kpis):
    column.metric(label, value)

eta_minutes = int(round(best["time_min"]))
eta_clock = (
    pd.Timestamp.combine(
        pd.Timestamp.today().date(), departure_time
    )
    + pd.Timedelta(minutes=eta_minutes)
).strftime("%I:%M %p")

street_text = " → ".join(best["streets"][:8])
if len(best["streets"]) > 8:
    street_text += " → …"

st.info(
    f"Recommended Route {best_idx + 1}: {street_text}\n\n"
    f"Estimated arrival: {eta_clock}. "
    "This uses the selected workbook time-window estimate."
)

rows = []
for index, metrics in enumerate(infos, start=1):
    rows.append({
        "Route": f"Route {index}",
        "Recommended": "✓" if index - 1 == best_idx else "",
        "Distance (km)": round(metrics["distance_km"], 2),
        "ETA (min)": round(metrics["time_min"], 1),
        "Average speed (km/h)": round(metrics["speed"], 1),
        "Traffic": metrics["traffic"],
        "Travel-time ratio": round(metrics["ratio"], 2),
        "Estimated fuel (L)": round(metrics["fuel_l"], 2),
        "Severe segments": metrics["severe"],
        "High segments": metrics["high"],
        "Route score": round(metrics["score"], 2),
    })

comparison = pd.DataFrame(rows)

st.subheader("Interactive route map and map layers")
shape_lookup = get_shape_lookup(shape)
center = [
    float(work["Start Latitude"].mean()),
    float(work["Start Longitude"].mean()),
]

map_view = route_map(
    paths, infos, shape_lookup, center, traffic_df=work
)
st_folium(
    map_view,
    use_container_width=True,
    height=620,
    returned_objects=[],
)
st.caption(
    "Use the layer selector to switch base maps and toggle traffic "
    "congestion and route layers. Traffic colors use the workbook's "
    "travel-time ratio; they are not live traffic data."
)

st.subheader("Multiple route comparison")
st.dataframe(
    comparison, use_container_width=True, hide_index=True
)

left, right = st.columns(2)
with left:
    st.subheader("Selected route traffic details")
    detail = pd.DataFrame({
        "Metric": [
            "Congestion category",
            "Travel-time ratio",
            "Severe segments",
            "High segments",
            "Moderate segments",
            "Road sections",
        ],
        "Value": [
            best["traffic"],
            f"{best['ratio']:.2f}",
            best["severe"],
            best["high"],
            best["moderate"],
            len(route_edges(graph, paths[best_idx])),
        ],
    })
    st.table(detail)

with right:
    st.subheader("Congestion distribution")
    distribution = (
        work["Travel Time Ratio"]
        .apply(traffic_label)
        .value_counts()
        .reindex(["Low", "Moderate", "High", "Severe"])
        .fillna(0)
    )
    st.bar_chart(distribution)

st.subheader("Street-level traffic summary")
street_summary = (
    work.groupby("Street Name", dropna=False)
    .agg(
        Segments=("Segment ID", "count"),
        Distance_km=("Distance (m)", lambda values: values.sum() / 1000),
        Average_speed_kmh=("Average Speed (km/h)", "mean"),
        Average_travel_time_ratio=("Travel Time Ratio", "mean"),
    )
    .reset_index()
)

street_summary["Congestion"] = (
    street_summary["Average_travel_time_ratio"].apply(traffic_label)
)
street_summary = street_summary.sort_values(
    "Average_travel_time_ratio", ascending=False
)
st.dataframe(
    street_summary.round(2),
    use_container_width=True,
    hide_index=True,
)

st.subheader("Dataset parameters used")
st.markdown(
    "**Available in workbook:** coordinates, time window, segment "
    "distance, speed, travel time, travel-time ratio, speed reduction, "
    "street and segment information."
)
st.markdown(
    "**Not available as measured fields:** traffic volume, traffic "
    "density, vehicle count and fuel consumption. The dashboard does "
    "not invent those measurements; fuel is estimated from the "
    "vehicle fuel-economy value entered in the sidebar."
)

st.download_button(
    "Download route comparison CSV",
    comparison.to_csv(index=False).encode("utf-8"),
    "route_comparison.csv",
    "text/csv",
)
