
import os
from io import BytesIO

import numpy as np
import pandas as pd
import networkx as nx
import streamlit as st
import folium
from streamlit_folium import st_folium

st.set_page_config(
    page_title="Bengaluru Smart Traffic Dashboard",
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


def clean_num(df, columns):
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column], errors="coerce"
            )
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


def safe_float(value, default=0.0):
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def build_graph(df):
    G = nx.DiGraph()

    required = [
        "Start Latitude",
        "Start Longitude",
        "End Latitude",
        "End Longitude",
    ]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(
            "Missing required workbook columns: "
            + ", ".join(missing)
        )

    for _, row in df.iterrows():
        values = [row.get(column) for column in required]

        if any(pd.isna(value) for value in values):
            continue

        a = (
            round(float(row["Start Latitude"]), 7),
            round(float(row["Start Longitude"]), 7),
        )
        b = (
            round(float(row["End Latitude"]), 7),
            round(float(row["End Longitude"]), 7),
        )

        if a == b:
            continue

        distance = max(
            safe_float(row.get("Distance (m)")) / 1000,
            0.001,
        )
        travel_time = max(
            safe_float(row.get("Average Travel Time (s)")) / 60,
            0.01,
        )
        ratio = max(
            safe_float(row.get("Travel Time Ratio"), 1.0),
            0.0,
        )

        data = row.to_dict()
        data.update(
            {
                "distance_km": distance,
                "time_min": travel_time,
                "ratio": ratio,
            }
        )

        # Preserve the original direction and add a reverse edge.
        if G.has_edge(a, b):
            if travel_time < G[a][b]["time_min"]:
                G[a][b].update(data)
        else:
            G.add_edge(a, b, **data)

        reverse_data = dict(data)
        reverse_data["_reversed"] = True

        if not G.has_edge(b, a):
            G.add_edge(b, a, **reverse_data)

    return G


def edge_cost(data, time_weight, distance_weight, congestion_weight):
    return (
        time_weight * data["time_min"]
        + distance_weight * data["distance_km"]
        + congestion_weight * max(0, data["ratio"] - 1)
    )


def route_edges(G, path):
    edges = []

    for a, b in zip(path[:-1], path[1:]):
        data = G.get_edge_data(a, b)
        if data is None:
            return []
        edges.append(data)

    return edges


def route_metrics(path, edges, fuel_kmpl):
    distance = sum(edge["distance_km"] for edge in edges)
    time_min = sum(edge["time_min"] for edge in edges)

    weighted_ratio = sum(
        edge["ratio"] * edge["time_min"] for edge in edges
    )
    ratio = weighted_ratio / time_min if time_min else np.nan
    speed = distance / (time_min / 60) if time_min else np.nan

    categories = [
        traffic_label(edge["ratio"]) for edge in edges
    ]

    streets = list(
        dict.fromkeys(
            str(edge.get("Street Name", "Unnamed Road"))
            for edge in edges
        )
    )

    return {
        "distance_km": distance,
        "time_min": time_min,
        "ratio": ratio,
        "speed": speed,
        "traffic": traffic_label(ratio),
        "severe": categories.count("Severe"),
        "high": categories.count("High"),
        "moderate": categories.count("Moderate"),
        "streets": streets,
        "fuel_l": distance / fuel_kmpl if fuel_kmpl > 0 else np.nan,
    }


def nearest_nodes(G, points, n=20):
    if not points or not G.nodes:
        return []

    candidates = []

    for node in G.nodes:
        distance = min(
            (node[0] - point[0]) ** 2
            + (node[1] - point[1]) ** 2
            for point in points
        )
        candidates.append((distance, node))

    candidates.sort(key=lambda item: item[0])
    return [node for _, node in candidates[:n]]


def get_shape_lookup(shape_df):
    lookup = {}

    if "New Segment ID" not in shape_df.columns:
        return lookup

    shape_df = clean_num(
        shape_df.copy(),
        ["Latitude", "Longitude", "Point Sequence"],
    )

    for segment_id, group in shape_df.groupby(
        "New Segment ID", dropna=True
    ):
        points = (
            group.sort_values("Point Sequence")[
                ["Latitude", "Longitude"]
            ]
            .dropna()
            .values.tolist()
        )

        if len(points) >= 2:
            lookup[str(segment_id)] = points

    return lookup


def route_map(
    route_paths,
    route_info,
    shape_lookup,
    center,
    traffic_df=None,
):
    map_view = folium.Map(
        location=center,
        zoom_start=13,
        control_scale=True,
        tiles=None,
    )

    # Base map layers
    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr="© OpenStreetMap contributors",
        name="Street map",
        control=True,
        max_zoom=19,
    ).add_to(map_view)

    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Street_Map/MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Tiles © Esri",
        name="Detailed road map",
        control=True,
        max_zoom=19,
    ).add_to(map_view)

    folium.TileLayer(
        tiles=(
            "https://{s}.basemaps.cartocdn.com/light_all/"
            "{z}/{x}/{y}{r}.png"
        ),
        attr="© OpenStreetMap contributors © CARTO",
        name="Light map",
        control=True,
        max_zoom=19,
    ).add_to(map_view)

    folium.TileLayer(
        tiles=(
            "https://{s}.basemaps.cartocdn.com/dark_all/"
            "{z}/{x}/{y}{r}.png"
        ),
        attr="© OpenStreetMap contributors © CARTO",
        name="Dark map",
        control=True,
        max_zoom=19,
    ).add_to(map_view)

    # Congestion layer based on workbook travel-time ratios
    if traffic_df is not None:
        traffic_group = folium.FeatureGroup(
            name="Traffic congestion layer",
            show=True,
        )

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

                if not all(np.isfinite(start + end)):
                    continue

                ratio = safe_float(
                    row.get("Travel Time Ratio"), 1.0
                )
                category = traffic_label(ratio)

                colors = {
                    "Low": "#16a34a",
                    "Moderate": "#eab308",
                    "High": "#f97316",
                    "Severe": "#dc2626",
                    "Unknown": "#64748b",
                }

                segment_id = str(
                    row.get(
                        "New Segment ID",
                        row.get("Segment ID", ""),
                    )
                )

                points = shape_lookup.get(segment_id)
                coordinates = (
                    points if points else [start, end]
                )

                street = str(
                    row.get("Street Name", "Unknown road")
                )
                speed = row.get("Average Speed (km/h)", np.nan)

                tooltip = (
                    f"{street} | {category} congestion"
                    f" | Travel-time ratio: {ratio:.2f}"
                )

                if pd.notna(speed):
                    tooltip += f" | Speed: {float(speed):.1f} km/h"

                folium.PolyLine(
                    coordinates,
                    color=colors[category],
                    weight=4,
                    opacity=0.8,
                    tooltip=tooltip,
                ).add_to(traffic_group)

            except (TypeError, ValueError, KeyError):
                continue

        traffic_group.add_to(map_view)

    # Draw each candidate route as a separate toggleable layer.
    route_colors = [
        "#059669",
        "#2563eb",
        "#9333ea",
        "#f97316",
        "#db2777",
        "#0891b2",
    ]

    for index, (path, info) in enumerate(
        zip(route_paths, route_info)
    ):
        route_group = folium.FeatureGroup(
            name=f"Route {index + 1} — {info['traffic']}",
            show=True,
        )

        coordinates = []

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
                        edge["Start Latitude"],
                        edge["Start Longitude"],
                    ],
                    [
                        edge["End Latitude"],
                        edge["End Longitude"],
                    ],
                ]

                if edge.get("_reversed"):
                    points.reverse()

            if coordinates and points:
                coordinates.extend(points[1:])
            else:
                coordinates.extend(points)

        if coordinates:
            folium.PolyLine(
                coordinates,
                color=route_colors[index % len(route_colors)],
                weight=8 if index == 0 else 5,
                opacity=0.95 if index == 0 else 0.8,
                tooltip=(
                    f"Route {index + 1}: "
                    f"{info['distance_km']:.2f} km, "
                    f"{info['time_min']:.1f} min, "
                    f"{info['traffic']} traffic"
                ),
            ).add_to(route_group)

        route_group.add_to(map_view)

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


# ---------------- Dashboard interface ----------------

st.title("🚦 Bengaluru Smart Traffic Dashboard")
st.caption(
    "Road-network route planning using your Excel workbook. "
    "Traffic estimates are based on recorded segment data, "
    "not a live traffic feed."
)

with st.sidebar:
    st.header("1. Dataset")
    uploaded_file = st.file_uploader(
        "Upload Excel workbook",
        type=["xlsx"],
    )

    source = (
        uploaded_file
        if uploaded_file is not None
        else (
            DEFAULT_FILE
            if os.path.exists(DEFAULT_FILE)
            else None
        )
    )

    if source is None:
        st.warning(
            "Upload Bengaluru_Traffic_Analysis_Converted.xlsx "
            "or add it to the same folder as app.py in GitHub."
        )
        st.stop()

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
    st.error(
        "The workbook is missing these sheets: "
        + ", ".join(missing_sheets)
    )
    st.stop()

workbook_df = sheets["Segment Time Analysis"].copy()
shape_df = sheets["Shape Points"].copy()

workbook_df = clean_num(
    workbook_df,
    [
        "Start Latitude",
        "Start Longitude",
        "End Latitude",
        "End Longitude",
        "Distance (m)",
        "Average Speed (km/h)",
        "Harmonic Avg Speed (km/h)",
        "Average Travel Time (s)",
        "Travel Time Ratio",
        "Speed Reduction (%)",
    ],
)

if "Time Window" not in workbook_df.columns:
    st.error("The workbook has no 'Time Window' column.")
    st.stop()

windows = sorted(
    workbook_df["Time Window"].dropna().astype(str).unique()
)

if not windows:
    st.error("No usable traffic time windows were found.")
    st.stop()

with st.sidebar:
    st.header("2. Route settings")

    selected_window = st.selectbox(
        "Traffic time window",
        windows,
    )

    departure_time = st.time_input(
        "Planned departure time",
        value=pd.to_datetime("09:00").time(),
    )

    number_routes = st.slider(
        "Number of route alternatives",
        min_value=2,
        max_value=6,
        value=3,
    )

    fuel_kmpl = st.number_input(
        "Vehicle fuel economy (km/L)",
        min_value=1.0,
        max_value=100.0,
        value=15.0,
        step=0.5,
    )

    st.caption(
        "Fuel use is estimated from your mileage input; "
        "it is not a measured workbook value."
    )

    st.subheader("Route optimization weights")

    time_weight = st.slider(
        "Travel time weight", 0.0, 5.0, 3.0, 0.1
    )
    distance_weight = st.slider(
        "Distance weight", 0.0, 5.0, 1.0, 0.1
    )
    congestion_weight = st.slider(
        "Congestion weight", 0.0, 5.0, 2.0, 0.1
    )

work = workbook_df[
    workbook_df["Time Window"].astype(str) == selected_window
].copy()

if work.empty:
    st.error("No records found for the selected time window.")
    st.stop()

try:
    graph = build_graph(work)
except Exception as error:
    st.error(f"Could not build the road network: {error}")
    st.stop()

if graph.number_of_edges() == 0:
    st.error("No usable road segments were found in the workbook.")
    st.stop()

# Select starting and destination streets
if "Street Name" not in work.columns:
    work["Street Name"] = "Unnamed Road"

street_names = sorted(
    work["Street Name"].dropna().astype(str).unique()
)

if not street_names:
    st.error("No street names were found in the selected dataset.")
    st.stop()

with st.sidebar:
    st.header("3. Start and destination")

    start_street = st.selectbox(
        "Starting street",
        street_names,
        index=0,
    )

    end_index = min(1, len(street_names) - 1)

    end_street = st.selectbox(
        "Destination street",
        street_names,
        index=end_index,
    )

def street_points(street_name):
    selected = work[
        work["Street Name"].astype(str) == street_name
    ]

    points = []

    for _, row in selected.iterrows():
        for lat_col, lon_col in [
            ("Start Latitude", "Start Longitude"),
            ("End Latitude", "End Longitude"),
        ]:
            lat = row.get(lat_col)
            lon = row.get(lon_col)

            if pd.notna(lat) and pd.notna(lon):
                points.append((float(lat), float(lon)))

    return points


start_candidates = nearest_nodes(
    graph, street_points(start_street), 40
)
end_candidates = nearest_nodes(
    graph, street_points(end_street), 40
)

if not start_candidates or not end_candidates:
    st.error("Could not find start or destination points.")
    st.stop()

node_labels = {
    node: f"{node[0]:.6f}, {node[1]:.6f}"
    for node in graph.nodes
}

with st.sidebar:
    start_label = st.selectbox(
        "Starting point coordinates",
        [node_labels[node] for node in start_candidates],
    )

    end_label = st.selectbox(
        "Destination coordinates",
        [node_labels[node] for node in end_candidates],
        index=min(1, len(end_candidates) - 1),
    )

start_node = next(
    node for node in start_candidates
    if node_labels[node] == start_label
)

end_node = next(
    node for node in end_candidates
    if node_labels[node] == end_label
)

if start_node == end_node:
    st.warning(
        "Start and destination are the same point. "
        "Choose different points."
    )
    st.stop()

# Calculate route weights
for _, _, edge_data in graph.edges(data=True):
    edge_data["route_weight"] = max(
        edge_cost(
            edge_data,
            time_weight,
            distance_weight,
            congestion_weight,
        ),
        0.000001,
    )

# Find candidate routes
paths = []

try:
    path_generator = nx.shortest_simple_paths(
        graph,
        start_node,
        end_node,
        weight="route_weight",
    )

    for path in path_generator:
        paths.append(path)

        if len(paths) >= number_routes:
            break

except (nx.NetworkXNoPath, nx.NodeNotFound):
    paths = []
except Exception as error:
    st.error(f"Route calculation failed: {error}")
    st.stop()

if not paths:
    st.error(
        "No route connects these points in the selected time window. "
        "Try another starting street, destination, or time window."
    )
    st.stop()

route_information = []

for path in paths:
    edges = route_edges(graph, path)
    metrics = route_metrics(path, edges, fuel_kmpl)
    metrics["_graph"] = graph
    metrics["score"] = sum(
        edge_cost(
            edge,
            time_weight,
            distance_weight,
            congestion_weight,
        )
        for edge in edges
    )
    route_information.append(metrics)

best_index = int(
    np.argmin([item["score"] for item in route_information])
)
best_route = route_information[best_index]

# ---------------- Results ----------------

st.subheader("Recommended route")

columns = st.columns(6)

kpis = [
    ("Travel time", f"{best_route['time_min']:.1f} min"),
    ("Distance", f"{best_route['distance_km']:.2f} km"),
    ("Average speed", f"{best_route['speed']:.1f} km/h"),
    ("Traffic", best_route["traffic"]),
    ("Travel-time ratio", f"{best_route['ratio']:.2f}"),
    ("Estimated fuel", f"{best_route['fuel_l']:.2f} L"),
]

for column, (label, value) in zip(columns, kpis):
    column.metric(label, value)

arrival_minutes = int(round(best_route["time_min"]))
arrival_time = (
    pd.Timestamp.combine(
        pd.Timestamp.today().date(),
        departure_time,
    )
    + pd.Timedelta(minutes=arrival_minutes)
).strftime("%I:%M %p")

street_route = " → ".join(best_route["streets"][:8])

if len(best_route["streets"]) > 8:
    street_route += " → …"

st.info(
    f"Recommended Route {best_index + 1}\n\n"
    f"Streets: {street_route}\n\n"
    f"Estimated arrival: {arrival_time}. "
    "This uses the selected time-window estimate, not live traffic."
)

# Route comparison table
comparison_rows = []

for index, metrics in enumerate(route_information, start=1):
    comparison_rows.append(
        {
            "Route": f"Route {index}",
            "Recommended": "Yes" if index - 1 == best_index else "",
            "Distance (km)": round(metrics["distance_km"], 2),
            "Travel time (min)": round(metrics["time_min"], 1),
            "Average speed (km/h)": round(metrics["speed"], 1),
            "Traffic": metrics["traffic"],
            "Travel-time ratio": round(metrics["ratio"], 2),
            "Estimated fuel (L)": round(metrics["fuel_l"], 2),
            "Severe segments": metrics["severe"],
            "High segments": metrics["high"],
            "Route score": round(metrics["score"], 2),
        }
    )

comparison_df = pd.DataFrame(comparison_rows)

st.subheader("Interactive map and map layers")

shape_lookup = get_shape_lookup(shape_df)

center = [
    float(work["Start Latitude"].mean()),
    float(work["Start Longitude"].mean()),
]

traffic_map = route_map(
    paths,
    route_information,
    shape_lookup,
    center,
    traffic_df=work,
)

st_folium(
    traffic_map,
    use_container_width=True,
    height=620,
    returned_objects=[],
)

st.caption(
    "Use the layer selector on the map to switch between street, "
    "detailed road, light and dark maps, and toggle traffic and route "
    "layers. Congestion colors are based on workbook travel-time ratios."
)

st.subheader("Route comparison")
st.dataframe(
    comparison_df,
    use_container_width=True,
    hide_index=True,
)

# Traffic details
left, right = st.columns(2)

with left:
    st.subheader("Recommended route traffic details")

    details_df = pd.DataFrame(
        {
            "Metric": [
                "Congestion category",
                "Travel-time ratio",
                "Severe segments",
                "High congestion segments",
                "Moderate congestion segments",
                "Road sections",
            ],
            "Value": [
                best_route["traffic"],
                f"{best_route['ratio']:.2f}",
                best_route["severe"],
                best_route["high"],
                best_route["moderate"],
                len(route_edges(graph, paths[best_index])),
            ],
        }
    )

    st.table(details_df)

with right:
    st.subheader("Network congestion distribution")

    congestion_distribution = (
        work["Travel Time Ratio"]
        .apply(traffic_label)
        .value_counts()
        .reindex(
            ["Low", "Moderate", "High", "Severe"],
            fill_value=0,
        )
    )

    st.bar_chart(congestion_distribution)

# Street-level summary
st.subheader("Street-level traffic summary")

if "Segment ID" not in work.columns:
    work["Segment ID"] = np.arange(len(work))

if "Distance (m)" not in work.columns:
    work["Distance (m)"] = np.nan

if "Average Speed (km/h)" not in work.columns:
    work["Average Speed (km/h)"] = np.nan

street_summary = (
    work.groupby("Street Name", dropna=False)
    .agg(
        Segments=("Segment ID", "count"),
        Distance_km=(
            "Distance (m)",
            lambda values: values.sum() / 1000,
        ),
        Avg_Speed_kmh=("Average Speed (km/h)", "mean"),
        Avg_Travel_Time_Ratio=("Travel Time Ratio", "mean"),
    )
    .reset_index()
)

street_summary["Congestion"] = (
    street_summary["Avg_Travel_Time_Ratio"].apply(traffic_label)
)

street_summary = street_summary.sort_values(
    "Avg_Travel_Time_Ratio",
    ascending=False,
)

st.dataframe(
    street_summary.round(2),
    use_container_width=True,
    hide_index=True,
)

st.subheader("Dataset information")

st.markdown(
    "**Workbook measurements used:** coordinates, time window, "
    "segment distance, average speed, average travel time, "
    "travel-time ratio, street and segment details."
)

st.markdown(
    "**Data limitations:** Traffic volume, traffic density, vehicle "
    "count and measured fuel consumption are not available as measured "
    "fields in this workbook. Fuel is estimated from the mileage entered "
    "in the sidebar. Traffic conditions are not live."
)

st.download_button(
    "Download route comparison CSV",
    data=comparison_df.to_csv(index=False).encode("utf-8"),
    file_name="route_comparison.csv",
    mime="text/csv",
)
