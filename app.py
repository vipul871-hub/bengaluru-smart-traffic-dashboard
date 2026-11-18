import os
import math
from io import BytesIO

import numpy as np
import pandas as pd
import networkx as nx
import streamlit as st
import streamlit.components.v1 as components
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title='Bengaluru Traffic Route Dashboard', page_icon='🚦', layout='wide')

DEFAULT_FILE = 'Bengaluru_Traffic_Analysis_Converted.xlsx'

# Clean, responsive theme. This does not require a map API key.
st.markdown("""
<style>
    .stApp { background: linear-gradient(180deg, #f5f8ff 0%, #f8fafc 260px, #f8fafc 100%); }
    .block-container { max-width: 1580px; padding-top: 1.15rem; padding-bottom: 2rem; }
    [data-testid="stHeader"] { background: rgba(248,250,252,0.88); }
    [data-testid="stSidebar"] { background: #eef4ff; border-right: 1px solid #dbe5f4; }
    [data-testid="stMetric"] { background: #ffffff; border: 1px solid #dce6f3; border-radius: 15px; padding: 14px 16px; box-shadow: 0 4px 14px rgba(25, 54, 94, 0.045); min-width: 0; overflow: visible; }
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] div { color: #52657d; white-space: normal !important; overflow-wrap: anywhere; }
    /* Keep the entire metric value and its unit visible instead of Streamlit's ellipsis. */
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] > div { font-size: clamp(1.35rem, 1.75vw, 1.9rem) !important; line-height: 1.22 !important; white-space: nowrap !important; overflow: visible !important; text-overflow: clip !important; }
    [data-testid="stMetricValue"] { min-width: 0; }
    div[data-testid="stDataFrame"], div[data-testid="stTable"] { border-radius: 12px; overflow: hidden; }
    .stButton > button, .stDownloadButton > button { border-radius: 12px; font-weight: 600; }
    h1 { color: #122642; letter-spacing: -0.035em; }
    h2, h3 { color: #193858; letter-spacing: -0.02em; }
    @media (max-width: 768px) {
        .block-container { padding: 0.75rem 0.65rem 1.25rem; }
        [data-testid="stMetric"] { padding: 9px 10px; }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_workbook(source, filename=""):
    """Read either a multi-sheet Excel workbook or a single CSV dataset."""
    if isinstance(source, str):
        ext = os.path.splitext(source)[1].lower()
        if ext == '.csv':
            df = pd.read_csv(source)
            return {"Segment Time Analysis": df, "Segments": df.copy(), "Shape Points": pd.DataFrame(), "Route Summary": pd.DataFrame()}
        book = pd.ExcelFile(source)
        result = {}
        for sheet in book.sheet_names:
            # Existing project workbook has its headings on row 3; other-city
            # workbooks commonly have headings on row 1. Keep whichever parses best.
            a = pd.read_excel(source, sheet_name=sheet, header=2)
            b = pd.read_excel(source, sheet_name=sheet, header=0)
            result[sheet] = a if len(a.columns) >= len(b.columns) and a.notna().sum().sum() >= b.notna().sum().sum() else b
        return result
    raw = source.getvalue()
    name = (filename or getattr(source, 'name', '')).lower()
    if name.endswith('.csv'):
        df = pd.read_csv(BytesIO(raw))
        return {"Segment Time Analysis": df, "Segments": df.copy(), "Shape Points": pd.DataFrame(), "Route Summary": pd.DataFrame()}
    bio = BytesIO(raw)
    book = pd.ExcelFile(bio)
    result = {}
    for sheet in book.sheet_names:
        a = pd.read_excel(BytesIO(raw), sheet_name=sheet, header=2)
        b = pd.read_excel(BytesIO(raw), sheet_name=sheet, header=0)
        result[sheet] = a if len(a.columns) >= len(b.columns) and a.notna().sum().sum() >= b.notna().sum().sum() else b
    return result


def _norm_col(value):
    return ''.join(ch.lower() for ch in str(value) if ch.isalnum())


def _pick_column(df, aliases):
    lookup = {_norm_col(c): c for c in df.columns}
    for alias in aliases:
        key = _norm_col(alias)
        if key in lookup:
            return lookup[key]
    # Allow common prefixes/suffixes such as avg_speed_kmph or start_latitude_deg.
    for alias in aliases:
        key = _norm_col(alias)
        for candidate, original in lookup.items():
            if candidate.startswith(key) or candidate.endswith(key):
                return original
    return None


def _haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2-lat1)
    dl = np.radians(lon2-lon1)
    a = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2*radius*np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def standardize_dataset(sheets):
    """Convert varied city datasets into the dashboard's common road-edge schema."""
    # Prefer the sheet with the most usable road/network fields.
    candidates = []
    for name, frame in sheets.items():
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        cols = {_norm_col(c) for c in frame.columns}
        score = sum(any(_norm_col(a) in cols for a in aliases) for aliases in [
            ['Start Latitude','start_lat','from_lat'], ['End Latitude','end_lat','to_lat'],
            ['Latitude','lat'], ['Longitude','lon','lng'], ['Street Name','road_name','street'],
            ['Distance (m)','distance_m','length_m'], ['Average Speed (km/h)','speed_kmh','avg_speed']])
        candidates.append((score, name, frame))
    if not candidates:
        raise ValueError('The uploaded file has no readable data rows.')
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, data_sheet_name, raw = candidates[0]
    df = raw.copy()

    aliases = {
        'Start Latitude':['Start Latitude','start_lat','from_lat','origin_lat','source_lat'],
        'Start Longitude':['Start Longitude','start_lon','start_lng','from_lon','from_lng','origin_lon','source_lon'],
        'End Latitude':['End Latitude','end_lat','to_lat','destination_lat','target_lat'],
        'End Longitude':['End Longitude','end_lon','end_lng','to_lon','to_lng','destination_lon','target_lon'],
        'Latitude':['Latitude','lat','point_latitude','y'],
        'Longitude':['Longitude','lon','lng','long','point_longitude','x'],
        'Street Name':['Street Name','street','street_name','road','road_name','roadname','name'],
        'Segment ID':['New Segment ID','Segment ID','segment_id','road_id','link_id','edge_id','id'],
        'Distance (m)':['Distance (m)','distance_m','length_m','segment_length_m','distance'],
        'Average Speed (km/h)':['Average Speed (km/h)','avg_speed_kmh','average_speed','speed_kmh','speed'],
        'Average Travel Time (s)':['Average Travel Time (s)','travel_time_s','travel_time_seconds','avg_travel_time_s','duration_s'],
        'Travel Time Ratio':['Travel Time Ratio','travel_time_ratio','congestion_ratio','tt_ratio'],
        'Time Window':['Time Window','time_window','period','time_period','traffic_period'],
        'Sequence':['Point Sequence','sequence','seq','order','point_order','timestamp','time']}
    found = {target:_pick_column(df, opts) for target,opts in aliases.items()}
    # Direct edge coordinates are ideal. For point datasets, create edges between
    # consecutive points within each road/group, preserving a supplied sequence.
    if found['Start Latitude'] and found['Start Longitude'] and found['End Latitude'] and found['End Longitude']:
        out = pd.DataFrame({k:df[v] for k,v in found.items() if v is not None and k not in ['Latitude','Longitude','Sequence']})
    elif found['Latitude'] and found['Longitude']:
        work = df.copy()
        latc, lonc = found['Latitude'], found['Longitude']
        groupc = found['Street Name']
        if found['Sequence']:
            work = work.sort_values(found['Sequence'], kind='stable')
        groups = work.groupby(groupc, dropna=False, sort=False) if groupc else [("Road network", work)]
        pieces=[]
        for _, g in groups:
            g=g.copy()
            g['_next_lat']=pd.to_numeric(g[latc],errors='coerce').shift(-1)
            g['_next_lon']=pd.to_numeric(g[lonc],errors='coerce').shift(-1)
            g['_this_lat']=pd.to_numeric(g[latc],errors='coerce')
            g['_this_lon']=pd.to_numeric(g[lonc],errors='coerce')
            g=g.dropna(subset=['_this_lat','_this_lon','_next_lat','_next_lon'])
            if not g.empty:
                pieces.append(pd.DataFrame({'Start Latitude':g['_this_lat'],'Start Longitude':g['_this_lon'],
                    'End Latitude':g['_next_lat'],'End Longitude':g['_next_lon'],
                    'Street Name':g[found['Street Name']] if found['Street Name'] else 'Road network',
                    'Segment ID':g[found['Segment ID']] if found['Segment ID'] else range(len(g)),
                    **({'Distance (m)':g[found['Distance (m']]} if False else {})}))
        if not pieces:
            raise ValueError('This file contains latitude/longitude points, but no consecutive point pairs. Add a sequence/order column and at least two points per road, or upload a road-segment dataset with start/end coordinates.')
        out=pd.concat(pieces,ignore_index=True)
        for target in ['Distance (m)','Average Speed (km/h)','Average Travel Time (s)','Travel Time Ratio','Time Window']:
            src=found.get(target)
            if src and src in work.columns:
                # Keep point-row attributes aligned where possible.
                vals=[]
                for _,g in groups:
                    vals.extend(g[src].iloc[:-1].tolist())
                if len(vals)==len(out): out[target]=vals
    else:
        raise ValueError('Could not identify road coordinates. Provide either Start Latitude/Start Longitude/End Latitude/End Longitude, or Latitude/Longitude points with a road ID/name and sequence column.')

    # Ensure the required canonical fields exist and are numeric.
    for col in ['Start Latitude','Start Longitude','End Latitude','End Longitude']:
        if col not in out.columns: raise ValueError(f'Missing required coordinate field: {col}')
        out[col]=pd.to_numeric(out[col],errors='coerce')
    out=out.dropna(subset=['Start Latitude','Start Longitude','End Latitude','End Longitude']).copy()
    if out.empty: raise ValueError('No valid coordinate pairs were found in the uploaded dataset.')
    if 'Street Name' not in out: out['Street Name']='Road segment'
    out['Street Name']=out['Street Name'].fillna('Unnamed Road').astype(str)
    if 'Segment ID' not in out: out['Segment ID']=np.arange(1,len(out)+1)
    if 'Distance (m)' in out: out['Distance (m)']=pd.to_numeric(out['Distance (m)'],errors='coerce')
    else: out['Distance (m)']=np.nan
    missing_dist=out['Distance (m)'].isna() | (out['Distance (m)']<=0)
    if missing_dist.any():
        out.loc[missing_dist,'Distance (m)']=_haversine_km(
            out.loc[missing_dist,'Start Latitude'].to_numpy(),out.loc[missing_dist,'Start Longitude'].to_numpy(),
            out.loc[missing_dist,'End Latitude'].to_numpy(),out.loc[missing_dist,'End Longitude'].to_numpy())*1000
    if 'Average Speed (km/h)' in out: out['Average Speed (km/h)']=pd.to_numeric(out['Average Speed (km/h)'],errors='coerce')
    else: out['Average Speed (km/h)']=np.nan
    if 'Average Travel Time (s)' in out: out['Average Travel Time (s)']=pd.to_numeric(out['Average Travel Time (s)'],errors='coerce')
    else: out['Average Travel Time (s)']=np.nan
    if 'Travel Time Ratio' in out: out['Travel Time Ratio']=pd.to_numeric(out['Travel Time Ratio'],errors='coerce')
    else: out['Travel Time Ratio']=np.nan
    # Convert common minute-based travel-time columns if detected separately.
    if out['Average Travel Time (s)'].isna().all():
        mins=_pick_column(raw,['travel_time_min','average_travel_time_min','duration_min','travel_time_minutes'])
        if mins: out['Average Travel Time (s)']=pd.to_numeric(raw[mins],errors='coerce').reindex(out.index)*60
    # Derive missing travel time from distance and speed; never fabricate traffic ratios.
    derived=(out['Distance (m)']/1000)/out['Average Speed (km/h)'].replace(0,np.nan)*3600
    out['Average Travel Time (s)']=out['Average Travel Time (s)'].where(out['Average Travel Time (s)'].notna(),derived)
    out['Average Travel Time (s)']=out['Average Travel Time (s)'].fillna(60.0)
    out['Average Speed (km/h)']=out['Average Speed (km/h)'].where(out['Average Speed (km/h)'].notna(),(out['Distance (m)']/1000)/(out['Average Travel Time (s)']/3600))
    out['Travel Time Ratio']=out['Travel Time Ratio'].fillna(1.0)
    if 'Time Window' not in out: out['Time Window']='All data'
    out['Time Window']=out['Time Window'].fillna('All data').astype(str)
    out['Average Speed (km/h)']=out['Average Speed (km/h)'].replace([np.inf,-np.inf],np.nan)
    # Keep a consistent shape frame if the workbook provides point geometry.
    shape=pd.DataFrame()
    for name, frame in sheets.items():
        if _norm_col(name) in ['shapepoints','geometry','roadgeometry']:
            sc={_norm_col(c):c for c in frame.columns}
            lat=next((sc[k] for k in ['latitude','lat'] if k in sc),None)
            lon=next((sc[k] for k in ['longitude','lon','lng'] if k in sc),None)
            sid=next((sc[k] for k in ['newsegmentid','segmentid','roadid','linkid'] if k in sc),None)
            seq=next((sc[k] for k in ['pointsequence','sequence','seq','order'] if k in sc),None)
            if lat and lon and sid:
                shape=pd.DataFrame({'New Segment ID':frame[sid],'Latitude':frame[lat],'Longitude':frame[lon],
                                    'Point Sequence':frame[seq] if seq else np.arange(len(frame))})
            break
    return out.reset_index(drop=True), shape, data_sheet_name


def clean_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def traffic_label(ratio):
    if pd.isna(ratio): return 'Unknown'
    if ratio <= 1.15: return 'Low'
    if ratio <= 1.35: return 'Moderate'
    if ratio <= 1.75: return 'High'
    return 'Severe'


def build_graph(df):
    G = nx.DiGraph()
    for _, r in df.iterrows():
        if pd.isna(r.get('Start Latitude')) or pd.isna(r.get('Start Longitude')) or pd.isna(r.get('End Latitude')) or pd.isna(r.get('End Longitude')):
            continue
        a = (round(float(r['Start Latitude']), 7), round(float(r['Start Longitude']), 7))
        b = (round(float(r['End Latitude']), 7), round(float(r['End Longitude']), 7))
        if a == b: continue
        dist_value = pd.to_numeric(r.get('Distance (m)', np.nan), errors='coerce')
        time_value = pd.to_numeric(r.get('Average Travel Time (s)', np.nan), errors='coerce')
        ratio_value = pd.to_numeric(r.get('Travel Time Ratio', 1), errors='coerce')
        dist_km = max(float(dist_value) / 1000, 0.001) if pd.notna(dist_value) else max(_haversine_km(a[0],a[1],b[0],b[1]),0.001)
        tmin = max(float(time_value) / 60, 0.01) if pd.notna(time_value) else 1.0
        ratio = float(ratio_value) if pd.notna(ratio_value) else 1.0
        data = r.to_dict()
        data.update({'distance_km': dist_km, 'time_min': tmin, 'ratio': ratio})
        G.add_edge(a, b, **data)
        G.add_edge(b, a, **data, _reversed=True)
    return G


def edge_cost(data, wt, wd, wc):
    return wt * data['time_min'] + wd * data['distance_km'] + wc * max(0, data['ratio'] - 1)


def route_edges(G, path):
    edges = []
    for a, b in zip(path[:-1], path[1:]):
        data = G.get_edge_data(a, b)
        if not data: return []
        edges.append(data)
    return edges


def route_metrics(path, edges, fuel_kmpl):
    distance = sum(e['distance_km'] for e in edges)
    time = sum(e['time_min'] for e in edges)
    ratio_num = sum(e['ratio'] * e['time_min'] for e in edges)
    ratio = ratio_num / time if time else np.nan
    speed = distance / (time / 60) if time else np.nan
    severe = sum(traffic_label(e['ratio']) == 'Severe' for e in edges)
    high = sum(traffic_label(e['ratio']) == 'High' for e in edges)
    moderate = sum(traffic_label(e['ratio']) == 'Moderate' for e in edges)
    streets = list(dict.fromkeys(str(e.get('Street Name','Unnamed Road')) for e in edges))
    fuel = distance / fuel_kmpl if fuel_kmpl > 0 else np.nan
    return dict(distance_km=distance, time_min=time, ratio=ratio, speed=speed,
                traffic=traffic_label(ratio), severe=severe, high=high,
                moderate=moderate, streets=streets, fuel_l=fuel)


def nearest_nodes(G, points, n=10):
    if not points: return []
    out=[]
    for node in G.nodes:
        d=min((node[0]-p[0])**2+(node[1]-p[1])**2 for p in points)
        out.append((d,node))
    return [node for _,node in sorted(out)[:n]]


def get_shape_lookup(shape_df):
    if shape_df is None or shape_df.empty or not {'New Segment ID','Latitude','Longitude'}.issubset(shape_df.columns):
        return {}
    shape_df = clean_num(shape_df.copy(), ['Latitude','Longitude','Point Sequence'])
    lookup={}
    for sid,g in shape_df.groupby('New Segment ID', dropna=False):
        pts=g.sort_values('Point Sequence')[['Latitude','Longitude']].dropna().values.tolist()
        if len(pts)>=2: lookup[str(sid)]=pts
    return lookup


def route_map(route_paths, route_info, shape_lookup, center, traffic_df=None):
    # Base maps: key-free CARTO raster tiles. Keep the layers mutually
    # exclusive so the user can switch basemaps without hiding route overlays.
    m = folium.Map(
        location=center,
        zoom_start=13,
        control_scale=True,
        tiles=None,
    )

    # 1. Standard OpenStreetMap
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="Street map",
        overlay=False,
        control=True,
        show=True,
        max_zoom=19,
    ).add_to(m)

    carto_attribution = '&copy; OpenStreetMap contributors &copy; CARTO'

    # 2. Light map — detailed roads and labels
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        name="Light map — detailed",
        attr=carto_attribution,
        subdomains="abcd",
        overlay=False,
        control=True,
        show=False,
        max_zoom=20,
    ).add_to(m)

    # 3. Dark map — navigation-style roads and labels
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        name="Dark map — navigation",
        attr=carto_attribution,
        subdomains="abcd",
        overlay=False,
        control=True,
        show=False,
        max_zoom=20,
    ).add_to(m)

    # Traffic layer: each segment is colored using the workbook's travel-time ratio.
    if traffic_df is not None:
        traffic_fg = folium.FeatureGroup(name="Traffic congestion layer", show=True)
        for _, row in traffic_df.iterrows():
            try:
                a = [float(row['Start Latitude']), float(row['Start Longitude'])]
                b = [float(row['End Latitude']), float(row['End Longitude'])]
                ratio = float(row.get('Travel Time Ratio', 1.0))
                label = traffic_label(ratio)
                color = {'Low': '#16a34a', 'Moderate': '#eab308', 'High': '#f97316', 'Severe': '#dc2626'}.get(label, '#64748b')
                sid = str(row.get('New Segment ID', row.get('Segment ID', '')))
                pts = shape_lookup.get(sid)
                coords = pts if pts and len(pts) >= 2 else [a, b]
                street = str(row.get('Street Name', 'Unknown road'))
                speed = row.get('Average Speed (km/h)', np.nan)
                tip = f"{street} | {label} | travel-time ratio {ratio:.2f}× | speed {speed:.1f} km/h" if pd.notna(speed) else f"{street} | {label} | travel-time ratio {ratio:.2f}×"
                folium.PolyLine(coords, color=color, weight=4, opacity=0.72, tooltip=tip).add_to(traffic_fg)
            except (TypeError, ValueError, KeyError):
                continue
        traffic_fg.add_to(m)

    route_colors = ['#059669', '#2563eb', '#9333ea', '#f97316', '#db2777', '#0891b2']
    for i, (path, info) in enumerate(zip(route_paths, route_info)):
        fg = folium.FeatureGroup(name=f"Route {i+1} — {info['traffic']}", show=True)
        coords = []
        for e in route_edges(info['_graph'], path):
            sid = str(e.get('New Segment ID', ''))
            pts = shape_lookup.get(sid)
            if pts:
                # Reverse geometry when this edge traverses the source segment backwards.
                if e.get('_reversed'):
                    pts = list(reversed(pts))
                coords.extend(pts if not coords else pts[1:])
            else:
                segment_coords = [[e['Start Latitude'], e['Start Longitude']], [e['End Latitude'], e['End Longitude']]]
                if e.get('_reversed'):
                    segment_coords.reverse()
                coords.extend(segment_coords if not coords else segment_coords[1:])
        if coords:
            folium.PolyLine(coords, color=route_colors[i % len(route_colors)], weight=8 if i == 0 else 5,
                            opacity=0.95 if i == 0 else 0.78,
                            tooltip=f"Route {i+1}: {info['distance_km']:.2f} km | {info['time_min']:.1f} min | {info['traffic']}").add_to(fg)
        fg.add_to(m)

    if route_paths:
        folium.Marker(route_paths[0][0], tooltip='Selected start', icon=folium.Icon(color='green', icon='play')).add_to(m)
        folium.Marker(route_paths[0][-1], tooltip='Destination', icon=folium.Icon(color='red', icon='flag')).add_to(m)
    folium.LayerControl(collapsed=False, position="topright").add_to(m)

    return m

# The uploaded filename becomes the displayed city/dataset label.
with st.sidebar:
    st.header('1. Upload city dataset')
    uploaded=st.file_uploader(
        'Choose any city road-network file',
        type=['xlsx','xls','csv'],
        help='Upload a road-segment file with start/end coordinates, or ordered road points with latitude, longitude and sequence.'
    )
    source=uploaded if uploaded is not None else (DEFAULT_FILE if os.path.exists(DEFAULT_FILE) else None)
    if source is None:
        st.info('Upload an Excel (.xlsx/.xls) or CSV road-network dataset to start. No GitHub change is needed when uploading a new dataset after this upload feature is deployed.')
        st.stop()
    st.success('Dataset file selected')

city_label = os.path.splitext(getattr(uploaded, 'name', 'Bengaluru Traffic Dataset') if uploaded is not None else 'Bengaluru')[0].replace('_',' ').replace('-',' ').title()
st.title(f'🚦 {city_label} Traffic Dashboard')
st.caption('Upload a different city dataset here whenever you need it. The dashboard detects supported road fields and builds the map and route analysis from the uploaded file.')

try:
    sheets=load_workbook(source, getattr(uploaded, 'name', ''))
    sta, shape, detected_sheet = standardize_dataset(sheets)
except Exception as e:
    st.error(f'Could not prepare this city dataset: {e}')
    st.info('Supported layouts: road segments with start/end latitude-longitude columns, or ordered latitude-longitude points grouped by road/street. Excel and CSV are supported.')
    st.stop()

segments=sta.copy()
route_summary=pd.DataFrame()
sta=clean_num(sta,['Start Latitude','Start Longitude','End Latitude','End Longitude','Distance (m)','Average Speed (km/h)','Average Travel Time (s)','Travel Time Ratio'])

windows=sorted(sta['Time Window'].dropna().astype(str).unique()) or ['All data']
with st.sidebar:
    st.header('2. Route settings')
    window=st.selectbox('Traffic time window', windows)
    departure_time=st.time_input('Planned departure time', value=pd.to_datetime('09:00').time())
    n_routes=st.slider('Number of route alternatives',2,6,3)
    fuel_kmpl=st.number_input('Vehicle fuel economy (km/L)', min_value=1.0, max_value=100.0, value=15.0, step=0.5)
    st.caption('Fuel is an estimate because measured fuel-consumption data is not present in this workbook.')
    st.subheader('Route optimization weights')
    wt=st.slider('Travel time weight',0.0,5.0,3.0,0.1)
    wd=st.slider('Distance weight',0.0,5.0,1.0,0.1)
    wc=st.slider('Congestion weight',0.0,5.0,2.0,0.1)

work=sta[sta['Time Window'].astype(str)==window].copy()
if work.empty: st.error('No records found for the selected time window.'); st.stop()
G=build_graph(work)
if G.number_of_edges()==0: st.error('No usable road segments were found.'); st.stop()

# Use segment endpoints as user-selectable points.
node_options=list(G.nodes)
node_labels={n:f"{n[0]:.6f}, {n[1]:.6f}" for n in node_options}

# Street filters make point selection easier.
streets=sorted(work['Street Name'].dropna().astype(str).unique()) or ['Road segment']
with st.sidebar:
    st.header('3. Start / destination')
    start_street=st.selectbox('Start street', streets, index=0)
    end_street=st.selectbox('Destination street', streets, index=min(1,len(streets)-1))

start_pts=work[work['Street Name'].astype(str)==start_street][['Start Latitude','Start Longitude','End Latitude','End Longitude']].dropna().values.tolist()
end_pts=work[work['Street Name'].astype(str)==end_street][['Start Latitude','Start Longitude','End Latitude','End Longitude']].dropna().values.tolist()
start_candidates=nearest_nodes(G, [(float(p[0]),float(p[1])) for row in start_pts for p in [row[:2],row[2:]]], 40)
end_candidates=nearest_nodes(G, [(float(p[0]),float(p[1])) for row in end_pts for p in [row[:2],row[2:]]], 40)

if not start_candidates or not end_candidates: st.error('Could not find selectable start/destination points.'); st.stop()
with st.sidebar:
    start_label=st.selectbox('Start point', [node_labels[n] for n in start_candidates])
    end_label=st.selectbox('Destination point', [node_labels[n] for n in end_candidates], index=min(1,len(end_candidates)-1))
start=next(n for n in start_candidates if node_labels[n]==start_label)
end=next(n for n in end_candidates if node_labels[n]==end_label)

# K shortest simple paths, then score them using the selected weights.
for u, v, data in G.edges(data=True):
    data['route_weight'] = edge_cost(data, wt, wd, wc)

try:
    gen=nx.shortest_simple_paths(G,start,end,weight='route_weight')
    paths=[]
    seen=set()
    for p in gen:
        key=tuple(p)
        if key in seen: continue
        seen.add(key); paths.append(p)
        if len(paths)>=n_routes: break
except Exception:
    paths=[]

if not paths:
    st.error('No route could be found between the selected points in this time window. Try different streets/points.')
    st.stop()

infos=[]
for p in paths:
    es=route_edges(G,p)
    met=route_metrics(p,es,fuel_kmpl)
    met['_graph']=G
    met['score']=sum(edge_cost(e,wt,wd,wc) for e in es)
    infos.append(met)

best_idx=int(np.argmin([x['score'] for x in infos]))
best=infos[best_idx]

# Compact voice assistant panel. The voice still attempts to start after the map
# has rendered; the small panel provides a visible fallback/replay control.
welcome_html = r"""
<!doctype html>
<html><head><meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  * { box-sizing: border-box; }
  body { margin:0; font-family:Arial,sans-serif; background:transparent; color:#17324f; }
  .voice-card { display:flex; align-items:center; justify-content:space-between; gap:10px;
    padding:9px 11px; border:1px solid #d8e4f2; border-radius:12px;
    background:linear-gradient(135deg,#ffffff 0%,#f1f6ff 100%);
    box-shadow:0 3px 12px rgba(28,55,90,.08); }
  .voice-copy { min-width:0; }
  .voice-title { font-size:13px; font-weight:700; line-height:1.25; }
  .voice-sub { font-size:11px; color:#64748b; margin-top:2px; }
  button { flex:0 0 auto; border:0; border-radius:9px; padding:8px 11px;
    background:#173f70; color:#fff; font-size:12px; font-weight:600; cursor:pointer; }
  button:focus { outline:2px solid #60a5fa; outline-offset:2px; }
  @media(max-width:480px) { .voice-card { padding:8px; } .voice-sub { display:none; } button { padding:8px; } }
</style></head>
<body>
<div class="voice-card">
  <div class="voice-copy">
    <div class="voice-title">🔊 Bengaluru AI Navigation</div>
    <div class="voice-sub" id="voiceStatus">English cinematic voice · Welcome assistant</div>
  </div>
  <button id="voiceButton" type="button">Replay voice</button>
</div>
<script>
(function () {
  const greeting = "Good day. Welcome to the City Traffic Map. Navigation systems are online. Please select your starting point and destination. I will help you compare routes, estimated travel times, and congestion levels. Have a smooth and safe journey.";
  const status = document.getElementById('voiceStatus');
  const button = document.getElementById('voiceButton');
  let hasStarted = false;
  let attemptCount = 0;
  const maxAttempts = 3;

  function chooseVoice(voices) {
    return voices.find(v => /^en-GB$/i.test(v.lang) && /natural|enhanced|premium/i.test(v.name)) ||
      voices.find(v => /^en-GB$/i.test(v.lang)) ||
      voices.find(v => /^en-IN$/i.test(v.lang)) ||
      voices.find(v => /^en/i.test(v.lang)) || null;
  }

  function speakWelcome(isManual) {
    if (!('speechSynthesis' in window) || !('SpeechSynthesisUtterance' in window)) {
      status.textContent = 'Speech is not supported by this browser';
      return;
    }
    if (hasStarted && !isManual) return;
    try {
      const synth = window.speechSynthesis;
      synth.cancel();
      const utterance = new SpeechSynthesisUtterance(greeting);
      const voice = chooseVoice(synth.getVoices());
      if (voice) utterance.voice = voice;
      utterance.lang = voice ? voice.lang : 'en-GB';
      utterance.rate = 0.88;
      utterance.pitch = 0.78;
      utterance.volume = 1;
      utterance.onstart = () => {
        hasStarted = true;
        status.textContent = 'Welcome message playing';
      };
      utterance.onend = () => { status.textContent = 'English AI voice · Ready'; };
      utterance.onerror = (event) => {
        status.textContent = 'Browser may require a tap on Replay voice';
      };
      synth.speak(utterance);
      if (isManual) status.textContent = 'Starting welcome message…';
    } catch (e) {
      status.textContent = 'Tap Replay voice if automatic audio is blocked';
    }
  }

  button.addEventListener('click', () => speakWelcome(true));

  // The component is rendered immediately after st_folium. Try only after
  // the map/dashboard has had time to paint, and retry briefly if voices load late.
  function autoStart() {
    if (hasStarted || attemptCount >= maxAttempts) return;
    attemptCount += 1;
    speakWelcome(false);
  }
  window.addEventListener('load', () => {
    window.setTimeout(autoStart, 1800);
    window.setTimeout(autoStart, 3500);
  });
  if (document.readyState === 'complete') {
    window.setTimeout(autoStart, 1800);
  }
  if ('speechSynthesis' in window) {
    window.speechSynthesis.onvoiceschanged = () => {
      if (!hasStarted) window.setTimeout(autoStart, 500);
    };
  }
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden && !hasStarted && attemptCount < maxAttempts) {
      window.setTimeout(autoStart, 900);
    }
  });
})();
</script>
</body></html>
"""

st.subheader(f'🗺️ Interactive {city_label} map')
shape_lookup = get_shape_lookup(shape)
center = [float(work['Start Latitude'].mean()), float(work['Start Longitude'].mean())]
traffic_map = route_map(paths, infos, shape_lookup, center, traffic_df=work)
st_folium(traffic_map, use_container_width=True, height=620, returned_objects=[])
st.caption('Use the top-right layer control to switch Street, Light, and Dark base maps. Select exactly one base map; traffic congestion and route lines remain independent overlays. Light and Dark maps use key-free Esri ArcGIS tiles. Traffic colors use the workbook travel-time ratio; they are not live traffic feeds.')
# Small, unobtrusive voice assistant panel below the map; auto-welcome is attempted
# after the map renders, and Replay voice is available if browser autoplay is blocked.
components.html(welcome_html, height=66, scrolling=False)

# KPIs — use two rows of three cards so values and units remain readable on desktop and mobile.
st.subheader('Optimal route')
kpi_rows = [
    [
        ('ETA', f"{best['time_min']:.1f} min"),
        ('Distance', f"{best['distance_km']:.2f} km"),
        ('Average speed', f"{best['speed']:.1f} km/h"),
    ],
    [
        ('Traffic condition', str(best['traffic'])),
        ('Travel-time ratio (×)', f"{best['ratio']:.2f}×"),
        ('Estimated fuel', f"{best['fuel_l']:.2f} L"),
    ],
]
for row_items in kpi_rows:
    cols = st.columns(3, gap="medium")
    for col, (label, value) in zip(cols, row_items):
        with col:
            st.metric(label, value)

eta_minutes = int(round(best['time_min']))
eta_clock = (pd.Timestamp.combine(pd.Timestamp.today().date(), departure_time) + pd.Timedelta(minutes=eta_minutes)).strftime('%I:%M %p')
st.caption('Unit guide: ETA and route duration are shown in minutes (min); distance in kilometres (km); speed in kilometres per hour (km/h); fuel in litres (L). Travel-time ratio is a relative multiplier (×), not a time duration.')
st.info(f"Recommended Route {best_idx+1}: {' → '.join(best['streets'][:8])}{' → …' if len(best['streets'])>8 else ''}\n\nEstimated arrival: {eta_clock} (using the selected time-window traffic estimate).")

# Route table
rows=[]
for i,m in enumerate(infos,1):
    rows.append({'Route':f'Route {i}','Recommended':'✓' if i-1==best_idx else '',
                 'Distance (km)':round(m['distance_km'],2),'ETA (min)':round(m['time_min'],1),
                 'Avg Speed (km/h)':round(m['speed'],1),'Traffic':m['traffic'],
                 'Travel Time Ratio (×)':f"{m['ratio']:.2f}×",'Fuel (L)':round(m['fuel_l'],2),
                 'Severe Segments':m['severe'],'High Segments':m['high'],'Route Score':round(m['score'],2)})
comp=pd.DataFrame(rows)
st.subheader('Multiple route comparison')
st.dataframe(comp,use_container_width=True,hide_index=True)
st.caption('Travel-time ratio unit: × (times). This is a dimensionless multiplier comparing segment travel time with its reference/free-flow travel time; it is not measured in minutes. For example, 1.93× means approximately 1.93 times the reference travel time.')

# Traffic details and analysis
c1,c2=st.columns(2)
with c1:
    st.subheader('Selected route traffic details')
    detail=pd.DataFrame({
        'Metric':['Congestion category','Travel-time ratio (×)','Severe segments','High segments','Moderate segments','Road sections'],
        'Value':[best['traffic'],f"{best['ratio']:.2f}×",best['severe'],best['high'],best['moderate'],len(route_edges(G,paths[best_idx]))]})
    st.table(detail)
    st.caption('Travel-time ratio is displayed in × (times), not min. ETA and travel duration use minutes (min).')
with c2:
    st.subheader('Congestion distribution')
    dist=work['Travel Time Ratio'].apply(traffic_label).value_counts().reindex(['Low','Moderate','High','Severe']).fillna(0)
    st.bar_chart(dist)

st.subheader('Street-level traffic summary')
street_summary=(work.groupby('Street Name',dropna=False)
                .agg(Segments=('Segment ID','count'),
                     Distance_km=('Distance (m)',lambda s:s.sum()/1000),
                     Avg_Speed_kmh=('Average Speed (km/h)','mean'),
                     Avg_Travel_Time_Ratio=('Travel Time Ratio','mean'))
                .reset_index())
street_summary['Congestion']=street_summary['Avg_Travel_Time_Ratio'].apply(traffic_label)
street_summary=street_summary.sort_values('Avg_Travel_Time_Ratio',ascending=False)
street_summary=street_summary.rename(columns={'Avg_Travel_Time_Ratio':'Avg Travel-time ratio (×)'})
street_summary_display=street_summary.round(2)
street_summary_display['Avg Travel-time ratio (×)']=street_summary_display['Avg Travel-time ratio (×)'].map(lambda v: f"{v:.2f}×" if pd.notna(v) else '—')
st.dataframe(street_summary_display,use_container_width=True,hide_index=True)
st.caption('Street-level average travel-time ratio is expressed as × (times); distance is km and average speed is km/h.')

st.subheader('Dataset parameters used by this dashboard')
st.markdown('**Detected dataset fields:** road coordinates, street or road ID, distance, speed, travel time, travel-time ratio and time window where supplied. Missing distance is calculated from coordinates; missing travel time can be estimated from distance and speed. Missing congestion ratio defaults to 1.0 (baseline/unknown), not a measured congestion value.')
st.markdown('**Important:** Traffic volume, density, vehicle count and fuel consumption are shown only when present in your dataset. Fuel displayed here is an estimate based on the fuel-economy value entered in the sidebar. This dashboard uses uploaded historical data, not live traffic feeds.')

# Download comparison CSV
st.download_button('Download route comparison CSV',comp.to_csv(index=False).encode('utf-8'),'route_comparison.csv','text/csv')
