import os
import math
from io import BytesIO

import numpy as np
import pandas as pd
import networkx as nx
import streamlit as st
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title='Bengaluru Traffic Route Dashboard', page_icon='🚦', layout='wide')

DEFAULT_FILE = 'Bengaluru_Traffic_Analysis_Converted.xlsx'

@st.cache_data(show_spinner=False)
def load_workbook(source):
    if isinstance(source, str):
        return {s: pd.read_excel(source, sheet_name=s, header=2) for s in pd.ExcelFile(source).sheet_names}
    raw = source.getvalue()
    return {s: pd.read_excel(BytesIO(raw), sheet_name=s, header=2) for s in pd.ExcelFile(BytesIO(raw)).sheet_names}


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
        if pd.isna(r.get('Start Latitude')) or pd.isna(r.get('End Latitude')):
            continue
        a = (round(float(r['Start Latitude']), 7), round(float(r['Start Longitude']), 7))
        b = (round(float(r['End Latitude']), 7), round(float(r['End Longitude']), 7))
        if a == b: continue
        dist_km = max(float(r.get('Distance (m)', 0) or 0) / 1000, 0.001)
        tmin = max(float(r.get('Average Travel Time (s)', 0) or 0) / 60, 0.01)
        ratio = float(r.get('Travel Time Ratio', 1) or 1)
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
    shape_df = clean_num(shape_df.copy(), ['Latitude','Longitude','Point Sequence'])
    lookup={}
    for sid,g in shape_df.groupby('New Segment ID', dropna=False):
        pts=g.sort_values('Point Sequence')[['Latitude','Longitude']].dropna().values.tolist()
        if len(pts)>=2: lookup[str(sid)]=pts
    return lookup


def route_map(route_paths, route_info, shape_lookup, center, traffic_df=None):
    m = folium.Map(location=center, zoom_start=13, control_scale=True, tiles=None)
    folium.TileLayer("OpenStreetMap", name="Street map", control=True).add_to(m)
    folium.TileLayer("CartoDB positron", name="Light map", control=True).add_to(m)
    folium.TileLayer("CartoDB dark_matter", name="Dark map", control=True).add_to(m)

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
                tip = f"{street} | {label} | travel-time ratio {ratio:.2f} | speed {speed:.1f} km/h" if pd.notna(speed) else f"{street} | {label} | travel-time ratio {ratio:.2f}"
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
    folium.LayerControl(collapsed=False).add_to(m)
    return m

st.title('🚦 Bengaluru Traffic & Optimal Route Dashboard')
st.caption('Route planning is calculated from the road-segment network in the uploaded Excel workbook.')

with st.sidebar:
    st.header('1. Dataset')
    uploaded=st.file_uploader('Upload Excel file', type=['xlsx'])
    source=uploaded if uploaded is not None else (DEFAULT_FILE if os.path.exists(DEFAULT_FILE) else None)
    if source is None:
        st.error('Upload Bengaluru_Traffic_Analysis_Converted.xlsx to continue.')
        st.stop()
    st.success('Dataset loaded')

try:
    sheets=load_workbook(source)
except Exception as e:
    st.error(f'Could not read the Excel workbook: {e}')
    st.stop()

sta=sheets['Segment Time Analysis'].copy()
segments=sheets['Segments'].copy()
shape=sheets['Shape Points'].copy()
route_summary=sheets['Route Summary'].copy()
sta=clean_num(sta,['Start Latitude','Start Longitude','End Latitude','End Longitude','Distance (m)','Harmonic Avg Speed (km/h)','Average Speed (km/h)','Average Travel Time (s)','Travel Time Ratio','Speed Reduction (%)'])
segments=clean_num(segments,['Start Latitude','Start Longitude','End Latitude','End Longitude','Distance (m)'])

windows=sorted(sta['Time Window'].dropna().astype(str).unique())
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
streets=sorted(work['Street Name'].dropna().astype(str).unique())
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

# KPIs
st.subheader('Optimal route')
cols=st.columns(6)
labels=[('ETA',f"{best['time_min']:.1f} min"),('Distance',f"{best['distance_km']:.2f} km"),('Avg speed',f"{best['speed']:.1f} km/h"),('Traffic',best['traffic']),('Travel-time ratio',f"{best['ratio']:.2f}"),('Fuel estimate',f"{best['fuel_l']:.2f} L")]
for c,(lab,val) in zip(cols,labels): c.metric(lab,val)

eta_minutes = int(round(best['time_min']))
eta_clock = (pd.Timestamp.combine(pd.Timestamp.today().date(), departure_time) + pd.Timedelta(minutes=eta_minutes)).strftime('%I:%M %p')
st.info(f"Recommended Route {best_idx+1}: {' → '.join(best['streets'][:8])}{' → …' if len(best['streets'])>8 else ''}\n\nEstimated arrival: {eta_clock} (using the selected time-window traffic estimate).")

# Route table
rows=[]
for i,m in enumerate(infos,1):
    rows.append({'Route':f'Route {i}','Recommended':'✓' if i-1==best_idx else '',
                 'Distance (km)':round(m['distance_km'],2),'ETA (min)':round(m['time_min'],1),
                 'Avg Speed (km/h)':round(m['speed'],1),'Traffic':m['traffic'],
                 'Travel Time Ratio':round(m['ratio'],2),'Fuel (L)':round(m['fuel_l'],2),
                 'Severe Segments':m['severe'],'High Segments':m['high'],'Route Score':round(m['score'],2)})
comp=pd.DataFrame(rows)
st.subheader('Interactive route map and map layers')
shape_lookup = get_shape_lookup(shape)
center = [float(work['Start Latitude'].mean()), float(work['Start Longitude'].mean())]
traffic_map = route_map(paths, infos, shape_lookup, center, traffic_df=work)
st_folium(traffic_map, use_container_width=True, height=620, returned_objects=[])
st.caption('Use the layer control on the map to toggle traffic congestion, individual routes, and base-map styles. Traffic colors are based on travel-time ratio, not live traffic feeds.')

st.subheader('Multiple route comparison')
st.dataframe(comp,use_container_width=True,hide_index=True)

# Traffic details and analysis
c1,c2=st.columns(2)
with c1:
    st.subheader('Selected route traffic details')
    detail=pd.DataFrame({
        'Metric':['Congestion category','Travel Time Ratio','Severe segments','High segments','Moderate segments','Road sections'],
        'Value':[best['traffic'],f"{best['ratio']:.2f}",best['severe'],best['high'],best['moderate'],len(route_edges(G,paths[best_idx]))]})
    st.table(detail)
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
st.dataframe(street_summary.round(2),use_container_width=True,hide_index=True)

st.subheader('Dataset parameters used by this dashboard')
st.markdown('**Available in workbook:** latitude, longitude, time window, segment distance, speed, travel time, travel-time ratio, speed reduction, congestion category, street/segment information and route percentiles.')
st.markdown('**Not available as measured columns:** Traffic Volume, Traffic Density, Vehicle Count and Fuel Consumption. The dashboard therefore does not invent those measurements; fuel is estimated from the fuel-economy value entered in the sidebar.')

# Download comparison CSV
st.download_button('Download route comparison CSV',comp.to_csv(index=False).encode('utf-8'),'route_comparison.csv','text/csv')
