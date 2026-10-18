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
