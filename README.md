# Bengaluru Smart Traffic Dashboard — Render deployment

A Streamlit traffic and route-planning dashboard built from `Bengaluru_Traffic_Analysis_Converted.xlsx`.

## Included dashboard features

- Start street and destination street selection, plus selectable network endpoint coordinates.
- Route scoring based on travel time, distance and the workbook's travel-time ratio.
- Route comparison table, route metrics, ETA estimate and fuel-use estimate from user-entered km/L.
- Interactive map layers: street map, light map, dark map, congestion segments, and individual computed routes.
- Street-level summary and congestion distribution for the selected time window.
- Upload control to use a different compatible workbook.

## Important data notes

The workbook contains road geometry, distance, speed, average travel time, travel-time ratio, congestion-related fields, and two time windows. It does not provide measured traffic volume, traffic density, vehicle count, or fuel consumption. Fuel is therefore an estimate from route distance and user-entered vehicle economy. Traffic colors represent the dataset's travel-time ratio, not a live traffic feed.

The supplied segment network may not contain multiple independent paths between every pair of endpoints. The app reports routes found in the graph and does not fabricate alternatives.

## Deploy on Render (stable platform subdomain)

1. Extract this ZIP on your computer.
2. Create a GitHub repository, for example `bengaluru-smart-traffic-dashboard`.
3. Upload **all files in this folder**, including the Excel workbook, `app.py`, `requirements.txt`, and `render.yaml`, to the repository root. Commit the changes.
4. Sign in to Render: https://dashboard.render.com/ .
5. Choose **New + → Blueprint** and connect the GitHub repository. Render will read `render.yaml`; review the service name and create/deploy it. Alternatively, choose **New + → Web Service** and use the settings below.
6. Wait for the first build/deploy to finish successfully. Open the `onrender.com` URL shown on the Render service page. The URL is assigned by Render; this package cannot reserve it in advance.
7. To update the app later, commit and push changes to the connected branch. Render auto-deploys from the linked branch when enabled.

### Manual Web Service settings

- Runtime: Python 3
- Build command: `pip install -r requirements.txt`
- Start command: `streamlit run app.py --server.address 0.0.0.0 --server.port $PORT --server.headless true`
- Health check path: `/_stcore/health`
- Include `Bengaluru_Traffic_Analysis_Converted.xlsx` in the repository root.

## Stable URL and uptime

The assigned `your-service.onrender.com` subdomain remains the service's address across code updates as long as the same Render service is retained. A free Render web service may spin down after inactivity and take time to wake; a paid always-on service is better if continuous availability is required. No deployment is performed by this ZIP itself: you must connect the repository and deploy it in your Render account.
