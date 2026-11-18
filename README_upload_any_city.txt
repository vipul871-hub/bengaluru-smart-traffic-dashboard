MULTI-CITY DASHBOARD — UPLOAD DATASETS DIRECTLY

WHAT THIS UPDATE DOES
- Adds a clear city-dataset upload area in the dashboard sidebar.
- Accepts CSV, Excel .xlsx and Excel .xls files.
- Uses the uploaded filename as the dashboard city label.
- Keeps the existing routing, map layers, route comparison, voice panel and traffic summaries in the app.
- After this version is installed once, uploading another city's dataset does NOT require another GitHub change or Render redeployment.

IMPORTANT HOSTING LIMIT
A currently deployed website cannot gain new upload functionality without its application code being installed once. To add this feature to your EXISTING Render URL, the updated app.py and requirements.txt must be deployed to that same Render service one time. This does not require creating a new Render service, and the existing URL/QR should stay the same. After that one-time deployment, upload future city datasets directly in the dashboard; no GitHub changes for each dataset.

IF YOU DO NOT WANT TO TOUCH THE EXISTING GITHUB REPOSITORY AT ALL
Run this package locally on Windows instead. This creates a local dashboard, not an update to the existing public Render URL:
1. Install Python 3.10 or newer.
2. Extract this ZIP into a folder.
3. Open Command Prompt in that folder.
4. Run:  py -m pip install -r requirements.txt
5. Run:  py -m streamlit run app.py
6. Open the local URL shown by Streamlit (usually http://localhost:8501).
7. Upload a city's CSV/XLSX/XLS file in the sidebar.

SUPPORTED DATA SHAPES
A) Road segments: start_lat, start_lon, end_lat, end_lon (also common naming variations).
B) Ordered road points: latitude, longitude, and a road/street name or ID; include a sequence/order column and at least two points per road.

The app can derive segment distance from coordinates and estimate travel time when distance and speed are available. If a traffic ratio is absent, it uses a neutral 1.0 baseline and must not be interpreted as measured congestion. The app cannot reliably route a file that contains only city name, traffic volume or GPS points without enough road connectivity.

CHECKS
- Python syntax compilation passed for app.py.
- Real-world city datasets still need to be tested individually because column names and road-network structures vary.
