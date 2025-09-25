import os
import subprocess
from threading import Thread
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

# === FLASK API ===
app = Flask(__name__)
CORS(app)

@app.route("/groundwater", methods=["GET"])
def groundwater():
    state = request.args.get("state", "Odisha")
    district = request.args.get("district", "Baleshwar")
    agency = request.args.get("agency", "CGWB")
    startdate = request.args.get("startdate", "2024-09-22")
    enddate = request.args.get("enddate", "2025-09-22")
    size = int(request.args.get("size", 500))

    url = "https://indiawris.gov.in/Dataset/Ground%20Water%20Level"
    payload = {
        "stateName": state,
        "districtName": district,
        "agencyName": agency,
        "startdate": startdate,
        "enddate": enddate,
        "download": "false",
        "page": 0,
        "size": size
    }

    try:
        response = requests.post(url, data=payload, timeout=90)  # increased timeout
        response.raise_for_status()
        data = response.json().get("data", [])
        return jsonify({"status": "success", "records": len(data), "data": data})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# === STREAMLIT ===
def run_streamlit():
    """
    Launch Streamlit app on port 8501 as a subprocess.
    """
    os.system("streamlit run app.py --server.port 8501 --server.headless true")

# Start Streamlit in a separate thread to not block Flask
thread = Thread(target=run_streamlit)
thread.daemon = True
thread.start()

# === RUN FLASK ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
