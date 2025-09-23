from flask import Flask, request, jsonify
import requests
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # allow cross-origin requests

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
        response = requests.post(url, data=payload, timeout=15)
        response.raise_for_status()
        data = response.json().get("data", [])
        return jsonify({"status": "success", "records": len(data), "data": data})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
