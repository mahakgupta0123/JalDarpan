from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
import logging

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route("/groundwater", methods=["GET", "POST"])
def groundwater():
    logger.info("Groundwater endpoint called")
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

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json'
    }

    logger.info(f"Sending request to {url} with payload: {payload} and headers: {headers}")
    try:
        response = requests.post(url, data=payload, headers=headers, timeout=90)
        response.raise_for_status()
        json_data = response.json()
        data = json_data.get("data", [])
        logger.info(f"Received {len(data)} records. Keys: {list(json_data.keys())}")
        logger.debug(f"Raw API response: {response.text[:500]}")
        return jsonify({"status": "success", "records": len(data), "data": data})
    except requests.exceptions.JSONDecodeError:
        logger.error(f"Invalid JSON response: {response.text[:500]}")
        return jsonify({"status": "error", "message": "Invalid JSON from WRIS"}), 502
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 502

@app.route("/health")
def health():
    logger.info("Health check requested")
    return jsonify({"status": "ok"})

@app.route("/debug-ip")
def debug_ip():
    logger.info("Debug IP requested")
    try:
        response = requests.get('https://ipinfo.io/json', timeout=5)
        return jsonify(response.json())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)