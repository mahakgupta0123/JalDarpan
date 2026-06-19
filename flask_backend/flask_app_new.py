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

WRIS_BASE = os.environ.get("WRIS_BASE_URL", "https://indiawris.gov.in")
WRIS_API_BASE = os.environ.get("WRIS_API_BASE", "https://indiawris.gov.in/wris")
WRIS_VERIFY = os.environ.get("WRIS_VERIFY", "false").lower() in ("true", "1", "yes")

COMMON_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'application/json',
}


def _normalize_wris_list(payload, keys):
    values = []
    if isinstance(payload, dict) and 'data' in payload:
        raw_items = payload['data']
    else:
        raw_items = payload

    if not isinstance(raw_items, list):
        return values

    for item in raw_items:
        if isinstance(item, dict):
            for key in keys:
                if key in item and item[key] is not None:
                    values.append(str(item[key]).strip())
                    break
        else:
            values.append(str(item).strip())

    return [v for v in values if v]


def _request_wris(url, method='get', data=None):
    try:
        if method == 'post':
            resp = requests.post(url, data=data, headers=COMMON_HEADERS, timeout=90, verify=WRIS_VERIFY)
        else:
            resp = requests.get(url, headers=COMMON_HEADERS, timeout=30, verify=WRIS_VERIFY)
        resp.raise_for_status()
        return resp
    except Exception as exc:
        logger.warning("WRIS request failed %s %s: %s", method.upper(), url, exc)
        return None


@app.route("/groundwater", methods=["GET", "POST"])
def groundwater():
    logger.info("Groundwater endpoint called")
    state = request.args.get("state", "Odisha")
    district = request.args.get("district", "Baleshwar")
    agency = request.args.get("agency", "CGWB")
    startdate = request.args.get("startdate", "2024-09-22")
    enddate = request.args.get("enddate", "2025-09-22")
    size = int(request.args.get("size", 500))

    url = f"{WRIS_BASE}/Dataset/Ground%20Water%20Level"
    payload = {
        "stateName": state,
        "districtName": district,
        "agencyName": agency,
        "startdate": startdate,
        "enddate": enddate,
        "download": "false",
        "page": 0,
        "size": size,
    }

    logger.info("Sending request to %s with payload state=%s district=%s agency=%s", url, state, district, agency)
    response = _request_wris(url, method='post', data=payload)
    if response is None:
        return jsonify({"status": "error", "message": "Could not reach WRIS groundwater API"}), 502

    try:
        json_data = response.json()
    except ValueError:
        logger.error("Invalid JSON response from WRIS: %s", response.text[:500])
        return jsonify({"status": "error", "message": "Invalid JSON from WRIS"}), 502

    data = json_data.get("data", [])
    return jsonify({"status": "success", "records": len(data) if isinstance(data, list) else 0, "data": data})


@app.route("/states")
def states():
    logger.info("State list requested")
    url = f"{WRIS_API_BASE}/getStatesListDataViewPage/"
    response = _request_wris(url, method='get')
    if response is None:
        return jsonify({"status": "error", "message": "Could not fetch states from WRIS"}), 502

    try:
        payload = response.json()
    except ValueError:
        return jsonify({"status": "error", "message": "Invalid JSON from WRIS"}), 502

    states = _normalize_wris_list(payload, ["state", "State", "stateName", "StateName", "name"])
    return jsonify({"status": "success", "states": sorted(set(states))})


@app.route("/districts/<path:state>")
def districts(state):
    logger.info("District list requested for state=%s", state)
    encoded_state = requests.utils.requote_uri(state)
    url = f"{WRIS_API_BASE}/getAllDistrictListDataViewPage/{encoded_state}/"
    response = _request_wris(url, method='get')
    if response is None:
        return jsonify({"status": "error", "message": "Could not fetch districts from WRIS"}), 502

    try:
        payload = response.json()
    except ValueError:
        return jsonify({"status": "error", "message": "Invalid JSON from WRIS"}), 502

    districts = _normalize_wris_list(payload, ["district", "District", "districtName", "DistrictName", "name"])
    return jsonify({"status": "success", "districts": sorted(set(districts))})


@app.route("/agencies")
def agencies():
    logger.info("Agency list requested")
    url = f"{WRIS_API_BASE}/getMasterAgencyListDataViewPage/"
    response = _request_wris(url, method='get')
    if response is None:
        return jsonify({"status": "error", "message": "Could not fetch agencies from WRIS"}), 502

    try:
        payload = response.json()
    except ValueError:
        return jsonify({"status": "error", "message": "Invalid JSON from WRIS"}), 502

    agencies = _normalize_wris_list(payload, ["agency", "Agency", "agencyName", "AgencyName", "name"])
    return jsonify({"status": "success", "agencies": sorted(set(agencies))})


@app.route("/health")
def health():
    logger.info("Health check requested")
    return jsonify({"status": "ok"})


@app.route("/debug-ip")
def debug_ip():
    logger.info("Debug IP requested")
    try:
        response = requests.get('https://ipinfo.io/json', timeout=5)
        response.raise_for_status()
        return jsonify(response.json())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
