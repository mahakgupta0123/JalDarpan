from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
import logging
import functools
import urllib3
import math
from datetime import datetime, timedelta

import pandas as pd

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WRIS_BASE = os.environ.get("WRIS_BASE_URL", "https://indiawris.gov.in")
WRIS_API_BASE = os.environ.get("WRIS_API_BASE", "https://india-water.gov.in/wris")
WRIS_VERIFY = True

COMMON_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'application/json',
}

FALLBACK_STATES = ["Odisha", "Andhra Pradesh", "Karnataka", "Tamil Nadu", "Maharashtra", "Rajasthan", "Uttar Pradesh", "West Bengal", "Gujarat", "Punjab"]
FALLBACK_DISTRICTS = {
    "odisha": ["Baleshwar", "Cuttack", "Puri", "Khurda", "Sundargarh"],
    "andhra pradesh": ["Guntur", "Vishakhapatnam", "Anantapur", "Krishna"],
    "karnataka": ["Bengaluru Urban", "Mysuru", "Belagavi", "Shimoga"],
    "tamil nadu": ["Chennai", "Coimbatore", "Madurai", "Salem"],
    "uttar pradesh": ["Lucknow", "Kanpur Nagar", "Varanasi", "Prayagraj"],
    "maharashtra": ["Pune", "Nagpur", "Nashik", "Thane"],
    "rajasthan": ["Jaipur", "Jodhpur", "Udaipur", "Kota"],
    "gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot"],
    "west bengal": ["Kolkata", "Howrah", "Darjeeling", "Medinipur"],
    "punjab": ["Ludhiana", "Amritsar", "Jalandhar", "Patiala"],
}
FALLBACK_AGENCIES = ["CGWB", "State DW", "Central Ground Water Board"]
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGGREGATION_DATASET = os.path.join(PROJECT_ROOT, "aggregation_outputs", "cleaned_dataset_used_for_aggregation.csv")


def _normalize_key(value):
    return str(value).strip().lower()


@functools.lru_cache(maxsize=1)
def _load_dataset_states():
    """Load all states from the aggregation dataset to ensure complete coverage."""
    if not os.path.exists(AGGREGATION_DATASET):
        return FALLBACK_STATES
    
    try:
        frame = pd.read_csv(AGGREGATION_DATASET, usecols=["state"])
    except Exception as exc:
        logger.warning("Could not load states from aggregation dataset: %s", exc)
        return FALLBACK_STATES
    
    states = sorted({str(value).strip() for value in frame["state"] if str(value).strip()})
    if states:
        return states
    return FALLBACK_STATES


@functools.lru_cache(maxsize=1)
def _load_dataset_district_map():
    if not os.path.exists(AGGREGATION_DATASET):
        return {}

    try:
        frame = pd.read_csv(AGGREGATION_DATASET, usecols=["state", "district"])
    except Exception as exc:
        logger.warning("Could not load aggregation dataset for district fallback: %s", exc)
        return {}

    district_map = {}
    for state_name, group in frame.dropna(subset=["state", "district"]).groupby("state"):
        districts = sorted({str(value).strip() for value in group["district"] if str(value).strip()})
        if districts:
            district_map[_normalize_key(state_name)] = districts
    return district_map


def _dataset_districts_for_state(state_name):
    district_map = _load_dataset_district_map()
    normalized_state = _normalize_key(state_name)
    if normalized_state in district_map:
        return district_map[normalized_state]

    for known_state, districts in district_map.items():
        if known_state in normalized_state or normalized_state in known_state:
            return districts
    return []


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
            resp = requests.post(url, data=data, headers=COMMON_HEADERS, timeout=90, verify=False)
        else:
            resp = requests.get(url, headers=COMMON_HEADERS, timeout=30, verify=False)
        resp.raise_for_status()
        return resp
    except Exception as exc:
        logger.warning("WRIS request failed %s %s: %s", method.upper(), url, exc)
        return None


def _resolve_state_code(state_name_or_code):
    if not state_name_or_code:
        return None
    candidate = str(state_name_or_code).strip()
    if candidate.isdigit():
        return candidate

    response = _request_wris(f"{WRIS_API_BASE}/getStatesListDataViewPage/", method='get')
    if response is None:
        return None
    try:
        payload = response.json()
    except ValueError:
        return None

    for row in payload if isinstance(payload, list) else payload.get('data', []):
        if not isinstance(row, dict):
            continue
        name = str(row.get('name') or row.get('state') or row.get('stateName') or '').strip()
        code = str(row.get('stateCode') or row.get('state_id') or '').strip()
        if name.lower() == candidate.lower() or code == candidate:
            return code
    return None


def _build_fallback_groundwater_data(state, district, agency, startdate, enddate, size):
    try:
        start_dt = datetime.strptime(startdate, "%Y-%m-%d")
        end_dt = datetime.strptime(enddate, "%Y-%m-%d")
    except ValueError:
        start_dt = datetime.now() - timedelta(days=180)
        end_dt = datetime.now()

    if end_dt < start_dt:
        start_dt, end_dt = end_dt, start_dt

    total_points = max(12, min(int(size) if int(size) > 0 else 60, 180))
    span_days = max(1, (end_dt - start_dt).days)
    if total_points == 1:
        step_days = 0
    else:
        step_days = max(1, span_days // (total_points - 1))

    seed = sum(ord(ch) for ch in f"{state}|{district}|{agency}")
    data = []
    for idx in range(total_points):
        current_dt = start_dt + timedelta(days=idx * step_days)
        if current_dt > end_dt:
            current_dt = end_dt
        seasonal = 1.2 * math.sin((idx + 1) / 6.0 + seed / 17.0)
        trend = 0.008 * idx
        base_level = 7.5 + ((seed % 9) * 0.3)
        level = base_level + seasonal + trend
        data.append({
            "state": state,
            "district": district,
            "agency": agency,
            "date": current_dt.strftime("%Y-%m-%d"),
            "datetime": current_dt.strftime("%Y-%m-%d"),
            "dataTime": current_dt.strftime("%Y-%m-%d"),
            "groundwater_level": round(level, 3),
            "waterLevel": round(level, 3),
            "dataValue": round(level, 3),
            "value": round(level, 3),
            "source": "fallback"
        })

    return data


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
        fallback_data = _build_fallback_groundwater_data(state, district, agency, startdate, enddate, size)
        logger.warning("WRIS timed out; returning fallback groundwater data for %s/%s", state, district)
        return jsonify({
            "status": "success",
            "records": len(fallback_data),
            "data": fallback_data,
            "message": "WRIS groundwater service was unavailable; returning offline fallback data"
        }), 200

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
    
    # First, try to get states from the aggregation dataset (most comprehensive)
    dataset_states = _load_dataset_states()
    if dataset_states and len(dataset_states) > len(FALLBACK_STATES):
        logger.info("Returning %d states from dataset", len(dataset_states))
        return jsonify({"status": "success", "states": dataset_states}), 200
    
    # Fall back to WRIS API if dataset states are insufficient
    url = f"{WRIS_API_BASE}/getStatesListDataViewPage/"
    response = _request_wris(url, method='get')
    if response is None:
        logger.warning("WRIS states endpoint unavailable; returning fallback state list")
        return jsonify({"status": "success", "states": FALLBACK_STATES}), 200

    try:
        payload = response.json()
    except ValueError:
        logger.warning("WRIS states response invalid; returning fallback state list")
        return jsonify({"status": "success", "states": FALLBACK_STATES}), 200

    states_list = _normalize_wris_list(payload, ["state", "State", "stateName", "StateName", "name"])
    if not states_list:
        return jsonify({"status": "success", "states": dataset_states or FALLBACK_STATES}), 200
    
    # Merge WRIS states with dataset states for maximum coverage
    merged_states = sorted(set(states_list) | set(dataset_states or []))
    return jsonify({"status": "success", "states": merged_states})


@app.route("/districts/<path:state>")
def districts(state):
    logger.info("District list requested for state=%s", state)
    
    # First priority: Get districts from the aggregation dataset (most comprehensive and reliable)
    dataset_districts = _dataset_districts_for_state(state)
    if dataset_districts:
        logger.info("Returning %d dataset-backed districts for state=%s", len(dataset_districts), state)
        return jsonify({"status": "success", "districts": dataset_districts}), 200
    
    # Second priority: Try WRIS API for this state
    state_code = _resolve_state_code(state)
    if state_code is None:
        # If state code can't be resolved, return minimal fallback
        normalized_state = str(state).strip().lower()
        fallback_districts = FALLBACK_DISTRICTS.get(normalized_state, ["Unknown District"])
        logger.warning("Could not resolve state code for %s; returning fallback districts", state)
        return jsonify({"status": "success", "districts": fallback_districts}), 200

    url = f"{WRIS_API_BASE}/getAllDistrictListDataViewPage/{state_code}/"
    response = _request_wris(url, method='get')
    if response is None:
        logger.warning("WRIS districts endpoint unavailable for state=%s; no dataset fallback", state)
        return jsonify({"status": "success", "districts": []}), 200

    try:
        payload = response.json()
    except ValueError:
        logger.warning("WRIS districts response invalid for state=%s; no dataset fallback", state)
        return jsonify({"status": "success", "districts": []}), 200

    wris_districts = _normalize_wris_list(payload, ["district", "District", "districtName", "DistrictName", "name"])
    if wris_districts:
        logger.info("Returning %d WRIS-backed districts for state=%s", len(wris_districts), state)
        return jsonify({"status": "success", "districts": sorted(set(wris_districts))}), 200
    
    # No data from any source
    logger.warning("No districts found for state=%s from any source", state)
    return jsonify({"status": "success", "districts": []}), 200


@app.route("/agencies")
def agencies():
    logger.info("Agency list requested")
    url = f"{WRIS_API_BASE}/getMasterAgencyListDataViewPage/"
    response = _request_wris(url, method='get')
    if response is None:
        logger.warning("WRIS agencies endpoint unavailable; returning fallback agency list")
        return jsonify({"status": "success", "agencies": FALLBACK_AGENCIES}), 200

    try:
        payload = response.json()
    except ValueError:
        logger.warning("WRIS agencies response invalid; returning fallback agency list")
        return jsonify({"status": "success", "agencies": FALLBACK_AGENCIES}), 200

    agencies = _normalize_wris_list(payload, ["agency", "Agency", "agencyName", "AgencyName", "name"])
    if not agencies:
        return jsonify({"status": "success", "agencies": FALLBACK_AGENCIES}), 200
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
