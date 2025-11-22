# predict_live.py
import os
import time
import signal
import joblib
import json
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils import (
    load_email_config, send_email, safe_get_value, load_model, model_path,
    write_json_atomic, append_csv, logger, acquire_hour_lock, clear_lock
)
from utils import MODEL_DIR

# Config (tweakable)
INPUT_FIELDS = [
    "TCTR.WindingTemp", "TCTR.OilTemp", "TCTR.Load", "TCTR.Tap",
    "DGA.H2", "DGA.C2H2", "TCTR.OilLevel", "DGA.Moisture",
    "TCTR.Buchholz", "TCTR.Cooling"
]
FEATURES_FILE_TEMPLATE = "features_v{}.json"
META_FILE_TEMPLATE = "meta_v{}.json"

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
LIVE_ALL_CSV = os.path.join(OUTPUT_DIR, "isolator_live_all.csv")
LATEST_STATUS = os.path.join(OUTPUT_DIR, "latest_status.json")
HOURLY_DIR = os.path.join(OUTPUT_DIR, "hourly")
os.makedirs(HOURLY_DIR, exist_ok=True)

# Behavior tuning
POLL_INTERVAL = 1.0  # seconds (your per-second)
ISO_CONTAMINATION = 0.03

# State
stop_event = threading.Event()

# Load latest model version automatically (pick highest timestamped version in models dir)
def latest_version() -> str:
    files = os.listdir(MODEL_DIR)
    versions = set()
    for fn in files:
        if fn.startswith("meta_v") and fn.endswith(".json"):
            v = fn[len("meta_v"):-len(".json")]
            versions.add(v)
    if not versions:
        return ""
    return sorted(versions)[-1]

def load_models(version: str):
    try:
        reg = load_model("reg", version)
        clf = load_model("clf", version)
        iso = load_model("iso", version)
        scaler = load_model("scaler", version)
        feats_path = os.path.join(MODEL_DIR, f"features_v{version}.json")
        with open(feats_path, "r") as f:
            feats = json.load(f)
        logger.info("Loaded models version %s", version)
        return reg, clf, iso, scaler, feats, True
    except Exception as e:
        logger.exception("Failed to load models v%s: %s", version, e)
        return None, None, None, None, INPUT_FIELDS, False

# Fallback heuristics (same as original but small improvements)
def fallback_health(row):
    wt = safe_get_value(row, "TCTR.WindingTemp", 60)
    load = safe_get_value(row, "TCTR.Load", 1)
    return float(np.clip(100 - (wt * 0.3 + load * 1.5), 5, 100))

def fallback_fault(row):
    wt = safe_get_value(row, "TCTR.WindingTemp", 60)
    h2 = safe_get_value(row, "DGA.H2", 5)
    p = min((wt / 120) + (h2 / 500), 1)
    return int(p > 0.5), float(p)

def fallback_anomaly(row):
    wt = safe_get_value(row, "TCTR.WindingTemp", 60)
    h2 = safe_get_value(row, "DGA.H2", 5)
    return int(wt > 110 or h2 > 700), -0.8

def compute_risk(health, fault_prob, anom_flag):
    risk = (1 - health / 100) * 60 + fault_prob * 40
    if anom_flag:
        risk += 15
    risk = float(np.clip(risk, 0, 100))
    if risk <= 40:
        return risk, "LOW"
    if risk <= 70:
        return risk, "MEDIUM"
    if risk <= 90:
        return risk, "HIGH"
    return risk, "CRITICAL"

# Create readable status color
def risk_to_color(level: str) -> str:
    return {
        "LOW": "GREEN",
        "MEDIUM": "YELLOW",
        "HIGH": "ORANGE",
        "CRITICAL": "RED"
    }.get(level, "GRAY")

# Placeholder for your real ingestion - replace with MQTT/OPC-UA/REST fetch
def fetch_live_data():
    """
    Replace this stub with your real data ingestion.
    Return a dict with keys matching INPUT_FIELDS and optionally TransformerID, Location, DeviceType.
    """
    # Simulated sensor row for demonstration
    import random
    return {
        "TransformerID": "TR-" + str(random.randint(1, 10)),
        "Location": "Plant-1",
        "DeviceType": "PowerTransformer",
        "TCTR.WindingTemp": 60 + random.random() * 60,
        "TCTR.OilTemp": 40 + random.random() * 40,
        "TCTR.Load": random.random() * 2,
        "TCTR.Tap": random.randint(0, 10),
        "DGA.H2": random.random() * 800,
        "DGA.C2H2": random.random() * 50,
        "TCTR.OilLevel": 70 + random.random() * 30,
        "DGA.Moisture": random.random() * 100,
        "TCTR.Buchholz": random.choice([0, 1]),
        "TCTR.Cooling": random.choice([0, 1]),
    }

# Prediction for one row using loaded models or fallback
def predict_single(row, reg, clf, iso, scaler, feats, model_ok):
    X = np.array([[safe_get_value(row, f) for f in feats]])
    try:
        Xs = scaler.transform(X) if model_ok and scaler is not None else X
    except Exception:
        Xs = X

    try:
        health = float(reg.predict(Xs)[0])
    except Exception:
        health = fallback_health(row)
    health = float(np.clip(health, 0, 100))

    try:
        fault = int(clf.predict(Xs)[0])
        fault_prob = float(clf.predict_proba(Xs)[:, 1][0])
    except Exception:
        fault, fault_prob = fallback_fault(row)

    try:
        anom_flag = int(iso.predict(Xs)[0] == -1)
        anom_score = float(iso.decision_function(Xs)[0])
    except Exception:
        anom_flag, anom_score = fallback_anomaly(row)

    risk, level = compute_risk(health, fault_prob, anom_flag)

    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    # construct details
    fault_details = []
    if fault == 1:
        fault_details.append("Electrical fault detected.")
    if anom_flag == 1:
        fault_details.append("Abnormal sensor behavior detected.")
    if level == "HIGH":
        fault_details.append("High risk operating condition.")
    if level == "CRITICAL":
        fault_details.append("Critical risk. Immediate shutdown recommended.")
    if not fault_details:
        fault_details = ["No specific fault category identified."]

    return {
        "TransformerID": row.get("TransformerID", "UNKNOWN"),
        "Location": row.get("Location", "UNKNOWN"),
        "DeviceType": row.get("DeviceType", "UNKNOWN"),
        "Timestamp": timestamp,
        "health": float(round(health, 3)),
        "fault": int(fault),
        "fault_prob": float(round(fault_prob, 4)),
        "anom_flag": int(anom_flag),
        "anom_score": float(round(anom_score, 4)),
        "risk": float(round(risk, 3)),
        "risk_level": level,
        "details": " ".join(fault_details)
    }

# Main loop
def run_loop(model_version: str = None):
    # load models
    if not model_version:
        model_version = latest_version()
    reg, clf, iso, scaler, feats, model_ok = None, None, None, None, INPUT_FIELDS, False
    if model_version:
        reg, clf, iso, scaler, feats, model_ok = load_models(model_version)
    else:
        logger.warning("No trained model version found; using fallbacks only.")

    hourly_events = []
    last_hour = datetime.utcnow().hour
    logger.info("Starting live loop (interval %0.2fs). Model OK: %s, version=%s", POLL_INTERVAL, model_ok, model_version)

    # ensure a clean lock for initial run
    clear_lock("high_alert")

    try:
        while not stop_event.is_set():
            row = fetch_live_data()
            result = predict_single(row, reg, clf, iso, scaler, feats, model_ok)

            # append to live all CSV (include raw row merged with result for traceability)
            out_row = {**row, **result}
            append_csv(LIVE_ALL_CSV, out_row)

            # write latest status json for frontend
            latest = {
                "transformer": result["TransformerID"],
                "location": result["Location"],
                "health": result["health"],
                "risk": result["risk"],
                "risk_level": result["risk_level"],
                "color": risk_to_color(result["risk_level"]),
                "timestamp": result["Timestamp"]
            }
            write_json_atomic(LATEST_STATUS, latest)

            # alert logic
            # CRITICAL -> immediate email always
            if result["risk_level"] == "CRITICAL":
                subject = f"[CRITICAL] Transformer {result['TransformerID']} at {result['Location']}"
                body = json.dumps(result, indent=2)
                send_email(subject, body)
                # also collect it for hourly archive
                hourly_events.append(result)

            # HIGH -> throttle to 1 per hour (lock)
            elif result["risk_level"] == "HIGH":
                if acquire_hour_lock("high_alert"):
                    logger.info("HIGH alert and lock acquired, sending email for %s", result["TransformerID"])
                    subject = f"[HIGH] Transformer {result['TransformerID']} at {result['Location']}"
                    body = json.dumps(result, indent=2)
                    send_email(subject, body)
                    hourly_events.append(result)
                else:
                    logger.info("HIGH alert suppressed due to hourly lock for %s", result["TransformerID"])
                    hourly_events.append(result)

            # MEDIUM/LOW -> no immediate email, but record if needed
            elif result["risk_level"] in ("MEDIUM", "LOW"):
                # record only if medium+ for hourly summary
                if result["risk_level"] == "MEDIUM":
                    hourly_events.append(result)

            # on-hour boundary: persist hourly CSVs + send summary email (one per hour)
            cur_hour = datetime.utcnow().hour
            if cur_hour != last_hour:
                # persist hourly file
                hour_ts = datetime.utcnow().replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
                filename = os.path.join(HOURLY_DIR, hour_ts.strftime("hour_%Y%m%d%H.csv"))
                # write full hour CSV
                if hourly_events:
                    df_hour = pd.DataFrame(hourly_events)
                    df_hour.to_csv(filename, index=False)
                    # high/critical filtering
                    high_df = df_hour[df_hour["risk_level"].isin(["HIGH", "CRITICAL"])]
                    if not high_df.empty:
                        high_filename = filename.replace(".csv", "_high.csv")
                        high_df.to_csv(high_filename, index=False)
                        # send hourly summary for high/critical if any
                        subject = f"Hourly HIGH/CRITICAL Summary {hour_ts.strftime('%Y-%m-%d %H:00')} UTC"
                        body = high_df.to_string(index=False)
                        send_email(subject, body)
                else:
                    logger.info("No events for hour %s", hour_ts.isoformat())

                # reset for next hour
                hourly_events = []
                last_hour = cur_hour
                # clear HIGH lock at hour boundary so next hour can send again
                clear_lock("high_alert")

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received — shutting down gracefully.")
    except Exception as e:
        logger.exception("Unhandled exception in loop: %s", e)
    finally:
        # final flush of hourly events
        if hourly_events:
            hour_ts = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
            filename = os.path.join(HOURLY_DIR, hour_ts.strftime("hour_shutdown_%Y%m%d%H.csv"))
            pd.DataFrame(hourly_events).to_csv(filename, index=False)
            logger.info("Flushed %d events to %s before shutdown", len(hourly_events), filename)
        logger.info("Live predictor stopped.")

if __name__ == "__main__":
    # load email config early to log if disabled
    from utils import EMAIL_CFG
    if EMAIL_CFG is None:
        logger.warning("Email is disabled. No alerts will be sent.")
    # start main loop with latest model version
    run_loop()
