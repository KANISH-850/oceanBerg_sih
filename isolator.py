#!/usr/bin/env python3
"""
isolator.py (single-email-per-run, option A cooldown)
- Trains models (health regressor, fault classifier + prob)
- IsolationForest anomaly detector
- Risk scoring + root-cause
- Saves:
    C:\SIH\output\isolator_predictions.csv         <- this run (all rows)
    C:\SIH\output\isolator_all_alerts.csv          <- append: all rows across runs
    C:\SIH\output\isolator_highrisk_alerts.csv     <- append: only HIGH/CRITICAL rows
- Sends ONE email per run when:
    * any CRITICAL rows exist -> send immediately listing CRITICAL rows
    * else if any HIGH rows with FaultFlag==1 and last HIGH-mail > 1 hour ago -> send listing those HIGH rows
- Training prints:
    TRAINING...
    MSE_health: ...
    MSE_faultprob: ...
    Accuracy: ...
    <classification report>
- Otherwise runtime is silent.
"""

import os
import json
import joblib
import glob
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timezone
import time

# -----------------------------
# EMAIL CONFIG (email_config.json)
# -----------------------------
CONFIG_FILE = "email_config.json"

def load_email_config():
    if not os.path.exists(CONFIG_FILE):
        return None
    try:
        with open(CONFIG_FILE, "r") as f:
            cfg = json.load(f)
        required = ["EMAIL_SENDER", "EMAIL_PASSWORD", "EMAIL_RECEIVER"]
        for key in required:
            if key not in cfg:
                return None
        return cfg
    except:
        return None

_email_cfg = load_email_config()
if _email_cfg:
    EMAIL_SENDER = _email_cfg["EMAIL_SENDER"]
    EMAIL_PASSWORD = _email_cfg["EMAIL_PASSWORD"]
    EMAIL_RECEIVER = _email_cfg["EMAIL_RECEIVER"]
    ENABLE_EMAIL_ALERTS = True
else:
    EMAIL_SENDER = EMAIL_PASSWORD = EMAIL_RECEIVER = None
    ENABLE_EMAIL_ALERTS = False

def send_email_alert(subject: str, body: str):
    """Send plain-text email. Silent on failure/success."""
    if not ENABLE_EMAIL_ALERTS:
        return
    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_SENDER
        msg["To"] = EMAIL_RECEIVER
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))
        server = smtplib.SMTP("smtp.gmail.com", 587, timeout=20)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        server.quit()
    except:
        pass

# -----------------------------
# Model & I/O settings
# -----------------------------
MODEL_DIR = os.path.join("models", "isolator")
os.makedirs(MODEL_DIR, exist_ok=True)

OUTPUT_DIR = r"C:\SIH\output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PREDICTION_RUN_FILE = os.path.join(OUTPUT_DIR, "isolator_predictions.csv")
ALL_ALERTS_FILE = os.path.join(OUTPUT_DIR, "isolator_all_alerts.csv")
HIGHRISK_ALERTS_FILE = os.path.join(OUTPUT_DIR, "isolator_highrisk_alerts.csv")
LAST_HIGH_EMAIL_FILE = os.path.join(OUTPUT_DIR, "last_high_email_time.json")

# -----------------------------
# Features / Targets
# -----------------------------
INPUT_FIELDS: List[str] = [
    "IsolatorStatus",
    "OpenClosed",
    "PositionSwitch",
    "NA",
    "CSWI.Pos",
    "SwitchingLogic"
]

TRAIN_TARGETS: List[str] = ["health_score", "fault_flag", "fault_prob"]

# -----------------------------
# Utilities
# -----------------------------
def safe_get_value(row: Dict[str, Any], key: str, default=0.0) -> float:
    try:
        v = row.get(key, default)
        if v is None:
            return float(default)
        if isinstance(v, str):
            vs = v.strip().upper()
            if vs in ("OPEN", "O"): return 0.0
            if vs in ("CLOSED", "C"): return 1.0
            if vs in ("TRUE", "T", "1"): return 1.0
            if vs in ("FALSE", "F", "0"): return 0.0
            try:
                return float(vs)
            except:
                return float(default)
        return float(v)
    except:
        return float(default)

def append_to_csv(filepath: str, data: dict):
    """Append a single-row dict to CSV (create file with header if not exists)."""
    df = pd.DataFrame([data])
    if not os.path.exists(filepath):
        df.to_csv(filepath, index=False)
    else:
        df.to_csv(filepath, mode="a", header=False, index=False)

# -----------------------------
# Training pipeline
# -----------------------------
def validate_training(df: pd.DataFrame):
    required = INPUT_FIELDS + TRAIN_TARGETS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in training CSV: {missing}")

def numeric_feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in INPUT_FIELDS if c in df.columns]

def train(df: pd.DataFrame,
          random_state: int = 42,
          test_size: float = 0.2,
          iso_contamination: float = 0.03) -> Dict[str, Any]:
    """
    Train models and save them to MODEL_DIR.
    Returns metadata with metrics.
    """
    validate_training(df)
    df = df.dropna(subset=INPUT_FIELDS + TRAIN_TARGETS)

    feats = numeric_feature_columns(df)
    if not feats:
        raise ValueError("No numeric features found for training.")

    # build numeric X
    X = np.array([[safe_get_value(row, f) for f in feats] for _, row in df.iterrows()])
    y_health = df["health_score"].astype(float).values
    y_faultprob = df["fault_prob"].astype(float).values
    y_faultflag = df["fault_flag"].astype(int).values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    Xtr, Xv, ytr_h, yv_h, ytr_fp, yv_fp, ytr_ff, yv_ff = train_test_split(
        Xs, y_health, y_faultprob, y_faultflag, test_size=test_size, random_state=random_state
    )

    reg_health = RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)
    reg_faultprob = RandomForestRegressor(n_estimators=150, random_state=random_state, n_jobs=-1)
    clf_fault = RandomForestClassifier(n_estimators=200, random_state=random_state,
                                       class_weight="balanced_subsample", n_jobs=-1)
    iso = IsolationForest(n_estimators=250, contamination=iso_contamination, random_state=random_state)

    reg_health.fit(Xtr, ytr_h)
    reg_faultprob.fit(Xtr, ytr_fp)
    clf_fault.fit(Xtr, ytr_ff)
    iso.fit(Xs)

    mse_health = float(mean_squared_error(yv_h, reg_health.predict(Xv)))
    mse_faultprob = float(mean_squared_error(yv_fp, reg_faultprob.predict(Xv)))
    acc_fault = float(accuracy_score(yv_ff, clf_fault.predict(Xv)))
    crep = classification_report(yv_ff, clf_fault.predict(Xv), zero_division=0)

    # Save models & scaler & features
    joblib.dump(reg_health, os.path.join(MODEL_DIR, "reg_health.joblib"))
    joblib.dump(reg_faultprob, os.path.join(MODEL_DIR, "reg_faultprob.joblib"))
    joblib.dump(clf_fault, os.path.join(MODEL_DIR, "clf_fault.joblib"))
    joblib.dump(iso, os.path.join(MODEL_DIR, "iso.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    joblib.dump(feats, os.path.join(MODEL_DIR, "features.joblib"))

    meta = {
        "mse_health": mse_health,
        "mse_faultprob": mse_faultprob,
        "acc_fault": acc_fault,
        "report": crep
    }
    return meta

# -----------------------------
# Safe model loader
# -----------------------------
def load_models_safe() -> Tuple[Any, Any, Any, Any, Any, List[str], bool]:
    try:
        reg_health = joblib.load(os.path.join(MODEL_DIR, "reg_health.joblib"))
        reg_faultprob = joblib.load(os.path.join(MODEL_DIR, "reg_faultprob.joblib"))
        clf_fault = joblib.load(os.path.join(MODEL_DIR, "clf_fault.joblib"))
        iso = joblib.load(os.path.join(MODEL_DIR, "iso.joblib"))
        scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
        feats = joblib.load(os.path.join(MODEL_DIR, "features.joblib"))
        return reg_health, reg_faultprob, clf_fault, iso, scaler, feats, True
    except Exception:
        return None, None, None, None, None, INPUT_FIELDS, False

# -----------------------------
# Fallback heuristics
# -----------------------------
def fallback_health(row: Dict[str, Any]) -> float:
    status = safe_get_value(row, "IsolatorStatus", 1)
    oc = safe_get_value(row, "OpenClosed", 1)
    pos = safe_get_value(row, "PositionSwitch", 0)
    base = 90.0 if status == 1.0 else 30.0
    base -= (abs(pos) / 100.0) * 20.0
    if oc == 0.0:
        base -= 10.0
    return float(np.clip(base, 5.0, 100.0))

def fallback_fault(row: Dict[str, Any]) -> Tuple[int, float]:
    oc = safe_get_value(row, "OpenClosed", 1)
    cswi = safe_get_value(row, "CSWI.Pos", 1)
    logic = safe_get_value(row, "SwitchingLogic", 1)
    mismatch = abs(oc - (cswi % 2))
    p = min(1.0, mismatch * 0.6 + (1.0 - logic) * 0.4)
    return (1 if p > 0.5 else 0), float(np.clip(p, 0.0, 1.0))

def fallback_anomaly(row: Dict[str, Any]) -> Tuple[int, float]:
    pos = safe_get_value(row, "PositionSwitch", 0)
    na = safe_get_value(row, "NA", 0)
    anom = int(pos > 100 or na != 0)
    score = -0.7 if anom else 0.2
    return anom, float(score)

# -----------------------------
# Risk computation & root cause
# -----------------------------
def compute_risk(health: float, fault_prob: float, anom_flag: int) -> Tuple[float, str]:
    risk = (1 - health / 100.0) * 60.0 + fault_prob * 40.0
    if anom_flag:
        risk += 12.0
    risk = float(np.clip(risk, 0.0, 100.0))
    if risk <= 40:
        return risk, "LOW"
    if risk <= 70:
        return risk, "MEDIUM"
    if risk <= 90:
        return risk, "HIGH"
    return risk, "CRITICAL"

def find_cause(row: Dict[str, Any]) -> str:
    causes = []
    if safe_get_value(row, "PositionSwitch") > 100:
        causes.append("PositionSwitch out of expected range")
    if safe_get_value(row, "NA") != 0:
        causes.append("NA flag present")
    if abs(safe_get_value(row, "CSWI.Pos") - safe_get_value(row, "OpenClosed", 0)) > 1:
        causes.append("CSWI.Pos mismatch with OpenClosed")
    return ", ".join(causes) if causes else "Normal"

# -----------------------------
# Prediction + alerting logic
# -----------------------------
def predict_single(row: Dict[str, Any]) -> Dict[str, Any]:
    reg_health, reg_fp, clf_fault, iso, scaler, feats, model_ok = load_models_safe()

    # build feature vector
    X = np.array([[safe_get_value(row, f) for f in feats]])
    try:
        Xs = scaler.transform(X) if model_ok and scaler is not None else X
    except Exception:
        Xs = X

    # Health
    try:
        health = float(reg_health.predict(Xs)[0]) if reg_health is not None else fallback_health(row)
    except Exception:
        health = fallback_health(row)
    health = float(np.clip(health, 0.0, 100.0))

    # Fault flag + probability
    try:
        if clf_fault is not None:
            fault_flag = int(clf_fault.predict(Xs)[0])
            try:
                fault_prob = float(clf_fault.predict_proba(Xs)[:, 1][0])
            except Exception:
                fault_prob = float(reg_fp.predict(Xs)[0]) if reg_fp is not None else fallback_fault(row)[1]
        else:
            fault_flag, fault_prob = fallback_fault(row)
    except Exception:
        fault_flag, fault_prob = fallback_fault(row)

    # Anomaly detection
    try:
        if iso is not None:
            anom_flag = int(iso.predict(Xs)[0] == -1)
            anom_score = float(iso.decision_function(Xs)[0])
        else:
            anom_flag, anom_score = fallback_anomaly(row)
    except Exception:
        anom_flag, anom_score = fallback_anomaly(row)

    # Risk
    risk, level = compute_risk(health, float(fault_prob), int(anom_flag))

    # Cause
    cause = find_cause(row)

    # ID + timestamp
    device_id = row.get("Isolator", "UNKNOWN")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # CSV row dict
    csv_row = {
        "Isolator": device_id,
        "Timestamp": ts,
        "Health": float(health),
        "FaultFlag": int(fault_flag),
        "FaultProb": float(fault_prob),
        "AnomalyFlag": int(anom_flag),
        "AnomalyScore": float(anom_score),
        "RiskScore": float(risk),
        "RiskLevel": level,
        "Cause": cause
    }

    # ALWAYS append to all-alerts CSV
    append_to_csv(ALL_ALERTS_FILE, csv_row)

    # IF HIGH or CRITICAL: append to highrisk CSV (and collect for caller)
    if level in ("HIGH", "CRITICAL"):
        append_to_csv(HIGHRISK_ALERTS_FILE, csv_row)

    # Return dict for aggregated predictions
    return {
        "Isolator": device_id,
        "timestamp": ts,
        "health": float(health),
        "fault_flag": int(fault_flag),
        "fault_prob": float(fault_prob),
        "anom_flag": int(anom_flag),
        "anom_score": float(anom_score),
        "risk": float(risk),
        "risk_level": level,
        "cause": cause
    }

# -----------------------------
# Helper: last-high-email persistence
# -----------------------------
def read_last_high_email_time() -> float:
    try:
        if not os.path.exists(LAST_HIGH_EMAIL_FILE):
            return 0.0
        with open(LAST_HIGH_EMAIL_FILE, "r") as f:
            data = json.load(f)
        return float(data.get("last_high_email_time", 0.0))
    except:
        return 0.0

def write_last_high_email_time(ts: float):
    try:
        with open(LAST_HIGH_EMAIL_FILE, "w") as f:
            json.dump({"last_high_email_time": ts}, f)
    except:
        pass

# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":
    BASE_PATH = r"C:\SIH"
    train_file = os.path.join(BASE_PATH, "isolator_train_numeric.csv")
    test_file = os.path.join(BASE_PATH, "isolator_test_numeric.csv")

    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file not found: {train_file}")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")

    # --- TRAIN ---
    df_train = pd.read_csv(train_file)
    metadata = train(df_train)

    # Print training metrics in requested simple format
    print("\nTRAINING...")
    # include both MSEs and accuracy + classification report
    print("MSE_health:", metadata.get("mse_health"))
    print("MSE_faultprob:", metadata.get("mse_faultprob"))
    print("Accuracy:", metadata.get("acc_fault"))
    print(metadata.get("report"))

    # --- PREDICT (silent) ---
    df_test = pd.read_csv(test_file)
    results: List[Dict[str, Any]] = []
    high_risk_rows: List[Dict[str, Any]] = []
    critical_rows: List[Dict[str, Any]] = []

    for _, r in df_test.iterrows():
        res = predict_single(r.to_dict())
        results.append(res)
        if res["risk_level"] == "CRITICAL":
            critical_rows.append(res)
        elif res["risk_level"] == "HIGH" and res["fault_flag"] == 1:
            high_risk_rows.append(res)

    # Save per-run predictions (overwrite per-run file)
    pd.DataFrame(results).to_csv(PREDICTION_RUN_FILE, index=False)

    # Prepare single email per run according to Option A
    email_body = None
    email_subject = None

    if len(critical_rows) > 0:
        # send CRITICAL rows immediately (single email)
        lines = ["CRITICAL Isolators Detected:\n"]
        for i, h in enumerate(critical_rows, start=1):
            lines.append(f"{i}) Isolator: {h['Isolator']}")
            lines.append(f"   Health: {h['health']:.2f}")
            lines.append(f"   Fault Probability: {h['fault_prob']:.3f}")
            lines.append(f"   Risk Level: {h['risk_level']}")
            lines.append(f"   Cause: {h['cause']}")
            lines.append("")
        email_body = "\n".join(lines)
        email_subject = "⚠️ Critical Isolator Fault Detected"
        send_email_alert(email_subject, email_body)

    else:
        # No criticals — consider HIGH (requires fault_flag==1 and cooldown)
        if len(high_risk_rows) > 0:
            last_ts = read_last_high_email_time()
            now_ts = time.time()
            # cooldown = 3600 seconds (1 hour)
            if now_ts - last_ts >= 3600:
                lines = ["HIGH Isolators Detected (fault_flag==1):\n"]
                for i, h in enumerate(high_risk_rows, start=1):
                    lines.append(f"{i}) Isolator: {h['Isolator']}")
                    lines.append(f"   Health: {h['health']:.2f}")
                    lines.append(f"   Fault Probability: {h['fault_prob']:.3f}")
                    lines.append(f"   Risk Level: {h['risk_level']}")
                    lines.append(f"   Cause: {h['cause']}")
                    lines.append("")
                email_body = "\n".join(lines)
                email_subject = "⚠️ High Isolator Faults Detected"
                send_email_alert(email_subject, email_body)
                write_last_high_email_time(now_ts)
            else:
                # cooldown not passed -> do not send
                pass

    # All alerts already appended to files during predict_single()
    # Append this run to "all alerts" and "highrisk" are handled per-row in predict_single()
    # Save per-run CSVs already done; no further prints (silent)

    # End of run (silent)
    # Note: training metrics were printed above as requested.
