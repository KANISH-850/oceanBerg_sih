import os
import json
import joblib
import glob
import time
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
from datetime import datetime

# -----------------------------
# EMAIL CONFIG
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
    """Silent send: on failure or success nothing is printed."""
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
# PATHS / IO
# -----------------------------
MODEL_DIR = "models/instrument_ct"
os.makedirs(MODEL_DIR, exist_ok=True)

BASE_PATH = r"C:\SIH"
OUTPUT_DIR = os.path.join(BASE_PATH, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PREDICTION_RUN_FILE = os.path.join(OUTPUT_DIR, "instrument_ct_predictions.csv")
ALL_ALERTS_FILE = os.path.join(OUTPUT_DIR, "all_ct_alerts.csv")
HIGHRISK_FILE = os.path.join(OUTPUT_DIR, "highrisk_ct_alerts.csv")
LAST_HIGH_EMAIL_FILE = os.path.join(OUTPUT_DIR, "last_high_email_time.json")

# -----------------------------
# FEATURES / TARGETS
# -----------------------------
INPUT_FIELDS = [
    "PrimaryCurrent_A",
    "SecondaryCurrent_A",
    "CT_Ratio",
    "Burden_VA",
    "Accuracy_Class",
    "Phase_Angle_Error_deg",
    "Core_Temperature_C",
    "Insulation_Resistance_MOhm",
    "CT_Saturation_Status",
    "Knee_Point_Alarm",
    "Secondary_Open_Circuit_Alarm",
    "Oil_Level_pct",
    "Oil_Moisture_ppm",
    "Partial_Discharge_pC",
    "EarthFaultOnSecondary_Alarm"
]

TRAIN_TARGETS = ["health_score", "fault_flag"]

# -----------------------------
# UTILITIES
# -----------------------------
def safe_get_value(row: Dict[str, Any], key: str, default=0.0):
    try:
        v = row.get(key, default)
        if v is None:
            return float(default)
        return float(v)
    except:
        return float(default)

def append_to_csv(filepath: str, data: dict):
    df = pd.DataFrame([data])
    if not os.path.exists(filepath):
        df.to_csv(filepath, index=False)
    else:
        df.to_csv(filepath, mode="a", header=False, index=False)

# -----------------------------
# TRAINING PIPELINE
# -----------------------------
def validate_training(df: pd.DataFrame):
    required = INPUT_FIELDS + TRAIN_TARGETS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

def numeric_feature_columns(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=[np.number]).columns.to_list()
    feats = [c for c in INPUT_FIELDS if c in num_cols]
    return feats

def train(df, random_state=42, test_size=0.2, iso_contamination=0.03):
    validate_training(df)
    df = df.dropna(subset=INPUT_FIELDS + TRAIN_TARGETS)

    feats = numeric_feature_columns(df)
    if len(feats) == 0:
        raise ValueError("No numeric features available.")

    X = df[feats].values
    y_reg = df["health_score"].astype(float).values
    y_clf = df["fault_flag"].astype(int).values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    Xtr, Xv, ytr_reg, yv_reg, ytr_clf, yv_clf = train_test_split(
        Xs, y_reg, y_clf, test_size=test_size, random_state=random_state
    )

    reg = RandomForestRegressor(n_estimators=160, random_state=random_state, n_jobs=-1)
    clf = RandomForestClassifier(n_estimators=160, random_state=random_state,
                                 class_weight="balanced", n_jobs=-1)
    iso = IsolationForest(n_estimators=250, contamination=iso_contamination, random_state=random_state)

    reg.fit(Xtr, ytr_reg)
    clf.fit(Xtr, ytr_clf)
    iso.fit(Xs)

    mse = float(mean_squared_error(yv_reg, reg.predict(Xv)))
    acc = float(accuracy_score(yv_clf, clf.predict(Xv)))
    report = classification_report(yv_clf, clf.predict(Xv), zero_division=0)

    joblib.dump(reg, os.path.join(MODEL_DIR, "reg.joblib"))
    joblib.dump(clf, os.path.join(MODEL_DIR, "clf.joblib"))
    joblib.dump(iso, os.path.join(MODEL_DIR, "iso.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    joblib.dump(feats, os.path.join(MODEL_DIR, "features.joblib"))

    return {"mse": mse, "accuracy": acc, "report": report, "features": feats}

# -----------------------------
# SAFE LOAD
# -----------------------------
def load_models_safe():
    try:
        reg = joblib.load(os.path.join(MODEL_DIR, "reg.joblib"))
        clf = joblib.load(os.path.join(MODEL_DIR, "clf.joblib"))
        iso = joblib.load(os.path.join(MODEL_DIR, "iso.joblib"))
        scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
        feats = joblib.load(os.path.join(MODEL_DIR, "features.joblib"))
        return reg, clf, iso, scaler, feats, True
    except:
        return None, None, None, None, INPUT_FIELDS, False

# -----------------------------
# FALLBACKS
# -----------------------------
def fallback_health(row):
    temp = safe_get_value(row, "Core_Temperature_C", 45)
    pd = safe_get_value(row, "Partial_Discharge_pC", 5)
    return float(np.clip(100 - (temp * 0.5 + pd * 0.1), 5, 100))

def fallback_fault(row):
    temp = safe_get_value(row, "Core_Temperature_C", 45)
    pd = safe_get_value(row, "Partial_Discharge_pC", 5)
    prob = min((temp / 120) + (pd / 500), 1)
    return int(prob > 0.5), float(prob)

def fallback_anomaly(row):
    temp = safe_get_value(row, "Core_Temperature_C", 45)
    pd = safe_get_value(row, "Partial_Discharge_pC", 5)
    return int(temp > 110 or pd > 1200), -0.9

# -----------------------------
# RISK & CAUSE
# -----------------------------
def compute_risk(health, fault_prob, anom_flag):
    risk = (1 - health / 100) * 60 + fault_prob * 40
    if anom_flag:
        risk += 15
    risk = float(np.clip(risk, 0, 100))
    if risk <= 40: return risk, "LOW"
    if risk <= 70: return risk, "MEDIUM"
    if risk <= 90: return risk, "HIGH"
    return risk, "CRITICAL"

def find_cause(row):
    causes = []
    if safe_get_value(row, "Core_Temperature_C") > 100:
        causes.append("High Core Temperature")
    if safe_get_value(row, "Partial_Discharge_pC") > 1000:
        causes.append("High Partial Discharge")
    if safe_get_value(row, "Knee_Point_Alarm") == 1:
        causes.append("Knee Point Alarm Active")
    if safe_get_value(row, "CT_Saturation_Status") == 1:
        causes.append("CT Saturation")
    if safe_get_value(row, "Secondary_Open_Circuit_Alarm") == 1:
        causes.append("Secondary Open Circuit")
    if safe_get_value(row, "EarthFaultOnSecondary_Alarm") == 1:
        causes.append("Earth Fault on Secondary")
    return ", ".join(causes) if causes else "Normal"

# -----------------------------
# PREDICT SINGLE
# -----------------------------
def predict_single(row):
    reg, clf, iso, scaler, feats, ok = load_models_safe()

    X = np.array([[safe_get_value(row, f) for f in feats]])
    try:
        Xs = scaler.transform(X) if ok and scaler is not None else X
    except:
        Xs = X

    # Health
    try:
        health = float(reg.predict(Xs)[0]) if reg is not None else fallback_health(row)
    except:
        health = fallback_health(row)
    health = float(np.clip(health, 0, 100))

    # Fault
    try:
        if clf is not None:
            fault = int(clf.predict(Xs)[0])
            try:
                fault_prob = float(clf.predict_proba(Xs)[:, 1][0])
            except:
                fault_prob = fallback_fault(row)[1]
        else:
            fault, fault_prob = fallback_fault(row)
    except:
        fault, fault_prob = fallback_fault(row)

    # Anomaly
    try:
        if iso is not None:
            anom_flag = int(iso.predict(Xs)[0] == -1)
            anom_score = float(iso.decision_function(Xs)[0])
        else:
            anom_flag, anom_score = fallback_anomaly(row)
    except:
        anom_flag, anom_score = fallback_anomaly(row)

    # Risk & cause
    risk, level = compute_risk(health, float(fault_prob), int(anom_flag))
    cause = find_cause(row)

    device_id = row.get("InstrumentTransformerID", "UNKNOWN")
    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # prepare CSV row
    csv_row = {
        "ID": device_id,
        "Time": time_now,
        "Severity": level,
        "Health": float(health),
        "FaultProbability": float(fault_prob),
        "Anomaly": int(anom_flag),
        "RiskScore": float(risk),
        "Cause": cause
    }

    # Append to all-alerts
    append_to_csv(ALL_ALERTS_FILE, csv_row)

    # If HIGH/CRITICAL append to highrisk file (for records)
    if level in ["HIGH", "CRITICAL"]:
        append_to_csv(HIGHRISK_FILE, csv_row)

    # return structured result
    return {
        "ID": device_id,
        "timestamp": time_now,
        "health": float(health),
        "fault": int(fault),
        "fault_prob": float(fault_prob),
        "anom_flag": int(anom_flag),
        "anom_score": float(anom_score),
        "risk": float(risk),
        "risk_level": level,
        "cause": cause
    }

# -----------------------------
# LAST HIGH EMAIL (persistence)
# -----------------------------
def read_last_high_email_time() -> float:
    try:
        if not os.path.exists(LAST_HIGH_EMAIL_FILE):
            return 0.0
        with open(LAST_HIGH_EMAIL_FILE, "r") as f:
            d = json.load(f)
        return float(d.get("last_high_email_time", 0.0))
    except:
        return 0.0

def write_last_high_email_time(ts: float):
    try:
        with open(LAST_HIGH_EMAIL_FILE, "w") as f:
            json.dump({"last_high_email_time": ts}, f)
    except:
        pass

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    train_file = os.path.join(BASE_PATH, "instrument_ct_train_5000.csv")
    test_file = os.path.join(BASE_PATH, "instrument_ct_test_1000.csv")

    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file not found: {train_file}")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")

    # TRAIN and print only metrics in requested format
    df_train = pd.read_csv(train_file)
    meta = train(df_train)

    print("\nTRAINING...")
    print("MSE:", meta.get("mse"))
    print("Accuracy:", meta.get("accuracy"))
    print(meta.get("report"))

    # PREDICT (silent)
    df_test = pd.read_csv(test_file)
    results: List[Dict[str, Any]] = []
    critical_rows: List[Dict[str, Any]] = []
    high_rows: List[Dict[str, Any]] = []

    for _, r in df_test.iterrows():
        out = predict_single(r.to_dict())
        results.append(out)
        if out["risk_level"] == "CRITICAL":
            critical_rows.append(out)
        elif out["risk_level"] == "HIGH" and out["fault"] == 1:
            high_rows.append(out)

    # Save per-run predictions (overwrite)
    pd.DataFrame(results).to_csv(PREDICTION_RUN_FILE, index=False)

    # Email decision (Option A)
    if len(critical_rows) > 0:
        lines = ["CRITICAL Instrument CTs Detected:\n"]
        for i, r in enumerate(critical_rows, start=1):
            lines.append(f"{i}) ID: {r['ID']}")
            lines.append(f"   Health: {r['health']:.2f}")
            lines.append(f"   FaultProb: {r['fault_prob']:.3f}")
            lines.append(f"   RiskLevel: {r['risk_level']}")
            lines.append(f"   Cause: {r['cause']}")
            lines.append("")
        body = "\n".join(lines)
        send_email_alert("⚠️ Critical Instrument CT Detected", body)
    else:
        if len(high_rows) > 0:
            last_ts = read_last_high_email_time()
            now_ts = time.time()
            if now_ts - last_ts >= 3600:
                lines = ["HIGH Instrument CTs Detected (fault==1):\n"]
                for i, r in enumerate(high_rows, start=1):
                    lines.append(f"{i}) ID: {r['ID']}")
                    lines.append(f"   Health: {r['health']:.2f}")
                    lines.append(f"   FaultProb: {r['fault_prob']:.3f}")
                    lines.append(f"   RiskLevel: {r['risk_level']}")
                    lines.append(f"   Cause: {r['cause']}")
                    lines.append("")
                body = "\n".join(lines)
                send_email_alert("⚠️ High Instrument CTs Detected", body)
                write_last_high_email_time(now_ts)
            else:
                # cooldown active — do not send
                pass

    # End (silent)
