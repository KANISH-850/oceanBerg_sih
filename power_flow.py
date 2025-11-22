import os
import json
import joblib
import glob
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

# --------------------------
# Email config loader (unchanged)
# --------------------------
CONFIG_FILE = "email_config.json"

def load_email_config():
    if not os.path.exists(CONFIG_FILE):
        print("Email config file not found. Email alerts disabled.")
        return None

    try:
        with open(CONFIG_FILE, "r") as f:
            cfg = json.load(f)

        required = ["EMAIL_SENDER", "EMAIL_PASSWORD", "EMAIL_RECEIVER"]
        for key in required:
            if key not in cfg:
                print(f"Missing '{key}' in email_config.json. Email disabled.")
                return None

        return cfg

    except Exception as e:
        print("Failed to load email config:", e)
        return None

email_cfg = load_email_config()

if email_cfg:
    EMAIL_SENDER = email_cfg["EMAIL_SENDER"]
    EMAIL_PASSWORD = email_cfg["EMAIL_PASSWORD"]
    EMAIL_RECEIVER = email_cfg["EMAIL_RECEIVER"]
    ENABLE_EMAIL_ALERTS = True
else:
    EMAIL_SENDER = None
    EMAIL_PASSWORD = None
    EMAIL_RECEIVER = None
    ENABLE_EMAIL_ALERTS = False

def send_email_alert(subject: str, message: str):
    if not ENABLE_EMAIL_ALERTS:
        print("Email alerts disabled.")
        return

    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_SENDER
        msg["To"] = EMAIL_RECEIVER
        msg["Subject"] = subject
        msg.attach(MIMEText(message, "plain"))

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        server.quit()

        print("Email alert sent to:", EMAIL_RECEIVER)

    except Exception as e:
        print("Email failed:", e)

# --------------------------
# Model / I/O paths
# --------------------------
MODEL_DIR = "models/powerflow"
os.makedirs(MODEL_DIR, exist_ok=True)

# NOTE: these INPUT_FIELDS match the CSV columns created previously.
INPUT_FIELDS = [
    "BusVoltage_kV", "LineCurrent_A", "MW", "MVAR",
    "PowerFactor", "Frequency_Hz", "VoltageAngle_deg", "CurrentAngle_deg",
    "ROCOF_Hz_per_s", "THD_percent"
]

# Updated targets for power-flow problem
TRAIN_TARGETS = ["stability_index", "violation_flag"]

# --------------------------
# Helpers
# --------------------------
def safe_get_value(row: Dict[str, Any], key: str, default=0.0):
    try:
        val = row.get(key, default)
        if val is None:
            return default
        # allow numeric strings with commas etc.
        return float(val)
    except:
        return default

def validate_training(df: pd.DataFrame):
    required = INPUT_FIELDS + TRAIN_TARGETS
    missing = [f for f in required if f not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

def numeric_feature_columns(df: pd.DataFrame):
    # keep only numeric columns from INPUT_FIELDS
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feats = [c for c in INPUT_FIELDS if c in num_cols]
    return feats

# --------------------------
# Training function
# --------------------------
def train(df, random_state=42, test_size=0.2, iso_contamination=0.02):
    validate_training(df)
    df = df.dropna(subset=INPUT_FIELDS + TRAIN_TARGETS)

    feats = numeric_feature_columns(df)
    if not feats:
        raise ValueError("No numeric features found for training.")

    X = df[feats].values
    y_reg = df["stability_index"].values
    y_clf = df["violation_flag"].astype(int).values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    Xtr, Xv, ytr_reg, yv_reg, ytr_clf, yv_clf = train_test_split(
        Xs, y_reg, y_clf, test_size=test_size, random_state=random_state
    )

    reg = RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)
    clf = RandomForestClassifier(
        n_estimators=200, random_state=random_state,
        class_weight="balanced_subsample", n_jobs=-1
    )
    iso = IsolationForest(
        n_estimators=200, contamination=iso_contamination, random_state=random_state
    )

    reg.fit(Xtr, ytr_reg)
    clf.fit(Xtr, ytr_clf)
    iso.fit(Xs)

    mse = float(mean_squared_error(yv_reg, reg.predict(Xv)))
    acc = float(accuracy_score(yv_clf, clf.predict(Xv)))
    crep = classification_report(yv_clf, clf.predict(Xv), zero_division=0)

    joblib.dump(reg, os.path.join(MODEL_DIR, "reg.joblib"))
    joblib.dump(clf, os.path.join(MODEL_DIR, "clf.joblib"))
    joblib.dump(iso, os.path.join(MODEL_DIR, "iso.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    joblib.dump(feats, os.path.join(MODEL_DIR, "features.joblib"))

    return {"mse": mse, "acc": acc, "report": crep}

def load_models_safe():
    try:
        reg = joblib.load(os.path.join(MODEL_DIR, "reg.joblib"))
        clf = joblib.load(os.path.join(MODEL_DIR, "clf.joblib"))
        iso = joblib.load(os.path.join(MODEL_DIR, "iso.joblib"))
        scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
        feats = joblib.load(os.path.join(MODEL_DIR, "features.joblib"))
        return reg, clf, iso, scaler, feats, True
    except Exception as e:
        print("Model load failed. Using fallback rules. Details:", e)
        return None, None, None, None, INPUT_FIELDS, False

# --------------------------
# Fallback rules (power-flow specific)
# --------------------------
def fallback_stability(row):
    v = safe_get_value(row, "BusVoltage_kV", 230)
    freq = safe_get_value(row, "Frequency_Hz", 50.0)
    thd = safe_get_value(row, "THD_percent", 1.0)
    # simple heuristic: stability drops with voltage deviation, freq deviation, and THD
    stability = 100 - (abs(v - 230) * 0.08) - (abs(freq - 50.0) * 8.0) - (thd * 1.2)
    return float(np.clip(stability, 0, 100))

def fallback_violation(row):
    # heuristic probability of violation
    v = safe_get_value(row, "BusVoltage_kV", 230)
    thd = safe_get_value(row, "THD_percent", 1.0)
    current = safe_get_value(row, "LineCurrent_A", 0.0)
    freq = safe_get_value(row, "Frequency_Hz", 50.0)
    prob = min(max((abs(v - 230) / 200) + (thd / 20) + (current / 2000) + (abs(freq - 50.0) * 0.5), 0.0), 1.0)
    flag = int(prob > 0.5)
    return flag, float(prob)

def fallback_anomaly(row):
    rocof = safe_get_value(row, "ROCOF_Hz_per_s", 0.0)
    thd = safe_get_value(row, "THD_percent", 0.0)
    # Anomaly if ROCOF large or THD extremely high
    flag = int(abs(rocof) > 3.0 or thd > 20.0)
    score = -0.8 if flag else 0.2
    return flag, float(score)

# --------------------------
# Risk computation & predict_single
# --------------------------
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

def predict_single(row: Dict[str, Any]):
    reg, clf, iso, scaler, feats, model_ok = load_models_safe()

    # build input vector using features saved during training (feats)
    X = np.array([[safe_get_value(row, f) for f in feats]])

    try:
        Xs = scaler.transform(X) if model_ok else X
    except Exception:
        Xs = X

    # stability index (regression)
    try:
        stability = float(reg.predict(Xs)[0]) if model_ok else fallback_stability(row)
    except Exception:
        stability = fallback_stability(row)
    stability = float(np.clip(stability, 0, 100))

    # violation flag & probability (classification)
    try:
        violation = int(clf.predict(Xs)[0]) if model_ok else 0
        violation_prob = float(clf.predict_proba(Xs)[:, 1][0]) if model_ok else 0.0
    except Exception:
        violation, violation_prob = fallback_violation(row)

    # anomaly detection
    try:
        anom_flag = int(iso.predict(Xs)[0] == -1) if model_ok else 0
        anom_score = float(iso.decision_function(Xs)[0]) if model_ok else 0.0
    except Exception:
        anom_flag, anom_score = fallback_anomaly(row)

    # reuse compute_risk to produce a simple "health-like" risk score
    risk, level = compute_risk(stability, violation_prob, anom_flag)

    # Compose alert message if needed
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fault_details = ""
    if violation == 1:
        fault_details += "Power flow violation detected.\n"
    if anom_flag == 1:
        fault_details += "Anomaly detected (ROCOF/THD). \n"
    if level == "HIGH":
        fault_details += "High risk operating condition.\n"
    if level == "CRITICAL":
        fault_details += "Critical risk. Immediate action recommended.\n"
    if fault_details == "":
        fault_details = "No specific fault category identified.\n"

    alert_msg = None
    if violation == 1 or level in ["HIGH", "CRITICAL"]:
        alert_msg = (
            f"Power Flow Fault Alert\n\n"
            f"Timestamp: {timestamp}\n"
            f"Severity: {level}\n\n"
            f"Stability Index: {stability:.2f}\n"
            f"Violation Probability: {violation_prob:.2f}\n"
            f"Anomaly Flag: {anom_flag}\n"
            f"Risk Score: {risk:.2f}\n\n"
            f"Details:\n{fault_details}"
        )
        print("ALERT TRIGGERED")
        print(alert_msg)
        send_email_alert("Power Flow Fault Alert", alert_msg)

    else:
        alert_msg = "No fault detected."

    return {
        "stability_index": stability,
        "violation_flag": int(violation),
        "violation_prob": float(violation_prob),
        "anom_flag": int(anom_flag),
        "anom_score": float(anom_score),
        "risk": float(risk),
        "risk_level": level,
        "alert_message": alert_msg
    }

# --------------------------
# MAIN
# --------------------------
if __name__ == "__main__":

    BASE_PATH = r"C:\SIH"   # change if needed

    train_file = os.path.join(BASE_PATH, "powerflow_train_5000.csv")
    test_file = os.path.join(BASE_PATH, "powerflow_test_1000.csv")

    print("Loading training dataset:", train_file)
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file not found: {train_file}")

    df = pd.read_csv(train_file)

    print("Training models...")
    meta = train(df)
    print(json.dumps(meta, indent=2))

    print("Loading test dataset:", test_file)
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")
    test_df = pd.read_csv(test_file)

    print("Running predictions...")
    results = []
    for _, row in test_df.iterrows():
        out = predict_single(row.to_dict())
        results.append(out)

    OUTPUT = os.path.join(BASE_PATH, "output")
    os.makedirs(OUTPUT, exist_ok=True)

    out_path = os.path.join(OUTPUT, "power_prediction.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)

    print("Predictions saved:", out_path)
