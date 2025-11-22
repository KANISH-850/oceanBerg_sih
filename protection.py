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


CONFIG_FILE = "email_config.json"


# ===================== EMAIL CONFIG ======================

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
        return  # silent

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

    except:
        pass  # silent


# ========================== MODEL CONSTANTS ===============================

MODEL_DIR = "models/protection"
os.makedirs(MODEL_DIR, exist_ok=True)

INPUT_FIELDS = [
    "Relay.PickupCurrent", "Relay.TripTime", "Relay.Zone",
    "CTRatioError_pct", "PhaseImbalance_pct",
    "Relay.OperCount", "Relay.HealthSensor", "IO_Status",
    "GroundFaultCurrent", "ResidualCurrent"
]

TRAIN_TARGETS = ["protection_health", "trip_flag"]


# ========================== SAFE GET ===============================

def safe_get(row, key, default=0.0):
    try:
        v = row.get(key, default)
        if v is None:
            return default
        v = float(v)
        if np.isnan(v):
            return default
        return v
    except:
        return default


# ========================== TRAINING ===============================

def validate_training(df):
    missing = [c for c in INPUT_FIELDS + TRAIN_TARGETS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def numeric_columns(df):
    nums = df.select_dtypes(include=[np.number]).columns.tolist()
    feats = [c for c in INPUT_FIELDS if c in nums]
    return feats


def train(df, test_size=0.2, random_state=42, iso_contam=0.03):
    validate_training(df)
    df = df.dropna(subset=INPUT_FIELDS + TRAIN_TARGETS)

    feats = numeric_columns(df)
    if not feats:
        raise ValueError("No numeric features found.")

    X = df[feats].values
    y_reg = df["protection_health"].values
    y_clf = df["trip_flag"].astype(int).values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    Xtr, Xv, ytr_reg, yv_reg, ytr_clf, yv_clf = train_test_split(
        Xs, y_reg, y_clf, test_size=test_size, random_state=random_state
    )

    reg = RandomForestRegressor(n_estimators=150, random_state=random_state, n_jobs=-1)
    clf = RandomForestClassifier(
        n_estimators=150,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1
    )
    iso = IsolationForest(
        n_estimators=200,
        contamination=iso_contam,
        random_state=random_state
    )

    reg.fit(Xtr, ytr_reg)
    clf.fit(Xtr, ytr_clf)
    iso.fit(Xs)

    # ======== PRINT TRAINING METRICS EXACT FORMAT ========
    print("\nTRAINING...")
    mse = mean_squared_error(yv_reg, reg.predict(Xv))
    acc = accuracy_score(yv_clf, clf.predict(Xv))
    rep = classification_report(yv_clf, clf.predict(Xv), zero_division=0)
    print("MSE:", mse)
    print("Accuracy:", acc)
    print(rep)

    # save
    joblib.dump(reg, os.path.join(MODEL_DIR, "reg.joblib"))
    joblib.dump(clf, os.path.join(MODEL_DIR, "clf.joblib"))
    joblib.dump(iso, os.path.join(MODEL_DIR, "iso.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    joblib.dump(feats, os.path.join(MODEL_DIR, "features.joblib"))

    return {"mse": float(mse), "accuracy": float(acc), "report": rep}


# ========================== LOAD MODELS ===============================

def load_models():
    try:
        reg = joblib.load(os.path.join(MODEL_DIR, "reg.joblib"))
        clf = joblib.load(os.path.join(MODEL_DIR, "clf.joblib"))
        iso = joblib.load(os.path.join(MODEL_DIR, "iso.joblib"))
        scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
        feats = joblib.load(os.path.join(MODEL_DIR, "features.joblib"))
        return reg, clf, iso, scaler, feats, True
    except:
        return None, None, None, None, INPUT_FIELDS, False


# ========================== FALLBACKS ===============================

def fallback_health(row):
    err = safe_get(row, "CTRatioError_pct", 2)
    imb = safe_get(row, "PhaseImbalance_pct", 1)
    return float(np.clip(100 - (err * 2 + imb * 1.5), 5, 100))


def fallback_trip(row):
    gf = safe_get(row, "GroundFaultCurrent", 20)
    res = safe_get(row, "ResidualCurrent", 10)
    p = min((gf / 200) + (res / 150), 1)
    return int(p > 0.5), p


def fallback_anomaly(row):
    gf = safe_get(row, "GroundFaultCurrent", 20)
    return int(gf > 200), -0.8


# ========================== RISK ===============================

def compute_risk(health, trip_prob, anom):
    risk = (1 - health / 100) * 60 + trip_prob * 40
    if anom:
        risk += 15
    risk = float(np.clip(risk, 0, 100))

    if risk <= 40:
        return risk, "LOW"
    if risk <= 70:
        return risk, "MEDIUM"
    if risk <= 90:
        return risk, "HIGH"
    return risk, "CRITICAL"


# ========================== PREDICT ===============================

def predict_single(row):
    reg, clf, iso, scaler, feats, ok = load_models()

    X = np.array([[safe_get(row, f) for f in feats]])

    try:
        Xs = scaler.transform(X) if ok else X
    except:
        Xs = X

    try:
        health = float(reg.predict(Xs)[0])
    except:
        health = fallback_health(row)

    try:
        trip = int(clf.predict(Xs)[0])
        trip_prob = float(clf.predict_proba(Xs)[:, 1][0])
    except:
        trip, trip_prob = fallback_trip(row)

    try:
        anom = int(iso.predict(Xs)[0] == -1)
        anom_score = float(iso.decision_function(Xs)[0])
    except:
        anom, anom_score = fallback_anomaly(row)

    risk, level = compute_risk(health, trip_prob, anom)

    relay_id = row.get("RelayID", "UNKNOWN")
    zone = row.get("Relay.Zone", "UNKNOWN")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    fault_msg = ""
    if trip == 1:
        fault_msg += "→ TRIP condition detected\n"
    if anom == 1:
        fault_msg += "→ Sensor abnormality\n"
    if level == "HIGH":
        fault_msg += "→ HIGH risk operation\n"
    if level == "CRITICAL":
        fault_msg += "→ CRITICAL – immediate inspection required\n"
    if fault_msg == "":
        fault_msg = "No fault conditions detected.\n"

    # EMAIL ONLY WHEN HIGH/CRITICAL/TRIP — NO PRINT
    if trip == 1 or level in ["HIGH", "CRITICAL"]:
        alert = (
            f"⚠ PROTECTION RELAY ALERT ⚠\n\n"
            f"Relay ID: {relay_id}\n"
            f"Zone: {zone}\n"
            f"Timestamp: {ts}\n"
            f"Severity: {level}\n\n"
            f"Health Score: {health:.2f}\n"
            f"Trip Probability: {trip_prob:.2f}\n"
            f"Anomaly: {anom}\n"
            f"Risk Score: {risk:.2f}\n\n"
            f"{fault_msg}"
        )
        send_email_alert("Protection Relay Trip Alert", alert)

    return {
        "health": health,
        "trip_flag": trip,
        "trip_prob": trip_prob,
        "anom_score": anom_score,
        "anom_flag": anom,
        "risk": risk,
        "risk_level": level,
        "details": fault_msg
    }


# ========================== MAIN ===============================

if __name__ == "__main__":

    BASE = r"C:\SIH"

    train_file = os.path.join(BASE, "protection_train_with_labels.csv")
    test_file = os.path.join(BASE, "protection_test_unlabeled.csv")

    print("Loading training dataset...")
    df = pd.read_csv(train_file)

    print("Training protection model...")
    meta = train(df)  # Already prints metrics

    print("Loading test dataset...")
    test_df = pd.read_csv(test_file)

    print("Running predictions...")
    results = [predict_single(r.to_dict()) for _, r in test_df.iterrows()]

    OUT_DIR = os.path.join(BASE, "output")
    os.makedirs(OUT_DIR, exist_ok=True)

    out_file = os.path.join(OUT_DIR, "protection_predictions.csv")
    pd.DataFrame(results).to_csv(out_file, index=False)

    print("Saved:", out_file)
