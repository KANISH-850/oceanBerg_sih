import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.preprocessing import StandardScaler
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime


# =====================================
# EMAIL CONFIG
# =====================================
CONFIG_FILE = "email_config.json"

def load_email_config():
    if not os.path.exists(CONFIG_FILE):
        print("⚠ Email config missing. Email disabled.")
        return None

    with open(CONFIG_FILE, "r") as f:
        cfg = json.load(f)

    required = ["EMAIL_SENDER", "EMAIL_PASSWORD", "EMAIL_RECEIVER"]
    for k in required:
        if k not in cfg:
            print(f"⚠ '{k}' missing in config.")
            return None

    return cfg


cfg = load_email_config()
if cfg:
    EMAIL_SENDER = cfg["EMAIL_SENDER"]
    EMAIL_PASSWORD = cfg["EMAIL_PASSWORD"]
    EMAIL_RECEIVER = cfg["EMAIL_RECEIVER"]
    ENABLE_EMAIL = True
else:
    ENABLE_EMAIL = False


def send_email(subject, message):
    if not ENABLE_EMAIL:
        print("⚠ Email disabled.")
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

        print("📧 Email sent.")
    except Exception as e:
        print("❌ Email failed:", e)


# =====================================
# MODEL FOLDER
# =====================================
MODEL_DIR = "models/cb_model"
os.makedirs(MODEL_DIR, exist_ok=True)


# =====================================
# FEATURE SETS
# =====================================
INPUT_FEATURES = ["OperationTime_ms", "SF6_Density_bar"]
TRAIN_TARGETS = ["breaker_health_index", "breaker_fault_flag", "breaker_fault_prob"]


# =====================================
# SAFE VALUE
# =====================================
def safe(row, key, default=0):
    try:
        return float(row.get(key, default))
    except:
        return default


# =====================================
# TRAINING FUNCTION
# =====================================
def train(df):
    print("Training rows:", len(df))

    X = df[INPUT_FEATURES].values

    y_health = df["breaker_health_index"].values
    y_fault = df["breaker_fault_flag"].astype(int).values
    y_prob = df["breaker_fault_prob"].values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    Xtr, Xts, yh_tr, yh_ts, yf_tr, yf_ts, yp_tr, yp_ts = train_test_split(
        Xs, y_health, y_fault, y_prob, test_size=0.2, random_state=42
    )

    reg_health = RandomForestRegressor(n_estimators=150, n_jobs=-1)
    clf_fault = RandomForestClassifier(n_estimators=150, n_jobs=-1, class_weight="balanced")
    reg_prob = RandomForestRegressor(n_estimators=100, n_jobs=-1)
    iso = IsolationForest(contamination=0.03, random_state=42)

    # Fit models
    reg_health.fit(Xtr, yh_tr)
    clf_fault.fit(Xtr, yf_tr)
    reg_prob.fit(Xtr, yp_tr)
    iso.fit(Xs)

    # Save
    joblib.dump(reg_health, f"{MODEL_DIR}/health.pkl")
    joblib.dump(clf_fault, f"{MODEL_DIR}/fault.pkl")
    joblib.dump(reg_prob, f"{MODEL_DIR}/prob.pkl")
    joblib.dump(iso, f"{MODEL_DIR}/iso.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")

    print("\n=== TEST RESULTS ===")
    pred_fault = clf_fault.predict(Xts)
    acc = accuracy_score(yf_ts, pred_fault)
    print("Accuracy:", acc)
    print("\nReport:\n", classification_report(yf_ts, pred_fault))

    return acc


# =====================================
# LOADING MODELS
# =====================================
def load_models():
    reg_health = joblib.load(f"{MODEL_DIR}/health.pkl")
    clf_fault = joblib.load(f"{MODEL_DIR}/fault.pkl")
    reg_prob = joblib.load(f"{MODEL_DIR}/prob.pkl")
    iso = joblib.load(f"{MODEL_DIR}/iso.pkl")
    scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
    return reg_health, clf_fault, reg_prob, iso, scaler


# =====================================
# RISK ENGINE
# =====================================
def compute_risk(health, prob, anomaly):
    risk = (100 - health) * 0.5 + prob * 100 * 0.4 + (15 if anomaly else 0)
    risk = min(max(risk, 0), 100)

    if risk <= 40:
        return risk, "LOW"
    elif risk <= 70:
        return risk, "MEDIUM"
    elif risk <= 90:
        return risk, "HIGH"
    else:
        return risk, "CRITICAL"


# =====================================
# PREDICT SINGLE ROW
# =====================================
def predict(row):
    reg_health, clf_fault, reg_prob, iso, scaler = load_models()

    X = np.array([[safe(row, "OperationTime_ms"), safe(row, "SF6_Density_bar")]])
    Xs = scaler.transform(X)

    health = float(reg_health.predict(Xs)[0])
    fault = int(clf_fault.predict(Xs)[0])
    prob = float(reg_prob.predict(Xs)[0])

    anomaly_flag = int(iso.predict(Xs)[0] == -1)
    anomaly_score = float(iso.decision_function(Xs)[0])

    risk, risk_level = compute_risk(health, prob, anomaly_flag)

    # EMAIL CONDITIONS
    if fault == 1 or risk_level in ["HIGH", "CRITICAL"]:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        body = (
            f"⚠ Circuit Breaker Fault Detected\n\n"
            f"Time: {timestamp}\n"
            f"BreakerStatus: {row.get('BreakerStatus')}\n\n"
            f"Predictions:\n"
            f"- Health Score: {health:.2f}\n"
            f"- Fault Flag: {fault}\n"
            f"- Fault Probability: {prob:.2f}\n"
            f"- Anomaly Flag: {anomaly_flag}\n"
            f"- Risk Score: {risk:.2f}\n"
            f"- Risk Level: {risk_level}\n"
        )
        send_email("Circuit Breaker Fault Alert", body)

    return {
        "health_score": health,
        "fault_flag": fault,
        "fault_probability": prob,
        "anomaly_flag": anomaly_flag,
        "anomaly_score": anomaly_score,
        "risk_score": risk,
        "risk_level": risk_level
    }


# =====================================
# MAIN EXECUTION
# =====================================
if __name__ == "__main__":

    BASE = r"C:\SIH"

    train_file = f"{BASE}/cb_train_5000.csv"
    test_file = f"{BASE}/cb_test_1000.csv"

    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)

    train(df_train)

    print("\nRunning predictions...")

    results = []
    for _, row in df_test.iterrows():
        results.append(predict(row))

    out_path = f"{BASE}/cb_predictions.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)

    print("Predictions saved to:", out_path)
# circuit_breaker.py
import os, joblib, numpy as np, pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from common_alerts import send_email_alert, compute_risk_level, make_high_risk_email

MODEL_DIR = "breaker_models"
os.makedirs(MODEL_DIR, exist_ok=True)

INPUT_COLS = ["TripCount", "TripTime_ms", "Temperature_C"]
TARGET_HEALTH = "breaker_health_index"
TARGET_FAULT = "breaker_fault_flag"
TARGET_PROB = "breaker_fault_prob"

def train(df):
    # keep your training logic — minimal example:
    missing = [c for c in INPUT_COLS + [TARGET_HEALTH, TARGET_FAULT, TARGET_PROB] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df.dropna()
    X = df[INPUT_COLS].values
    yh = df[TARGET_HEALTH].values
    yf = df[TARGET_FAULT].values
    yp = df[TARGET_PROB].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    reg = RandomForestRegressor(n_estimators=200)
    clf = RandomForestClassifier(n_estimators=200)
    regp = RandomForestRegressor(n_estimators=200)
    reg.fit(Xs, yh)
    clf.fit(Xs, yf)
    regp.fit(Xs, yp)
    joblib.dump(reg, os.path.join(MODEL_DIR, "reg.joblib"))
    joblib.dump(clf, os.path.join(MODEL_DIR, "clf.joblib"))
    joblib.dump(regp, os.path.join(MODEL_DIR, "faultprob.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    print("✔ Breaker models saved.")

def predict_single(row, reg, clf, regp, scaler):
    X = np.array([[row[c] for c in INPUT_COLS]])
    Xs = scaler.transform(X)
    health = float(reg.predict(Xs)[0])
    fault = int(clf.predict(Xs)[0])
    fault_prob = float(regp.predict(Xs)[0])
    risk_level = compute_risk_level(health)
    asset = row.get("BreakerID", row.get("CB", "UNKNOWN"))
    if risk_level == "HIGH":
        subj, body = make_high_risk_email("CIRCUIT_BREAKER", asset, {"timestamp": str(datetime.now()), "health_score": health, "fault_prob": fault_prob, "fault_flag": fault, "risk_level": risk_level, "message": "Breaker high risk."})
        send_email_alert(subj, body)
    return {"asset_id": asset, "health_score": health, "fault_flag": fault, "fault_prob": fault_prob, "risk_level": risk_level, "timestamp": str(datetime.now())}

if __name__ == "__main__":
    BASE = r"C:\SIH"
    train_file = os.path.join(BASE, "breaker_train.csv")
    test_file = os.path.join(BASE, "breaker_test.csv")
    df = pd.read_csv(train_file)
    train(df)
    test_df = pd.read_csv(test_file)
    reg = joblib.load(os.path.join(MODEL_DIR, "reg.joblib"))
    clf = joblib.load(os.path.join(MODEL_DIR, "clf.joblib"))
    regp = joblib.load(os.path.join(MODEL_DIR, "faultprob.joblib"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    results = [predict_single(r.to_dict(), reg, clf, regp, scaler) for _, r in test_df.iterrows()]
    out_path = os.path.join(BASE, "predictions_breaker.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)
    print("✔ Predictions saved to:", out_path)
