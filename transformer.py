"""
transformer.py – FINAL VERSION
(single email + PDF summary)
"""

import os
import json
import joblib
import glob
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

# ---------------------------
# GLOBAL ALERT STORAGE
# ---------------------------
GLOBAL_ALERT_LIST = []

# ---------------------------
# EMAIL CONFIG
# ---------------------------
EMAIL_SENDER = "your_email@gmail.com"
EMAIL_PASSWORD = "your_app_password"
EMAIL_RECEIVER = "receiver_mail@gmail.com"


def send_email_summary(subject, message, pdf_file):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_SENDER
        msg['To'] = EMAIL_RECEIVER
        msg['Subject'] = subject

        msg.attach(MIMEText(message, "plain"))

        with open(pdf_file, 'rb') as f:
            attach = MIMEApplication(f.read(), _subtype="pdf")
        attach.add_header('Content-Disposition', 'attachment', filename="report.pdf")
        msg.attach(attach)

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        server.quit()

        print("📧 PDF summary email sent")

    except Exception as e:
        print("❌ Email failed:", e)


# PDF GENERATOR
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet


def make_pdf(df: pd.DataFrame, filename="summary.pdf"):

    doc = SimpleDocTemplate(filename, pagesize=A4)
    styles = getSampleStyleSheet()

    elems = []
    elems.append(Paragraph("⚠ Transformer Summary Report", styles["Heading1"]))

    table_data = [df.columns.tolist()] + df.values.tolist()

    elems.append(Table(table_data))

    doc.build(elems)

    return filename


INPUT_FIELDS = [
    "TCTR.WindingTemp",
    "TCTR.OilTemp",
    "TCTR.Load",
    "TCTR.Tap",
    "DGA.H2",
    "DGA.C2H2",
    "TCTR.OilLevel",
    "DGA.Moisture",
    "TCTR.Buchholz",
    "TCTR.Cooling"
]

TRAIN_TARGETS = ["health_score", "fault_flag"]

MODEL_DIR = "models/transformer"
os.makedirs(MODEL_DIR, exist_ok=True)



def safe_get_value(row, key, default=0.0):
    try:
        v = row.get(key, default)
        if v is None: return default
        v = float(v)
        if np.isnan(v): return default
        return v
    except:
        return default



def validate_training(df):
    req = INPUT_FIELDS + TRAIN_TARGETS
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(miss)


def numeric_feature_columns(df: pd.DataFrame):
    n = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in INPUT_FIELDS if c in n]



def train(df: pd.DataFrame):

    validate_training(df)
    df = df.dropna()

    feats = numeric_feature_columns(df)

    X = df[feats].values
    y_reg = df["health_score"].values
    y_clf = df["fault_flag"].astype(int).values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    Xtr, Xv, ytr_r, yv_r, ytr_c, yv_c = train_test_split(
        Xs, y_reg, y_clf, test_size=0.2, random_state=42
    )

    reg = RandomForestRegressor(n_estimators=150)
    clf = RandomForestClassifier(n_estimators=150, class_weight="balanced_subsample")
    iso = IsolationForest(n_estimators=200, contamination=0.03)

    reg.fit(Xtr, ytr_r)
    clf.fit(Xtr, ytr_c)
    iso.fit(Xs)

    joblib.dump(reg, f"{MODEL_DIR}/reg.joblib")
    joblib.dump(clf, f"{MODEL_DIR}/clf.joblib")
    joblib.dump(iso, f"{MODEL_DIR}/iso.joblib")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.joblib")
    joblib.dump(feats, f"{MODEL_DIR}/features.joblib")

    return {
        "mse": float(mean_squared_error(yv_r, reg.predict(Xv))),
        "acc": float(accuracy_score(yv_c, clf.predict(Xv)))
    }



def load_models_safe():
    try:
        return (
            joblib.load(f"{MODEL_DIR}/reg.joblib"),
            joblib.load(f"{MODEL_DIR}/clf.joblib"),
            joblib.load(f"{MODEL_DIR}/iso.joblib"),
            joblib.load(f"{MODEL_DIR}/scaler.joblib"),
            joblib.load(f"{MODEL_DIR}/features.joblib"),
            True
        )
    except:
        return None, None, None, None, INPUT_FIELDS, False



def fallback_health(r):
    return 50.0

def fallback_fault(r):
    return 0, 0.01

def fallback_anom(r):
    return 0, 0.0



def compute_risk(h,p,a):
    r = (1 - h/100)*60 + p*40 + (15 if a else 0)
    if r<40: return r,"LOW"
    if r<70: return r,"MED"
    if r<90: return r,"HIGH"
    return r,"CRITICAL"



def predict_single(row):

    reg, clf, iso, scaler, feats, ok = load_models_safe()

    X=np.array([[safe_get_value(row,f) for f in feats]])

    try: Xs=scaler.transform(X)
    except: Xs=X

    try:     h=float(reg.predict(Xs)[0])
    except:  h=fallback_health(row)

    try:
        f=int(clf.predict(Xs)[0])
        p=float(clf.predict_proba(Xs)[:,1][0])
    except:
        f,p=fallback_fault(row)

    try:
        af = int(iso.predict(Xs)[0] == -1)
        ascore=float(iso.decision_function(Xs)[0])
    except:
        af,ascore=fallback_anom(row)


    risk,level=compute_risk(h,p,af)

    # collect for final summary
    if level in ["HIGH","CRITICAL"]:
        GLOBAL_ALERT_LIST.append({
            "health":h,
            "fault":f,
            "fault_prob":p,
            "anom_flag":af,
            "risk":risk,
            "risk_level":level
        })

    return{
        "health":h,
        "fault":f,
        "fault_prob":p,
        "anom_flag":af,
        "risk":risk,
        "risk_level":level
    }


# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":

    BASE=r"C:\SIH"

    train_file=f"{BASE}\\transformer_train_with_labels_5000.csv"
    test_file =f"{BASE}\\transformer_test_unlabeled_1000_with_faults.csv"

    print("loading train…")
    df=pd.read_csv(train_file)

    meta=train(df)
    print(meta)

    test=pd.read_csv(test_file)

    results=[predict_single(r.to_dict()) for _,r in test.iterrows()]

    OUT=f"{BASE}\\output"
    os.makedirs(OUT,exist_ok=True)

    out_csv=f"{OUT}\\prediction.csv"
    pd.DataFrame(results).to_csv(out_csv,index=False)

    print("saved csv",out_csv)

    #  SEND ONE SUMMARY EMAIL
    if len(GLOBAL_ALERT_LIST)>0:

        table=pd.DataFrame(GLOBAL_ALERT_LIST)

        pdf_path = make_pdf(table, filename=f"{OUT}\\summary.pdf")

        msg = f"Total risky units: {len(table)}\n\n"
        msg += table.to_string()

        send_email_summary(
            subject="⚠ TRANSFORMER SUMMARY REPORT",
            message=msg,
            pdf_file=pdf_path
        )

    else:
        print("No risky transformers. No email sent.")
