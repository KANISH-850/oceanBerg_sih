#!/usr/bin/env python3
"""
busbar.py — FINAL PRODUCTION VERSION (matches CB behaviour)
Silent prediction, silent email, smart risk + CSV logging.
"""

import os, time, json, joblib, numpy as np, pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


# ==========================================================
# PATHS
# ==========================================================
BASE = r"C:\SIH"
OUT = BASE+r"\output"

os.makedirs(OUT,exist_ok=True)

MODEL_DIR="models/busbar"
os.makedirs(MODEL_DIR,exist_ok=True)

TRAIN = BASE+r"\busbar_train_5000.csv"
TEST  = BASE+r"\busbar_test_1000.csv"

PRED_CSV = OUT+r"\busbar_predictions.csv"
ALL_ALERTS = OUT+r"\busbar_alerts_all.csv"
HIGH_CSV = OUT+r"\busbar_alerts_high.csv"
LAST_HIGH = OUT+r"\busbar_last_high.json"

# ==========================================================
# FEATURES
# ==========================================================
FEATS = ["BRDM.Diff_A","PhaseImbalance_pct","CTRatioError_pct"]

T_H = "bus_health_index"
T_F = "bus_fault_flag"
T_P = "bus_fault_prob"


# ==========================================================
# EMAIL CONFIG
# ==========================================================
cfg=None
if os.path.exists("email_config.json"):
    cfg=json.load(open("email_config.json"))
    
if cfg:
    E_SEND=cfg["EMAIL_SENDER"]
    E_PASS=cfg["EMAIL_PASSWORD"]
    E_RECV=cfg["EMAIL_RECEIVER"]
    EMAIL_ON=True
else:
    EMAIL_ON=False


def send_mail(sub,body):
    if not EMAIL_ON: return
    try:
        m=MIMEMultipart()
        m["From"]=E_SEND
        m["To"]=E_RECV
        m["Subject"]=sub
        m.attach(MIMEText(body,"plain"))

        s=smtplib.SMTP("smtp.gmail.com",587)
        s.starttls()
        s.login(E_SEND,E_PASS)
        s.sendmail(E_SEND,E_RECV,m.as_string())
        s.quit()
    except:
        pass


# ==========================================================
# TRAIN
# ==========================================================
def train(df):
    X=df[FEATS].values
    yh=df[T_H].values
    yf=df[T_F].astype(int).values
    yp=df[T_P].values
    
    from sklearn.model_selection import train_test_split
    Xtr,Xts,yht,yhs,yft,yfs,ypt,yps=train_test_split(
        X,yh,yf,yp,test_size=0.2,random_state=42
    )
    
    sc=StandardScaler()
    Xtr=sc.fit_transform(Xtr)
    Xts=sc.transform(Xts)
    
    reg=RandomForestRegressor(n_estimators=160,n_jobs=-1)
    clf=RandomForestClassifier(n_estimators=160,n_jobs=-1,class_weight="balanced")
    regp=RandomForestRegressor(n_estimators=120,n_jobs=-1)
    iso=IsolationForest(contamination=0.03)
    
    reg.fit(Xtr,yht)
    clf.fit(Xtr,yft)
    regp.fit(Xtr,ypt)
    iso.fit(sc.transform(df[FEATS].values))
    
    joblib.dump(reg,f"{MODEL_DIR}/health.pkl")
    joblib.dump(clf,f"{MODEL_DIR}/fault.pkl")
    joblib.dump(regp,f"{MODEL_DIR}/prob.pkl")
    joblib.dump(iso,f"{MODEL_DIR}/iso.pkl")
    joblib.dump(sc,f"{MODEL_DIR}/scaler.pkl")
    
    mse=mean_squared_error(yhs,reg.predict(Xts))
    acc=accuracy_score(yfs,clf.predict(Xts))
    rep=classification_report(yfs,clf.predict(Xts),zero_division=0)
    
    print("\nTRAINING...")
    print("MSE:",mse)
    print("Accuracy:",acc)
    print(rep)


# ==========================================================
# LOAD
# ==========================================================
def load():
    return (
        joblib.load(f"{MODEL_DIR}/health.pkl"),
        joblib.load(f"{MODEL_DIR}/fault.pkl"),
        joblib.load(f"{MODEL_DIR}/prob.pkl"),
        joblib.load(f"{MODEL_DIR}/iso.pkl"),
        joblib.load(f"{MODEL_DIR}/scaler.pkl")
    )

# ==========================================================
# RISK
# ==========================================================
def risk(h,p,a):
    r=(100-h)*0.55 + p*100*0.35 + (15 if a else 0)
    r=min(max(r,0),100)
    if r<=40: return r,"LOW"
    if r<=70: return r,"MEDIUM"
    if r<=90: return r,"HIGH"
    return r,"CRITICAL"


# ==========================================================
# EMAIL COOLDOWN
# ==========================================================
def last():
    if not os.path.exists(LAST_HIGH): return 0
    try:return float(json.load(open(LAST_HIGH))["ts"])
    except:return 0

def set_time():
    try:json.dump({"ts":time.time()},open(LAST_HIGH,"w"))
    except:pass


# ==========================================================
# PREDICT
# ==========================================================
def pred(row,mdl):
    reg,clf,regp,iso,sc=mdl
    
    X=np.array([[float(row[c]) for c in FEATS]])
    Xs=sc.transform(X)
    
    h=float(reg.predict(Xs))
    f=int(clf.predict(Xs))
    p=float(regp.predict(Xs))
    
    a=int(iso.predict(Xs)[0]==-1)
    r,lev=risk(h,p,a)
    
    d={
        "ID":row.get("ID","BUS_UNKNOWN"),
        "Time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Health":h,
        "Fault":f,
        "Prob":p,
        "Anom":a,
        "Risk":r,
        "Level":lev
    }
    
    append(ALL_ALERTS,d)
    if lev in ["HIGH","CRITICAL"]:
        append(HIGH_CSV,d)
    return d


def append(path,row):
    header=not os.path.exists(path)
    pd.DataFrame([row]).to_csv(path,mode="a",header=header,index=False)


# ==========================================================
# MAIN
# ==========================================================
if __name__=="__main__":
    
    df=pd.read_csv(TRAIN)
    train(df)
    
    mdl=load()
    
    res=[]
    high=[]
    crit=[]
    
    for _,r in pd.read_csv(TEST).iterrows():
        o=pred(r,mdl)
        res.append(o)
        if o["Level"]=="CRITICAL":crit.append(o)
        elif o["Level"]=="HIGH":high.append(o)
    
    pd.DataFrame(res).to_csv(PRED_CSV,index=False)
    
    if crit:
        txt="\n".join(f"{x['ID']} | H={x['Health']:.1f} P={x['Prob']:.2f} L={x['Level']}" for x in crit)
        send_mail("CRITICAL - BUSBAR",txt)
    
    else:
        if high and time.time()-last()>3600:
            txt="\n".join(f"{x['ID']} | H={x['Health']:.1f} P={x['Prob']:.2f} L={x['Level']}" for x in high)
            send_mail("HIGH - BUSBAR",txt)
            set_time()
