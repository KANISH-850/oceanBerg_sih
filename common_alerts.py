# common_alerts.py
import os
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

CONFIG_FILE = "email_config.json"

def load_email_config():
    if not os.path.exists(CONFIG_FILE):
        print("⚠️ Email config file not found. Email alerts disabled.")
        return None
    try:
        with open(CONFIG_FILE, "r") as f:
            cfg = json.load(f)
        required = ["EMAIL_SENDER", "EMAIL_PASSWORD", "EMAIL_RECEIVER"]
        for key in required:
            if key not in cfg:
                print(f"⚠️ Missing '{key}' in {CONFIG_FILE}. Email disabled.")
                return None
        return cfg
    except Exception as e:
        print("⚠️ Failed to load email config:", e)
        return None

_email_cfg = load_email_config()
if _email_cfg:
    EMAIL_SENDER = _email_cfg["EMAIL_SENDER"]
    EMAIL_PASSWORD = _email_cfg["EMAIL_PASSWORD"]
    EMAIL_RECEIVER = _email_cfg["EMAIL_RECEIVER"]
    ENABLE_EMAIL = True
else:
    EMAIL_SENDER = EMAIL_PASSWORD = EMAIL_RECEIVER = None
    ENABLE_EMAIL = False

def send_email_alert(subject: str, body: str):
    """
    Send a plain-text email if enabled. Exceptions are caught and printed.
    """
    if not ENABLE_EMAIL:
        print("⚠️ Email disabled. Not sending:", subject)
        return
    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_SENDER
        msg["To"] = EMAIL_RECEIVER
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        server.quit()
        print("📧 Email sent to", EMAIL_RECEIVER, "| Subject:", subject)
    except Exception as e:
        print("❌ Email failed:", e)

def compute_risk_level(health_score: float) -> str:
    """
    Unified risk engine: returns "LOW", "MEDIUM" or "HIGH".
    (unchanged from plan)
    """
    try:
        h = float(health_score)
    except Exception:
        # If health_score invalid, be conservative and return HIGH
        return "HIGH"
    if h >= 80:
        return "LOW"
    if 50 <= h < 80:
        return "MEDIUM"
    return "HIGH"

def make_high_risk_email(component_name: str, asset_id: str, details: dict) -> (str, str):
    """
    Build subject and body for HIGH risk email.
    details: dictionary with keys like health_score, fault_prob, fault_flag, timestamp, extras...
    """
    subj = f"⚠ HIGH RISK ALERT: {component_name} - {asset_id}"
    body_lines = [
        f"HIGH RISK ALERT - {component_name}",
        f"Asset ID      : {asset_id}",
        f"Timestamp     : {details.get('timestamp', str(datetime.now()))}",
        f"Health Score  : {details.get('health_score', 'N/A')}",
        f"Fault Prob    : {details.get('fault_prob', 'N/A')}",
        f"Fault Flag    : {details.get('fault_flag', 'N/A')}",
        f"Risk Level    : {details.get('risk_level', 'HIGH')}",
        "",
        "Detailed telemetry / suggestion:",
        details.get('message', 'Inspect asset immediately.'),
        "",
        "This is an automated alert."
    ]
    return subj, "\n".join(body_lines)
