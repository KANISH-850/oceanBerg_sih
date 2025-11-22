import smtplib
from email.mime.text import MIMEText

# UPDATE THESE 3 VALUES
SENDER_EMAIL = "your_email@gmail.com"
APP_PASSWORD = "your16digitapppassword"
RECEIVER_EMAIL = "receiver_email@gmail.com"


class SubstationModel:

    def __init__(self):
        self.fault = False
        self.status = "UNKNOWN"

    def power_flow(self):
        print("Power Flow: Power is flowing into the system.")
        return True

    def transformer(self):
        print("Transformer: Voltage level adjusted.")
        return True

    def busbar(self):
        print("Busbar: Power distributed to multiple circuits.")
        return True

    def protection(self):
        if self.fault:
            print("Protection: Relay activated, sending trip signal.")
            return False
        print("Protection: System normal.")
        return True

    def circuit_breaker(self, protection_status):
        if not protection_status:
            print("Circuit Breaker: Fault detected! Breaker tripped.")
            return False
        print("Circuit Breaker: Circuit closed, power allowed.")
        return True

    def isolator(self, breaker_status):
        if not breaker_status:
            print("Isolator: System isolated safely.")
            return True
        print("Isolator: Cannot isolate while breaker is ON.")
        return False

    def predict_system_status(self, protection_status, breaker_status):
        if not protection_status or not breaker_status:
            self.status = "FAULT_DETECTED"
        else:
            self.status = "NORMAL_OPERATION"
        return self.status

    def send_email(self):
        if self.status != "FAULT_DETECTED":
            return  # Only send on fault

        msg = MIMEText(
            "⚠️ Fault detected in the substation system.\nImmediate attention required!"
        )
        msg["Subject"] = "⚠️ Substation Fault Alert"
        msg["From"] = SENDER_EMAIL
        msg["To"] = RECEIVER_EMAIL

        try:
            print("\nSending fault email...")
            server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
            server.login(SENDER_EMAIL, APP_PASSWORD)
            server.send_message(msg)
            server.quit()
            print("EMAIL SENT SUCCESSFULLY!")
        except Exception as e:
            print("Email sending failed:", e)

    def run(self, fault=False):
        self.fault = fault

        print("\n--- Substation Combined Model ---\n")

        pf = self.power_flow()
        tr = self.transformer()
        bb = self.busbar()
        prot = self.protection()
        cb = self.circuit_breaker(protection_status=prot)
        iso = self.isolator(breaker_status=cb)

        prediction = self.predict_system_status(prot, cb)
        print(f"Prediction Output: {prediction}")

        if prediction == "FAULT_DETECTED":
            self.send_email()

        print("\n--- Model Execution Complete ---\n")


# RUN MODEL
model = SubstationModel()
model.run(fault=False)   # Normal operation
model.run(fault=True)    # Fault → Email will be sent
